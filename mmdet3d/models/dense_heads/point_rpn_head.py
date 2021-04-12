import numpy as np
import torch
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule
from torch import nn as nn

from mmdet3d.core import limit_period, xywhr2xyxyr
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.models import HEADS, build_loss
from mmdet.core import build_bbox_coder, multi_apply
from .anchor3d_head import Anchor3DHead
from mmdet3d.core.bbox.structures import (DepthInstance3DBoxes,
                                          LiDARInstance3DBoxes,
                                          rotation_3d_in_axis)


@HEADS.register_module()
class PointRPNHead(nn.Module):
    def __init__(self,
                 num_classes,
                 num_dir_bins,
                 input_channels,
                 train_cfg,
                 test_cfg,
                 bbox_codesize=52,
                 conv_channels=(256, 256),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 cls_loss=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_class_loss=None,
                 size_res_loss=None,
                 semantic_loss=None,
                 bbox_coder=None,
                 predict_boxes_when_training=False):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_dir_bins = num_dir_bins
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bbox_codesize = bbox_codesize
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        
        # build loss function
        self.center_loss = build_loss(center_loss)
        self.dir_res_loss = build_loss(dir_res_loss)
        self.dir_class_loss = build_loss(dir_class_loss)
        self.size_res_loss = build_loss(size_res_loss)
        self.cls_loss = build_loss(cls_loss)
        if semantic_loss is not None:
            self.semantic_loss = build_loss(semantic_loss)

        # build box coder
        self.bbox_coder = build_bbox_coder(bbox_coder)
        
        self.cls_layers = self.make_conv_layers(
                            conv_channels=conv_channels,
                            input_channels=self.input_channels,
                            output_channels=self.num_classes,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)
        self.bbox_layers = self.make_conv_layers(
                            conv_channels=conv_channels,
                            input_channels=self.input_channels,
                            output_channels=self.bbox_codesize,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)
    
    def init_weights(self):
        """Initialize weights of VoteHead."""
        pass

    @staticmethod
    def make_conv_layers(conv_channels, 
                         input_channels, 
                         output_channels,
                         conv_cfg,
                         norm_cfg,
                         act_cfg):
        # prev_channels = input_channels
        prev_channels = 256
        conv_list = list()
        for k in range(len(conv_channels)):
            conv_list.append(
                ConvModule(
                    prev_channels,
                    conv_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True))
            prev_channels = conv_channels[k]
        conv_list.append(
            ConvModule(
                prev_channels,
                output_channels,
                1,
                padding=0,
                conv_cfg=conv_cfg,
                bias=True,
                inplace=True))
        return nn.Sequential(*conv_list)
        

    def forward(self, feat_dict):
        point_features = feat_dict['sa_features'][-1]
        point_features = point_features
        point_cls_preds = self.cls_layers(point_features)
        point_box_preds = self.bbox_layers(point_features)

        point_cls_scores = torch.sigmoid(point_cls_preds)        
        ret_dict = {
            'point_cls_preds': point_cls_scores,
            'point_box_preds': point_box_preds,
            'seed_points': feat_dict['sa_xyz'],
            'seed_indices': feat_dict['sa_indices'],
            'seed_features': feat_dict['sa_features']
        }
        decode_res = self.bbox_coder.split_pred(point_cls_scores,
                                                point_box_preds,
                                                feat_dict['sa_xyz'][-1])
        ret_dict.update(decode_res)
        return ret_dict
    
    @force_fp32(apply_to=('bbox_preds'))
    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of PointRCNNHead.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            img_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses of PointRCNN.
        """
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask,
                                   bbox_preds)
        (center_targets, size_res_targets, dir_class_targets,
         dir_res_targets, mask_targets, positive_mask, 
         negative_mask, box_loss_weights) = targets
        # calculate center loss
        center_loss = self.center_loss(
            bbox_preds['center_offset'],
            center_targets,
            weight=box_loss_weights.unsqueeze(-1))
        # calculate direction class loss
        dir_class_loss = self.dir_class_loss(
            bbox_preds['dir_class'].transpose(1, 2),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        dir_res_loss = self.dir_res_loss(
            bbox_preds['dir_res_norm'],
            dir_res_targets.unsqueeze(-1).repeat(1, 1, self.num_dir_bins),
            weight=1.0)

        # calculate size residual loss
        size_loss = self.size_res_loss(
            bbox_preds['size'],
            size_res_targets,
            weight=box_loss_weights.unsqueeze(-1))
        # calculate semantic loss
        semantic_loss = self.cls_loss(
            bbox_preds['point_cls_preds'].transpose(2, 1).reshape(-1,1),
            mask_targets.reshape(-1))

        losses = dict(
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_res_loss=size_loss,
            semantic_loss=semantic_loss)
        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of ssd3d head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of ssd3d head.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        # find empty example
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]
        
        (center_targets, size_res_targets, dir_class_targets,
         dir_res_targets, mask_targets, positive_mask, 
         negative_mask) = multi_apply(self.get_targets_single, points, 
                                     gt_bboxes_3d, gt_labels_3d,
                                     pts_semantic_mask, pts_instance_mask)

        center_targets = torch.stack(center_targets)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)
        positive_mask = torch.stack(positive_mask)
        negative_mask = torch.stack(negative_mask)
        box_loss_weights = positive_mask / (positive_mask.sum() + 1e-6)
        
        return (center_targets, size_res_targets,
                dir_class_targets, dir_res_targets, mask_targets,
                positive_mask, negative_mask, box_loss_weights)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None,
                           seed_points=None):
        """Generate targets of ssd3d head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                candidate points layer.
            seed_points (torch.Tensor): Seed points of candidate points.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        assert self.bbox_coder.with_rot or pts_semantic_mask is not None
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)
        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]

        (center_targets, size_targets, dir_class_targets,
         dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)
    
        points_mask, assignment = self._assign_targets_by_points_inside(
            gt_bboxes_3d, points)

        center_targets = center_targets[assignment]
        size_res_targets = size_targets[assignment]
        mask_targets = gt_labels_3d[assignment]
        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        mask_targets = gt_labels_3d[assignment]

        positive_mask = (points_mask.max(1)[0] > 0) 
        negative_mask = (points_mask.max(1)[0] == 0)

        return (center_targets, size_res_targets,dir_class_targets, 
                dir_res_targets, mask_targets, positive_mask,negative_mask)

    def get_bboxes(self, points, bbox_preds, input_metas, rescale=False):
        """Generate bboxes from sdd3d head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from sdd3d head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # decode boxes
        sem_scores = F.sigmoid(bbox_preds['obj_scores']).transpose(1, 2)
        obj_scores = sem_scores.max(-1)[0]
        bbox3d = self.bbox_coder.decode(bbox_preds)

        batch_size = bbox3d.shape[0]
        results = list()

        for b in range(batch_size):
            bbox_selected, score_selected, labels = self.multiclass_nms_single(
                obj_scores[b], sem_scores[b], bbox3d[b], points[b, ..., :3],
                input_metas[b])
            bbox = input_metas[b]['box_type_3d'](
                bbox_selected.clone(),
                box_dim=bbox_selected.shape[-1],
                with_yaw=self.bbox_coder.with_rot)
            results.append((bbox, score_selected, labels))

        return results

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        num_bbox = bbox.shape[0]
        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 1.0))

        if isinstance(bbox, LiDARInstance3DBoxes):
            box_idx = bbox.points_in_boxes(points)
            box_indices = box_idx.new_zeros([num_bbox + 1])
            box_idx[box_idx == -1] = num_bbox
            box_indices.scatter_add_(0, box_idx.long(),
                                     box_idx.new_ones(box_idx.shape))
            box_indices = box_indices[:-1]
            nonempty_box_mask = box_indices >= 0
        elif isinstance(bbox, DepthInstance3DBoxes):
            box_indices = bbox.points_in_boxes(points)
            nonempty_box_mask = box_indices.T.sum(1) >= 0
        else:
            raise NotImplementedError('Unsupported bbox type!')

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = batched_nms(
            minmax_box3d[nonempty_box_mask][:, [0, 1, 3, 4]],
            obj_scores[nonempty_box_mask], bbox_classes[nonempty_box_mask],
            self.test_cfg.nms_cfg)[1]

        if nms_selected.shape[0] > self.test_cfg.max_output_num:
            nms_selected = nms_selected[:self.test_cfg.max_output_num]

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores >= self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels

    def _assign_targets_by_points_inside(self, bboxes_3d, points):
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (BaseInstance3DBoxes): Instance of bounding boxes.
            points (torch.Tensor): Points of a batch.

        Returns:
            tuple[torch.Tensor]: Flags indicating whether each point is
                inside bbox and the index of box where each point are in.
        """
        # TODO: align points_in_boxes function in each box_structures
        num_bbox = bboxes_3d.tensor.shape[0]
        if isinstance(bboxes_3d, LiDARInstance3DBoxes):
            assignment = bboxes_3d.points_in_boxes(points).long()
            points_mask = assignment.new_zeros(
                [assignment.shape[0], num_bbox + 1])
            assignment[assignment == -1] = num_bbox
            points_mask.scatter_(1, assignment.unsqueeze(1), 1)
            points_mask = points_mask[:, :-1]
            assignment[assignment == num_bbox] = num_bbox - 1
        elif isinstance(bboxes_3d, DepthInstance3DBoxes):
            points_mask = bboxes_3d.points_in_boxes(points)
            assignment = points_mask.argmax(dim=-1)
        else:
            raise NotImplementedError('Unsupported bbox type!')

        return points_mask, assignment
