# from mmcv.runner import load_checkpoint
# from torch import nn as nn
import warnings
from abc import ABCMeta
from mmcv.runner import BaseModule


class BasePointNet(BaseModule, metaclass=ABCMeta):
    """Base class for PointNet.

    args:
        init_cfg (dict or list[dict], optional): Initialization config dict.
        Default: None
    """

    def __init__(self, init_cfg=None, pretrained=None):
        super(BasePointNet, self).__init__()
        self.fp16_enabled = False
        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @staticmethod
    def _split_point_feats(points):
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features
