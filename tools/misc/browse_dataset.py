import argparse
from mmcv import Config

from mmdet3d.datasets import build_dataset

# try:
#    import open3d.ml.torch as ml3d
#   from open3d.ml.vis import Visualizer, BoundingBox3D
# except ImportError:
#   raise ImportError(
#      'please run "pip install open3d" to install open3d first. ')


def parse_args():
    parser = argparse.ArgumentParser(description='MMdet3D browse the dataset')
    parser.add_argument('config', help='config file path')
    parser.add_argument('split', help='train, test or val')
    args = parser.parse_args()
    return args


def get_data_cfg(config, split):
    cfg = Config.fromfile(config)
    if split == 'train':
        cfg = cfg.data.train.dataset
    elif split == 'test':
        cfg = cfg.data.test
    elif split == 'val':
        cfg = cfg.data.val
    else:
        print("[ERROR] '" + split + "' is not train, test or val")


#        print_usage_and_exit()
    return cfg


def convert_bboxes_for_open3d(bboxes, label_class, confidence=1):
    import numpy as np

    bounding_boxes = []
    for i in len(bboxes):
        box = bboxes[i]
        center = [box[0], box[1], box[2] + box[5] / 2]
        size = box[3:6]
        yaw = box[6]
        # x-axis
        left = [np.cos(yaw), -np.sin(yaw), 0]
        # y-axis
        front = [np.sin(yaw), np.cos(yaw), 0]
        # z-axis
        up = [0, 0, 1]


#        box = BoundingBox3D(center, front, up, left, size, label_class,
#                           confidence)
#        bounding_bboxes.append(box)
    return bounding_boxes, left, front, up, center, size


def main():
    args = parse_args()
    cfg = get_data_cfg(args.config, args.split)
    dataset = build_dataset(cfg)
    vis_points = []
    vis_bboxes = []
    for item in dataset:
        points = item['points']._data.numpy()
        gt_bboxes = item['gt_bboxes_3d']._data.tensor.numpy()
        #        gt_labels_3d = item['gt_labels_3d']._data.numpy()
        print(item['img_metas']._data['pts_filename'])
        filename = item['img_metas']._data['pts_filename']
        vis_p = {'name': filename, 'points': points}
        vis_points.append(vis_p),
        vis_bboxes.append(gt_bboxes)


if __name__ == '__main__':
    main()
