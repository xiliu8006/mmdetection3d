import torch
from collections import OrderedDict
from mmcv import Config

from mmdet3d.models import build_detector


def parse_cfg(config_path):
    config = Config.fromfile(config_path)
    return config


def convert_backbone(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('backbone_3d', 'backbone')
    split_keys = new_key.split('.')
    if len(split_keys) == 7:
        backbone, sa, sa_num, mlps, layer, block, module = split_keys
        block_id = int(block)
        conv_list_set = set([0, 3, 6])
        if block_id in conv_list_set:
            block_name = f'layer{int(block_id / 3)}'
            module = 'conv.weight'
            new_key = f'{backbone}.{sa}.{sa_num}.{mlps}.{layer}.{block_name}.{module}'
        else:
            block_name = f'layer{int(block_id / 3)}'
            module = 'bn.' + module
            new_key = f'{backbone}.{sa}.{sa_num}.{mlps}.{layer}.{block_name}.{module}'

    if len(split_keys) == 6:
        backbone, fp, fp_num, mlps, block, module = split_keys
        block_id = int(block)
        fp_num = str(3 - int(fp_num))
        conv_list_set = set([0, 3])
        if block_id in conv_list_set:
            block_name = f'layer{int(block_id / 3)}'
            module = 'conv.weight'
            new_key = f'{backbone}.{fp}.{fp_num}.mlps.{block_name}.{module}'
        else:
            block_name = f'layer{int(block_id / 3)}'
            module = 'bn.' + module
            new_key = f'{backbone}.{fp}.{fp_num}.mlps.{block_name}.{module}'

    state_dict[new_key] = model_weight
    converted_names.add(model_key)


def convert_rpn_head(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('point_head', 'rpn_head')
    split_keys = new_key.split('.')
    head, task_layer, block, module = split_keys
    block_id = int(block)
    conv_list_set = set([0, 3])
    if task_layer == 'cls_layers':
        new_key = f'{head}.conv_pred.{task_layer}.{block}.{module}'
    else:
        task_layer = 'reg_layers'
        new_key = f'{head}.conv_pred.{task_layer}.{block}.{module}'
    state_dict[new_key] = model_weight
    converted_names.add(model_key)


def convert_roi_head(model_key, model_weight, state_dict, converted_names):
    split_keys = model_key.split('.')
    split_keys[0] = 'roi_head.bbox_head'
    if split_keys[1] == 'SA_modules':
        roi_head, sa, sa_num, mlps, layer, block, module = split_keys
        block_id = int(block)
        conv_list_set = set([0, 3, 6])
        if block_id in conv_list_set:
            block_name = f'layer{int(block_id / 3)}'
            module = 'conv.weight'
            new_key = f'{roi_head}.{sa}.{sa_num}.{mlps}.{layer}.{block_name}.{module}'
        else:
            block_name = f'layer{int(block_id / 3)}'
            module = 'bn.' + module
            new_key = f'{roi_head}.{sa}.{sa_num}.{mlps}.{layer}.{block_name}.{module}'
    elif split_keys[1] == 'xyz_up_layer':
        roi_head, xyz_up_layer, block, module = split_keys
        block_id = int(int(block) / 2)
        module = 'conv.' + module
        new_key = f'{roi_head}.{xyz_up_layer}.{block_id}.{module}'
    elif split_keys[1] == 'merge_down_layer':
        roi_head, merge_down_layer, block, module = split_keys
        block_id = int(int(block) / 2)
        module = 'conv.' + module
        new_key = f'{roi_head}.{merge_down_layer}.{module}'
    else:
        head, task_layer, block, module = split_keys
        block_id = int(block)
        conv_list_set = set([0, 4])
        if task_layer == 'cls_layers':
            if block_id in conv_list_set:
                task_layer = 'cls_convs'
                block_name = f'layer{int(block_id / 4)}'
                module = 'conv.weight'
                new_key = f'{head}.conv_pred.{task_layer}.{block_name}.{module}'
            elif block_id < 6:
                task_layer = 'cls_convs'
                block_name = f'layer{int(block_id / 3)}'
                module = 'bn.' + module
                new_key = f'{head}.conv_pred.{task_layer}.{block_name}.{module}'
            else:
                task_layer = 'conv_cls'
                new_key = f'{head}.conv_pred.{task_layer}.{module}'
        else:
            if block_id in conv_list_set:
                task_layer = 'reg_convs'
                block_name = f'layer{int(block_id / 3)}'
                module = 'conv.weight'
                new_key = f'{head}.conv_pred.{task_layer}.{block_name}.{module}'
            elif block_id < 6:
                task_layer = 'reg_convs'
                block_name = f'layer{int(block_id / 3)}'
                module = 'bn.' + module
                new_key = f'{head}.conv_pred.{task_layer}.{block_name}.{module}'
            else:
                task_layer = 'conv_reg'
                new_key = f'{head}.conv_pred.{task_layer}.{module}'
    state_dict[new_key] = model_weight
    converted_names.add(model_key)


def main():
    checkpoint = torch.load(
        './work_dirs/pointrcnn_2x8_kitti-3d-car/pointrcnn_7870.pth')
    pointrcnn = torch.load(
        './work_dirs/pointrcnn_2x8_kitti-3d-car/test/latest.pth')
    blobs = checkpoint['model_state']
    src = pointrcnn['state_dict']
    state_dict = OrderedDict()
    converted_names = set()
    for key, weight in blobs.items():
        print(key)
        if 'backbone_3d' in key:
            convert_backbone(key, weight, state_dict, converted_names)
        elif 'point_head' in key:
            convert_rpn_head(key, weight, state_dict, converted_names)
        elif 'roi_head' in key:
            convert_roi_head(key, weight, state_dict, converted_names)
    for key in blobs:
        if key not in converted_names:
            print(f'not converted: {key}')
    # save checkpoint
    save_checkpoint = dict()
    save_checkpoint['state_dict'] = state_dict
    torch.save(
        save_checkpoint,
        './work_dirs/pointrcnn_2x8_kitti-3d-car/convert_pointrcnn_7870.pth',
        _use_new_zipfile_serialization=False)
    print('-------------------------PointRCNN-------------------------')
    for key, weight in src.items():
        print(key)
    print('-------------------------convert-------------------------')
    for key, weight in state_dict.items():
        print(key)
    return 0
    '''
    cfg = parse_cfg('')
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    orig_ckpt = checkpoint['state_dict']
    convert_ckpt = ori
    '''


if __name__ == '__main__':
    main()
