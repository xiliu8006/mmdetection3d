from mmcv.cnn.bricks import build_conv_layer
from torch import nn as nn

from mmdet.models.builder import HEADS
from .base_conv_bbox_head import BaseConvBboxHead


@HEADS.register_module()
class BaseSeparateConvBboxHead(BaseConvBboxHead):
    r"""More general bbox head, with shared conv layers and two optional
    separated branches.

    .. code-block:: none

           /-> cls convs -> cls_score
    featurs
           \-> reg convs -> bbox_pred
    """

    def __init__(self,
                 in_channels=0,
                 cls_conv_channels=(),
                 num_cls_out_channels=0,
                 reg_conv_channels=(),
                 num_reg_out_channels=0,
                 init_cfg=None,
                 bias=False,
                 replace_conv=False,
                 *args,
                 **kwargs):
        super().__init__(
            init_cfg=init_cfg,
            in_channels=in_channels,
            cls_conv_channels=cls_conv_channels,
            num_cls_out_channels=num_cls_out_channels,
            reg_conv_channels=reg_conv_channels,
            num_reg_out_channels=num_reg_out_channels,
            *args,
            **kwargs)
        self.replace_conv = replace_conv
        # add cls specific branch
        self.cls_convs = self._add_conv_branch(self.in_channels,
                                               self.cls_conv_channels)
        prev_channel = self.cls_conv_channels[-1]
        self.conv_cls = build_conv_layer(
            self.conv_cfg,
            in_channels=prev_channel,
            out_channels=num_cls_out_channels,
            kernel_size=1)

        # add reg specific branch
        self.reg_convs = self._add_conv_branch(self.in_channels,
                                               self.reg_conv_channels)
        prev_channel = self.reg_conv_channels[-1]
        self.conv_reg = build_conv_layer(
            self.conv_cfg,
            in_channels=prev_channel,
            out_channels=num_reg_out_channels,
            kernel_size=1)

        if self.replace_conv:
            self.cls_layers = self._make_fc_layers(self.cls_conv_channels,
                                                   self.in_channels,
                                                   num_cls_out_channels)

            self.reg_layers = self._make_fc_layers(self.reg_conv_channels,
                                                   self.in_channels,
                                                   num_reg_out_channels)

    def _make_fc_layers(self, fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def forward(self, feats):
        """Forward.

        Args:
            feats (Tensor): Input features

        Returns:
            Tensor: Class scores predictions
            Tensor: Regression predictions
        """

        # separate branches
        x_cls = feats
        x_reg = feats

        if self.replace_conv:
            bs = x_cls.shape[0]
            x_cls = x_cls.transpose(2, 1).reshape(-1, self.in_channels)
            x_reg = x_reg.transpose(2, 1).reshape(-1, self.in_channels)
            cls_score = self.cls_layers(x_cls)
            bbox_pred = self.reg_layers(x_reg)
            return cls_score.reshape(bs, -1, 3).transpose(2, 1), \
                bbox_pred.reshape(bs, -1, 8).transpose(2, 1)
        else:
            x_cls = self.cls_convs(x_cls)
            cls_score = self.conv_cls(x_cls)

            x_reg = self.reg_convs(x_reg)
            bbox_pred = self.conv_reg(x_reg)
            return cls_score, bbox_pred
