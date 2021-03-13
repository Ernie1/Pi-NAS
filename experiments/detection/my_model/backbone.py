from encoding.models.sseg.base import get_backbone

import logging
import types
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from .resnest import build_resnest_backbone

logger = logging.getLogger(__name__)

__all__ = ['build_my_backbone', 'build_my_fpn_backbone', 'build_resnest_fpn_backbone']


def get_norm(norm):
    if len(norm) == 0:
        return None
    return {
        "BN": nn.BatchNorm2d,
        "SyncBN": nn.SyncBatchNorm
    }[norm]


class MyBackbone(Backbone):
    def __init__(self, model, out_features, out_feature_channels, out_feature_strides):
        super().__init__()
        self.model = model
        self._out_features = out_features
        self._out_feature_channels = out_feature_channels
        self._out_feature_strides = out_feature_strides

    def forward(self, x):
        outputs = {}
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if "stem" in self._out_features:
            outputs['stem'] = x
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if not hasattr(self.model, name):
                break
            x = getattr(self.model, name)(x)
            if name in self._out_features:
                outputs[name] = x
        return outputs


@BACKBONE_REGISTRY.register()
def build_my_backbone(cfg, input_shape):
    model = get_backbone(cfg.MODEL.BACKBONE.MODEL,
                         norm_layer=get_norm(cfg.MODEL.RESNETS.NORM),
                         backbone_pretrained_path=cfg.MODEL.BACKBONE.PRETRAINED)
    for name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if name > max(cfg.MODEL.RESNETS.OUT_FEATURES):
            delattr(model, name)
    del model.fc
    return MyBackbone(model,
                     cfg.MODEL.RESNETS.OUT_FEATURES,
                     {'stem': 64, 'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048},
                     {'stem': 4, 'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32})


@BACKBONE_REGISTRY.register()
def build_my_fpn_backbone(cfg, input_shape):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_my_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_resnest_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnest_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone