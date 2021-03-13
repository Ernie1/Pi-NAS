import torch
import torch.nn as nn
from .resnet import ResNet, Bottleneck
import math

__all__ = ['alone_resnest50']


class AloneResnet50(ResNet):
    def __init__(self, *args, **kwargs):
        self.block_args = kwargs.pop('block_args')
        self.choice_indices = kwargs.pop('choice_indices')
        super(AloneResnet50, self).__init__(*args, **kwargs)

    def _layer_block_args(self, planes):
        start, end = [[0, 3], [3, 7], [7, 13], [13, 16]][int(math.log(planes // 64, 2))]
        return [self.block_args[c] for c in self.choice_indices[start : end]]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        layer_block_args = self._layer_block_args(planes)
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=layer_block_args[0]['radix'],
                                cardinality=layer_block_args[0]['cardinality'],
                                bottleneck_width=layer_block_args[0]['base_width'],
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=layer_block_args[0]['radix'],
                                cardinality=layer_block_args[0]['cardinality'],
                                bottleneck_width=layer_block_args[0]['base_width'],
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=layer_block_args[i]['radix'],
                                cardinality=layer_block_args[i]['cardinality'],
                                bottleneck_width=layer_block_args[i]['base_width'],
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)


block_args=[
    dict(radix=1, cardinality=1, base_width=64),
    dict(radix=2, cardinality=1, base_width=64),
    dict(radix=1, cardinality=2, base_width=42),
    dict(radix=2, cardinality=2, base_width=40),
]


def alone_resnest50(pretrained=False, root='~/.encoding/models', backbone_pretrained_path='', **kwargs):
    model = AloneResnet50(Bottleneck, [3, 4, 6, 3],
                          deep_stem=True, stem_width=32, avg_down=True,
                          avd=True, avd_first=False, block_args=block_args, **kwargs)
    if backbone_pretrained_path:
        print(model.load_state_dict(torch.load(backbone_pretrained_path, map_location='cpu')['state_dict'], strict=False), flush=True)
        print("=> loaded checkpoint '{}'".format(backbone_pretrained_path), flush=True)
    return model