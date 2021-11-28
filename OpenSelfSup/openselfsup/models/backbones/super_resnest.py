# timm/models/resnet.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SelectAdaptivePool2d, DropBlock2d, DropPath, AvgPool2dSame, create_attn, BlurPool2d

__all__ = ['SuperResNet']  # model_registry will add each entrypoint fn to this


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 conv_layer=nn.Conv2d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None

        self.conv1 = conv_layer(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.aa = aa_layer(channels=first_planes) if stride == 2 and use_aa else None

        self.conv2 = conv_layer(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 conv_layer=nn.Conv2d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        # width = int(math.floor(planes * (base_width / 64)) * cardinality)
        width = planes * cardinality
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None

        self.conv1 = conv_layer(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = conv_layer(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width) if stride == 2 and use_aa else None

        self.conv3 = conv_layer(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None,
        norm_layer=None, conv_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    conv_layer = conv_layer or nn.Conv2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        conv_layer(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None,
        norm_layer=None, conv_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    conv_layer = conv_layer or nn.Conv2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        conv_layer(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


# timm/models/resnest.py
from .layers.split_attn import SplitAttnConv2d


class ResNestBottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, base_width=64, avd=False, avd_first=False, is_first=False,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 conv_layer=nn.Conv2d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(ResNestBottleneck, self).__init__()
        assert reduce_first == 1  # not supported
        assert attn_layer is None  # not supported
        assert aa_layer is None  # TODO not yet supported
        assert drop_path is None  # TODO not yet supported

        group_width = int(planes * (base_width / 64.)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix
        self.drop_block = drop_block

        self.conv1 = conv_layer(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.act1 = act_layer(inplace=True)
        self.avd_first = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and avd_first else None

        if self.radix >= 1:
            self.conv2 = SplitAttnConv2d(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, radix=radix, norm_layer=norm_layer, drop_block=drop_block)
            self.bn2 = None  # FIXME revisit, here to satisfy current torchscript fussyness
            self.act2 = None
        else:
            self.conv2 = conv_layer(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
            self.act2 = act_layer(inplace=True)
        self.avd_last = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and not avd_first else None

        self.conv3 = conv_layer(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.drop_block is not None:
            out = self.drop_block(out)
        out = self.act1(out)

        if self.avd_first is not None:
            out = self.avd_first(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
            if self.drop_block is not None:
                out = self.drop_block(out)
            out = self.act2(out)

        if self.avd_last is not None:
            out = self.avd_last(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.drop_block is not None:
            out = self.drop_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act3(out)
        return out


from ..utils import norm_cfg as Norm_cfg
from ..utils import conv_cfg as Conv_cfg


# timm/models/resnet.py
class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64
        Number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width//4 * 6, stem_width * 2
          * 'deep_tiered_narrow' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs, 3x3 for senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    output_stride : int, default 32
        Set the output stride of the network, 32, 16, or 8. Typically used in segmentation.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, block=ResNestBottleneck, layers=[3, 4, 6, 3], num_classes=1000, in_chans=3,
                 cardinality=1, base_width=64, stem_width=64, stem_type='',
                 block_reduce_first=1, down_kernel_size=1, avg_down=False, output_stride=32,
                 act_layer=nn.ReLU, norm_cfg=dict(type='BN'), conv_cfg=dict(type='Conv'), aa_layer=None, drop_rate=0.0,
                 drop_path_rate=0., drop_block_rate=0., global_pool='avg', zero_init_last_bn=True, block_args=None):
        block_args = block_args or dict()
        self.num_classes = num_classes
        deep_stem = 'deep' in stem_type
        self.inplanes = stem_width * 2 if deep_stem else stem_width
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        # self.expansion = block.expansion
        # my
        if norm_cfg['type'] == 'SwitchableBN':
            norm_layer = Norm_cfg['SwitchableBN'](norm_cfg)
        else:
            norm_layer = Norm_cfg[norm_cfg['type']][1]
        conv_layer = Conv_cfg[conv_cfg['type']]
        #
        super(ResNet, self).__init__()

        # Stem
        if deep_stem:
            stem_chs_1 = stem_chs_2 = stem_width
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (stem_width // 4)
                stem_chs_2 = stem_width if 'narrow' in stem_type else 6 * (stem_width // 4)
            self.conv1 = nn.Sequential(*[
                conv_layer(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs_1),
                act_layer(inplace=True),
                conv_layer(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_2),
                act_layer(inplace=True),
                conv_layer(stem_chs_2, self.inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = conv_layer(in_chans, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act1 = act_layer(inplace=True)
        # Stem Pooling
        if aa_layer is not None:
            self.maxpool = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                aa_layer(channels=self.inplanes, stride=2)
            ])
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        dp = DropPath(drop_path_rate) if drop_path_rate else None
        db_3 = DropBlock2d(drop_block_rate, 7, 0.25) if drop_block_rate else None
        db_4 = DropBlock2d(drop_block_rate, 7, 1.00) if drop_block_rate else None
        channels, strides, dilations = [base_width * 2 ** i for i in range(4)], [1, 2, 2, 2], [1] * 4 # my
        if output_stride == 16:
            strides[3] = 1
            dilations[3] = 2
        elif output_stride == 8:
            strides[2:4] = [1, 1]
            dilations[2:4] = [2, 4]
        else:
            assert output_stride == 32
        layer_args = list(zip(channels, layers, strides, dilations))
        layer_kwargs = dict(
            reduce_first=block_reduce_first, act_layer=act_layer, norm_layer=norm_layer,
            conv_layer=conv_layer, aa_layer=aa_layer,
            avg_down=avg_down, down_kernel_size=down_kernel_size, drop_path=dp, **block_args)
        self.layer1 = self._make_layer(block, *layer_args[0], **layer_kwargs)
        self.layer2 = self._make_layer(block, *layer_args[1], **layer_kwargs)
        self.layer3 = self._make_layer(block, drop_block=db_3, *layer_args[2], **layer_kwargs)
        self.layer4 = self._make_layer(block, drop_block=db_4, *layer_args[3], **layer_kwargs)

        # Head (Pooling and Classifier)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, reduce_first=1,
                    avg_down=False, down_kernel_size=1, **kwargs):
        downsample = None
        first_dilation = 1 if dilation in (1, 2) else 2
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_args = dict(
                in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=first_dilation,
                norm_layer=kwargs.get('norm_layer'), conv_layer=kwargs.get('conv_layer'))
            downsample = downsample_avg(**downsample_args) if avg_down else downsample_conv(**downsample_args)

        block_kwargs = dict(
            cardinality=self.cardinality, base_width=self.base_width, reduce_first=reduce_first,
            dilation=dilation, **kwargs)
        layers = [block(self.inplanes, planes, stride, downsample, first_dilation=first_dilation, **block_kwargs)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, **block_kwargs) for _ in range(1, blocks)]

        return nn.Sequential(*layers)
    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x


def get_block(**kwargs):
    blocks = {
        'Bottleneck': Bottleneck,
        'ResNestBottleneck': ResNestBottleneck,
    }
    name = kwargs.pop('block', 'ResNestBottleneck')

    block = blocks[name](**kwargs)
    return block


from torch.nn.modules.batchnorm import _BatchNorm
# timm/models/super_resnest.py
from ..registry import BACKBONES
from mmcv.runner import load_checkpoint
from openselfsup.utils import get_root_logger
@BACKBONES.register_module
class SuperResNet(ResNet):
    layer_indices = (('layer1', 0), ('layer1', 1), ('layer1', 2), ('layer2', 0),
                     ('layer2', 1), ('layer2', 2), ('layer2', 3), ('layer3', 0),
                     ('layer3', 1), ('layer3', 2), ('layer3', 3), ('layer3', 4),
                     ('layer3', 5), ('layer4', 0), ('layer4', 1), ('layer4', 2))

    def __init__(self, *args, candidate_num=0, frozen_stages=-1, norm_eval='', **kwargs):
        self.candidate_num = candidate_num
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        super(SuperResNet, self).__init__(*args, **kwargs)
        self._freeze_stages()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, reduce_first=1,
                    avg_down=False, down_kernel_size=1, **kwargs):
        # 默认 block.expansion = 4，block在这里不用
        downsample_share = kwargs.pop('downsample_share', True)
        candidate_args = kwargs.pop('candidate_args')

        downsample = [None for _ in range(len(candidate_args))]
        first_dilation = 1 if dilation in (1, 2) else 2
        if stride != 1 or self.inplanes != planes * 4:
            downsample_args = dict(
                in_channels=self.inplanes, out_channels=planes * 4, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=first_dilation,
                norm_layer=kwargs.get('norm_layer'), conv_layer=kwargs.get('conv_layer'))
            if downsample_share:
                downsample = [downsample_avg(**downsample_args) if avg_down else downsample_conv(**downsample_args)] *\
                                                                                                    len(candidate_args)
            else:
                downsample = [downsample_avg(**downsample_args) if avg_down else downsample_conv(**downsample_args) \
                                                                                    for _ in range(len(candidate_args))]

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, **kwargs)

        layers = [nn.ModuleList(
                    [get_block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample[i],
                           first_dilation=first_dilation, **block_kwargs, **candidate_args[i]
                           ) for i in range(len(candidate_args))])]
            
        self.inplanes = planes * 4
        layers += [nn.ModuleList(
                     [get_block(inplanes=self.inplanes, planes=planes, **block_kwargs, **candidate_arg
                           ) for candidate_arg in candidate_args]) for _ in range(1, blocks)]

        return nn.ModuleList(layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.bn1]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x, choice_indices):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        for layer_index, choice_index in zip(self.layer_indices, choice_indices):
            x = getattr(self, layer_index[0])[layer_index[1]][choice_index](x)
#         x = self.global_pool(x).flatten(1)
#         if self.drop_rate:
#             x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return (x, )

    def ema_blocks(self, choice_indices):
        blocks = [self.conv1, self.bn1]
        for layer_index, choice_index in zip(self.layer_indices, choice_indices):
            blocks.append(getattr(self, layer_index[0])[layer_index[1]][choice_index])
        return blocks

    def train(self, mode=True):
        super(SuperResNet, self).train(mode)
        self._freeze_stages()
        if mode:
            if self.norm_eval == 'all_true':
                for m in self.modules():
                    # trick: eval have effect on BatchNorm only
                    if isinstance(m, _BatchNorm):
                        m.eval()
            elif self.norm_eval == 'all_false':
                for m in self.modules():
                    # trick: eval have effect on BatchNorm only
                    if isinstance(m, _BatchNorm):
                        m.train()
            elif self.norm_eval != '':
                raise NotImplementedError
