import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class RotationPred(nn.Module):

    def __init__(self, backbone, head=None, pretrained=None):
        super(RotationPred, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if head is not None:
            self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.head.init_weights(init_linear='kaiming')

    def forward_backbone(self, img):
        """Forward backbone

        Returns:
            x (tuple): backbone outputs
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, rot_label, **kwargs):
        x = self.forward_backbone(img)
        outs = self.head(x)
        loss_inputs = (outs, rot_label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, img, **kwargs):
        x = self.forward_backbone(img)  # tuple
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward(self, img, rot_label=None, mode='train', **kwargs):
        if mode != "extract" and img.dim() == 5:
            assert rot_label.dim() == 2
            img = img.view(
                img.size(0) * img.size(1), img.size(2), img.size(3),
                img.size(4))
            rot_label = torch.flatten(rot_label)
        if mode == 'train':
            return self.forward_train(img, rot_label, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
