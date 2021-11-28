import numpy as np

import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
from .utils import Sobel


@MODELS.register_module
class SuperClassification(nn.Module):
    def __init__(self,
                 backbone,
                 with_sobel=False,
                 neck=None,
                 head=None,
                 pretrained=None,
                 fairnas=False):
        super(SuperClassification, self).__init__()
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if head is not None:
            self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)
        self.fairnas = fairnas
        if fairnas:
            self.choice_indices = np.expand_dims(np.arange(self.backbone.candidate_num), 1).repeat(16, 1)
            self.ptr = self.backbone.candidate_num

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        if hasattr(self, 'neck'):
            self.neck.init_weights()
        self.head.init_weights()

    def forward_backbone(self, img, choice_indices=None):
        """Forward backbone

        Returns:
            x (tuple): backbone outputs
        """
        if self.with_sobel:
            img = self.sobel_layer(img)
        if choice_indices is None:
            choice_indices = np.random.randint(0, self.backbone.candidate_num, 16)
        x = self.backbone(img, choice_indices)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        choice_indices = kwargs.pop('choice_indices', None)
        if not self.fairnas:
            x = self.forward_backbone(img, choice_indices)
            if hasattr(self, 'neck'):
                x = self.neck(x)
            outs = self.head(x)
            loss_inputs = (outs, gt_label)
            losses = self.head.loss(*loss_inputs)
        else:
            assert choice_indices is None
            
            if self.ptr == self.backbone.candidate_num:
                for i in range(16):
                    np.random.shuffle(self.choice_indices[:, i])
                self.ptr = 0

            x = self.forward_backbone(img, self.choice_indices[self.ptr])
            if hasattr(self, 'neck'):
                x = self.neck(x)
            outs = self.head(x)
            loss_inputs = (outs, gt_label)
            losses = self.head.loss(*loss_inputs)

            self.ptr += 1

        return losses

    def forward_test(self, img, **kwargs):
        x = self.forward_backbone(img, choice_indices=kwargs.pop('choice_indices', None))
        if hasattr(self, 'neck'):
            x = self.neck(x)
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))
