import torch
import torch.nn as nn
import numpy as np

from .registry import MODELS
from .moco import MOCO, concat_all_gather
from collections import OrderedDict


@MODELS.register_module
class PiNAS(MOCO):
    @torch.no_grad()
    def _momentum_update_key_encoder(self, choice_indices):
        """
        Momentum update of the key encoder
        """
        for block_q, block_k in zip(self.encoder_q[0].ema_blocks(choice_indices) + [self.encoder_q[1]],
                                    self.encoder_k[0].ema_blocks(choice_indices) + [self.encoder_k[1]]):
            for param_q, param_k in zip(block_q.parameters(),
                                        block_k.parameters()):
                param_k.data = param_k.data * self.momentum + \
                               param_q.data * (1. - self.momentum)

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_q_0 = img[:, 0, ...].contiguous()
        im_k_0 = img[:, 1, ...].contiguous()
        im_q_1 = img[:, 2, ...].contiguous()
        im_k_1 = img[:, 3, ...].contiguous()
        # compute query features
        assert 'choice_indices' not in kwargs and 'extract' not in kwargs

        choice_indices = np.array([np.random.choice(self.encoder_q[0].candidate_num,
                                    2, replace=False) for _ in range(16)])
        choice_indices_0 = choice_indices[:, 0]
        choice_indices_1 = choice_indices[:, 1]

        # step 0
        q_0 = self.encoder_q[0](im_q_0, choice_indices_0)  # queries: NxC
        q_0 = self.encoder_q[1](q_0)[0]
        q_0 = nn.functional.normalize(q_0, dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder(choice_indices_0)  # update the key encoder

            # shuffle for making use of BN
            im_k_0, idx_unshuffle = self._batch_shuffle_ddp(im_k_0)

            k_0 = self.encoder_k[0](im_k_0, choice_indices_0)  # keys: NxC
            k_0 = self.encoder_k[1](k_0)[0]
            k_0 = nn.functional.normalize(k_0, dim=1)

            # undo shuffle
            k_0 = self._batch_unshuffle_ddp(k_0, idx_unshuffle)

        # step 1
        q_1 = self.encoder_q[0](im_q_1, choice_indices_1)  # queries: NxC
        q_1 = self.encoder_q[1](q_1)[0]
        q_1 = nn.functional.normalize(q_1, dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder(choice_indices_1)  # update the key encoder

            # shuffle for making use of BN
            im_k_1, idx_unshuffle = self._batch_shuffle_ddp(im_k_1)

            k_1 = self.encoder_k[0](im_k_1, choice_indices_1)  # keys: NxC
            k_1 = self.encoder_k[1](k_1)[0]
            k_1 = nn.functional.normalize(k_1, dim=1)

            # undo shuffle
            k_1 = self._batch_unshuffle_ddp(k_1, idx_unshuffle)
        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_0 = torch.einsum('nc,nc->n', [q_0, k_1]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_0 = torch.einsum('nc,ck->nk', [q_0, self.queue.clone().detach()])
        # positive logits: Nx1
        l_pos_1 = torch.einsum('nc,nc->n', [q_1, k_0]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_1 = torch.einsum('nc,ck->nk', [q_1, self.queue.clone().detach()])

        losses_0 = self.head(l_pos_0, l_neg_0)
        losses_1 = self.head(l_pos_1, l_neg_1)
        losses = {k: (losses_0[k] + losses_1[k]) for k in losses_0}
        self._dequeue_and_enqueue((k_0 + k_1) / 2)

        return losses

    def forward_test(self, img, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_q_0 = img[:, 0, ...].contiguous()
        im_k_0 = img[:, 1, ...].contiguous()
        im_q_1 = img[:, 2, ...].contiguous()
        im_k_1 = img[:, 3, ...].contiguous()
        # compute query features
        choice_indices_0 = kwargs.pop('choice_indices')
        choice_indices_1 = np.array([np.random.choice(
            list(set(range(self.encoder_q[0].candidate_num)) - \
                set([choice_indices_0[i]]))) for i in range(16)])

        # step 0
        q_0 = self.encoder_q[0](im_q_0, choice_indices_0)  # queries: NxC
        q_0 = self.encoder_q[1](q_0)[0]
        q_0 = nn.functional.normalize(q_0, dim=1)

        k_0 = self.encoder_k[0](im_k_0, choice_indices_0)  # keys: NxC
        k_0 = self.encoder_k[1](k_0)[0]
        k_0 = nn.functional.normalize(k_0, dim=1)

        # step 1
        q_1 = self.encoder_q[0](im_q_1, choice_indices_1)  # queries: NxC
        q_1 = self.encoder_q[1](q_1)[0]
        q_1 = nn.functional.normalize(q_1, dim=1)

        k_1 = self.encoder_k[0](im_k_1, choice_indices_1)  # keys: NxC
        k_1 = self.encoder_k[1](k_1)[0]
        k_1 = nn.functional.normalize(k_1, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_0 = torch.einsum('nc,nc->n', [q_0, k_1]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_0 = torch.einsum('nc,ck->nk', [q_0, self.queue.clone().detach()])
        # positive logits: Nx1
        l_pos_1 = torch.einsum('nc,nc->n', [q_1, k_0]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_1 = torch.einsum('nc,ck->nk', [q_1, self.queue.clone().detach()])

        losses_0 = self.head(l_pos_0, l_neg_0)  
        losses_1 = self.head(l_pos_1, l_neg_1)
        losses = {k: (losses_0[k] + losses_1[k]) for k in losses_0}

        return losses

    def forward_key(self, img, **kwargs):
        assert img.dim() == 4, \
            "Input must have 4 dims, got: {}".format(img.dim())

        choice_indices = kwargs.pop('choice_indices')

        k = self.encoder_k[0](img, choice_indices)  # keys: NxC
        k = self.encoder_k[1](k)[0]
        k = nn.functional.normalize(k, dim=1)

        return k

    def forward_extract(self, img, **kwargs):
        assert img.dim() == 4, \
            "Input must have 4 dims, got: {}".format(img.dim())
        im_q = img
        # compute query features
        choice_indices = kwargs.pop('choice_indices')
        q = self.encoder_q[0](im_q, choice_indices)  # queries: NxC
        q = self.encoder_q[1].avgpool(q[0])
        q = nn.functional.normalize(q, dim=1)
        q = concat_all_gather(q)

        return q

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'key':
            return self.forward_key(img, **kwargs)
        elif mode == 'extract':
            return self.forward_extract(img, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))
