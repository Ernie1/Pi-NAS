import mmcv
from mmcv.runner import Hook

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from openselfsup.utils import nondist_forward_collect, dist_forward_collect
from openselfsup.apis.search_phase import SearchPhase
from openselfsup.models.moco import concat_all_gather

from .registry import HOOKS

import numpy as np
import itertools
from typing import List, Tuple, Type


BN_MODULE_TYPES: Tuple[Type[torch.nn.Module]] = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)


def get_bn_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Find all BatchNorm (BN) modules that are in training mode. See
    fvcore.precise_bn.BN_MODULE_TYPES for a list of all modules that are
    included in this search.
    Args:
        model (nn.Module): a model possibly containing BN modules.
    Returns:
        list[nn.Module]: all BN modules in the model.
    """
    # Finds all the bn layers.
    bn_layers = [
        m for m in model.modules() if m.training and isinstance(m, BN_MODULE_TYPES)
    ]
    return bn_layers


# encoding/utils/precise_bn.py
def update_bn_stats(data_loader, runner, choice_indices, ema=False, num_iters=200, num_epochs=1):
    runner.model.train()

    bn_layers = get_bn_modules(runner.model)
    for m in bn_layers:
        m.track_running_stats = True
    
    if len(bn_layers) == 0:
        return

    # In order to make the running stats only reflect the current batch, the
    # momentum is disabled.
    # bn.running_mean = (1 - momentum) * bn.running_mean + momentum * batch_mean
    # Setting the momentum to 1.0 to compute the stats without momentum.
    momentum_actual = [bn.momentum for bn in bn_layers]  # pyre-ignore
    for bn in bn_layers:
        bn.momentum = 1.0

    # Note that running_var actually means "running average of variance"
    running_mean = [
        torch.zeros_like(bn.running_mean) for bn in bn_layers  # pyre-ignore
    ]
    running_var = [torch.zeros_like(bn.running_var) for bn in bn_layers]  # pyre-ignore

    if runner.rank == 0:
        prog_bar = mmcv.ProgressBar(num_iters * num_epochs)
    idx = -1
    for e in range(num_epochs):
        for idx, data in enumerate(itertools.islice(data_loader, num_iters)):
            with torch.no_grad():  # No need to backward
                runner.model(data['img'], choice_indices=choice_indices, mode='extract', ema=ema)

            for i, bn in enumerate(bn_layers):
                # Accumulates the bn stats.
                running_mean[i] += (bn.running_mean - running_mean[i]) / (e * num_iters + idx + 1)
                running_var[i] += (bn.running_var - running_var[i]) / (e * num_iters + idx + 1)
                # We compute the "average of variance" across iterations.

            if runner.rank == 0:
                prog_bar.update()   

    assert idx == num_iters - 1, (
        "update_bn_stats is meant to run for {} iterations, "
        "but the dataloader stops at {} iterations.".format(num_iters, idx)
    )

    for i, bn in enumerate(bn_layers):
        # Sets the precise bn stats.
        bn.running_mean = running_mean[i]
        bn.running_var = running_var[i]
        bn.momentum = momentum_actual[i]

    runner.model.eval()
