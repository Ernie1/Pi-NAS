from mmcv.runner import Hook

import torch
from torch.utils.data import Dataset

from openselfsup.utils import nondist_forward_collect, dist_forward_collect
from .registry import HOOKS


@HOOKS.register_module
class ValidateHook(Hook):

    def __init__(self,
                 dataset,
                 dist_mode=True,
                 initial=True,
                 interval=1,
                 **eval_kwargs):
        from openselfsup import datasets
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.data_loader = datasets.build_dataloader(
            self.dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode,
            shuffle=False)
        self.dist_mode = dist_mode
        self.initial = initial
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        # my
        self.update_bn = eval_kwargs.get('update_bn', False)
        if self.update_bn:
            self.train_dataset = datasets.build_dataset(eval_kwargs['train_dataset'])
            self.train_data_loader = datasets.build_dataloader(
                self.train_dataset,
                eval_kwargs['imgs_per_gpu'],
                eval_kwargs['workers_per_gpu'],
                dist=dist_mode,
                shuffle=False)
        #

    def before_run(self, runner):
        if self.initial:
            self._run_validate(runner)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        self._run_validate(runner)

    def _run_validate(self, runner):
        # my
        if self.update_bn:
            update_bn_stats(self.train_data_loader, runner, num_iters=196)
        #
        runner.model.eval()
        func = lambda **x: runner.model(mode='test', **x)
        if self.dist_mode:
            results = dist_forward_collect(
                func, self.data_loader, runner.rank,
                len(self.dataset))  # dict{key: np.ndarray}
        else:
            results = nondist_forward_collect(func, self.data_loader,
                                              len(self.dataset))
        if runner.rank == 0:
            for name, val in results.items():
                self._evaluate(runner, torch.from_numpy(val), name)
        runner.model.train()

    def _evaluate(self, runner, results, keyword):
        eval_res = self.dataset.evaluate(
            results,
            keyword=keyword,
            logger=runner.logger,
            **self.eval_kwargs['eval_param'])
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


import mmcv
import itertools
from .contrastive_search_hook import get_bn_modules
def update_bn_stats(data_loader, runner, num_iters=200):
    runner.model.train()

    bn_layers = get_bn_modules(runner.model)

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
        runner.logger.info('update bn')
        prog_bar = mmcv.ProgressBar(num_iters)
    idx = -1
    for idx, data in enumerate(itertools.islice(data_loader, num_iters)):
        with torch.no_grad():  # No need to backward
            # runner.model(**data, choice_indices=choice_indices, extract=True)
            runner.model(**data)

        for i, bn in enumerate(bn_layers):
            # Accumulates the bn stats.
            running_mean[i] += (bn.running_mean - running_mean[i]) / (idx + 1)
            running_var[i] += (bn.running_var - running_var[i]) / (idx + 1)
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