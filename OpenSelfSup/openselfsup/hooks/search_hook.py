from mmcv.runner import Hook

import torch
from torch.utils.data import Dataset

from openselfsup.utils import nondist_forward_collect, dist_forward_collect
from openselfsup.apis.search_phase import SearchPhase
from .registry import HOOKS

import numpy as np
from .contrastive_search_hook import update_bn_stats
import os
import torch.distributed as dist


@HOOKS.register_module
class SearchHook(Hook):

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 dist_mode=True,
                 **eval_kwargs):
        from openselfsup import datasets
        if isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset
        elif isinstance(train_dataset, dict):
            self.train_dataset = datasets.build_dataset(train_dataset)
        else:
            raise TypeError(
                'train_dataset must be a Dataset object or a dict, not {}'.format(
                    type(train_dataset)))
        self.train_data_loader = datasets.build_dataloader(
            self.train_dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode,
            shuffle=False)
        if isinstance(val_dataset, Dataset):
            self.val_dataset = val_dataset
        elif isinstance(val_dataset, dict):
            self.val_dataset = datasets.build_dataset(val_dataset)
        else:
            raise TypeError(
                'val_dataset must be a Dataset object or a dict, not {}'.format(
                    type(val_dataset)))
        self.val_data_loader = datasets.build_dataloader(
            self.val_dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode,
            shuffle=False)
        self.dist_mode = dist_mode
        self.eval_kwargs = eval_kwargs
        self.lookup = None
        if 'lookup' in eval_kwargs and eval_kwargs['lookup'] is not None:
            self.lookup = torch.load(eval_kwargs['lookup'])

    def before_run(self, runner):
        #
        runner.model.eval()
        if self.eval_kwargs['bn'] == 'train':
            for m in runner.model.module.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.track_running_stats = False
        #
        sample_func = lambda: np.random.randint(0,
                runner.model.module.backbone.candidate_num, 16)
        get_net_acc = lambda choice_indices: self._get_net_acc(runner, choice_indices)
        self._searcher = SearchPhase(
            initial_samples=self.eval_kwargs['initial_samples'],
            initial_sample=self.eval_kwargs['initial_sample'],
            selects=self.eval_kwargs['selects'],
            height_level=self.eval_kwargs['height_level'],
            sample_func=sample_func,
            get_net_acc=get_net_acc,
            logger=runner.logger,
            work_dir=runner.work_dir)
        self._searcher.run(target_accuracy=100, max_samples=self.eval_kwargs['max_samples'])
        #
        runner.model.train()
        if self.eval_kwargs['bn'] == 'train':
            for m in runner.model.module.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.track_running_stats = True
        #

    def _get_net_acc(self, runner, choice_indices):
        if self.lookup is not None:
            choice_indices_text = ''.join([str(i) for i in choice_indices])
            if choice_indices_text in self.lookup:
                return self.lookup[choice_indices_text]
            
        if self.eval_kwargs['bn'] == 'update':
            frozen_stages = runner.model.module.backbone.frozen_stages
            runner.model.module.backbone.frozen_stages = -1
            update_bn_stats(self.train_data_loader, runner, choice_indices, num_iters=49)
            runner.model.module.backbone.frozen_stages = frozen_stages

        func = lambda **x: runner.model(mode='test', **x, choice_indices=choice_indices)
        if self.dist_mode:
            results = dist_forward_collect(
                func, self.val_data_loader, runner.rank,
                len(self.val_dataset))  # dict{key: np.ndarray}
        else:
            results = nondist_forward_collect(func, self.val_data_loader,
                                              len(self.val_dataset))

        results = {name: self._evaluate(runner, torch.from_numpy(val), name)\
                       for name, val in results.items()}
        return results['head0']['head0_top1']

    def _evaluate(self, runner, results, keyword):
        eval_res = self.val_dataset.evaluate(
            results,
            keyword=keyword,
            **self.eval_kwargs['eval_param'])
        return eval_res
