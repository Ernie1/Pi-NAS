import mmcv
from mmcv.runner import EpochBasedRunner


class SearchRunner(EpochBasedRunner):
    def search(self, **kwargs):
        pass # search is called by hook before this function

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        self.data_loader = data_loaders[0]
        super().run(data_loaders, workflow, max_epochs, **kwargs)