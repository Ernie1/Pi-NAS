from .registry import DATASETS
from .base import BaseDataset


@DATASETS.register_module
class ExtractDataset(BaseDataset):
    """Dataset for feature extraction
    """

    def __init__(self, data_source, pipeline):
        super(ExtractDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        # my
        if isinstance(img, tuple):
            img = img[0]
        #
        img = self.pipeline(img)
        return dict(img=img)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented


@DATASETS.register_module
class C2ExtractDataset(BaseDataset):
    """Dataset for feature extraction
    """

    def __init__(self, data_source, pipeline):
        super(C2ExtractDataset, self).__init__(data_source, pipeline)
        self.storage1 = []
        self.storage2 = []
        self.count = 0

    def __getitem__(self, idx):
        if len(self.storage1) < self.data_source.get_length():
            img = self.data_source.get_sample(idx)
            img = self.pipeline(img)
            self.storage1.append(img)
        elif len(self.storage2) < self.data_source.get_length():
            img = self.data_source.get_sample(idx)
            img = self.pipeline(img)
            self.storage2.append(img)
        elif self.count < self.data_source.get_length():
            img = self.storage1[self.count]
        else:
            img = self.storage2[self.count - self.data_source.get_length()]

        self.count += 1
        if self.count == self.data_source.get_length() * 2:
            self.count = 0
        return dict(img=img)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented