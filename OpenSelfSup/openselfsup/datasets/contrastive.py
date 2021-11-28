# import torch
from PIL import Image
# from .registry import DATASETS
# from .base import BaseDataset
from .base import *


@DATASETS.register_module
class ContrastiveDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline):
        super(ContrastiveDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented


@DATASETS.register_module
class ContrastiveDatasetX(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline):
        super(ContrastiveDatasetX, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        img3 = self.pipeline(img)
        img4 = self.pipeline(img)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0),
                             img3.unsqueeze(0), img4.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
