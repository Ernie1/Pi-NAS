# Π-NAS
This repository provides the evaluation code of our submitted paper: ***Π-NAS: Improving Neural Architecture Search by Reducing Supernet Training Consistency Shift***.

## Our Trained Models 
- Here is a summary of our searched models:

    |    ImageNet    |  FLOPs    |   Params |   Acc@1   |   Acc@5   |
    |:---------:|:---------:|:---------:|:---------:|:---------:|
    | Π-NAS-*cls*    |   5.38G     |	27.1M    |      81.6%    |      95.7%   |

    |    Mask-RCNN on COCO 2017   |  APbb   |   APmk |
    |:---------:|:---------:|:---------:|
    | Π-NAS-*trans*    |      44.07   |      39.50   |

    |    DeeplabV3 on ADE20K    |  pixAcc   |   mIoU |
    |:---------:|:---------:|:---------:|
    | Π-NAS-*trans*    |      81.27    |      45.47   |

    |    DeeplabV3 on Cityscapes   |   mIoU |
    |:---------:|:---------:|
    | Π-NAS-*trans*    |     80.70   |

## Usage
### 1. Requirements
- Install third-party requirements with command `pip install -e .`
	- We adapt the code from [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding) and [detectron2](https://github.com/facebookresearch/detectron2) to validate our models.
- Prepare ImageNet, COCO 2017, ADE20K and Cityscapes datasets
    - Our data paths are at `/data/ImageNet`, `/data/coco`, `/data/ADEChallengeData2016` and `/data/citys`, respectively.
    - You can specify COCO's data path through environment variable `DETECTRON2_DATASETS` and others in `experiments/recognition/verify.py`, `encoding/datasets/ade20k.py` and `encoding/datasets/cityscapes.py`.
- [Download our checkpoint files](https://drive.google.com/drive/folders/1GAOAtLz8appoEdMpO2_s6DajJZ4RZJGv?usp=sharing)

### 2. Evaluate our models

- You can evaluate our models with the following command:

    |    ImageNet    |  FLOPs    |   Params |   Acc@1   |   Acc@5   |
    |:---------:|:---------:|:---------:|:---------:|:---------:|
    | Π-NAS-*cls*    |   5.38G     |	27.1M    |      81.6%    |      95.7%   |

    ```bash
    python experiments/recognition/verify.py --dataset imagenet --model alone_resnest50 --choice-indices 3 0 1 3 2 3 1 2 0 3 2 1 3 0 3 2 --resume /path/to/PiNAS_cls.pth.tar
    ```

    |    Mask-RCNN on COCO 2017    |  APbb   |   APmk |
    |:---------:|:---------:|:---------:|
    | Π-NAS-*trans*    |      44.07   |      39.50   |

    ```bash
    DETECTRON2_DATASETS=/data python experiments/detection/plain_train_net.py --config-file experiments/detection/configs/mask_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS /path/to/PiNAS_trans_COCO.pth MODEL.RESNETS.CHOICE_INDICES [3,3,3,3,1,1,3,3,3,0,0,1,1,0,2,1]
    ```

    |    DeeplabV3 on ADE20K    |  pixAcc   |   mIoU |
    |:---------:|:---------:|:---------:|
    | Π-NAS-*trans*    |      81.27    |      45.47   |

    ```bash
    python experiments/segmentation/test.py --dataset ADE20K --model deeplab --backbone alone_resnest50 --choice-indices 3 3 3 3 1 1 3 3 3 0 0 1 1 0 2 1 --aux --se-loss --resume /path/to/PiNAS_trans_ade.pth.tar --eval
    ```

    |    DeeplabV3 on Cityscapes   |   mIoU |
    |:---------:|:---------:|
    | Π-NAS-*trans*    |     80.70   |

    ```bash
    python experiments/segmentation/test.py --dataset citys --base-size 2048 --crop-size 768 --model deeplab --backbone alone_resnest50 --choice-indices 3 3 3 3 1 1 3 3 3 0 0 1 1 0 2 1 --aux --se-loss --resume /path/to/PiNAS_trans_citys.pth.tar --eval
    ```

## TODO
Training and Searching code will be released in the future.
