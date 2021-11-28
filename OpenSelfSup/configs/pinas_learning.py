_base_ = './base.py'
# model settings
model = dict(
    type='PiNAS',
    pretrained=None,
    queue_len=49152,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='SuperResNet',
        num_classes=1000, in_chans=3,
        stem_type='deep', stem_width=32, avg_down=True,
        block_args=dict(candidate_args=[
            dict(block='ResNestBottleneck', radix=1, cardinality=1,
                    base_width=64, avd=True, avd_first=False),
            dict(block='ResNestBottleneck', radix=2, cardinality=1,
                    base_width=64, avd=True, avd_first=False),
            dict(block='ResNestBottleneck', radix=1, cardinality=2, 
                    base_width=42, avd=True, avd_first=False),
            dict(block='ResNestBottleneck', radix=2, cardinality=2, 
                    base_width=40, avd=True, avd_first=False),
        ]),
        candidate_num=4),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = '/data/ImageNet/meta/train.txt'
data_train_root = '/data/ImageNet/train'
dataset_type = 'ContrastiveDatasetX'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.4)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=0.5),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=24,
    workers_per_gpu=8,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=5)
# runtime settings
total_epochs = 100
use_fp16 = True
optimizer_config = dict(use_fp16=use_fp16)
