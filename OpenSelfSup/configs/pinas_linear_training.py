_base_ = './base.py'
# model settings
model = dict(
    type='SuperClassification',
    pretrained=None,
    with_sobel=False,
    backbone=dict(
        type='SuperResNet',
        num_classes=1000, in_chans=3,
        stem_type='deep', stem_width=32, avg_down=True,
        block_args=dict(
            candidate_args=[
                dict(block='ResNestBottleneck', radix=1, cardinality=1,
                        base_width=64, avd=True, avd_first=False),
                dict(block='ResNestBottleneck', radix=2, cardinality=1,
                        base_width=64, avd=True, avd_first=False),
                dict(block='ResNestBottleneck', radix=1, cardinality=2, 
                        base_width=42, avd=True, avd_first=False),
                dict(block='ResNestBottleneck', radix=2, cardinality=2, 
                        base_width=40, avd=True, avd_first=False)],
            downsample_share=True),
        candidate_num=4,
        norm_eval='all_false',
        frozen_stages=4),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=1000))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = '/data/ImageNet/meta/train_labeled.txt'
data_train_root = '/data/ImageNet/train'
data_test_list = '/data/ImageNet/meta/my_val_labeled.txt'
data_test_root = '/data/ImageNet/val'
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=True,
        interval=1,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
optimizer = dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.)
# learning policy
lr_config = dict(policy='step', step=[60, 80])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 100
use_fp16 = True
optimizer_config = dict(use_fp16=use_fp16)
