_base_ = './pinas_linear_training.py'
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = '/data/ImageNet/meta/train_labeled_50000_random_shuffle.txt'
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
    imgs_per_gpu=128,
    workers_per_gpu=8,
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
        type='SearchHook',
        train_dataset=data['train'],
        val_dataset=data['val'],
        imgs_per_gpu=128,
        workers_per_gpu=8,
        eval_param=dict(topk=(1, )),
        max_samples=1000,
        initial_sample=80,
        selects=8,
        height_level=[400, 800, 1600, 3200],
        bn='update',)
]
# runtime settings
total_epochs = 1
workflow = [('search', 1)]
