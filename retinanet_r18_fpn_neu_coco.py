'''
Author: Iccccy.xie
Date: 2023-03-17 15:14:08
LastEditTime: 2023-03-17 21:49:36
LastEditors: Iccccy.xie(binicey@outlook.com)
FilePath: /AI-lab/model-all/mmdetection-self-config/retinanet_r18_fpn_neu_coco.py
'''
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# optimizer
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]),
    bbox_head=dict(num_classes=6))
optimizer = dict(type='SGD', lr=0.005, momentum=0.5, weight_decay=0.0001)


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('crazing', 'inclusion', 'patches', 'pitted_surface','rolled-in_scale', 'scratches',)
           
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_prefix=
        '/kaggle/input/neu-coco-train-val/neu-coco-train-val/images/train2017',
        classes=classes,
        ann_file=
        '/kaggle/input/neu-coco-train-val/neu-coco-train-val/annotations/instances_train2017.json',
        pipeline=train_pipeline
    ),
    val=dict(
        img_prefix=
        '/kaggle/input/neu-coco-train-val/neu-coco-train-val/images/val2017',
        classes=classes,
        ann_file=
        '/kaggle/input/neu-coco-train-val/neu-coco-train-val/annotations/instances_val2017.json',
        pipeline=test_pipeline

    ),
    test=dict(
        img_prefix=
        '/mnt/work/AI-lab/datasets/neu-coco-train-val/images/val2017',
        classes=classes,
        ann_file=
        '/mnt/work/AI-lab/datasets/neu-coco-train-val/annotations/instances_val2017.json',
        pipeline=test_pipeline
    ))

