_base_ = './gfl_r50_fpn_1x_coco_pcb.py'

model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    bbox_head=dict(
        num_classes=6
    )
)

# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
# multi-scale training
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

classes = ("missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper")
data = dict(
    samples_per_gpu=8,  # batch size
    workers_per_gpu=2,  # num_workers
    train=dict(
        img_prefix='/mnt/data/yx/defectdet/PCB/images/train',
        classes=classes,
        ann_file='/mnt/data/yx/defectdet/PCB/annotations/instances_train.json'),
    val=dict(
        img_prefix='/mnt/data/yx/defectdet/PCB/images/val',
        classes=classes,
        ann_file='/mnt/data/yx/defectdet/PCB/annotations/instances_val.json'),
    test=dict(
        img_prefix='/mnt/data/yx/defectdet/PCB/images/test',
        classes=classes,
        ann_file='/mnt/data/yx/defectdet/PCB/annotations/instances_test.json'))

load_from = '/home/yx/fgd/checkpoints/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
