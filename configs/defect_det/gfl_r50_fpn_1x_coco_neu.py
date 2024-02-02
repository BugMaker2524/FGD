_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='GFL',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=6,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

classes = ("crazing", "patches", "inclusion", "pitted_surface", "rolled-in_scale", "scratches")
data = dict(
    samples_per_gpu=16,  # batch size
    workers_per_gpu=2,  # num_workers
    train=dict(
        img_prefix='/mnt/data/yx/defectdet/NEU-DET/images/train',
        classes=classes,
        ann_file='/mnt/data/yx/defectdet/NEU-DET/annotations/instances_train.json'),
    val=dict(
        img_prefix='/mnt/data/yx/defectdet/NEU-DET/images/val',
        classes=classes,
        ann_file='/mnt/data/yx/defectdet/NEU-DET/annotations/instances_val.json'),
    test=dict(
        img_prefix='/mnt/data/yx/defectdet/NEU-DET/images/test',
        classes=classes,
        ann_file='/mnt/data/yx/defectdet/NEU-DET/annotations/instances_test.json'))

log_config = dict(
    interval=1
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

load_from = '/home/yx/fgd/checkpoints/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth'
