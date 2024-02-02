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

# model settings
find_unused_parameters = True
temp = 0.5
alpha_fgd = 0.001
beta_fgd = 0.0005
gamma_fgd = 0.0005
lambda_fgd = 0.000005
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained='https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth',
    init_student=True,
    distill_cfg=[dict(student_module='neck.fpn_convs.4.conv',
                      teacher_module='neck.fpn_convs.4.conv',
                      output_hook=True,
                      methods=[dict(type='FeatureLoss',
                                    name='loss_fgd_fpn_4',
                                    student_channels=256,
                                    teacher_channels=256,
                                    temp=temp,
                                    alpha_fgd=alpha_fgd,
                                    beta_fgd=beta_fgd,
                                    gamma_fgd=gamma_fgd,
                                    lambda_fgd=lambda_fgd,
                                    )
                               ]
                      ),
                 dict(student_module='neck.fpn_convs.3.conv',
                      teacher_module='neck.fpn_convs.3.conv',
                      output_hook=True,
                      methods=[dict(type='FeatureLoss',
                                    name='loss_fgd_fpn_3',
                                    student_channels=256,
                                    teacher_channels=256,
                                    temp=temp,
                                    alpha_fgd=alpha_fgd,
                                    beta_fgd=beta_fgd,
                                    gamma_fgd=gamma_fgd,
                                    lambda_fgd=lambda_fgd,
                                    )
                               ]
                      ),
                 dict(student_module='neck.fpn_convs.2.conv',
                      teacher_module='neck.fpn_convs.2.conv',
                      output_hook=True,
                      methods=[dict(type='FeatureLoss',
                                    name='loss_fgd_fpn_2',
                                    student_channels=256,
                                    teacher_channels=256,
                                    temp=temp,
                                    alpha_fgd=alpha_fgd,
                                    beta_fgd=beta_fgd,
                                    gamma_fgd=gamma_fgd,
                                    lambda_fgd=lambda_fgd,
                                    )
                               ]
                      ),
                 dict(student_module='neck.fpn_convs.1.conv',
                      teacher_module='neck.fpn_convs.1.conv',
                      output_hook=True,
                      methods=[dict(type='FeatureLoss',
                                    name='loss_fgd_fpn_1',
                                    student_channels=256,
                                    teacher_channels=256,
                                    temp=temp,
                                    alpha_fgd=alpha_fgd,
                                    beta_fgd=beta_fgd,
                                    gamma_fgd=gamma_fgd,
                                    lambda_fgd=lambda_fgd,
                                    )
                               ]
                      ),
                 dict(student_module='neck.fpn_convs.0.conv',
                      teacher_module='neck.fpn_convs.0.conv',
                      output_hook=True,
                      methods=[dict(type='FeatureLoss',
                                    name='loss_fgd_fpn_0',
                                    student_channels=256,
                                    teacher_channels=256,
                                    temp=temp,
                                    alpha_fgd=alpha_fgd,
                                    beta_fgd=beta_fgd,
                                    gamma_fgd=gamma_fgd,
                                    lambda_fgd=lambda_fgd,
                                    )
                               ]
                      ),

                 ]
)

student_cfg = '../configs/gfl/gfl_r50_fpn_1x_coco.py'
teacher_cfg = '../configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py'
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

classes = ("missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper")
data = dict(
    samples_per_gpu=8,  # batch size
    workers_per_gpu=8,  # num_workers
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

log_config = dict(
    interval=1
)
