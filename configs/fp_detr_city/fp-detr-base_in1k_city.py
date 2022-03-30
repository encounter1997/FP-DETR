_base_ = [
    '../_base_/datasets/cityscapes_detection.py', '../_base_/default_runtime.py'
]
model = dict(
    type='DeformableDETR',
    backbone=dict(
        type='PatchTokenizerMS',
        pretrain_img_size=224,
        in_chans=3,
        embed_dim=384,
        out_indices=(8, 16, 32),
        new_norms=True,
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[384, 384, 384],
        kernel_size=1,
        out_channels=480,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_query=300,
        num_classes=8,
        in_channels=480,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        transformer=dict(
            type='DeformableDetrTransformerEncOnly',
            concate_layer=6,
            query_shape=(15, 20),
            num_feature_levels=4,
            encoder1=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=480,
                        num_heads=10,
                        dropout=0.1),
                    feedforward_channels=1920,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            encoder2=dict(
                type='DeformableDetrTransformerEncOnlyEncoder',
                num_layers=6,
                return_intermediate=(True, True, True, True, True, True),
                transformerlayers=dict(
                    type='BaseTransformerLayerCustom',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',  # only applied to the object queries
                            embed_dims=480,  # scale
                            num_heads=10,
                            dropout=0.1,
                            shared=True),
                        dict(
                            type='MultiScaleDeformableAttentionCustom',  # Custom with mask_object_query supported
                            mask_object_query=True,
                            num_levels=5,  # treat object queries as additional feature level
                            dropout=0.1,
                            num_heads=10,
                            embed_dims=480),
                    ],
                    feedforward_channels=1920,
                    ffn_dropout=0.1,
                    operation_order=('prompt_self_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm'))),
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=240,  # scale
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(dataset=dict(filter_empty_gt=False, pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
# runtime config
checkpoint_config = dict(interval=1)
# load imagenet pretrained model
load_from = "pretrained_weights/fp-detr-base/pretrained_epoch_299.pth"

# log config
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='FP-DETR',
                name='fp-detr-base_in1k_city'))
    ])