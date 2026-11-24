# RTMDet-L v0 (Phase 6, option A): 2 classes (swimmer, person), input 1024, AMP, EMA, fine-tuned from COCO.
# Backbone frozen for the first 10 % of epochs, then trained at 0.1x LR (FreezeBackboneHook in tools\train_rtmdet.py).
# Mosaic for the first 85 %, then off (stage 2). Copy-paste is not used: it needs instance masks, the seed is boxes only.
# Same COCO export as D-FINE (seed R1: reviewed x2 + auto-accepted x1; empty frames kept).
_base_ = '../third_party/mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'

TR = 'C:/lifeguard/training/'
size = 1024
max_epochs = 60
stage2_num_epochs = 9   # last 15 %
freeze_epochs = 6       # first 10 %
base_lr = 0.001
classes = ('swimmer', 'person')
metainfo = dict(classes=classes, palette=[(0, 160, 255), (255, 128, 0)])

norm = dict(type='BN')  # one GPU, no SyncBN
model = dict(
    backbone=dict(norm_cfg=norm),
    neck=dict(norm_cfg=norm),
    bbox_head=dict(num_classes=2, norm_cfg=norm))

# pool photometrics (MASTER_PROMPT Phase 6): pixel-level only, boxes unchanged
albu = [
    dict(type='RandomBrightnessContrast', brightness_limit=0.25, contrast_limit=0.25, p=0.5),
    dict(type='RandomSunFlare', flare_roi=(0, 0, 1, 1), src_radius=120, num_flare_circles_range=(2, 5), p=0.15),
    dict(type='MotionBlur', blur_limit=(3, 9), p=0.25),
    dict(type='GaussNoise', std_range=(0.01, 0.05), p=0.25),
    dict(type='ImageCompression', quality_range=(35, 90), p=0.4),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(size, size), pad_val=114.0, max_cached_images=20),
    dict(type='RandomResize', scale=(size * 2, size * 2), ratio_range=(0.5, 2.0), keep_ratio=True),  # no strong warps
    dict(type='RandomCrop', crop_size=(size, size)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='Albu', transforms=albu),
    dict(type='RandomFlip', prob=0.5),  # horizontal only
    dict(type='Pad', size=(size, size), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=(size, size), ratio_range=(0.75, 1.25), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(size, size)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(size, size), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(size, size), keep_ratio=True),
    dict(type='Pad', size=(size, size), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=2,  # batch 4 unfrozen: 7.7 GB of 8 GB and ~5 s/step (spills); batch 2: 4.3 GB
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        data_root='',
        metainfo=metainfo,
        ann_file=TR + 'data/coco/seed_r1/train.json',
        data_prefix=dict(img=TR + 'data/frames/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=1),  # keep empty frames
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root='',
        metainfo=metainfo,
        ann_file=TR + 'data/coco/seed_r1/val.json',
        data_prefix=dict(img=TR + 'data/frames/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=TR + 'data/coco/seed_r1/val.json', metric='bbox', proposal_nums=(100, 1, 10))
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=max_epochs, val_interval=5, dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True,
                       custom_keys={'backbone': dict(lr_mult=0.1)}))  # 0.1x LR once unfrozen
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-3, by_epoch=False, begin=0, end=300),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=max_epochs // 2, end=max_epochs,
         T_max=max_epochs // 2, by_epoch=True, convert_to_iter_based=True),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP'),
    logger=dict(type='LoggerHook', interval=25))
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49),
    dict(type='PipelineSwitchHook', switch_epoch=max_epochs - stage2_num_epochs, switch_pipeline=train_pipeline_stage2),
    dict(type='FreezeBackboneHook', epochs=freeze_epochs),
]

load_from = TR + 'weights/rtmdet_l_coco_statedict.pth'
work_dir = TR + 'work_dirs/rtmdet_l_v0'
randomness = dict(seed=0)
auto_scale_lr = dict(enable=False)
