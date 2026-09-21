# -*- coding: utf-8 -*-
"""Strict FPN -> ZoomNeXt replacement for the existing Unvalue experiment.

The three-pool schedule and every Unvalue loss hyperparameter are intentionally
unchanged.  Only the segmentation architecture changes, so this run can be
compared directly with ``PvtV2B4_FPN_Unvalue``.
"""

has_test = True
deterministic = True
use_custom_worker_init = True
log_interval = 20
base_seed = 112358

__BATCHSIZE = 8
__NUM_EPOCHS = 150
__SAMPLES_PER_EPOCH = 4040
__ITER_PER_EPOCH = __SAMPLES_PER_EPOCH // __BATCHSIZE

model = dict(
    # ZoomNeXt architecture. The current repository implementation uses
    # image_l (1.5x) + image_m (1.0x) through shared PVTv2-B4.
    mid_dim=64,
    siu_groups=4,
    hmu_groups=6,

    # Unchanged Unvalue loss settings.
    mask_loss_mode="bce",
    q_switch_ratio=0.40,
    boundary_kernel=31,
    boundary_gain=5.0,
    box_dilate_kernel=9,
    teacher_confidence=0.90,
    teacher_disagreement=0.05,
    min_teacher_foreground=16,
    outside_weight=0.20,
    dynamic_weight=0.25,
    consistency_weight=0.05,
    dynamic_mix_max=0.50,
)

train = dict(
    batch_size=__BATCHSIZE,
    num_workers=4,
    use_amp=True,
    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    num_iters=None,
    lr=1e-4,
    grad_acc_step=1,
    val_start_epoch=30,
    val_interval=5,
    save_val_ckpt=True,

    unvalue_teacher=dict(
        start_epoch=41,
        freeze_epoch=120,
        ema_decay=0.99,
        hflip=True,
    ),

    curriculum=dict(
        enable=True,
        pools=dict(
            clean="./data/pseudo_pool/sam2/sam2_clean.txt",
            noisy="./data/pseudo_pool/sam2/sam2_noisy.txt",
            unvalue="./data/pseudo_pool/sam2/sam2_unvalue.txt",
        ),
        num_samples_per_epoch=__SAMPLES_PER_EPOCH,
        continuous_schedule=dict(enable=True),
        batch_schedule=[
            dict(
                name="mask_warmup",
                end_epoch=40,
                batch=dict(clean=4, noisy=4, unvalue=0),
            ),
            dict(
                name="box_introduction",
                end_epoch=60,
                batch=dict(clean=4, noisy=3, unvalue=1),
            ),
            dict(
                name="teacher_recovery",
                end_epoch=100,
                batch=dict(clean=3, noisy=3, unvalue=2),
            ),
            dict(
                name="hard_consolidation",
                end_epoch=120,
                batch=dict(clean=4, noisy=2, unvalue=2),
            ),
            dict(
                name="reliable_finetune",
                end_epoch=150,
                batch=dict(clean=5, noisy=2, unvalue=1),
            ),
        ],
    ),

    optimizer=dict(
        mode="adam",
        set_to_none=False,
        group_mode="finetune",
        cfg=dict(weight_decay=0, diff_factor=0.1),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(num_iters=0, initial_coef=0.01, mode="linear"),
        mode="step",
        cfg=dict(milestones=__ITER_PER_EPOCH * 120, gamma=0.1),
    ),
    bn=dict(
        freeze_status=True,
        freeze_affine=True,
        freeze_encoder=False,
    ),
    data=dict(
        shape=dict(h=384, w=384),
        names=["combined_tr"],
        use_box=True,
    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=8,
    clip_range=None,
    data=dict(
        shape=dict(h=384, w=384),
        names=["camo_te", "cod10k_te"],
    ),
)
