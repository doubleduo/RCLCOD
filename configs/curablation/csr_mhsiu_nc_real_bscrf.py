# -*- coding: utf-8 -*-
"""Exact continuous Clean/Noisy REAL config, with box masks loaded for fair
architectural ablations A0/A0M/A1/A1P/A2/A3.

Derived from: configs/curablation/csr_mhsiu_nc_real.py in RCLCOD main.
Only change to train settings is train.data.use_box=True; model-name controls
module variant, all share the same schedule, seed, optimizer and evaluation.
NOTE: Baseline model uses BCE (filename 'nc' alone does not enable NC).
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
__NUM_ITERS = __NUM_EPOCHS * __ITER_PER_EPOCH

train = dict(
    batch_size=__BATCHSIZE,
    num_workers=4,
    use_amp=True,
    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    num_iters=None,
    lr=1e-4,
    grad_acc_step=1,
    val_start_epoch=10,
    val_interval=5,
    save_val_ckpt=True,
    ema_kd=dict(enable=False, lambda_kd=0.0),
    curriculum=dict(
        enable=True,
        pools=dict(
            clean='./data/pseudo_pool/sam2/sam2_gt_clean.txt',
            noisy='./data/pseudo_pool/sam2/sam2_gt_noisy.txt',
        ),
        num_samples_per_epoch=__SAMPLES_PER_EPOCH,
        continuous_schedule=dict(
            enable=True,
            clean_weight=1.0,
            noisy=dict(
                hold_end_epoch=60,
                start_weight=1.0,
                anneal_start_epoch=61,
                anneal_end_epoch=140,
                end_weight=0.4,
                mode='cosine',
            ),
            final_start_epoch=141,
        ),
    ),
    optimizer=dict(
        mode='adam',
        set_to_none=False,
        group_mode='finetune',
        cfg=dict(
            weight_decay=0,
            diff_factor=0.1,
        ),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(
            num_iters=0,
            initial_coef=0.01,
            mode='linear',
        ),
        mode='step',
        cfg=dict(
            milestones=__ITER_PER_EPOCH * 120,
            gamma=0.1,
        ),
    ),
    bn=dict(
        freeze_status=True,
        freeze_affine=True,
        freeze_encoder=False,
    ),
    data=dict(
        shape=dict(h=384, w=384),
        names=['combined_tr'],
        use_box=True,  # <-- the ONLY training-setting change vs real config
    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=8,
    clip_range=None,
    data=dict(
        shape=dict(h=384, w=384),
        names=['camo_te', 'cod10k_te'],
    ),
)
