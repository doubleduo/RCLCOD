# -*- coding: utf-8 -*-
"""A4: PvtV2B4_FPN + RNC + RSC + Reliability-aware Boundary UAL."""

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
    clean_reliability=1.0,
    noisy_reliability=0.5,
    q_mode="binary",
    wbce_kernel=31,
    wbce_boundary_gain=5.0,
    boundary_kernel=7,
    sc_weight=0.10,
    sc_alpha=1.0,
    sc_max_level=1,
    sc_loss_clip=10.0,
    bual_weight=0.05,      # conservative start; only reliable boundaries contribute strongly
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
    ema_kd=dict(enable=False, lambda_kd=0.0),
    curriculum=dict(
        enable=True,
        pools=dict(
            clean="./data/pseudo_pool/real/clean.txt",
            noisy="./data/pseudo_pool/real/noisy.txt",
        ),
        num_samples_per_epoch=__SAMPLES_PER_EPOCH,
        continuous_schedule=dict(
            enable=True,
            clean_weight=1.0,
            noisy=dict(
                hold_end_epoch=150,
                start_weight=1.0,
                anneal_start_epoch=151,
                anneal_end_epoch=151,
                end_weight=1.0,
                mode="linear",
            ),
            final_start_epoch=151,
        ),
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
    bn=dict(freeze_status=True, freeze_affine=True, freeze_encoder=False),
    data=dict(shape=dict(h=384, w=384), names=["combined_tr"]),
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=8,
    clip_range=None,
    data=dict(shape=dict(h=384, w=384), names=["camo_te", "cod10k_te"]),
)
