# -*- coding: utf-8 -*-
"""Z3: dual-scale ZoomNeXt + NC + Box-MHSIU2 + background prototype."""

has_test = True
deterministic = True
use_custom_worker_init = True

log_interval = 20
base_seed = 112358

__BATCHSIZE = 8
__NUM_EPOCHS = 150
__SAMPLES_PER_EPOCH = 4040
__ITER_PER_EPOCH = __SAMPLES_PER_EPOCH // __BATCHSIZE

# Passed directly to PvtV2B4_ZoomNeXt_Z3 by basemain_continuous.py.
model = dict(
    q_switch_ratio=0.40,

    # Router supervision is strongest before the NC switch, then relaxes.
    scale_weight_start=0.20,
    scale_weight_end=0.05,

    # Background prototype is delayed to avoid perturbing early structure
    # acquisition, then reaches full weight at 40% progress.
    background_weight=0.10,
    background_start_ratio=0.10,
    background_full_ratio=0.40,

    # Soft two-scale prior from union-box area.
    small_box_ratio=0.10,
    large_box_ratio=0.50,
    large_branch_prior_min=0.20,
    large_branch_prior_max=0.80,

    # Reduce routing supervision for BO and border-touching/OV-like boxes.
    uncertain_box_weight=0.25,

    background_margin=0.10,
    background_temperature=0.20,
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
            clean="./data/pseudo_pool/shape/clean.txt",
            noisy="./data/pseudo_pool/shape/noisy.txt",
        ),
        num_samples_per_epoch=__SAMPLES_PER_EPOCH,
        continuous_schedule=dict(
            enable=True,
            clean_weight=1.0,
            noisy=dict(
                hold_end_epoch=60,
                start_weight=1.0,
                anneal_start_epoch=61,
                anneal_end_epoch=130,
                end_weight=0.0,
                mode="cosine",
            ),
            final_start_epoch=131,
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
