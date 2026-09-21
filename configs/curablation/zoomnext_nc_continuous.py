# -*- coding: utf-8 -*-
"""Strict loss-only ablation: current PVT-B4 ZoomNeXt + continuous NC.

Run from the APBOXNet repository root:

    python basemain_continuous.py \
        --config configs/curablation/zoomnext_nc_continuous.py \
        --model-name PvtV2B4_ZoomNeXt_NC_Curriculum

Comparison target:

    python basemain_continuous.py \
        --config configs/curablation/zoomnext_nc_continuous.py \
        --model-name PvtV2B4_ZoomNeXt

Both commands use the same 384x384 data, optimizer, seed and Clean/Noisy
sampling schedule.  Only the training loss differs.
"""

has_test = True
deterministic = True
use_custom_worker_init = True

log_interval = 20
base_seed = 112358

__BATCHSIZE = 8
__NUM_EPOCHS = 150
__SAMPLES_PER_EPOCH = 4040
__ITER_PER_EPOCH = __SAMPLES_PER_EPOCH // __BATCHSIZE  # 505

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

    # Deliberately disabled for this clean loss-only experiment.
    ema_kd=dict(
        enable=False,
        lambda_kd=0.0,
    ),

    curriculum=dict(
        enable=True,
        pools=dict(
            clean="./data/pseudo_pool/shape/clean.txt",
            noisy="./data/pseudo_pool/shape/noisy.txt",
        ),
        num_samples_per_epoch=__SAMPLES_PER_EPOCH,

        # Match the current Stageout Clean/Noisy trajectory exactly.
        continuous_schedule=dict(
            enable=True,
            clean_weight=1.0,
            noisy=dict(
                # Epoch 1-60: clean:noisy = 1:1.
                hold_end_epoch=60,
                start_weight=1.0,

                # Epoch 61-130: noisy mass cosine-anneals to zero.
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
        cfg=dict(
            weight_decay=0,
            # Backbone LR = 1e-5; MHSIU2/RGPU/head LR = 1e-4.
            diff_factor=0.1,
        ),
    ),

    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(
            num_iters=0,
            initial_coef=0.01,
            mode="linear",
        ),
        mode="step",
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
        names=["combined_tr"],
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
