# -*- coding: utf-8 -*-
"""
B2: PvtV2B4 FPN + Noisy-COD loss curriculum + CSR/MHSIU residual.

Recommended command:
    python basemain_continuous.py \
        --config configs/curablation/csr_mhsiu_nc_continuous.py \
        --model-name PvtV2B4_FPN_CSR_NC_Curriculum

Sampling schedule is kept the same as the current continuous baseline so the
only new architectural variable is CSR/MHSIU residual.

Model-internal schedules (driven by iter_percentage):
    epoch 1-60   : q=2, CSR alpha 0.15 -> 0.30
    epoch 61-100 : q=1, CSR alpha 0.30 -> 1.00
    epoch 101-150: q=1, CSR alpha = 1.00

This config intentionally excludes unvalue for the first CSR ablation.
After B2 is stable, use the same model with the previously prepared
Clean/Noisy/Unvalue continuous trainer/config.
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
__NUM_ITERS = __NUM_EPOCHS * __ITER_PER_EPOCH          # 75750

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

    ema_kd=dict(
        enable=False,
        lambda_kd=0.0,
    ),

    curriculum=dict(
        enable=True,

        pools=dict(
            clean="./data/pseudo_pool/best/clean.txt",
            noisy="./data/pseudo_pool/best/noisy.txt",
        ),

        num_samples_per_epoch=__SAMPLES_PER_EPOCH,

        continuous_schedule=dict(
            enable=True,
            clean_weight=1.0,

            noisy=dict(
                # Keep broad exposure during the early-learning stage.
                hold_end_epoch=60,
                start_weight=1.0,

                # Reliability annealing.
                anneal_start_epoch=91,
                anneal_end_epoch=140,
                end_weight=0.4,
                mode="cosine",
            ),

            final_start_epoch=141,
        ),
    ),

    optimizer=dict(
        mode="adam",
        set_to_none=False,
        group_mode="finetune",
        cfg=dict(
            weight_decay=0,
            # PVT backbone 1e-5; FPN/CSR/ZeroConv/head 1e-4.
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
            # Do not drop LR at the 60/100 curriculum transitions.
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
