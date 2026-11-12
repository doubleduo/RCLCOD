
# -*- coding: utf-8 -*-
"""
PIS: PvtV2B4 FPN + Clean/Noisy Continuous Curriculum.

Pseudo labels:
    SAM1 GT-Box -> Iter2

Candidate IoU grading:
    Clean:  IoU >= 0.95       (515)
    Noisy:  0.80 <= IoU < 0.95 (606)
    Reject: IoU < 0.80        (329, excluded)

Curriculum:
    Epoch 1-50:    Clean only
    Epoch 51-160:  Noisy weight 0.0 -> 0.8
    Epoch 161-170: Clean + Noisy, fixed weights

No EMA knowledge distillation.
"""

has_test = True
deterministic = True
use_custom_worker_init = True

log_interval = 20
base_seed = 112358

__BATCHSIZE = 12
__NUM_EPOCHS = 170

# Fixed epoch sampling budget.
# Sampling may repeat samples within an epoch.
__SAMPLES_PER_EPOCH = 1450

__ITER_PER_EPOCH = __SAMPLES_PER_EPOCH // __BATCHSIZE
__NUM_ITERS = __NUM_EPOCHS * __ITER_PER_EPOCH

use_box = True

train = dict(
    batch_size=__BATCHSIZE,
    num_workers=4,
    use_amp=True,

    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    num_iters=None,

    lr=1e-4,
    grad_acc_step=1,

    val_start_epoch=60,
    val_interval=10,
    save_val_ckpt=True,

    ema_kd=dict(
        enable=False,
        lambda_kd=0.0,
    ),

    # Clean/Noisy curriculum
    curriculum=dict(
        enable=True,

        pools=dict(
            clean="./Dataset_Polyp/clean (1).txt",
            noisy="./Dataset_Polyp/noisy (1).txt",
        ),

        num_samples_per_epoch=__SAMPLES_PER_EPOCH,

        continuous_schedule=dict(
            enable=True,

            clean_weight=1.0,

            noisy=dict(
                hold_end_epoch=50,
                start_weight=1.0,

                anneal_start_epoch=100,

                anneal_end_epoch=140,
                end_weight=0.4,
                mode="cosine",
            ),

            final_start_epoch=161,
        ),
    ),

    optimizer=dict(
        mode="adam",
        set_to_none=False,
        group_mode="finetune",
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
            mode="linear",
        ),
        mode="step",
        cfg=dict(
            milestones=int(__NUM_ITERS * 2 / 3),
            gamma=0.1,
        ),
    ),

    bn=dict(
        freeze_status=True,
        freeze_affine=True,
        freeze_encoder=False,
    ),

    data=dict(
        shape=dict(h=512, w=512),
        names=["train_tr"],
    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=8,
    clip_range=None,
    data=dict(
        shape=dict(h=512, w=512),
        names=[
            "cvc-300",
            "cvc-clinicdb",
            "cvc-colondb",
            "etis-laribpolypdb",
            "kvasir",
        ],
    ),
)
