# -*- coding: utf-8 -*-
"""
Continuous C3 reliability curriculum for APBOXNet.

Purpose
-------
First validate ONLY the path effect of Clean/Noisy sampling:

    Epoch   1-50 : clean:noisy = 1.0:1.0   -> 50.00% / 50.00%
    Epoch  51-100: noisy weight cosine anneals 1.0 -> 0.3
    Epoch 101-150: clean:noisy = 1.0:0.3   -> 76.92% / 23.08%

No unvalue.
No camo.
No saturation switch.
No EMA teacher.
No EMA-KD.
No Gaussian perturbation yet.

This is intended as the clean ablation after the original C3 result.
"""

has_test = True
deterministic = True
use_custom_worker_init = True

log_interval = 20
base_seed = 112358

__BATCHSIZE = 8
__NUM_EPOCHS = 150
__NUM_TR_SAMPLES = 4040

# WeightedRandomSampler draws exactly this many samples each epoch.
__SAMPLES_PER_EPOCH = 4040

# drop_last=True
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

    # Observe the whole trajectory closely.
    val_start_epoch=30,
    val_interval=5,
    save_val_ckpt=True,

    # This trainer intentionally does NOT use EMA/KD.
    ema_kd=dict(
        enable=False,
        lambda_kd=0.0,
    ),

    curriculum=dict(
        enable=True,

        # Only two pools in this ablation.
        pools=dict(
            clean="./data/pseudo_pool/shape/clean.txt",
            noisy="./data/pseudo_pool/shape/noisy.txt",
        ),

        num_samples_per_epoch=__SAMPLES_PER_EPOCH,

        # -------------------------------------------------------------
        # Continuous reliability schedule
        # -------------------------------------------------------------
        continuous_schedule=dict(
            enable=True,

            clean_weight=1.0,

            noisy=dict(
                # Stage I: Broad exposure.
                hold_end_epoch=50,
                start_weight=1.0,

                # Stage II: Reliability annealing.
                anneal_start_epoch=51,
                anneal_end_epoch=100,
                end_weight=0.3,

                # Smooth 1.0 -> 0.3.
                mode="cosine",
            ),

            # Stage III:
            # from epoch 101 onward, keep clean=1.0 / noisy=0.3.
            final_start_epoch=101,
        ),
    ),

    optimizer=dict(
        mode="adam",
        set_to_none=False,
        group_mode="finetune",
        cfg=dict(
            weight_decay=0,
            # pretrained backbone LR = 1e-5
            # retrained FPN/head LR = 1e-4
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
            # Do not collide LR drop with the 50 or 100 epoch
            # reliability-distribution transition.
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
        # Final paper evaluation:
        # names=["chameleon", "camo_te", "cod10k_te", "nc4k"],
    ),
)
