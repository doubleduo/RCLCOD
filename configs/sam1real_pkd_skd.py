# -*- coding: utf-8 -*-
"""SAM1 pseudo-label training with PKD and SKD on the plain FPN baseline.

Run:
    python basemain_continuous.py \
        --config configs/sam1/sam1real_pkd_skd.py \
        --model-name PvtV2B4_FPN_PKD_SKD \
        --output-dir outputs/1sam1/pkd_skd
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

# SAM-COD uses 0.5 * PKD and 1.0 * SKD.  The network architecture remains
# PvtV2B4_FPN_Baseline; only its training objective is extended.
model = dict(
    pkd_weight=0.5,
    skd_weight=1.0,
    pkd_erode_kernel=5,
    pkd_min_valid_pixels=16,
    skd_view="image_s",
    skd_ramp_start=0.0,
    skd_ramp_end=0.0,
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

    val_start_epoch=20,
    val_interval=5,
    save_val_ckpt=True,

    ema_kd=dict(
        enable=False,
        lambda_kd=0.0,
    ),

    curriculum=dict(
        enable=True,
        pools=dict(
            clean="./data/pseudo_pool/1sam1/IoU7090/clean.txt",
            noisy="./data/pseudo_pool/1sam1/IoU7090/noisy.txt",
        ),
        num_samples_per_epoch=__SAMPLES_PER_EPOCH,
        # Reuse the original SAM1 per-epoch dynamic curriculum unchanged.
        # Epoch 1-60: C:N = 1.0:1.0.
        # Epoch 61-120: Noisy mass cosine-anneals from 1.0 to 0.4.
        # Epoch 121-150: C:N = 1.0:0.4.
        continuous_schedule=dict(
            enable=True,
            clean_weight=1.0,
            noisy=dict(
                hold_end_epoch=60,
                start_weight=1.0,
                anneal_start_epoch=61,
                anneal_end_epoch=120,
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
        # Required by online box-PKD.  The box_json path is read from
        # dataset.yaml -> combined_tr -> box_json.
        use_box=True,
    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=8,
    clip_range=None,
    data=dict(
        shape=dict(h=384, w=384),
        names=["chameleon", "camo_te", "cod10k_te", "nc4k"],
    ),
)
