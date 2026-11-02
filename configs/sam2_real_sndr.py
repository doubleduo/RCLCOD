# -*- coding: utf-8 -*-
"""Strict SNDR ablation on the best FPN-ZoomNeXt training recipe.

Recommended formal runs:

    # D1: progressive upsampling head only
    python basemain_continuous.py \
        --config configs/sam2_real_sndr.py \
        --model-name PvtV2B4_FPN_ZoomNeXt_PUH \
        --output-dir realsam2/_SNDR_D1_PUH

    # D2: P2-only RGB/Sobel fusion
    python basemain_continuous.py \
        --config configs/sam2_real_sndr.py \
        --model-name PvtV2B4_FPN_ZoomNeXt_SNDR_Single \
        --output-dir realsam2/_SNDR_D2_Single

    # D3: recommended P2 + half-resolution RGB/Sobel fusion
    python basemain_continuous.py \
        --config configs/sam2_real_sndr.py \
        --model-name PvtV2B4_FPN_ZoomNeXt_SNDR \
        --output-dir realsam2/_SNDR_D3_Dual

Only the model architecture changes.  Pseudo labels, Clean/Noisy sampling,
loss, optimizer, scheduler, seed and input resolution match the successful
FPN-ZoomNeXt experiment.
"""

has_test = True
deterministic = True
use_custom_worker_init = True

log_interval = 20
base_seed = 112358
# Preserve the current dataset interface. SNDR itself never consumes box data.
use_box = True

__BATCHSIZE = 8
__NUM_EPOCHS = 150
__SAMPLES_PER_EPOCH = 4040
__ITER_PER_EPOCH = __SAMPLES_PER_EPOCH // __BATCHSIZE  # 505

# basemain_continuous.py forwards this dictionary to the model constructor.
# The named D1/D2/D3 wrappers fix their own detail mode, so the same config
# can be used for all three strict ablations.
model = dict(
    fpn_dim=64,
    siu_groups=4,
    hmu_groups=6,
    clean_only_highres=False,
    detail_half_channels=32,
    detail_full_channels=16,
    detail_gate_hidden=16,
    detail_gn_groups=8,
    zero_init_output=True,
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

    # Evaluation does not alter optimization.  Five-epoch spacing is used so
    # a new head whose optimum moves away from epoch 140 is not missed.
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
            clean="./data/pseudo_pool/sam2refine/high.txt",
            noisy="./data/pseudo_pool/sam2refine/middle.txt",
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
            # Encoder: 1e-5. Existing decoder and SNDR: 1e-4.
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
            milestones=__ITER_PER_EPOCH * 120,  # 60600 iterations
            gamma=0.1,
        ),
    ),

    # Existing encoder/decoder BN handling stays unchanged.  Every new SNDR
    # block uses GroupNorm, so none of its normalization is accidentally frozen.
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
        names=["chameleon", "camo_te", "cod10k_te", "nc4k"],
    ),
)
