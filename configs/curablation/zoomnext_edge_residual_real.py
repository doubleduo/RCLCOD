# -*- coding: utf-8 -*-
"""SAM2 Refine real-pool, fair ZoomNeXt EdgeRes / Edge-UAL ablations.

For EVERY model_name use this SAME config. Model toggles are selected by
class name, so the only variables are edge residual and edge-only UAL.

Dataset is the user's SAM2 high/middle quality split; no NC, no EMA/KD.
No basemain modifications are required by the new model.
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

# All ablations share these hyperparameters. Individual subclasses toggle
# edge residual and UAL scope; B0/A2 have zero UAL by design.
model = dict(
    edge_levels=(3, 2),          # P3/P2 only; don't corrupt deep semantics
    edge_strength=1.0,         # ZeroConv starts at EXACTLY 0 regardless
    edge_kernel=7,             # seven-pixel pseudo-label boundary band
    ual_weight=0.20,           # lower than original full-map ZoomNeXt UAL
    ual_start_ratio=0.40,      # delayed; 0 for early 40% of training
    ual_full_ratio=0.70,       # cosine ramp to full by 70% of training
    edge_ual_clean_only=False, # True requires pool_id; recommended extra ablation
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
    save_val_ckpt=False,
    ema_kd=dict(enable=False, lambda_kd=0.0),
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
                hold_end_epoch=100,
                start_weight=1.0,
                anneal_start_epoch=101,
                anneal_end_epoch=130,
                end_weight=0.4,
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
        cfg=dict(milestones=__ITER_PER_EPOCH*120, gamma=0.1),
    ),
    bn=dict(freeze_status=True, freeze_affine=True, freeze_encoder=False),
    data=dict(
        shape=dict(h=384, w=384),
        names=["combined_tr"],
        # This reproduces existing ZoomNeXt's 384 -> 576 interpolation.
        # Set native_large=True in BOTH train and test ONLY when evaluating
        # genuine original-image 576 input, and rerun all controls equally.
        native_large=False,
    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=8,
    clip_range=None,
    data=dict(
        shape=dict(h=384, w=384),
        names=["chameleon", "camo_te", "cod10k_te", "nc4k"],
        native_large=False,
    ),
)
