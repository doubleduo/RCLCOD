# -*- coding: utf-8 -*-
"""B3: Box-supervised CSR + continuous Noisy-COD curriculum.

The box is available only during training. The model remains RGB-only during
validation/inference. The dataset loader reads LabelMe rectangles/polygons
from ``dataset.yaml -> combined_tr.box_json`` and exposes ``data['box_mask']``.

Recommended command:
    python basemain_continuous.py \
      --config configs/curablation/bcsr_boxscale_nc_continuous.py \
      --model-name PvtV2B4_FPN_BCSR_NC_Curriculum
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

    # New: make LabelMe box masks available to the training dataset.
    use_box_mask=True,

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
        # Use 576 for the main run because your previous result showed a
        # substantial gain over 384. Keep 384 for fair module ablations.
        shape=dict(h=576, w=576),
        names=["combined_tr"],
    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=8,
    clip_range=None,
    data=dict(
        shape=dict(h=576, w=576),
        names=["camo_te", "cod10k_te"],
    ),
)
