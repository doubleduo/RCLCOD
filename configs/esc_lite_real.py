# -*- coding: utf-8 -*-
"""ESC-Lite model ablation: SAME continuous Clean/Noisy schedule as csr_mhsiu_nc_real.py.

Only architecture/loss settings in `model` below are changed.
Run:
  python basemain_continuous.py \
      --config configs/curablation/esc_lite_real.py \
      --model-name PvtV2B4_FPN_ESC_Lite
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

model = dict(
    fpn_dim=64,
    use_aetp=True,
    use_dsfa=False,        # A4: True; A2 default: False
    use_mfmm_mask=True,
    use_mfmm_edge=True,
    aux_coarse_weight=0.2, # set 0.0 for purely original BCE loss
    aux_edge_weight=0.0,   # noisy SAM2 boundary: do NOT enable initially
    dsfa_shift_px=1.5,
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
