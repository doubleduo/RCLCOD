# -*- coding: utf-8 -*-
"""Paper-form losses on RCLCOD; not the full original SAM-COD experiment.

Unspecified details are explicit conventions: inward box band, pixel mean,
uniform one_of T/F/G with S always applied, ignore translation padding.
The public repository differs from paper equations; see README_SAMCOD_V2.md.
"""
engine = dict(name="samcod_paper")
model_name = "PvtV2B4_FPN_Baseline"
model = dict()
has_test = True
deterministic = True
use_custom_worker_init = True
log_interval = 20
base_seed = 112358

objective = dict(
    pkd=dict(enable=True, extra_weight=1.0),  # total weight 2 inside band
    skd=dict(enable=True, weight=1.0, detach_view="augmented", ignore_padding=True),
)
skd_augment = dict(
    policy="one_of",  # implementation convention, not recovered paper probabilities
    align_corners=False,
    scale=dict(enable=True, factor=0.5),
    translate=dict(enable=True, ratio=0.15, p=0.5),
    flip=dict(enable=True, axes=["horizontal", "vertical"], p=0.5),
    blur=dict(enable=True, kernel_size=5, sigma=[0.1, 2.0], p=0.5),
    crop=dict(enable=False, ratio=0.7, p=0.5),
)

train = dict(
    batch_size=8, num_workers=4, use_amp=True,
    num_epochs=150, epoch_based=True, num_iters=None,
    lr=1e-4, grad_acc_step=1,
    val_start_epoch=20, val_interval=5, save_val_ckpt=True,
    ema_kd=dict(enable=False, lambda_kd=0.0),
    # Public dataset pipeline uses common horizontal flip. Extra RCLCOD
    # rotations/color jitter are OFF here, shared across all V2 ablations.
    augment=dict(p=0.5, flip=True, rotate=False, brightness_contrast=False,
                 hsv=False, affine=False, gamma=False, blur=False),
    curriculum=dict(
        enable=True, mode="continuous", clean_weight=1.0,
        pools=dict(clean="./data/pseudo_pool/1sam1/IoU7090/clean.txt",
                   noisy="./data/pseudo_pool/1sam1/IoU7090/noisy.txt"),
        num_samples_per_epoch=4040,
        noisy=dict(hold_end_epoch=60, start_weight=1.0,
                   anneal_start_epoch=61, anneal_end_epoch=120,
                   end_weight=0.4, mode="cosine"),
        # Set 141 explicitly to enable Clean-only epochs 141-150.
        clean_only_start_epoch=None,
    ),
    optimizer=dict(mode="adam", set_to_none=False, group_mode="finetune",
                   cfg=dict(weight_decay=0, diff_factor=0.1)),
    sche_usebatch=True,
    scheduler=dict(warmup=dict(num_iters=0, initial_coef=0.01, mode="linear"),
                   mode="step", cfg=dict(milestones_epochs=[120], gamma=0.1)),
    bn=dict(freeze_status=True, freeze_affine=True, freeze_encoder=False),
    data=dict(shape=dict(h=384, w=384), names=["combined_tr"], use_box=True,
              box_prompt=dict(enable=True, band_ratio=0.25)),
)
test = dict(batch_size=8, num_workers=8, clip_range=None,
            data=dict(shape=dict(h=384, w=384),
                      names=["chameleon", "camo_te", "cod10k_te", "nc4k"]))
