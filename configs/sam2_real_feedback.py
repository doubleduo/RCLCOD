# -*- coding: utf-8 -*-
"""Minimal two-pool feedback curriculum config for PvtV2B4_FPN_Baseline.

Copy to configs/sam2_real_feedback.py within doubleduo/RCLCOD.
Only sampling changes; base architecture, plain BCE loss and optimization stay.
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
    val_start_epoch=20,
    val_interval=5,
    save_val_ckpt=True,
    ema_kd=dict(enable=False,lambda_kd=0.0),
    curriculum=dict(
        enable=True,
        pools=dict(
                    clean='./data/pseudo_pool/real/clean.txt',
                    noisy='./data/pseudo_pool/real/noisy.txt'),
        # pools1=dict(
        #     clean='./data/pseudo_pool/sam2refine/high.txt',
        #     noisy='./data/pseudo_pool/sam2refine/middle.txt',
        # ),
        num_samples_per_epoch=__SAMPLES_PER_EPOCH,
        # Kept for original trainer compatibility; this schedule is ignored.
        continuous_schedule=dict(enable=True),
    ),
    feedback=dict(
        enable=True,  # False = static mixture using the SAME sampler for ablation
        initial_noisy_prob=0.50,
        noisy_min=0.20,
        noisy_max=0.60,
        clean_min=0.40,
        min_clean_per_batch=3,
        probe_interval=5,
        warmup_epochs=10,
        clean_probe_size=64,
        noisy_probe_size=64,
        probe_batch_size=4,
        probe_workers=2,
        ema_beta=0.70,
        loss_good_threshold=0.01,
        loss_bad_threshold=0.03,
        stability_safe_margin=0.005,
        stability_drop_trigger=0.02,
        dice_safe_margin=0.015,
        dice_drop_trigger=0.03,
        fg_ratio_min=0.50,
        fg_ratio_max=1.50,
        plateau_patience=2,
        bad_patience=2,
        cooldown_checks=1,
        advance_step=0.05,
        recover_step=0.10,
        # Diagnostic-only Noisy signal in V1, does not control the sampler.
        pixel_confidence=0.90,
        pixel_disagreement=0.05,
    ),
    optimizer=dict(
        mode='adam',set_to_none=False,group_mode='finetune',
        cfg=dict(weight_decay=0,diff_factor=0.1),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(num_iters=0,initial_coef=0.01,mode='linear'),
        mode='step',cfg=dict(milestones=__ITER_PER_EPOCH*120,gamma=0.1),
    ),
    bn=dict(freeze_status=True,freeze_affine=True,freeze_encoder=False),
    data=dict(shape=dict(h=384,w=384),names=['combined_tr']),
)

test=dict(
    batch_size=__BATCHSIZE,
    num_workers=8,
    clip_range=None,
    data=dict(shape=dict(h=384,w=384),
              names=['chameleon','camo_te','cod10k_te','nc4k']),
)
