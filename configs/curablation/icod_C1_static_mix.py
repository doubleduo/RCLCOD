has_test = True
deterministic = True
use_custom_worker_init = True
log_interval = 20
base_seed = 112358

__BATCHSIZE = 8
__NUM_EPOCHS = 150
__NUM_TR_SAMPLES = 4040
__ITER_PER_EPOCH = __NUM_TR_SAMPLES // __BATCHSIZE
__NUM_ITERS = __NUM_EPOCHS * __ITER_PER_EPOCH

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
    val_interval=10,
    save_val_ckpt=True,

    # 当前 basemain.py 的 ema_kd.enable 不能完全阻止 KD 路径，
    # 因此 lambda_kd=0.0，确保本组消融只比较样本调度。
    ema_kd=dict(
        enable=False,
        lambda_kd=0.0,
        decay=0.999,
        temperature=2.0,
        confidence_threshold=0.8,
        confidence_gamma=1.0,
        warmup_epochs=5,
    ),

    # ============================================================
    # C1 Static-Mix
    # Epoch 1-50   : 固定平均比例
    # Epoch 51-100 : 固定平均比例
    # Epoch 101-150: 固定平均比例
    #
    # 该比例近似匹配 C2 在 150 epoch 内的平均总曝光：
    # clean   ≈ 67.8%
    # noisy   ≈ 31.2%
    # unvalue ≈ 1.0%
    # ============================================================
    curriculum=dict(
        enable=True,

        pools=dict(
            clean="./data/pseudo_pool/shape/clean.txt",
            noisy="./data/pseudo_pool/shape/noisy.txt",
            unvalue="./data/pseudo_pool/shape/unvalue.txt",
            camo="./data/pseudo_pool/camoD6.txt",
        ),

        num_samples_per_epoch=4040,

        # trainer 要求两个 stage，因此放两个完全相同的 stage。
        # 虽然第100轮发生“形式上的切换”，采样比例完全不变。
        transition=dict(
            min_stage_epochs=100,
            max_stage_epochs=100,
            use_ema_consistency=False,
            window_size=5,
            loss_eps=0.005,
            max_loss_regression=0.01,
            patience=1,
        ),

        stages=[
            dict(
                name="C1_static_1_100",
                weights=dict(
                    clean=1.0,
                    noisy=0.46,
                    unvalue=0.015,
                    camo=0.0,
                ),
            ),
            dict(
                name="C1_static_101_150",
                weights=dict(
                    clean=1.0,
                    noisy=0.46,
                    unvalue=0.015,
                    camo=0.0,
                ),
            ),
        ],
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
            # 课程切换和 LR 降低不要同时发生。
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
