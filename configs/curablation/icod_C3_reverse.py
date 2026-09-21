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
    # C3 Reverse
    #
    # Epoch 1-50   : 难度扩展
    # Epoch 51-100 : 易
    # Epoch 101-150: 易
    #
    # 与 C2 使用相同的两种采样混合，只交换先后顺序：
    #
    # C2: 100 epoch 易 + 50 epoch 难
    # C3:  50 epoch 难 + 100 epoch 易
    #
    # 因此二者总曝光量基本一致，主要变量就是顺序。
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

        # 第50轮结束后切换
        transition=dict(
            min_stage_epochs=50,
            max_stage_epochs=50,
            use_ema_consistency=False,
            window_size=5,
            loss_eps=0.005,
            max_loss_regression=0.01,
            patience=1,
        ),

        stages=[
            dict(
                name="C3_expand_1_50",
                weights=dict(
                    clean=1.0,
                    noisy=0.6,
                    unvalue=0.05,
                    camo=0.0,
                ),
            ),
            dict(
                name="C3_easy_51_150",
                weights=dict(
                    clean=1.0,
                    noisy=0.4,
                    unvalue=0.0,
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
