has_test = True
deterministic = True
use_custom_worker_init = True
log_interval = 20
base_seed = 112358

__BATCHSIZE = 8
__NUM_EPOCHS = 100
__NUM_TR_SAMPLES = 3040 + 1000
__ITER_PER_EPOCH = __NUM_TR_SAMPLES // __BATCHSIZE  # drop_last is True
__NUM_ITERS = __NUM_EPOCHS * __ITER_PER_EPOCH


train = dict(
    batch_size=__BATCHSIZE,
    num_workers=4,
    use_amp=True,
    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    num_iters=None,
    lr=0.0001,
    grad_acc_step=1,

    # 第 20 轮开始，每 10 轮验证一次：
    # 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    val_start_epoch=30
    ,
    val_interval=5,
    save_val_ckpt=True,
    transition=dict(enable=False,),
    
    ema_kd=dict(enable=False,),

    # 三阶段课程学习
    curriculum=dict(
        enable=True,
        pools=dict(
            clean="./data/pseudo_pool/shape/clean.txt",
            noisy="./data/pseudo_pool/shape/noisy.txt",
            unvalue="./data/pseudo_pool/shape/unvalue.txt",
            camo="./data/pseudo_pool/camoD6.txt"
        ),

        # 保持每轮样本量接近原始 4040。
        # BS=24 且 drop_last=True 时，仍然约 168 iters/epoch。
        num_samples_per_epoch=4040,

        # 阶段结束强制验证：
        validate_at_stage_end=True,
         transition=dict(
            enable=False,
        ),

        stages = [

    dict(
           name="stage1",
           start_epoch=1,
           end_epoch=50,
            weights=dict(clean=0.0, noisy=0.4, camo=0.1, unvalue=0.1,)
            #weights=dict(clean=1.0, noisy=0.4, camo=0.05, unvalue=0.0),
       ),
    dict(
        name="stage2_expand_noisy",
        start_epoch=51,

        
        end_epoch=100,
        weights=dict(clean=1.0, noisy=0.6, camo=0.0, unvalue=0.00),
    ),]

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
            milestones=int(__NUM_ITERS * 2 / 3),
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
    num_workers=24,
    clip_range=None,
    data=dict(
        shape=dict(h=384, w=384),
        names=["camo_te", "cod10k_te"],
        # names=["chameleon", "camo_te", "cod10k_te", "nc4k"],
    ),
)