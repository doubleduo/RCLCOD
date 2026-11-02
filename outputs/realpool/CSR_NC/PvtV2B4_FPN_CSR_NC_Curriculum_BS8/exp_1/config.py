base_seed = 112358
config = 'configs/curablation/csr_mhsiu_nc_real.py'
data_cfg = './dataset.yaml'
dataset_infos = dict(
    camo_te=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/CAMO-TE'),
    chameleon=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/CHAMELEON'),
    cod10k_te=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/COD10K-TE'),
    combined_tr=dict(
        box_json=dict(path='./data/Train/box_label', suffix='.json'),
        image=dict(path='./data/Train/Imgs', suffix='.jpg'),
        mask=dict(path='./data/Train/S1_GT', suffix='.png'),
        pseudo_mask=dict(path='./data/Train/S1_GT', suffix='.png'),
        root='.'),
    nc4k=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/NC4K'))
deterministic = True
device = 'cuda:0'
evaluate = False
exp_name = 'PvtV2B4_FPN_CSR_NC_Curriculum_BS8'
has_test = True
info = None
load_from = None
log_interval = 20
metric_names = [
    'sm',
    'wfm',
    'mae',
    'em',
]
model_name = 'PvtV2B4_FPN_CSR_NC_Curriculum'
output_dir = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC'
path = dict(
    cfg_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC/PvtV2B4_FPN_CSR_NC_Curriculum_BS8/exp_1/config.py',
    excel=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC/PvtV2B4_FPN_CSR_NC_Curriculum_BS8/exp_1/results.xlsx',
    final_full_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC/PvtV2B4_FPN_CSR_NC_Curriculum_BS8/exp_1/pth/checkpoint_final.pth',
    final_state_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC/PvtV2B4_FPN_CSR_NC_Curriculum_BS8/exp_1/pth/state_final.pth',
    log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC/PvtV2B4_FPN_CSR_NC_Curriculum_BS8/exp_1/log_2026-09-22.txt',
    output_dir=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC',
    pth=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC/PvtV2B4_FPN_CSR_NC_Curriculum_BS8/exp_1/pth',
    pth_log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC/PvtV2B4_FPN_CSR_NC_Curriculum_BS8/exp_1',
    save=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC/PvtV2B4_FPN_CSR_NC_Curriculum_BS8/exp_1/pre',
    tb=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC/PvtV2B4_FPN_CSR_NC_Curriculum_BS8/exp_1/tb',
    trainer_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/realpool/CSR_NC/PvtV2B4_FPN_CSR_NC_Curriculum_BS8/exp_1/trainer.txt'
)
pretrained = True
proj_root = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD'
save_results = False
test = dict(
    batch_size=8,
    clip_range=None,
    data=dict(names=[
        'camo_te',
        'cod10k_te',
    ], shape=dict(h=384, w=384)),
    num_workers=8)
train = dict(
    batch_size=8,
    bn=dict(freeze_affine=True, freeze_encoder=False, freeze_status=True),
    curriculum=dict(
        continuous_schedule=dict(
            clean_weight=1.0,
            enable=True,
            final_start_epoch=141,
            noisy=dict(
                anneal_end_epoch=140,
                anneal_start_epoch=61,
                end_weight=0.4,
                hold_end_epoch=60,
                mode='cosine',
                start_weight=1.0)),
        enable=True,
        num_samples_per_epoch=4040,
        pools=dict(
            clean='./data/pseudo_pool/real/clean.txt',
            noisy='./data/pseudo_pool/real/noisy.txt')),
    data=dict(names=[
        'combined_tr',
    ], shape=dict(h=384, w=384)),
    ema_kd=dict(enable=False, lambda_kd=0.0),
    epoch_based=True,
    grad_acc_step=1,
    lr=0.0001,
    num_epochs=150,
    num_iters=None,
    num_workers=4,
    optimizer=dict(
        cfg=dict(diff_factor=0.1, weight_decay=0),
        group_mode='finetune',
        mode='adam',
        set_to_none=False),
    save_val_ckpt=True,
    sche_usebatch=True,
    scheduler=dict(
        cfg=dict(gamma=0.1, milestones=60600),
        mode='step',
        warmup=dict(initial_coef=0.01, mode='linear', num_iters=0)),
    use_amp=True,
    val_interval=5,
    val_start_epoch=30)
use_checkpoint = False
use_custom_worker_init = True
