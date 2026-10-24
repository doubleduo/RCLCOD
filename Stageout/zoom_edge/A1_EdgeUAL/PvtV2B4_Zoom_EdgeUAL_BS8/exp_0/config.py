base_seed = 112358
config = 'configs/curablation/zoomnext_edge_residual_real.py'
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
        mask=dict(
            path='./SGT/SAM2_FinalRefine_OriginalPools/03_Refine_NoFlip',
            suffix='.png'),
        root='.'),
    nc4k=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/NC4K'))
deterministic = True
device = 'cuda:0'
evaluate = False
exp_name = 'PvtV2B4_Zoom_EdgeUAL_BS8'
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
model = dict(
    edge_kernel=7,
    edge_levels=(
        3,
        2,
    ),
    edge_strength=1.0,
    edge_ual_clean_only=False,
    ual_full_ratio=0.7,
    ual_start_ratio=0.4,
    ual_weight=0.2)
model_name = 'PvtV2B4_Zoom_EdgeUAL'
output_dir = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL'
path = dict(
    cfg_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL/PvtV2B4_Zoom_EdgeUAL_BS8/exp_0/config.py',
    excel=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL/PvtV2B4_Zoom_EdgeUAL_BS8/exp_0/results.xlsx',
    final_full_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL/PvtV2B4_Zoom_EdgeUAL_BS8/exp_0/pth/checkpoint_final.pth',
    final_state_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL/PvtV2B4_Zoom_EdgeUAL_BS8/exp_0/pth/state_final.pth',
    log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL/PvtV2B4_Zoom_EdgeUAL_BS8/exp_0/log_2026-10-20.txt',
    output_dir=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL',
    pth=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL/PvtV2B4_Zoom_EdgeUAL_BS8/exp_0/pth',
    pth_log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL/PvtV2B4_Zoom_EdgeUAL_BS8/exp_0',
    save=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL/PvtV2B4_Zoom_EdgeUAL_BS8/exp_0/pre',
    tb=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL/PvtV2B4_Zoom_EdgeUAL_BS8/exp_0/tb',
    trainer_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/Stageout/zoom_edge/A1_EdgeUAL/PvtV2B4_Zoom_EdgeUAL_BS8/exp_0/trainer.txt'
)
pretrained = True
proj_root = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD'
save_results = False
test = dict(
    batch_size=8,
    clip_range=None,
    data=dict(
        names=[
            'chameleon',
            'camo_te',
            'cod10k_te',
            'nc4k',
        ],
        native_large=False,
        shape=dict(h=384, w=384)),
    num_workers=8)
train = dict(
    batch_size=8,
    bn=dict(freeze_affine=True, freeze_encoder=False, freeze_status=True),
    curriculum=dict(
        continuous_schedule=dict(
            clean_weight=1.0,
            enable=True,
            final_start_epoch=131,
            noisy=dict(
                anneal_end_epoch=130,
                anneal_start_epoch=101,
                end_weight=0.4,
                hold_end_epoch=100,
                mode='cosine',
                start_weight=1.0)),
        enable=True,
        num_samples_per_epoch=4040,
        pools=dict(
            clean='./data/pseudo_pool/sam2refine/high.txt',
            noisy='./data/pseudo_pool/sam2refine/middle.txt')),
    data=dict(
        names=[
            'combined_tr',
        ], native_large=False, shape=dict(h=384, w=384)),
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
    save_val_ckpt=False,
    sche_usebatch=True,
    scheduler=dict(
        cfg=dict(gamma=0.1, milestones=60600),
        mode='step',
        warmup=dict(initial_coef=0.01, mode='linear', num_iters=0)),
    use_amp=True,
    val_interval=5,
    val_start_epoch=20)
use_checkpoint = False
use_custom_worker_init = True
