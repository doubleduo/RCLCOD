base_seed = 112358
config = 'configs/sam1real_pkd_skd.py'
data_cfg = './dataset.yaml'
dataset_infos = dict({
    'camo_te':
    dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/CAMO-TE'),
    'chameleon':
    dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/CHAMELEON'),
    'cod10k_te':
    dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/COD10K-TE'),
    'combined_tr':
    dict(
        box_json=dict(path='./data/Train/box_label', suffix='.json'),
        image=dict(path='./data/Train/Imgs', suffix='.jpg'),
        mask=dict(
            path='softresult/SAM1_Optimized3/03_Refine_NoFlip', suffix='.png'),
        root='.'),
    'combined_trs1':
    dict(
        box_json=dict(path='./data/Train/box_label', suffix='.json'),
        image=dict(path='./data/Train/Imgs', suffix='.jpg'),
        mask=dict(path='softresult/SAM1_BoxOnly/01_BoxBase', suffix='.png'),
        root='.'),
    'combined_trs2':
    dict(
        box_json=dict(path='./data/Train/box_label', suffix='.json'),
        image=dict(path='./data/Train/Imgs', suffix='.jpg'),
        mask=dict(
            path='softresult/SAM1_Optimized3/02_SoftLogits_SO_MO',
            suffix='.png'),
        root='.'),
    'combinedsam3':
    dict(
        box_json=dict(path='./data/Train/box_label', suffix='.json'),
        image=dict(path='./data/Train/Imgs', suffix='.jpg'),
        mask=dict(
            path='softresult/SAM3_Optimized/02_SoftLogits_SO_MO',
            suffix='.png'),
        root='.'),
    'cvc-300':
    dict(
        image=dict(path='images', suffix='.png'),
        mask=dict(path='masks', suffix='.png'),
        root='./Dataset_Polyp/Test/CVC-300'),
    'cvc-clinicdb':
    dict(
        image=dict(path='images', suffix='.png'),
        mask=dict(path='masks', suffix='.png'),
        root='./Dataset_Polyp/Test/CVC-ClinicDB'),
    'cvc-colondb':
    dict(
        image=dict(path='images', suffix='.png'),
        mask=dict(path='masks', suffix='.png'),
        root='./Dataset_Polyp/Test/CVC-ColonDB'),
    'etis-laribpolypdb':
    dict(
        image=dict(path='images', suffix='.png'),
        mask=dict(path='masks', suffix='.png'),
        root='./Dataset_Polyp/Test/ETIS-LaribPolypDB'),
    'kvasir':
    dict(
        image=dict(path='images', suffix='.png'),
        mask=dict(path='masks', suffix='.png'),
        root='./Dataset_Polyp/Test/Kvasir'),
    'nc4k':
    dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/NC4K'),
    'train_tr':
    dict(
        image=dict(path='./Dataset_Polyp/Train/image', suffix='.png'),
        mask=dict(
            path='./softresult/SAM1_Polyp_GTBox/02_Iter2', suffix='.png'),
        root='.')
})
deterministic = True
device = 'cuda:0'
evaluate = False
exp_name = 'PvtV2B4_FPN_PKD_SKD_BS8'
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
    pkd_erode_kernel=5,
    pkd_min_valid_pixels=16,
    pkd_weight=0.5,
    skd_ramp_end=0.0,
    skd_ramp_start=0.0,
    skd_view='image_s',
    skd_weight=1.0)
model_name = 'PvtV2B4_FPN_PKD_SKD'
output_dir = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd'
path = dict(
    cfg_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd/PvtV2B4_FPN_PKD_SKD_BS8/exp_0/config.py',
    excel=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd/PvtV2B4_FPN_PKD_SKD_BS8/exp_0/results.xlsx',
    final_full_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd/PvtV2B4_FPN_PKD_SKD_BS8/exp_0/pth/checkpoint_final.pth',
    final_state_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd/PvtV2B4_FPN_PKD_SKD_BS8/exp_0/pth/state_final.pth',
    log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd/PvtV2B4_FPN_PKD_SKD_BS8/exp_0/log_2026-11-07.txt',
    output_dir=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd',
    pth=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd/PvtV2B4_FPN_PKD_SKD_BS8/exp_0/pth',
    pth_log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd/PvtV2B4_FPN_PKD_SKD_BS8/exp_0',
    save=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd/PvtV2B4_FPN_PKD_SKD_BS8/exp_0/pre',
    tb=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd/PvtV2B4_FPN_PKD_SKD_BS8/exp_0/tb',
    trainer_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/RCLCOD/outputs/1sam1/pkd_skd/PvtV2B4_FPN_PKD_SKD_BS8/exp_0/trainer.txt'
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
        shape=dict(h=384, w=384)),
    num_workers=8)
train = dict(
    batch_size=8,
    bn=dict(freeze_affine=True, freeze_encoder=False, freeze_status=True),
    curriculum=dict(
        enable=True,
        num_samples_per_epoch=4040,
        pools=dict(
            clean='./data/pseudo_pool/1sam1/IoU7090/clean.txt',
            noisy='./data/pseudo_pool/1sam1/IoU7090/noisy.txt'),
        stages=[
            dict(name='sam1_broad_1_120', weights=dict(clean=1.0, noisy=1.0)),
            dict(
                name='sam1_reliable_121_150',
                weights=dict(clean=1.0, noisy=0.4)),
        ],
        transition=dict(
            loss_eps=0.005,
            max_loss_regression=0.01,
            max_stage_epochs=120,
            min_stage_epochs=120,
            patience=1,
            use_ema_consistency=False,
            window_size=5)),
    data=dict(names=[
        'combined_tr',
    ], shape=dict(h=384, w=384), use_box=True),
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
    val_start_epoch=20)
use_checkpoint = False
use_custom_worker_init = True
