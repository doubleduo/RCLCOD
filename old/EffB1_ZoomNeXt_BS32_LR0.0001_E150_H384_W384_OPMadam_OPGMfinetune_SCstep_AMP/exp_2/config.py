base_seed = 112358
config = 'configs/icod_train.py'
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
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='./data/Train/'),
    nc4k=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='GT', suffix='.png'),
        root='data/Test/NC4K'))
deterministic = True
device = 'cuda:0'
evaluate = True
exp_name = 'EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP'
has_test = True
info = None
load_from = 'outputs/EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP/exp_0/pth/state_final.pth'
log_interval = 20
metric_names = [
    'sm',
    'wfm',
    'mae',
    'em',
    'fmeasure',
]
model_name = 'EffB1_ZoomNeXt'
output_dir = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs'
path = dict(
    cfg_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs/EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP/exp_2/config.py',
    excel=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs/EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP/exp_2/results.xlsx',
    final_full_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs/EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP/exp_2/pth/checkpoint_final.pth',
    final_state_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs/EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP/exp_2/pth/state_final.pth',
    log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs/EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP/exp_2/log_2026-03-17.txt',
    output_dir=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs',
    pth=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs/EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP/exp_2/pth',
    pth_log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs/EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP/exp_2',
    save=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs/EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP/exp_2/pre',
    tb=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs/EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP/exp_2/tb',
    trainer_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main/outputs/EffB1_ZoomNeXt_BS32_LR0.0001_E150_H384_W384_OPMadam_OPGMfinetune_SCstep_AMP/exp_2/trainer.txt'
)
pretrained = False
proj_root = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/hjw_WS-SAM/ZoomNeXt-main'
save_results = False
test = dict(
    batch_size=32,
    clip_range=None,
    data=dict(
        names=[
            'chameleon',
            'camo_te',
            'cod10k_te',
            'nc4k',
        ],
        shape=dict(h=384, w=384)),
    num_workers=2)
train = dict(
    batch_size=32,
    bn=dict(freeze_affine=True, freeze_encoder=False, freeze_status=True),
    data=dict(names=[
        'combined_tr',
    ], shape=dict(h=384, w=384)),
    epoch_based=True,
    grad_acc_step=1,
    lr=0.0001,
    num_epochs=150,
    num_iters=None,
    num_workers=2,
    optimizer=dict(
        cfg=dict(diff_factor=0.1, weight_decay=0),
        group_mode='finetune',
        mode='adam',
        set_to_none=False),
    sche_usebatch=True,
    scheduler=dict(
        cfg=dict(gamma=0.1, milestones=12600),
        mode='step',
        warmup=dict(initial_coef=0.01, mode='linear', num_iters=0)),
    use_amp=True)
use_checkpoint = False
use_custom_worker_init = True
