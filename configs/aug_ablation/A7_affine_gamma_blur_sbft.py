# Existing A0-A7 augmentation is applied first. Then SBFT on clean RGB only.
_base_ = "../aug_ablation/A7_affine_gamma_blur.py"

train = dict(
    sbft=dict(
        enable=True,
        start_epoch=60,  # apply starting epoch 60 (inclusive)
        r=0.95,          # CurriSeg circular frequency cutoff ratio
        p=0.5,           # probability per clean sample
        pool="clean",    # noisy samples NEVER receive SBFT
    ),
)
