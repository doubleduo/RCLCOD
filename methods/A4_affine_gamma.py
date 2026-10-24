# Config inheritance retains sam2_real.py training and curriculum settings.
_base_ = "../sam2_real.py"

train = dict(
    augment=dict(
        p=0.5,
        flip=True,
        rotate=True,
        brightness_contrast=True,
        hsv=True,
        affine=False,
        gamma=True,
        blur=False,
    ),
)
