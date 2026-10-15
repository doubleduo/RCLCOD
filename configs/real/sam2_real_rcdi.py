# -*- coding: utf-8 -*-
"""RCDI architectural ablation on the exact SAM2 real curriculum.

Use this same config for R1, R2, R3 and the final RCDI model.  No box mask,
box auxiliary loss, UAL or frequency loss is enabled, so the only variable is
the dual-resolution fusion architecture.
"""

_base_ = "../sam2_real.py"

train = dict(
    data=dict(
        use_box=False,
    ),
)
