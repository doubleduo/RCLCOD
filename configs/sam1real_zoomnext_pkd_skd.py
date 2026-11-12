# -*- coding: utf-8 -*-
"""SAM1 dynamic curriculum with PKD/SKD on FPN-ZoomNeXt.

Run:
    python basemain_pkd.py \
        --config configs/sam1real_zoomnext_pkd_skd.py \
        --model-name PvtV2B4_FPN_ZoomNeXt_PKD_SKD \
        --output-dir outputs/1sam1/zoomnext_pkd_skd

All data, optimizer, schedule, PKD/SKD weights, and evaluation settings are
inherited from sam1real_pkd_skd.py.  Only the architecture-specific ZoomNeXt
switches are added here.
"""

_base_ = "./sam1real_pkd_skd.py"

model = dict(
    use_mhsiu=True,
    use_rgpu=True,
    clean_only_highres=False,
    siu_groups=4,
    hmu_groups=6,
)
