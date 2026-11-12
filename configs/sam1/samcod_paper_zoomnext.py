_base_ = "./samcod_paper_base.py"
model_name = "PvtV2B4_FPN_ZoomNeXt"
model = dict(use_mhsiu=True, use_rgpu=True, clean_only_highres=False,
             siu_groups=4, hmu_groups=6)
