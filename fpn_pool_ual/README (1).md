# FPN Pool-aware UAL Ablation

The encoder/decoder and full-mask BCE are unchanged. High uses full-image UAL with a cosine weight (`high_ual_weight=0.2` default). Middle uses one of: no UAL, full-image UAL with high weight, full-image low-weight UAL, or pseudo-label boundary-only UAL.

## Quick start

Copy these three files to the root of RCLCOD or keep them together inside `fpn_pool_ual/`. From the RCLCOD root:

```bash
python fpn_pool_ual/install_and_make_configs.py --epochs 40
bash fpn_pool_ual/run_fpn_pool_ual.sh a1_high_only
# or all five:
bash fpn_pool_ual/run_fpn_pool_ual.sh all
```

For `--epochs 150`, regenerate configs first. This script uses your existing `configs/sam2_real.py` and High/Middle lists, and fixes pool *sampling masses* to 1:1. No Unvalue supervised; no UAL in Middle's BCE itself.

## Ablations

| Config | High | Middle |
|---|---|---|
| a0_no_ual | BCE | BCE |
| a1_high_only | BCE+UAL(.2) | BCE |
| a2_both_full | BCE+UAL(.2) | BCE+full-UAL(.2) |
| a3_middle_low | BCE+UAL(.2) | BCE+full-UAL(.05) |
| a4_middle_edge | BCE+UAL(.2) | BCE+boundary-UAL(.05) |

All UAL weights are multiplied by `0.5 * (1-cos(pi*iter_percentage))`, which ramps over the full configured training horizon. The edge UAL is normalized by the number of edge pixels, not all H×W pixels; with the same nominal lambda it can be more concentrated, so treat coefficient selection and boundary width as explicit factors and monitor `middle_ual_term` and `edge_coverage`.

**Critical:** All experiments must use the exact same fixed High/Middle list, seed, and training settings. FPN remains full BCE for Middle in every variant; changes vs previous Seed/Adaptive runs thus test UAL only, NOT masked supervision. SAM2 labels and box annotations are not changed. No GT fields are used in supervision.

The `loss_metrics.csv` includes raw and weighted UAL contributions and edge coverage.
