# -*- coding: utf-8 -*-
"""PVTv2-B4 ZoomNeXt with the existing APBOXNet Unvalue objective.

The RGB inference graph is exactly ``PvtV2B4_ZoomNeXt``.  Clean/noisy
samples keep dense-mask supervision; unvalue samples use box-out background
constraints and reliable EMA-teacher pixels, exactly as in
``PvtV2B4_FPN_Unvalue``.
"""

import torch

from .fpn_unvalue_nc import MaskAnchoredUnvalueLoss, _resize
from .zoomnext.zoomnext import PvtV2B4_ZoomNeXt


class PvtV2B4_ZoomNeXt_Unvalue(PvtV2B4_ZoomNeXt):
    """Dual-scale ZoomNeXt plus the unchanged three-pool Unvalue loss."""

    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
        **kwargs,
    ):
        loss_keys = {
            "mask_loss_mode",
            "q_switch_ratio",
            "boundary_kernel",
            "boundary_gain",
            "box_dilate_kernel",
            "teacher_confidence",
            "teacher_disagreement",
            "min_teacher_foreground",
            "outside_weight",
            "dynamic_weight",
            "consistency_weight",
            "dynamic_mix_max",
        }
        loss_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if key in loss_keys
        }
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected ZoomNeXt-Unvalue arguments: {unknown}")

        super().__init__(
            pretrained=pretrained,
            num_frames=num_frames,
            input_norm=input_norm,
            mid_dim=mid_dim,
            siu_groups=siu_groups,
            hmu_groups=hmu_groups,
            use_checkpoint=use_checkpoint,
        )
        self.unvalue_loss = MaskAnchoredUnvalueLoss(**loss_kwargs)

    def forward(self, data, iter_percentage=1.0, **kwargs):
        del kwargs
        # The repository's current ZoomNeXt body consumes image_l + image_m.
        logits = self.body(data=data)

        if not self.training:
            return logits

        if "mask" not in data:
            raise KeyError("Training requires data['mask'].")

        target = data["mask"].to(
            device=logits.device,
            dtype=logits.dtype,
        )
        target = _resize(target, logits.shape[-2:]).clamp(0.0, 1.0)

        total, items = self.unvalue_loss(
            logits=logits,
            target=target,
            data=data,
            iter_percentage=iter_percentage,
        )
        items["total"] = total.detach()

        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid()},
            "loss": total,
            "loss_items": items,
            "loss_str": (
                    f"L:{total.detach().item():.4f} "
                    f"BCE:{items['bce'].item():.4f} "
                    f"CIOU:{items['clean_iou'].item():.4f} "
                    f"OUT:{items['unvalue_out'].item():.4f} "
                    f"DYN:{items['unvalue_dynamic'].item():.4f} "
                    f"CONS:{items['unvalue_consistency'].item():.4f} "
                    f"ACC:{items['unvalue_accepted'].item():.2f}"
                ),
        }


__all__ = ["PvtV2B4_ZoomNeXt_Unvalue"]
