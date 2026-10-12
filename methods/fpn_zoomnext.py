# -*- coding: utf-8 -*-
"""FPN baseline + native ZoomNeXt MHSIU2 / RGPU modules.

Built for RCLCOD commit b37a6b52359ae2f5d2c9a8ff99d1c47442bbc18e.

Unchanged from PvtV2B4_FPN_Baseline:
  - PVTv2-B4 encoder and its normalization
  - four 1x1 lateral projections and four 3x3 FPN smooth operators
  - the top-down FPN addition, prediction head, BCE training objective
  - forward() output contract and optimizer parameter grouping

Inserted:
  - ZoomNeXt MHSIU2: 1.5x/1.0x feature fusion at each FPN level
  - ZoomNeXt RGPU: progressive residual grouping after each FPN fusion

The two modules can be ablated independently without touching the trainer.
"""

import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline
from .zoomnext.layers import MHSIU2, RGPU


class PvtV2B4_FPN_ZoomNeXt(PvtV2B4_FPN_Baseline):
    """Dual-scale MHSIU2 + FPN top-down aggregation + RGPU refinement.

    The default is the full version. For ablations, set ``use_mhsiu`` or
    ``use_rgpu`` to False. ``image_l`` is already emitted by the RCLCOD
    dataset pipeline at 1.5x resolution; when absent, a bilinear fallback
    is synthesized from ``image_m`` (which is less informative).

    All training/inference interfaces are inherited from the FPN baseline.
    """

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        num_frames=1,
        siu_groups=4,
        hmu_groups=6,
        use_mhsiu=True,
        use_rgpu=True,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.use_mhsiu = bool(use_mhsiu)
        self.use_rgpu = bool(use_rgpu)

        if self.use_mhsiu:
            if siu_groups < 1 or fpn_dim % siu_groups != 0:
                raise ValueError(
                    f"MHSIU2 requires fpn_dim ({fpn_dim}) divisible by "
                    f"siu_groups ({siu_groups})"
                )
            self.siu_2 = MHSIU2(fpn_dim, siu_groups)
            self.siu_3 = MHSIU2(fpn_dim, siu_groups)
            self.siu_4 = MHSIU2(fpn_dim, siu_groups)
            self.siu_5 = MHSIU2(fpn_dim, siu_groups)

        if self.use_rgpu:
            if hmu_groups < 2:
                raise ValueError("RGPU requires hmu_groups >= 2")
            self.hmu_2 = RGPU(fpn_dim, hmu_groups, num_frames=num_frames)
            self.hmu_3 = RGPU(fpn_dim, hmu_groups, num_frames=num_frames)
            self.hmu_4 = RGPU(fpn_dim, hmu_groups, num_frames=num_frames)
            self.hmu_5 = RGPU(fpn_dim, hmu_groups, num_frames=num_frames)

    def body(self, data):
        image_m = data["image_m"]
        # Reuses baseline normalization, encoder weights and PVT stage mapping.
        m2, m3, m4, m5 = self.normalize_encoder(image_m)

        # Original FPN lateral projections (1x1), shared across scales.
        z2 = self.lateral_2(m2)
        z3 = self.lateral_3(m3)
        z4 = self.lateral_4(m4)
        z5 = self.lateral_5(m5)

        if self.use_mhsiu:
            image_l = data.get("image_l")
            if image_l is None:
                image_l = F.interpolate(
                    image_m,
                    scale_factor=1.5,
                    mode="bilinear",
                    align_corners=False,
                )

            l2, l3, l4, l5 = self.normalize_encoder(image_l)
            z2 = self.siu_2(l=self.lateral_2(l2), m=z2)
            z3 = self.siu_3(l=self.lateral_3(l3), m=z3)
            z4 = self.siu_4(l=self.lateral_4(l4), m=z4)
            z5 = self.siu_5(l=self.lateral_5(l5), m=z5)

        # Preserve the baseline FPN top-down structure. RGPU is an optional
        # post-fusion refinement, matching ZoomNeXt's decode-stage ordering.
        p5 = self.smooth_5(z5)
        if self.use_rgpu:
            p5 = self.hmu_5(p5)

        p4 = self.smooth_4(z4 + self._resize_like(p5, m4))
        if self.use_rgpu:
            p4 = self.hmu_4(p4)

        p3 = self.smooth_3(z3 + self._resize_like(p4, m3))
        if self.use_rgpu:
            p3 = self.hmu_3(p3)

        p2 = self.smooth_2(z2 + self._resize_like(p3, m2))
        if self.use_rgpu:
            p2 = self.hmu_2(p2)

        logits = self.predictor(p2)
        return F.interpolate(
            logits,
            size=image_m.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class PvtV2B4_FPN_ZoomNeXt_MHSIU(PvtV2B4_FPN_ZoomNeXt):
    """Ablation: FPN + 2-scale MHSIU2, but no RGPU."""

    def __init__(self, **kwargs):
        kwargs.pop("use_mhsiu", None)
        kwargs.pop("use_rgpu", None)
        super().__init__(use_mhsiu=True, use_rgpu=False, **kwargs)


class PvtV2B4_FPN_ZoomNeXt_RGPU(PvtV2B4_FPN_ZoomNeXt):
    """Ablation: single-scale FPN + RGPU, but no MHSIU2."""

    def __init__(self, **kwargs):
        kwargs.pop("use_mhsiu", None)
        kwargs.pop("use_rgpu", None)
        super().__init__(use_mhsiu=False, use_rgpu=True, **kwargs)


__all__ = [
    "PvtV2B4_FPN_ZoomNeXt",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU",
    "PvtV2B4_FPN_ZoomNeXt_RGPU",
]
