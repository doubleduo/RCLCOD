# -*- coding: utf-8 -*-
"""Minimal PVTv2-B4 FPN 384+576 fusion ablation for RCLCOD.

Requires the repository's existing ImageTrainDataset/ImageTestDataset providing:
    image_m: [B, 3, 384, 384]
    image_l: [B, 3, 576, 576]

Training objective and evaluator contract are inherited unchanged from
PvtV2B4_FPN_NC_Curriculum. No box input or auxiliary supervision is used.

Fusion modes:
    'gate' : zero-initialized spatial gated residual (main experiment)
    'mean' : parameter-free arithmetic average (optional control)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_NC_Curriculum


class _SpatialGatedResidual(nn.Module):
    """Start as identity on the medium/base branch, learn safe scale corrections."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.delta = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, base: torch.Tensor, large: torch.Tensor) -> torch.Tensor:
        difference = large - base
        score = self.gate(torch.cat((base, large, difference.abs()), dim=1))
        return base + score * self.delta(difference)


class PvtV2B4_FPN_ZoomMini(PvtV2B4_FPN_NC_Curriculum):
    """Shared-backbone, two-resolution FPN. Fuse C2/C3 only by default.

    - Medium input: data['image_m'] (384x384)
    - Large input : data['image_l'] (576x576)
    - One shared encoder and shared lateral projections.
    - Fusion operates on 64-channel lateral features on medium resolution.
    - C4/C5 and the entire top-down FPN/prediction head stay unchanged.
    - FPN NC curriculum loss, return dict and parameter grouping inherited.
    """

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        q_switch_ratio=0.40,
        fusion_mode="gate",
        fusion_levels=(2, 3),
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            q_switch_ratio=q_switch_ratio,
            **kwargs,
        )
        if fusion_mode not in ("gate", "mean"):
            raise ValueError("fusion_mode must be 'gate' or 'mean'")
        fusion_levels = tuple(int(x) for x in fusion_levels)
        if not fusion_levels or any(x not in (2, 3, 4, 5) for x in fusion_levels):
            raise ValueError("fusion_levels must be a non-empty subset of (2, 3, 4, 5)")
        if len(set(fusion_levels)) != len(fusion_levels):
            raise ValueError("Duplicate fusion levels")

        self.fusion_mode = fusion_mode
        self.fusion_levels = fusion_levels
        self.fusers = nn.ModuleDict(
            {str(level): _SpatialGatedResidual(fpn_dim)
             for level in fusion_levels}
            if fusion_mode == "gate" else {}
        )

    @staticmethod
    def _align_large_to_medium(large: torch.Tensor, medium: torch.Tensor):
        size = medium.shape[-2:]
        if large.shape[-2:] == size:
            return large
        # Features from whole-image high resolution share normalized coordinates
        # with whole-image medium resolution, so area downsampling is valid.
        if all(a >= b for a, b in zip(large.shape[-2:], size)):
            return F.adaptive_avg_pool2d(large, size)
        return F.interpolate(large, size=size, mode="bilinear", align_corners=False)

    def _merge(self, level: int, medium: torch.Tensor, large: torch.Tensor):
        if level not in self.fusion_levels:
            return medium
        large = self._align_large_to_medium(large, medium)
        if self.fusion_mode == "mean":
            return 0.5 * (medium + large)
        return self.fusers[str(level)](medium, large)

    def body(self, data):
        if "image_l" not in data or "image_m" not in data:
            raise KeyError("ZoomMini requires data['image_m'] and data['image_l']")

        image_m = data["image_m"]
        image_l = data["image_l"]
        if image_m.shape[0] != image_l.shape[0]:
            raise ValueError("image_m and image_l must have identical batch size")
        if image_m.shape[-2:] == image_l.shape[-2:]:
            raise ValueError("image_l must be larger than image_m for scale ablation")

        c2m, c3m, c4m, c5m = self.normalize_encoder(image_m)
        c2l, c3l, c4l, c5l = self.normalize_encoder(image_l)

        # Same lateral projections used on both resolutions (weight sharing).
        # Only compute large lateral features for levels selected for fusion.
        stages_m = (c2m, c3m, c4m, c5m)
        stages_l = (c2l, c3l, c4l, c5l)
        lateral_layers = (self.lateral_2, self.lateral_3, self.lateral_4, self.lateral_5)
        medium = {lvl: proj(f) for lvl, proj, f in zip((2, 3, 4, 5), lateral_layers, stages_m)}

        for lvl in self.fusion_levels:
            large = lateral_layers[lvl - 2](stages_l[lvl - 2])
            medium[lvl] = self._merge(lvl, medium[lvl], large)

        p5 = self.smooth_5(medium[5])
        p4 = self.smooth_4(medium[4] + self._resize_like(p5, medium[4]))
        p3 = self.smooth_3(medium[3] + self._resize_like(p4, medium[3]))
        p2 = self.smooth_2(medium[2] + self._resize_like(p3, medium[2]))

        logits = self.predictor(p2)
        return F.interpolate(
            logits, size=image_m.shape[-2:], mode="bilinear", align_corners=False
        )
