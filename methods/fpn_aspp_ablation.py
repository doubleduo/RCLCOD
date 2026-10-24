# -*- coding: utf-8 -*-
"""PVT-only / FPN / ASPP / ASPP->FPN controlled architectural ablations.

Drop this file in RCLCOD/methods/fpn_aspp_ablation.py.

This module deliberately reuses the original fpn_baseline.py for:
  - PVTv2-B4 backbone, normalizer, and pretrained initialization
  - FPN lateral/smoothing blocks and predictor
  - training BCE and output dictionary
  - inference logits and optimizer parameter grouping

Architectures:
  A-1: PvtV2B4_PVT_Only
       C5 -> 1x1 binary head -> upsample (NO decoder / FPN / ASPP)

  A0: PvtV2B4_FPN_Baseline  (original class, unchanged)
      C2,C3,C4,C5 -> standard FPN -> predictor

  A1: PvtV2B4_ASPP_Baseline
      C5 -> lateral_5 -> ASPP -> predictor (no FPN)

  A2: PvtV2B4_ASPP_FPN
      C5 -> lateral_5 -> ASPP -> original top-down FPN with C2,C3,C4

ASPP is applied to the stride-32 C5 feature (12x12 at 384 input), so the
moderate dilations (1, 2, 3) are intentional. Larger rates such as 12/18
would waste much of their spatial support at this resolution.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline
from .zoomnext.ops import ConvBNReLU


class PvtV2B4_PVT_Only(PvtV2B4_FPN_Baseline):
    """A-1: PVTv2-B4 backbone + essential 1x1 binary prediction ONLY.

    No lateral projection, FPN, ASPP, 3x3 refinement, or intermediate decoder.
    Predict directly from the last PVT feature C5, then bilinearly resize.
    Parent forward() keeps the exact BCE and train/eval return signatures.

    PVT is a feature encoder: a 1x1 map-to-one-channel layer is necessary
    to produce a binary segmentation logit; it is not a decoder module.
    """

    def __init__(self, pretrained=True, input_norm=True, fpn_dim=64,
                 use_checkpoint=False, **kwargs):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

        # Erase ALL originally constructed FPN/segmentation-head layers.
        # Only encoder, normalizer and the new one-convolution head remain.
        for name in (
            "lateral_2", "lateral_3", "lateral_4", "lateral_5",
            "smooth_2", "smooth_3", "smooth_4", "smooth_5",
            "predictor",
        ):
            delattr(self, name)

        self.predictor = nn.Conv2d(self.embed_dims[-1], 1, kernel_size=1)

    def body(self, data):
        image = data["image_m"]
        c5 = self.normalize_encoder(image)[-1]
        logits = self.predictor(c5)
        return F.interpolate(
            logits, size=image.shape[-2:], mode="bilinear",
            align_corners=False,
        )


class ASPP(nn.Module):
    """Lightweight atrous spatial pyramid pooling with image-level context.

    Branches: 1x1, 3x3 dilation r for each rate, and global average pooling.
    Image-pooling branch has no BatchNorm because its spatial size is 1x1 and
    training must remain valid even when a final local batch has size 1.
    """

    def __init__(self, in_channels=64, out_channels=64,
                 rates=(1, 2, 3), branch_channels=None):
        super().__init__()
        rates = tuple(int(r) for r in rates)
        if not rates or any(r < 1 for r in rates):
            raise ValueError("rates must contain positive integer dilations")
        if branch_channels is None:
            branch_channels = max(16, out_channels // 2)

        self.branch_1x1 = ConvBNReLU(in_channels, branch_channels, 1)
        self.atrous_branches = nn.ModuleList([
            ConvBNReLU(
                in_channels, branch_channels,
                kernel_size=3, stride=1, padding=rate, dilation=rate,
            )
            for rate in rates
        ])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        total_branches = 2 + len(rates)
        self.project = ConvBNReLU(
            total_branches * branch_channels, out_channels, kernel_size=1,
        )

    def forward(self, x):
        size = x.shape[-2:]
        pooled = F.interpolate(
            self.image_pool(x), size=size,
            mode="bilinear", align_corners=False,
        )
        all_branches = (
            [self.branch_1x1(x)]
            + [branch(x) for branch in self.atrous_branches]
            + [pooled]
        )
        return self.project(torch.cat(all_branches, dim=1))


class PvtV2B4_ASPP_Baseline(PvtV2B4_FPN_Baseline):
    """A1: ONLY ASPP on C5, with FPN decoder completely removed.

    The original predictor is retained to isolate the effect of the decoder.
    Inherited forward() uses the SAME BCE loss and training output schema.
    """

    def __init__(self, pretrained=True, input_norm=True, fpn_dim=64,
                 use_checkpoint=False, aspp_rates=(1, 2, 3),
                 aspp_branch_channels=None, **kwargs):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.aspp = ASPP(
            in_channels=fpn_dim, out_channels=fpn_dim,
            rates=aspp_rates, branch_channels=aspp_branch_channels,
        )
        # Remove every FPN-only parameter so it does not inflate parameter
        # count / optimizer groups. Keep lateral_5 for a shared 64-dim neck.
        for name in (
            "lateral_2", "lateral_3", "lateral_4",
            "smooth_2", "smooth_3", "smooth_4", "smooth_5",
        ):
            delattr(self, name)

    def body(self, data):
        image = data["image_m"]
        c5 = self.normalize_encoder(image)[-1]
        x = self.aspp(self.lateral_5(c5))
        logits = self.predictor(x)
        return F.interpolate(
            logits, size=image.shape[-2:],
            mode="bilinear", align_corners=False,
        )


class PvtV2B4_ASPP_FPN(PvtV2B4_FPN_Baseline):
    """A2: ASPP on C5 followed by EXACT original top-down FPN decoder."""

    def __init__(self, pretrained=True, input_norm=True, fpn_dim=64,
                 use_checkpoint=False, aspp_rates=(1, 2, 3),
                 aspp_branch_channels=None, **kwargs):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.aspp = ASPP(
            in_channels=fpn_dim, out_channels=fpn_dim,
            rates=aspp_rates, branch_channels=aspp_branch_channels,
        )

    def body(self, data):
        image = data["image_m"]
        c2, c3, c4, c5 = self.normalize_encoder(image)

        # Only difference from the original FPN is ASPP before smooth_5.
        p5 = self.smooth_5(self.aspp(self.lateral_5(c5)))
        p4 = self.smooth_4(
            self.lateral_4(c4) + self._resize_like(p5, c4)
        )
        p3 = self.smooth_3(
            self.lateral_3(c3) + self._resize_like(p4, c3)
        )
        p2 = self.smooth_2(
            self.lateral_2(c2) + self._resize_like(p3, c2)
        )
        logits = self.predictor(p2)
        return F.interpolate(
            logits, size=image.shape[-2:],
            mode="bilinear", align_corners=False,
        )
