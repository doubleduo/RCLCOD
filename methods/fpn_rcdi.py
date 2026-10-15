# -*- coding: utf-8 -*-
"""RCDI: reliability-aware consensus-discrepancy interaction for RCLCOD.

This file is a drop-in architectural replacement for the two-scale MHSIU2
fusion used by ``PvtV2B4_FPN_ZoomNeXt_MHSIU``.  It deliberately keeps the
existing PVTv2-B4 encoder, lateral projections, FPN smooth layers, predictor,
training loss, output contract and optimizer grouping from
``PvtV2B4_FPN_Baseline``.

Inputs at pyramid level i:
    medium: lateral feature from image_m (the semantic anchor)
    large:  lateral feature from image_l (higher-resolution evidence)
    context: upsampled decoder feature P_{i+1}, when available

Core design:
    1. anti-aliased large-to-medium alignment;
    2. optional, zero-initialized sub-pixel offset refinement;
    3. explicit low-frequency region and high-frequency detail discrepancy;
    4. independent reliability gates (no compulsory scale softmax);
    5. zero-initialized residual injection into the medium-scale anchor.

The final model is ``PvtV2B4_FPN_RCDI``.  R1/R2/R3 are controlled ablations:
    R1: anchored discrepancy routing only;
    R2: R1 + region/detail decomposition and cross-scale detail consistency;
    R3: R2 + top-down FPN context verification;
    Final: R3 + constrained content-adaptive alignment at P2/P3.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline


def _valid_group_count(channels: int, preferred: int = 8) -> int:
    """Return a GroupNorm group count that always divides ``channels``."""
    return max(1, math.gcd(int(channels), int(preferred)))


class ConvGNAct(nn.Sequential):
    """Small convolution block independent of the trainer's frozen-BN option."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_valid_group_count(out_channels), out_channels),
        ]
        if activate:
            layers.append(nn.GELU())
        super().__init__(*layers)


class ReliabilityGate(nn.Module):
    """Generate independent group-wise reliability gates in [0, 1]."""

    def __init__(self, in_channels: int, channels: int, num_groups: int) -> None:
        super().__init__()
        hidden = max(16, channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GroupNorm(_valid_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(_valid_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, num_groups, 1, bias=True),
        )
        # A conservative initial reliability before the residual scale opens.
        nn.init.constant_(self.net[-1].bias, -1.0)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(evidence))


class RCDIBlock(nn.Module):
    """Reliability-aware consensus-discrepancy interaction.

    ``medium`` is always retained as the identity anchor.  New evidence is
    injected only through learnable residual scales initialized to zero.
    Consequently a newly constructed block returns ``medium`` exactly.
    """

    def __init__(
        self,
        channels: int = 64,
        num_groups: int = 4,
        frequency_split: bool = True,
        use_detail: bool = True,
        use_context: bool = True,
        use_offset: bool = False,
        max_offset: float = 0.5,
    ) -> None:
        super().__init__()
        if channels < 1 or num_groups < 1 or channels % num_groups:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_groups "
                f"({num_groups})"
            )
        if max_offset < 0:
            raise ValueError("max_offset must be non-negative")

        self.channels = int(channels)
        self.num_groups = int(num_groups)
        self.group_width = channels // num_groups
        self.frequency_split = bool(frequency_split)
        self.use_detail = bool(use_detail and frequency_split)
        self.use_context = bool(use_context)
        self.use_offset = bool(use_offset)
        self.max_offset = float(max_offset)

        # Scale-specific preprocessing before agreement/discrepancy analysis.
        self.medium_proj = ConvGNAct(channels, channels, 3, groups=channels)
        self.large_proj = ConvGNAct(channels, channels, 3, groups=channels)
        if self.use_context:
            self.context_proj = ConvGNAct(channels, channels, 1)

        if self.use_offset:
            hidden = max(16, channels // 2)
            self.offset_generator = nn.Sequential(
                nn.Conv2d(3 * channels, hidden, 3, padding=1, bias=False),
                nn.GroupNorm(_valid_group_count(hidden), hidden),
                nn.GELU(),
                nn.Conv2d(hidden, 2, 1, bias=True),
            )
            # Identity sampling at initialization.
            nn.init.zeros_(self.offset_generator[-1].weight)
            nn.init.zeros_(self.offset_generator[-1].bias)

        # Evidence = consensus(C) + |discrepancy|(C) + context(C) + score(1).
        gate_in_channels = 3 * channels + 1
        self.region_gate = ReliabilityGate(
            gate_in_channels, channels, num_groups
        )
        self.region_delta = nn.Sequential(
            ConvGNAct(channels, channels, 3, groups=channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.alpha_region = nn.Parameter(torch.zeros(()))

        if self.use_detail:
            self.detail_gate = ReliabilityGate(
                gate_in_channels, channels, num_groups
            )
            self.detail_delta = nn.Sequential(
                ConvGNAct(channels, channels, 3, groups=channels),
                nn.Conv2d(channels, channels, 1, bias=False),
            )
            self.alpha_detail = nn.Parameter(torch.zeros(()))

        # Runtime diagnostics.  These are not checkpoints or loss terms.
        for name in (
            "_last_region_ratio",
            "_last_detail_ratio",
            "_last_agreement",
            "_last_detail_consistency",
            "_last_offset_magnitude",
        ):
            self.register_buffer(name, torch.tensor(0.0), persistent=False)

    @staticmethod
    def _anti_alias_resize(source: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
        size = tuple(int(x) for x in size)
        if source.shape[-2:] == size:
            return source
        shrinking = (
            size[0] < source.shape[-2] or size[1] < source.shape[-1]
        )
        return F.interpolate(
            source,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=shrinking,
        )

    @staticmethod
    def _base_grid(
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Pixel-center grid for grid_sample(..., align_corners=False).
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5)
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5)
        y = 2.0 * y / max(height, 1) - 1.0
        x = 2.0 * x / max(width, 1) - 1.0
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(
            batch, -1, -1, -1
        )

    def _content_align(
        self,
        medium: torch.Tensor,
        large: torch.Tensor,
    ) -> torch.Tensor:
        large = self._anti_alias_resize(large, medium.shape[-2:])
        if not self.use_offset:
            self._last_offset_magnitude.zero_()
            return large

        evidence = torch.cat((medium, large, (large - medium).abs()), dim=1)
        offset = self.max_offset * torch.tanh(self.offset_generator(evidence))
        batch, _, height, width = offset.shape
        grid = self._base_grid(
            batch, height, width, offset.device, offset.dtype
        )
        offset_normalized = torch.stack(
            (
                2.0 * offset[:, 0] / max(width, 1),
                2.0 * offset[:, 1] / max(height, 1),
            ),
            dim=-1,
        )
        aligned = F.grid_sample(
            large,
            grid + offset_normalized,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        with torch.no_grad():
            self._last_offset_magnitude.copy_(
                offset.float().square().sum(1).sqrt().mean()
            )
        return aligned

    @staticmethod
    def _cosine_map(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = F.normalize(first.float(), dim=1, eps=1e-6)
        second = F.normalize(second.float(), dim=1, eps=1e-6)
        return (0.5 + 0.5 * (first * second).sum(1, keepdim=True)).clamp(0, 1)

    def _expand_gate(self, gate: torch.Tensor) -> torch.Tensor:
        return gate.repeat_interleave(self.group_width, dim=1)

    @staticmethod
    def _residual_ratio(residual: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        numerator = residual.float().square().mean().sqrt()
        denominator = anchor.float().square().mean().sqrt().clamp_min(1e-6)
        return numerator / denominator

    def forward(
        self,
        large: torch.Tensor,
        medium: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if large.ndim != 4 or medium.ndim != 4:
            raise ValueError("large and medium must be BCHW tensors")
        if large.shape[:2] != medium.shape[:2]:
            raise ValueError("large and medium must share batch and channel sizes")

        large = self._content_align(medium, large)
        medium_evidence = self.medium_proj(medium)
        large_evidence = self.large_proj(large)

        agreement = self._cosine_map(medium_evidence, large_evidence).to(
            dtype=medium.dtype
        )
        consensus = 0.5 * (medium_evidence + large_evidence)

        if context is None or not self.use_context:
            context_evidence = torch.zeros_like(consensus)
        else:
            context = self._anti_alias_resize(context, medium.shape[-2:])
            context_evidence = self.context_proj(context)

        if self.frequency_split:
            low_medium = F.avg_pool2d(
                medium_evidence, 3, stride=1, padding=1
            )
            low_large = F.avg_pool2d(
                large_evidence, 3, stride=1, padding=1
            )
            high_medium = medium_evidence - low_medium
            high_large = large_evidence - low_large
            region_discrepancy = low_large - low_medium
            detail_discrepancy = high_large - high_medium
        else:
            region_discrepancy = large_evidence - medium_evidence
            detail_discrepancy = torch.zeros_like(region_discrepancy)
            high_medium = detail_discrepancy
            high_large = detail_discrepancy

        region_evidence = torch.cat(
            (
                consensus,
                region_discrepancy.abs(),
                context_evidence,
                agreement,
            ),
            dim=1,
        )
        region_gate = self._expand_gate(self.region_gate(region_evidence))
        region_update = region_gate * self.region_delta(region_discrepancy)
        region_scale = torch.tanh(self.alpha_region)
        region_residual = region_scale * region_update
        result = medium + region_residual

        detail_consistency = medium.new_zeros(
            (medium.shape[0], 1, *medium.shape[-2:])
        )
        detail_residual = torch.zeros_like(medium)
        if self.use_detail:
            detail_consistency = self._cosine_map(
                high_medium, high_large
            ).to(dtype=medium.dtype)
            detail_evidence = torch.cat(
                (
                    consensus,
                    detail_discrepancy.abs(),
                    context_evidence,
                    detail_consistency,
                ),
                dim=1,
            )
            detail_gate = self._expand_gate(self.detail_gate(detail_evidence))
            # Cross-resolution consistency suppresses one-scale-only texture.
            detail_update = (
                detail_gate
                * detail_consistency
                * self.detail_delta(detail_discrepancy)
            )
            detail_scale = torch.tanh(self.alpha_detail)
            detail_residual = detail_scale * detail_update
            result = result + detail_residual

        with torch.no_grad():
            self._last_region_ratio.copy_(
                self._residual_ratio(region_residual, medium)
            )
            self._last_detail_ratio.copy_(
                self._residual_ratio(detail_residual, medium)
            )
            self._last_agreement.copy_(agreement.float().mean())
            self._last_detail_consistency.copy_(
                detail_consistency.float().mean()
            )
        return result


class _PvtV2B4_FPN_RCDIBase(PvtV2B4_FPN_Baseline):
    """PVTv2-B4 FPN with RCDI replacing all MHSIU units."""

    def __init__(
        self,
        pretrained: bool = True,
        input_norm: bool = True,
        fpn_dim: int = 64,
        use_checkpoint: bool = False,
        rcdi_groups: int = 4,
        frequency_split: bool = True,
        use_context: bool = True,
        offset_levels: Iterable[int] = (2, 3),
        detail_levels: Iterable[int] = (2, 3),
        max_offset: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        if fpn_dim % rcdi_groups:
            raise ValueError("fpn_dim must be divisible by rcdi_groups")
        offset_levels = {int(x) for x in offset_levels}
        detail_levels = {int(x) for x in detail_levels}
        valid_levels = {2, 3, 4, 5}
        if not offset_levels <= valid_levels or not detail_levels <= valid_levels:
            raise ValueError("RCDI levels must be selected from (2, 3, 4, 5)")

        self.rcdi = nn.ModuleDict()
        for level in (2, 3, 4, 5):
            self.rcdi[str(level)] = RCDIBlock(
                channels=fpn_dim,
                num_groups=rcdi_groups,
                frequency_split=frequency_split,
                use_detail=level in detail_levels,
                use_context=use_context and level != 5,
                use_offset=level in offset_levels,
                max_offset=max_offset,
            )

    def _fuse_level(
        self,
        level: int,
        medium: torch.Tensor,
        large: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.rcdi[str(level)](
            large=large,
            medium=medium,
            context=context,
        )

    def body(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_m = data["image_m"]
        image_l = data.get("image_l")
        if image_l is None:
            image_l = F.interpolate(
                image_m,
                scale_factor=1.5,
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )

        m2, m3, m4, m5 = self.normalize_encoder(image_m)
        l2, l3, l4, l5 = self.normalize_encoder(image_l)

        medium = {
            2: self.lateral_2(m2),
            3: self.lateral_3(m3),
            4: self.lateral_4(m4),
            5: self.lateral_5(m5),
        }
        large = {
            2: self.lateral_2(l2),
            3: self.lateral_3(l3),
            4: self.lateral_4(l4),
            5: self.lateral_5(l5),
        }

        # Sequential top-down decoding lets high-level semantics verify which
        # mixed-resolution discrepancies are useful at shallower levels.
        z5 = self._fuse_level(5, medium[5], large[5], context=None)
        p5 = self.smooth_5(z5)

        t4 = self._resize_like(p5, medium[4])
        z4 = self._fuse_level(4, medium[4], large[4], context=t4)
        p4 = self.smooth_4(z4 + t4)

        t3 = self._resize_like(p4, medium[3])
        z3 = self._fuse_level(3, medium[3], large[3], context=t3)
        p3 = self.smooth_3(z3 + t3)

        t2 = self._resize_like(p3, medium[2])
        z2 = self._fuse_level(2, medium[2], large[2], context=t2)
        p2 = self.smooth_2(z2 + t2)

        logits = self.predictor(p2)
        return F.interpolate(
            logits,
            size=image_m.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, data, iter_percentage=1, **kwargs):
        result = super().forward(
            data=data,
            iter_percentage=iter_percentage,
            **kwargs,
        )
        if not self.training or not isinstance(result, dict):
            return result

        # Diagnostics only; they do not change the inherited BCE objective.
        for level in (2, 3, 4, 5):
            block = self.rcdi[str(level)]
            prefix = f"rcdi_p{level}"
            result["loss_items"][f"{prefix}_region_ratio"] = (
                block._last_region_ratio.detach()
            )
            result["loss_items"][f"{prefix}_detail_ratio"] = (
                block._last_detail_ratio.detach()
            )
            result["loss_items"][f"{prefix}_agreement"] = (
                block._last_agreement.detach()
            )
            result["loss_items"][f"{prefix}_detail_consistency"] = (
                block._last_detail_consistency.detach()
            )
            result["loss_items"][f"{prefix}_offset"] = (
                block._last_offset_magnitude.detach()
            )
            result["loss_items"][f"{prefix}_alpha_region"] = torch.tanh(
                block.alpha_region
            ).detach()
            if block.use_detail:
                result["loss_items"][f"{prefix}_alpha_detail"] = torch.tanh(
                    block.alpha_detail
                ).detach()
        return result


class PvtV2B4_FPN_RCDI_R1(_PvtV2B4_FPN_RCDIBase):
    """R1: anchored scale-discrepancy routing, without frequency split."""

    def __init__(self, **kwargs) -> None:
        for key in ("frequency_split", "use_context", "offset_levels", "detail_levels"):
            kwargs.pop(key, None)
        super().__init__(
            frequency_split=False,
            use_context=False,
            offset_levels=(),
            detail_levels=(),
            **kwargs,
        )


class PvtV2B4_FPN_RCDI_R2(_PvtV2B4_FPN_RCDIBase):
    """R2: R1 + region/detail decomposition and detail consistency."""

    def __init__(self, **kwargs) -> None:
        for key in ("frequency_split", "use_context", "offset_levels", "detail_levels"):
            kwargs.pop(key, None)
        super().__init__(
            frequency_split=True,
            use_context=False,
            offset_levels=(),
            detail_levels=(2, 3),
            **kwargs,
        )


class PvtV2B4_FPN_RCDI_R3(_PvtV2B4_FPN_RCDIBase):
    """R3: R2 + top-down semantic context verification."""

    def __init__(self, **kwargs) -> None:
        for key in ("frequency_split", "use_context", "offset_levels", "detail_levels"):
            kwargs.pop(key, None)
        super().__init__(
            frequency_split=True,
            use_context=True,
            offset_levels=(),
            detail_levels=(2, 3),
            **kwargs,
        )


class PvtV2B4_FPN_RCDI(_PvtV2B4_FPN_RCDIBase):
    """Final RCDI: R3 + constrained content alignment at P2/P3."""

    def __init__(self, **kwargs) -> None:
        for key in ("frequency_split", "use_context", "offset_levels", "detail_levels"):
            kwargs.pop(key, None)
        super().__init__(
            frequency_split=True,
            use_context=True,
            offset_levels=(2, 3),
            detail_levels=(2, 3),
            **kwargs,
        )


__all__ = [
    "RCDIBlock",
    "PvtV2B4_FPN_RCDI_R1",
    "PvtV2B4_FPN_RCDI_R2",
    "PvtV2B4_FPN_RCDI_R3",
    "PvtV2B4_FPN_RCDI",
]
