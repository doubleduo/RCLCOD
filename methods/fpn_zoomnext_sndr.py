# -*- coding: utf-8 -*-
"""FPN-ZoomNeXt with progressive upsampling and native RGB/Sobel detail.

This module is a strict architectural extension of
``PvtV2B4_FPN_ZoomNeXt``:

* the PVTv2-B4 encoder is unchanged;
* C2--C5 MHSIU2 fusion is unchanged;
* the FPN top-down path and C2--C5 RGPU blocks are unchanged;
* the original predictor is retained as an identity/anchor path;
* training still uses the inherited BCE objective.

The only addition is a zero-initialized residual prediction path:

    raw RGB + fixed Sobel -> shallow GroupNorm detail stem
                            -> semantic-gated P2 correction
    corrected P2 -> learned 2x refinement
                 -> optional semantic-gated 1/2 detail correction
                 -> learned 2x refinement -> residual logit

At initialization the residual logit is exactly zero, so the complete model
starts with the same output as the original FPN-ZoomNeXt.  GroupNorm is used
in every new trainable block because the RCLCOD training configuration freezes
BatchNorm statistics/affine parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_zoomnext import PvtV2B4_FPN_ZoomNeXt


def _valid_group_count(channels: int, preferred: int = 8) -> int:
    """Return the largest valid GroupNorm group count up to ``preferred``."""
    for groups in range(min(int(preferred), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    """Convolution + GroupNorm + GELU used by the new detail path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        gn_groups: int = 8,
        activate: bool = True,
    ):
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(
                _valid_group_count(out_channels, gn_groups),
                out_channels,
            ),
        ]
        if activate:
            layers.append(nn.GELU())
        super().__init__(*layers)


class DepthwiseSeparableGN(nn.Sequential):
    """Low-cost spatial refinement followed by channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        gn_groups: int = 8,
    ):
        super().__init__(
            ConvGNAct(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                gn_groups=gn_groups,
            ),
            ConvGNAct(
                in_channels,
                out_channels,
                1,
                gn_groups=gn_groups,
            ),
        )


class FixedSobelMagnitude(nn.Module):
    """Parameter-free luminance Sobel magnitude computed from raw RGB."""

    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ) / 4.0
        kernel_y = kernel_x.t().contiguous()
        self.register_buffer(
            "kernel_x",
            kernel_x.view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "kernel_y",
            kernel_y.view(1, 1, 3, 3),
            persistent=False,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "FixedSobelMagnitude expects RGB input shaped [B, 3, H, W]."
            )

        input_dtype = image.dtype
        # Keep the fixed finite-difference operation numerically stable in AMP.
        with torch.autocast(device_type=image.device.type, enabled=False):
            rgb = image.float()
            luminance = (
                0.2989 * rgb[:, 0:1]
                + 0.5870 * rgb[:, 1:2]
                + 0.1140 * rgb[:, 2:3]
            )
            luminance = F.pad(luminance, (1, 1, 1, 1), mode="replicate")
            grad_x = F.conv2d(luminance, self.kernel_x.float())
            grad_y = F.conv2d(luminance, self.kernel_y.float())
            magnitude = torch.sqrt(
                grad_x.square() + grad_y.square() + 1e-12
            ).clamp_(0.0, 1.0)
        return magnitude.to(dtype=input_dtype)


class NativeDetailStem(nn.Module):
    """Produce 1/2 and 1/4 detail features from raw RGB + Sobel."""

    def __init__(
        self,
        half_channels: int = 32,
        quarter_channels: int = 64,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.to_half = ConvGNAct(
            4,
            half_channels,
            5,
            stride=2,
            padding=2,
            gn_groups=gn_groups,
        )
        self.half_refine = DepthwiseSeparableGN(
            half_channels,
            half_channels,
            gn_groups=gn_groups,
        )
        self.to_quarter = DepthwiseSeparableGN(
            half_channels,
            quarter_channels,
            stride=2,
            gn_groups=gn_groups,
        )

    def forward(
        self,
        image: torch.Tensor,
        sobel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.half_refine(self.to_half(torch.cat([image, sobel], dim=1)))
        quarter = self.to_quarter(half)
        return half, quarter


class SemanticGatedDetailResidual(nn.Module):
    """Inject detail only where it is compatible with the semantic feature.

    A one-channel spatial gate is intentionally used instead of an unrestricted
    channel gate.  The final 1x1 projection is zero-initialized, making this
    module an exact identity mapping at initialization.
    """

    def __init__(
        self,
        channels: int,
        gate_hidden: int = 16,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.detail_refine = DepthwiseSeparableGN(
            channels,
            channels,
            gn_groups=gn_groups,
        )
        self.gate = nn.Sequential(
            ConvGNAct(
                channels * 3,
                gate_hidden,
                1,
                gn_groups=min(gn_groups, gate_hidden),
            ),
            nn.Conv2d(gate_hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.delta = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        semantic: torch.Tensor,
        detail: torch.Tensor,
    ) -> torch.Tensor:
        if detail.shape[-2:] != semantic.shape[-2:]:
            detail = F.interpolate(
                detail,
                size=semantic.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        detail = self.detail_refine(detail.to(dtype=semantic.dtype))
        gate_input = torch.cat(
            [semantic, detail, (semantic - detail).abs()],
            dim=1,
        )
        gate = self.gate(gate_input)
        return semantic + self.delta(gate * detail)


class ProgressiveResidualHead(nn.Module):
    """Learned 1/4 -> 1/2 -> 1x residual mask reconstruction."""

    def __init__(
        self,
        quarter_channels: int = 64,
        half_channels: int = 32,
        full_channels: int = 16,
        gate_hidden: int = 16,
        gn_groups: int = 8,
        fuse_half_detail: bool = True,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.fuse_half_detail = bool(fuse_half_detail)
        self.to_half = DepthwiseSeparableGN(
            quarter_channels,
            half_channels,
            gn_groups=gn_groups,
        )
        if self.fuse_half_detail:
            self.half_fusion = SemanticGatedDetailResidual(
                half_channels,
                gate_hidden=gate_hidden,
                gn_groups=gn_groups,
            )
        self.to_full = DepthwiseSeparableGN(
            half_channels,
            full_channels,
            gn_groups=gn_groups,
        )
        self.residual_predictor = nn.Conv2d(
            full_channels,
            1,
            kernel_size=1,
            bias=True,
        )
        if zero_init_output:
            nn.init.zeros_(self.residual_predictor.weight)
            nn.init.zeros_(self.residual_predictor.bias)

    def forward(
        self,
        quarter: torch.Tensor,
        output_size: tuple[int, int],
        half_detail: torch.Tensor | None = None,
    ) -> torch.Tensor:
        half_size = (
            max(1, int(output_size[0]) // 2),
            max(1, int(output_size[1]) // 2),
        )
        half = F.interpolate(
            quarter,
            size=half_size,
            mode="bilinear",
            align_corners=False,
        )
        half = self.to_half(half)

        if self.fuse_half_detail:
            if half_detail is None:
                raise ValueError(
                    "Dual-level SNDR requires the 1/2 RGB/Sobel detail feature."
                )
            half = self.half_fusion(half, half_detail)

        full = F.interpolate(
            half,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        full = self.to_full(full)
        return self.residual_predictor(full)


class _PvtV2B4_FPN_ZoomNeXt_SNDR_Base(PvtV2B4_FPN_ZoomNeXt):
    """Shared implementation for PUH, single-level SNDR and dual SNDR."""

    def __init__(
        self,
        pretrained: bool = True,
        input_norm: bool = True,
        fpn_dim: int = 64,
        use_checkpoint: bool = False,
        num_frames: int = 1,
        siu_groups: int = 4,
        hmu_groups: int = 6,
        clean_only_highres: bool = False,
        detail_mode: str = "dual",
        detail_half_channels: int = 32,
        detail_full_channels: int = 16,
        detail_gate_hidden: int = 16,
        detail_gn_groups: int = 8,
        zero_init_output: bool = True,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            num_frames=num_frames,
            siu_groups=siu_groups,
            hmu_groups=hmu_groups,
            use_mhsiu=True,
            use_rgpu=True,
            clean_only_highres=clean_only_highres,
            **kwargs,
        )

        detail_mode = str(detail_mode).lower()
        if detail_mode not in {"none", "single", "dual"}:
            raise ValueError("detail_mode must be 'none', 'single', or 'dual'.")
        if fpn_dim != 64:
            raise ValueError(
                "The first SNDR ablation fixes fpn_dim=64 for strict comparison."
            )
        self.detail_mode = detail_mode

        if self.detail_mode != "none":
            self.sobel = FixedSobelMagnitude()
            self.detail_stem = NativeDetailStem(
                half_channels=detail_half_channels,
                quarter_channels=fpn_dim,
                gn_groups=detail_gn_groups,
            )
            self.p2_detail_fusion = SemanticGatedDetailResidual(
                fpn_dim,
                gate_hidden=detail_gate_hidden,
                gn_groups=detail_gn_groups,
            )

        self.progressive_head = ProgressiveResidualHead(
            quarter_channels=fpn_dim,
            half_channels=detail_half_channels,
            full_channels=detail_full_channels,
            gate_hidden=detail_gate_hidden,
            gn_groups=detail_gn_groups,
            fuse_half_detail=self.detail_mode == "dual",
            zero_init_output=zero_init_output,
        )

    def _fpn_feature(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the unchanged FPN-ZoomNeXt trunk and return ``(image_m, p2)``."""
        image_m = data["image_m"]

        m2, m3, m4, m5 = self.normalize_encoder(image_m)
        z2 = self.lateral_2(m2)
        z3 = self.lateral_3(m3)
        z4 = self.lateral_4(m4)
        z5 = self.lateral_5(m5)

        # Preserve the parent's optional Clean-only high-resolution routing.
        if self.use_mhsiu:
            if self.training and self.clean_only_highres:
                pool_id = data.get("pool_id")
                if pool_id is None:
                    raise KeyError(
                        "clean_only_highres=True requires data['pool_id'] "
                        "during training (0=Clean, 1=Noisy)."
                    )
                pool_id = pool_id.reshape(-1).to(device=image_m.device)
                if pool_id.numel() != image_m.shape[0]:
                    raise ValueError("pool_id must have one value per sample.")
                if not torch.all((pool_id == 0) | (pool_id == 1)):
                    raise ValueError("pool_id must contain only 0 or 1.")
                clean_indices = torch.nonzero(pool_id == 0, as_tuple=True)[0]
                if clean_indices.numel() > 0:
                    image_l = self._large_image(data, image_m)
                    z2, z3, z4, z5 = self._fuse_large_subset(
                        (z2, z3, z4, z5),
                        image_l,
                        clean_indices,
                    )
            else:
                image_l = self._large_image(data, image_m)
                l2, l3, l4, l5 = self.normalize_encoder(image_l)
                z2 = self.siu_2(l=self.lateral_2(l2), m=z2)
                z3 = self.siu_3(l=self.lateral_3(l3), m=z3)
                z4 = self.siu_4(l=self.lateral_4(l4), m=z4)
                z5 = self.siu_5(l=self.lateral_5(l5), m=z5)

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

        return image_m, p2

    def body(self, data):
        image_m, p2 = self._fpn_feature(data)
        output_size = tuple(image_m.shape[-2:])

        # Original FPN-ZoomNeXt predictor remains the anchor path.
        base_logits = self.predictor(p2)
        base_logits = F.interpolate(
            base_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        half_detail = None
        detail_p2 = p2
        if self.detail_mode != "none":
            edge = self.sobel(image_m)
            half_detail, quarter_detail = self.detail_stem(image_m, edge)
            detail_p2 = self.p2_detail_fusion(p2, quarter_detail)

        residual_logits = self.progressive_head(
            detail_p2,
            output_size=output_size,
            half_detail=half_detail if self.detail_mode == "dual" else None,
        )
        return base_logits + residual_logits


class PvtV2B4_FPN_ZoomNeXt_PUH(_PvtV2B4_FPN_ZoomNeXt_SNDR_Base):
    """D1 control: progressive residual head only, without RGB/Sobel."""

    def __init__(self, **kwargs):
        kwargs.pop("detail_mode", None)
        super().__init__(detail_mode="none", **kwargs)


class PvtV2B4_FPN_ZoomNeXt_SNDR_Single(
    _PvtV2B4_FPN_ZoomNeXt_SNDR_Base
):
    """D2 control: RGB/Sobel semantic fusion only at P2 (1/4)."""

    def __init__(self, **kwargs):
        kwargs.pop("detail_mode", None)
        super().__init__(detail_mode="single", **kwargs)


class PvtV2B4_FPN_ZoomNeXt_SNDR(_PvtV2B4_FPN_ZoomNeXt_SNDR_Base):
    """D3 recommended: P2 and 1/2 dual-level RGB/Sobel detail fusion."""

    def __init__(self, **kwargs):
        kwargs.pop("detail_mode", None)
        super().__init__(detail_mode="dual", **kwargs)


__all__ = [
    "PvtV2B4_FPN_ZoomNeXt_PUH",
    "PvtV2B4_FPN_ZoomNeXt_SNDR_Single",
    "PvtV2B4_FPN_ZoomNeXt_SNDR",
]
