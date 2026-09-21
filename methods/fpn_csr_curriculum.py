# -*- coding: utf-8 -*-
"""
APBOXNet: PVTv2-B4 FPN + Curriculum Scale Router (CSR).

Design
------
1. Keep the original PVT-FPN path permanently.
2. Add an MHSIU-inspired three-branch scale router only at P4 and P3.
3. Three branches at each level:
       fine    = higher-resolution lateral feature
       current = current lateral feature
       context = top-down FPN feature
4. The router is injected by a zero-initialized 1x1 residual projection:
       P_i = P_i_base + alpha(t) * ZeroConv(CSR_i)
5. alpha(t), for a 150-epoch run:
       epoch 1-60   : 0.15 -> 0.30
       epoch 61-100 : 0.30 -> alpha_max
       epoch 101-150: alpha_max
   The model uses iter_percentage, so basemain_continuous.py needs no change.
6. Exported ablation models include:
       PvtV2B4_FPN_CSR_BCE
       PvtV2B4_FPN_CSR_NC_Curriculum
       PvtV2B4_FPN_CSR_NC_A05 / A06 / A07
       PvtV2B4_FPN_CSR_RGPU_NC_Curriculum

No box is consumed in this B2 implementation. The returned scale attention is
kept for logging and for the next B3 experiment (box-supervised scale routing).
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.pvt_v2_eff import pvt_v2_eff_b4
from .zoomnext.layers import RGPU
from .zoomnext.ops import ConvBNReLU, PixelNormalizer

LOGGER = logging.getLogger("main")


def _cosine_lerp(start: float, end: float, x: float) -> float:
    """Cosine interpolation from start to end for x in [0, 1]."""
    x = min(max(float(x), 0.0), 1.0)
    coef = 0.5 * (1.0 - math.cos(math.pi * x))
    return float(start + (end - start) * coef)


def csr_alpha_from_progress(
    progress: float,
    alpha_max: float = 1.0,
) -> float:
    """150-epoch schedule expressed through normalized training progress.

    60 / 150 = 0.4
    100 / 150 = 2/3
    alpha_max changes only the second-stage endpoint and final plateau.
    """
    alpha_max = float(alpha_max)
    if not 0.30 <= alpha_max <= 1.00:
        raise ValueError(
            "csr_alpha_max must be in [0.30, 1.00], "
            f"but got {alpha_max}."
        )

    p = min(max(float(progress), 0.0), 1.0)
    stage1_end = 60.0 / 150.0
    stage2_end = 100.0 / 150.0

    if p <= stage1_end:
        local_p = p / stage1_end
        return _cosine_lerp(0.15, 0.30, local_p)

    if p <= stage2_end:
        local_p = (p - stage1_end) / (stage2_end - stage1_end)
        return _cosine_lerp(0.30, alpha_max, local_p)

    return alpha_max


class ZeroResidualProjection(nn.Conv2d):
    """A strict zero-initialized residual projection."""

    def __init__(self, channels: int):
        super().__init__(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class CurriculumScaleRouter(nn.Module):
    """MHSIU-inspired grouped three-scale competition.

    Inputs
    ------
    fine:
        feature from one finer FPN/lateral level
    current:
        current-level lateral feature; defines the target grid
    context:
        coarser top-down FPN feature

    Returns
    -------
    fused:
        [B, C, H, W]
    attn:
        [B, G, 3, H, W], branch order = fine/current/context

    This is intentionally not the original ZoomNeXt MHSIU. It preserves the
    grouped three-branch competition mechanism while replacing three image
    scales with three FPN semantic/spatial scales.
    """

    def __init__(
        self,
        channels: int = 64,
        num_groups: int = 4,
    ):
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by "
                f"num_groups={num_groups}."
            )

        self.channels = int(channels)
        self.num_groups = int(num_groups)
        group_dim = channels // num_groups

        # Branch-specific alignment/modeling.
        self.fine_pre = ConvBNReLU(channels, channels, 3, 1, 1)
        self.current_pre = ConvBNReLU(channels, channels, 3, 1, 1)
        self.context_pre = ConvBNReLU(channels, channels, 3, 1, 1)

        self.fine_refine = ConvBNReLU(channels, channels, 3, 1, 1)
        self.current_refine = ConvBNReLU(channels, channels, 3, 1, 1)
        self.context_refine = ConvBNReLU(channels, channels, 3, 1, 1)

        # Inter-branch evidence and value transforms.
        self.attn_mix = ConvBNReLU(3 * channels, 3 * channels, 1)
        self.value_mix = ConvBNReLU(3 * channels, 3 * channels, 1)

        # Same core idea as MHSIU: each channel group predicts three
        # spatially-varying scale weights.
        self.route = nn.Sequential(
            ConvBNReLU(3 * group_dim, group_dim, 1),
            ConvBNReLU(group_dim, group_dim, 3, 1, 1),
            nn.Conv2d(group_dim, 3, kernel_size=1, bias=True),
        )

    @staticmethod
    def _align_fine(
        x: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        # MHSIU-style max + average information when reducing resolution.
        if x.shape[-2:] == size:
            return x
        max_x = F.adaptive_max_pool2d(x, output_size=size)
        avg_x = F.adaptive_avg_pool2d(x, output_size=size)
        return max_x + avg_x

    @staticmethod
    def _align_context(
        x: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        fine: torch.Tensor,
        current: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_size = current.shape[-2:]

        fine = self._align_fine(
            self.fine_pre(fine),
            target_size,
        )
        current = self.current_pre(current)
        context = self._align_context(
            self.context_pre(context),
            target_size,
        )

        fine = self.fine_refine(fine)
        current = self.current_refine(current)
        context = self.context_refine(context)

        branches = torch.cat(
            [fine, current, context],
            dim=1,
        )
        b, _, h, w = branches.shape
        g = self.num_groups
        c = self.channels
        d = c // g

        # Attention evidence:
        # [B, 3C, H, W] -> [B*G, 3D, H, W].
        attn_feat = self.attn_mix(branches)
        attn_feat = (
            attn_feat
            .view(b, 3, g, d, h, w)
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(b * g, 3 * d, h, w)
        )
        attn = torch.softmax(
            self.route(attn_feat),
            dim=1,
        )  # [B*G, 3, H, W]

        # Values:
        # [B, 3C, H, W] -> [B*G, 3, D, H, W].
        values = self.value_mix(branches)
        values = (
            values
            .view(b, 3, g, d, h, w)
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(b * g, 3, d, h, w)
        )

        fused = (
            attn.unsqueeze(2) * values
        ).sum(dim=1)
        fused = (
            fused
            .view(b, g, d, h, w)
            .reshape(b, c, h, w)
        )

        attn = attn.view(b, g, 3, h, w)
        return fused, attn


class NoisyCODCurriculumLoss(nn.Module):
    """Faithful loss-only adaptation of the official Noisy-COD PNet NCLoss."""

    def __init__(
        self,
        q_switch_ratio: float = 0.40,
        boundary_kernel: int = 31,
        boundary_gain: float = 5.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.q_switch_ratio = float(q_switch_ratio)
        self.boundary_kernel = int(boundary_kernel)
        self.boundary_gain = float(boundary_gain)
        self.eps = float(eps)

    def weighted_bce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pad = self.boundary_kernel // 2
        local_mean = F.avg_pool2d(
            target,
            kernel_size=self.boundary_kernel,
            stride=1,
            padding=pad,
        )
        weight = (
            1.0
            + self.boundary_gain
            * torch.abs(local_mean - target)
        )
        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        wbce = (
            (weight * bce_map).sum(dim=(2, 3))
            / weight.sum(dim=(2, 3)).clamp_min(self.eps)
        )
        return wbce.mean()

    def nc_term(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        q: float,
    ) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        prob_flat = prob.flatten(1)
        target_flat = target.flatten(1)

        numerator = torch.sum(
            torch.abs(prob_flat - target_flat).pow(q),
            dim=1,
        )
        intersection = torch.sum(
            prob_flat * target_flat,
            dim=1,
        )
        denominator = (
            torch.sum(prob_flat, dim=1)
            + torch.sum(target_flat, dim=1)
            - intersection
        ).clamp_min(self.eps)

        return (numerator / denominator).mean()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        iter_percentage: float,
    ):
        progress = float(iter_percentage)
        q = (
            2.0
            if progress <= self.q_switch_ratio
            else 1.0
        )

        wbce = self.weighted_bce(logits, target)
        nc = self.nc_term(logits, target, q=q)

        # Match the released Noisy-COD PNet behavior.
        if q == 2.0:
            total = nc + wbce
        else:
            total = 2.0 * nc

        return total, q, nc.detach(), wbce.detach()


class _PvtV2B4_FPN_CSR_Base(nn.Module):
    """Original PVTv2-B4 FPN path plus P4/P3 CSR residual branches."""

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        csr_groups=4,
        csr_alpha_max=1.0,
        use_rgpu=False,
        rgpu_groups=6,
        rgpu_levels=(5, 4, 3, 2),
        rgpu_residual_max=1.0,
        num_frames=1,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.csr_alpha_max = float(csr_alpha_max)
        if not 0.30 <= self.csr_alpha_max <= 1.00:
            raise ValueError(
                "csr_alpha_max must be in [0.30, 1.00], "
                f"but got {self.csr_alpha_max}."
            )

        self.encoder = pvt_v2_eff_b4(
            pretrained=pretrained,
            use_checkpoint=use_checkpoint,
        )
        self.embed_dims = list(self.encoder.embed_dims)
        self.normalizer = (
            PixelNormalizer()
            if input_norm
            else nn.Identity()
        )

        # Keep the original FPN lateral path.
        self.lateral_2 = nn.Conv2d(
            self.embed_dims[0], fpn_dim, 1, bias=False
        )
        self.lateral_3 = nn.Conv2d(
            self.embed_dims[1], fpn_dim, 1, bias=False
        )
        self.lateral_4 = nn.Conv2d(
            self.embed_dims[2], fpn_dim, 1, bias=False
        )
        self.lateral_5 = nn.Conv2d(
            self.embed_dims[3], fpn_dim, 1, bias=False
        )

        self.smooth_5 = ConvBNReLU(
            fpn_dim, fpn_dim, 3, 1, 1
        )
        self.smooth_4 = ConvBNReLU(
            fpn_dim, fpn_dim, 3, 1, 1
        )
        self.smooth_3 = ConvBNReLU(
            fpn_dim, fpn_dim, 3, 1, 1
        )
        self.smooth_2 = ConvBNReLU(
            fpn_dim, fpn_dim, 3, 1, 1
        )

        # Only P4 and P3 receive scale-course residuals.
        self.csr_4 = CurriculumScaleRouter(
            channels=fpn_dim,
            num_groups=csr_groups,
        )
        self.csr_3 = CurriculumScaleRouter(
            channels=fpn_dim,
            num_groups=csr_groups,
        )
        self.zero_4 = ZeroResidualProjection(fpn_dim)
        self.zero_3 = ZeroResidualProjection(fpn_dim)

        # Optional original ZoomNeXt RGPU refinement.  Keeping this behind a
        # switch guarantees that the existing CSR and CSR+NC models remain
        # exact architectural baselines for the RGPU ablation.
        allowed_rgpu_levels = {2, 3, 4, 5}
        requested_rgpu_levels = tuple(
            dict.fromkeys(int(level) for level in rgpu_levels)
        )
        unknown_levels = (
            set(requested_rgpu_levels) - allowed_rgpu_levels
        )
        if unknown_levels:
            raise ValueError(
                "rgpu_levels can only contain 2, 3, 4, 5, "
                f"but got {sorted(unknown_levels)}."
            )

        self.rgpu_levels = (
            requested_rgpu_levels if use_rgpu else tuple()
        )
        self.rgpu_residual_max = float(rgpu_residual_max)
        if self.rgpu_residual_max <= 0:
            raise ValueError(
                "rgpu_residual_max must be positive, "
                f"but got {self.rgpu_residual_max}."
            )
        self.rgpu = nn.ModuleDict(
            {
                str(level): RGPU(
                    in_c=fpn_dim,
                    num_groups=rgpu_groups,
                    num_frames=num_frames,
                )
                for level in self.rgpu_levels
            }
        )
        # ReZero-style per-level mixing keeps the new model exactly equal to
        # its no-RGPU control at initialization.  tanh bounds every learned
        # correction while still giving the mixing scalar a gradient at zero.
        self.rgpu_mix = nn.ParameterDict(
            {
                str(level): nn.Parameter(torch.zeros(()))
                for level in self.rgpu_levels
            }
        )

        self.predictor = nn.Sequential(
            ConvBNReLU(fpn_dim, 32, 3, 1, 1),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def normalize_encoder(self, image):
        image = self.normalizer(image)
        features = self.encoder(image)
        return (
            features["reduction_2"],
            features["reduction_3"],
            features["reduction_4"],
            features["reduction_5"],
        )

    @staticmethod
    def _resize_like(source, target):
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _mean_route(attn: torch.Tensor) -> torch.Tensor:
        # [B,G,3,H,W] -> [3]
        return attn.detach().mean(dim=(0, 1, 3, 4))

    def _apply_rgpu(self, level, feature, diagnostics):
        """Refine one FPN level and record its relative correction size."""
        key = str(level)
        if key not in self.rgpu:
            return feature

        candidate = self.rgpu[key](feature)
        mix = self.rgpu_residual_max * torch.tanh(
            self.rgpu_mix[key]
        )
        refined = feature + mix.to(feature.dtype) * (
            candidate - feature
        )
        with torch.no_grad():
            delta_magnitude = (refined - feature).float().abs().mean()
            base_magnitude = feature.float().abs().mean().clamp_min(1e-6)
            diagnostics[f"rgpu_p{level}_ratio"] = (
                delta_magnitude / base_magnitude
            )
            diagnostics[f"rgpu_p{level}_mix"] = mix.detach()
        return refined

    def body(
        self,
        data,
        iter_percentage=1.0,
    ):
        image = data["image_m"]
        c2, c3, c4, c5 = self.normalize_encoder(image)

        l2 = self.lateral_2(c2)
        l3 = self.lateral_3(c3)
        l4 = self.lateral_4(c4)
        l5 = self.lateral_5(c5)

        rgpu_diagnostics = {}

        # Permanent original FPN path at P5.
        p5 = self.smooth_5(l5)
        p5 = self._apply_rgpu(5, p5, rgpu_diagnostics)

        # ---------------------------
        # P4: original FPN + CSR residual
        # ---------------------------
        top4 = self._resize_like(p5, l4)
        p4_base = self.smooth_4(l4 + top4)

        z4, a4 = self.csr_4(
            fine=l3,
            current=l4,
            context=p5,
        )

        alpha = csr_alpha_from_progress(
            float(iter_percentage),
            alpha_max=self.csr_alpha_max,
        )
        p4 = p4_base + alpha * self.zero_4(z4)
        p4 = self._apply_rgpu(4, p4, rgpu_diagnostics)

        # ---------------------------
        # P3: original FPN + CSR residual
        # ---------------------------
        top3 = self._resize_like(p4, l3)
        p3_base = self.smooth_3(l3 + top3)

        z3, a3 = self.csr_3(
            fine=l2,
            current=l3,
            context=p4,
        )
        p3 = p3_base + alpha * self.zero_3(z3)
        p3 = self._apply_rgpu(3, p3, rgpu_diagnostics)

        # P2 has no CSR branch; the RGPU variant may still refine it.
        p2 = self.smooth_2(
            l2 + self._resize_like(p3, l2)
        )
        p2 = self._apply_rgpu(2, p2, rgpu_diagnostics)

        logits = self.predictor(p2)
        logits = F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        aux = {
            "alpha": float(alpha),
            "a4": a4,
            "a3": a3,
            "rgpu_diagnostics": rgpu_diagnostics,
        }
        return logits, aux

    def get_grouped_params(self):
        param_groups = {
            "pretrained": [],
            "fixed": [],
            "retrained": [],
        }

        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                param_groups["retrained"].append(param)

        LOGGER.info(
            "Parameter Groups:{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}"
            "}"
        )
        return param_groups


class PvtV2B4_FPN_CSR_BCE(_PvtV2B4_FPN_CSR_Base):
    """B2a: original BCE + CSR residual. Useful for pure architecture ablation."""

    def forward(
        self,
        data,
        iter_percentage=1,
        **kwargs,
    ):
        del kwargs
        logits, aux = self.body(
            data=data,
            iter_percentage=iter_percentage,
        )

        if not self.training:
            return logits

        target = data["mask"].float()
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(
                target,
                size=logits.shape[-2:],
                mode="nearest",
            )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="mean",
        )

        a3_mean = self._mean_route(aux["a3"])
        a4_mean = self._mean_route(aux["a4"])

        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid()},
            "loss": bce,
            "loss_items": {
                "bce": bce.detach(),
                "total": bce.detach(),
                "csr_alpha": torch.tensor(
                    aux["alpha"],
                    device=logits.device,
                ),
                "p3_fine": a3_mean[0],
                "p3_current": a3_mean[1],
                "p3_context": a3_mean[2],
                "p4_fine": a4_mean[0],
                "p4_current": a4_mean[1],
                "p4_context": a4_mean[2],
            },
            "loss_str": (
                f"L:{bce.detach().item():.4f} "
                f"A:{aux['alpha']:.3f}"
            ),
        }


class PvtV2B4_FPN_CSR_NC_Curriculum(
    _PvtV2B4_FPN_CSR_Base
):
    """B2b: Noisy-COD q curriculum + CSR residual."""

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        csr_groups=4,
        csr_alpha_max=1.0,
        use_checkpoint=False,
        q_switch_ratio=0.40,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            csr_groups=csr_groups,
            csr_alpha_max=csr_alpha_max,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.nc_loss = NoisyCODCurriculumLoss(
            q_switch_ratio=q_switch_ratio
        )

    def forward(
        self,
        data,
        iter_percentage=1,
        **kwargs,
    ):
        del kwargs
        logits, aux = self.body(
            data=data,
            iter_percentage=iter_percentage,
        )

        if not self.training:
            return logits

        target = data["mask"].float()
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(
                target,
                size=logits.shape[-2:],
                mode="nearest",
            )

        total, q, nc, wbce = self.nc_loss(
            logits=logits,
            target=target,
            iter_percentage=iter_percentage,
        )

        a3_mean = self._mean_route(aux["a3"])
        a4_mean = self._mean_route(aux["a4"])

        loss_items = {
            # Preserve trainer compatibility.
            "bce": wbce,
            "nc": nc,
            "q": torch.tensor(
                q,
                device=logits.device,
                dtype=logits.dtype,
            ),
            "csr_alpha": torch.tensor(
                aux["alpha"],
                device=logits.device,
                dtype=logits.dtype,
            ),
            "p3_fine": a3_mean[0],
            "p3_current": a3_mean[1],
            "p3_context": a3_mean[2],
            "p4_fine": a4_mean[0],
            "p4_current": a4_mean[1],
            "p4_context": a4_mean[2],
            "total": total.detach(),
        }
        loss_items.update(aux["rgpu_diagnostics"])

        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid()},
            "loss": total,
            "loss_items": loss_items,
            "loss_str": (
                f"L:{total.detach().item():.4f} "
                f"NC:{nc.item():.4f} "
                f"WBCE:{wbce.item():.4f} "
                f"Q:{q:.1f} "
                f"A:{aux['alpha']:.3f}"
            ),
        }


class PvtV2B4_FPN_CSR_RGPU_NC_Curriculum(
    PvtV2B4_FPN_CSR_NC_Curriculum
):
    """CSR+NC with zero-gated ZoomNeXt RGPU at P5/P4/P3/P2."""

    def __init__(
        self,
        rgpu_groups=6,
        rgpu_levels=(5, 4, 3, 2),
        rgpu_residual_max=1.0,
        **kwargs,
    ):
        kwargs["use_rgpu"] = True
        kwargs["rgpu_groups"] = rgpu_groups
        kwargs["rgpu_levels"] = rgpu_levels
        kwargs["rgpu_residual_max"] = rgpu_residual_max
        super().__init__(**kwargs)


class PvtV2B4_FPN_CSR_NC_A05(
    PvtV2B4_FPN_CSR_NC_Curriculum
):
    """CSR+NC ablation with alpha capped at 0.50."""

    def __init__(self, **kwargs):
        kwargs["csr_alpha_max"] = 0.50
        super().__init__(**kwargs)


class PvtV2B4_FPN_CSR_NC_A06(
    PvtV2B4_FPN_CSR_NC_Curriculum
):
    """CSR+NC ablation with alpha capped at 0.60."""

    def __init__(self, **kwargs):
        kwargs["csr_alpha_max"] = 0.60
        super().__init__(**kwargs)


class PvtV2B4_FPN_CSR_NC_A07(
    PvtV2B4_FPN_CSR_NC_Curriculum
):
    """CSR+NC ablation with alpha capped at 0.70."""

    def __init__(self, **kwargs):
        kwargs["csr_alpha_max"] = 0.70
        super().__init__(**kwargs)
