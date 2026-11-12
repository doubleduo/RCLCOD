# -*- coding: utf-8 -*-
"""PVTv2-B4 + plain FPN baseline used by the B2 curriculum trainer."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.pvt_v2_eff import pvt_v2_eff_b4
from .zoomnext.ops import ConvBNReLU, PixelNormalizer

LOGGER = logging.getLogger("main")
import timm

class PvtV2B4_FPN_Baseline(nn.Module):
    """Single-scale PVTv2-B4 encoder with a conventional top-down FPN."""

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        del kwargs

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

        self.lateral_2 = nn.Conv2d(
            self.embed_dims[0],
            fpn_dim,
            kernel_size=1,
            bias=False,
        )
        self.lateral_3 = nn.Conv2d(
            self.embed_dims[1],
            fpn_dim,
            kernel_size=1,
            bias=False,
        )
        self.lateral_4 = nn.Conv2d(
            self.embed_dims[2],
            fpn_dim,
            kernel_size=1,
            bias=False,
        )
        self.lateral_5 = nn.Conv2d(
            self.embed_dims[3],
            fpn_dim,
            kernel_size=1,
            bias=False,
        )

        self.smooth_5 = ConvBNReLU(
            fpn_dim,
            fpn_dim,
            3,
            1,
            1,
        )
        self.smooth_4 = ConvBNReLU(
            fpn_dim,
            fpn_dim,
            3,
            1,
            1,
        )
        self.smooth_3 = ConvBNReLU(
            fpn_dim,
            fpn_dim,
            3,
            1,
            1,
        )
        self.smooth_2 = ConvBNReLU(
            fpn_dim,
            fpn_dim,
            3,
            1,
            1,
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

    def body(self, data):
        image = data["image_m"]
        c2, c3, c4, c5 = self.normalize_encoder(image)

        p5 = self.smooth_5(self.lateral_5(c5))
        p4 = self.smooth_4(
            self.lateral_4(c4)
            + self._resize_like(p5, c4)
        )
        p3 = self.smooth_3(
            self.lateral_3(c3)
            + self._resize_like(p4, c3)
        )
        p2 = self.smooth_2(
            self.lateral_2(c2)
            + self._resize_like(p3, c2)
        )

        logits = self.predictor(p2)
        return F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        data,
        iter_percentage=1,
        **kwargs,
    ):
        del iter_percentage, kwargs
        logits = self.body(data=data)

        if not self.training:
            return logits

        mask = data["mask"].float()
        if mask.shape[-2:] != logits.shape[-2:]:
            mask = F.interpolate(
                mask,
                size=logits.shape[-2:],
                mode="nearest",
            )

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            mask,
            reduction="mean",
        )

        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid()},
            "loss": bce_loss,
            "loss_items": {
                "bce": bce_loss.detach(),
                "total": bce_loss.detach(),
            },
            "loss_str": (
                f"L:{bce_loss.detach().item():.4f} "
                f"BCE:{bce_loss.detach().item():.4f}"
            ),
        }

    def get_grouped_params(self):
        param_groups = {
            "pretrained": [],
            "fixed": [],
            "retrained": [],
        }

        for name, param in self.named_parameters():
            if name.startswith(
                "encoder.patch_embed1."
            ):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                param_groups["retrained"].append(param)

        LOGGER.info(
            "Parameter Groups:{"
            f"Pretrained: "
            f"{len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: "
            f"{len(param_groups['retrained'])}"
            "}"
        )
        return param_groups
















# -*- coding: utf-8 -*-
"""
PVTv2-B4 + plain FPN + LFP ablation for APBOXNet.

Drop this file into:
    APBOXNet/methods/fpn_lfp.py

This keeps PvtV2B4_FPN_Baseline unchanged except for inserting LFP after
all four lateral 1x1 projections and before the original top-down FPN.

LFP follows the official NS-FPN logic/hyperparameters:
- 1-level Haar DWT
- LL-guided spatial attention for LH/HL/HH
- gated 3x3 Gaussian filtering of weak HF responses
- learnable sigma initialized to 1.0
- gauss_gate = 0.5
- inverse DWT reconstruction

For easier APBOXNet ablation this port implements Haar DWT/IDWT directly
in PyTorch, so no extra pytorch_wavelets dependency is required.
"""




class HaarDWT(nn.Module):
    """Orthonormal one-level 2D Haar DWT."""

    def forward(self, x):
        # Guard odd shapes. PVTv2-B4 @ 384 normally gives even 96/48/24/12.
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]

        # Orthonormal Haar. Sign/order conventions of HF bands do not affect
        # LFP because all three bands share the same spatial gate and Gaussian.
        ll = (a + b + c + d) * 0.5
        lh = (-a - b + c + d) * 0.5
        hl = (-a + b - c + d) * 0.5
        hh = (a - b - c + d) * 0.5
        yh = torch.cat([lh, hl, hh], dim=1)
        return ll, yh


class HaarIDWT(nn.Module):
    """Inverse of HaarDWT."""

    def forward(self, ll, yh):
        lh, hl, hh = torch.chunk(yh, 3, dim=1)

        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5

        bsz, ch, h, w = ll.shape
        out = ll.new_empty(bsz, ch, h * 2, w * 2)
        out[..., 0::2, 0::2] = a
        out[..., 0::2, 1::2] = b
        out[..., 1::2, 0::2] = c
        out[..., 1::2, 1::2] = d
        return out


class SpatialAttention(nn.Module):
    """Low-frequency spatial attention used by LFP."""

    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True).values
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class LearnableGaussianFilter(nn.Module):
    """Depthwise Gaussian filter with learnable sigma, initialized to 1.0."""

    def __init__(self, channels, kernel_size=3, sigma_init=1.0):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))

    def _kernel(self, x):
        radius = self.kernel_size // 2
        coords = torch.arange(
            -radius,
            radius + 1,
            device=x.device,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        sigma = self.sigma.float().clamp_min(1e-4)
        kernel = torch.exp(
            -(xx.square() + yy.square()) / (2.0 * sigma.square())
        )
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, x):
        kernel = self._kernel(x)
        weight = kernel.repeat(self.channels, 1, 1, 1)
        x_pad = F.pad(
            x.float(),
            (self.padding, self.padding, self.padding, self.padding),
            mode="replicate",
        )
        y = F.conv2d(x_pad, weight, groups=self.channels)
        return y.to(dtype=x.dtype)


class LFP(nn.Module):
    """Low-frequency Guided Feature Purification."""

    def __init__(
        self,
        in_channels,
        with_gauss=True,
        gauss_gate=0.5,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.with_gauss = bool(with_gauss)
        self.gauss_gate = float(gauss_gate)

        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        self.attention = SpatialAttention(kernel_size=7)

        if self.with_gauss:
            self.gaussian_filter = LearnableGaussianFilter(
                channels=3 * self.in_channels,
                kernel_size=3,
                sigma_init=1.0,
            )

    def forward(self, x):
        _, c, h, w = x.shape
        if c != self.in_channels:
            raise RuntimeError(
                f"LFP expected {self.in_channels} channels, got {c}"
            )

        # Official implementation performs wavelet operations in fp32.
        input_dtype = x.dtype
        x_work = x.float()

        ll, yh = self.dwt(x_work)

        # Stage 1: LL predicts potential target locations and gates HF bands.
        att = self.attention(ll)
        yh = yh * att

        # Stage 2: only weak HF responses are Gaussian-smoothed.
        if self.with_gauss:
            yh_blurred = self.gaussian_filter(yh)
            weak_mask = (yh.abs() < self.gauss_gate).to(yh.dtype)
            yh = yh * (1.0 - weak_mask) + yh_blurred * weak_mask

        x_rec = self.idwt(ll, yh)
        x_rec = x_rec[..., :h, :w]
        return x_rec.to(dtype=input_dtype)


class PvtV2B4_FPN_LFP(PvtV2B4_FPN_Baseline):
    """
    Clean LFP-only ablation for APBOXNet's PvtV2B4_FPN_Baseline.

    Unchanged:
      encoder / fpn_dim / bilinear top-down / smooth convs / predictor /
      BCE loss / optimizer parameter grouping.

    Changed:
      lateral_2..5 output -> LFP -> original top-down FPN.
    """

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        lfp_with_gauss=True,
        lfp_gauss_gate=0.5,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

        lfp_kwargs = dict(
            in_channels=fpn_dim,
            with_gauss=lfp_with_gauss,
            gauss_gate=lfp_gauss_gate,
        )
        self.lfp_2 = LFP(**lfp_kwargs)
        self.lfp_3 = LFP(**lfp_kwargs)
        self.lfp_4 = LFP(**lfp_kwargs)
        self.lfp_5 = LFP(**lfp_kwargs)

    def body(self, data):
        image = data["image_m"]
        c2, c3, c4, c5 = self.normalize_encoder(image)

        # Only change vs baseline: LFP after each lateral projection.
        l2 = self.lfp_2(self.lateral_2(c2))
        l3 = self.lfp_3(self.lateral_3(c3))
        l4 = self.lfp_4(self.lateral_4(c4))
        l5 = self.lfp_5(self.lateral_5(c5))

        p5 = self.smooth_5(l5)
        p4 = self.smooth_4(l4 + self._resize_like(p5, c4))
        p3 = self.smooth_3(l3 + self._resize_like(p4, c3))
        p2 = self.smooth_2(l2 + self._resize_like(p3, c2))

        logits = self.predictor(p2)
        return F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )



# -*- coding: utf-8 -*-
"""
Loss-only curriculum ablation for APBOXNet FPN.

V1 keeps the encoder/decoder identical to PvtV2B4_FPN_Baseline and only
replaces BCE with the Noisy-COD-style Noise Correction Loss curriculum:

    first 40% of training: q = 2
    remaining training:    q = 1

For the current 150-epoch setup:
    epoch 1-60   -> q=2
    epoch 61-150 -> q=1

No box input, no MHSIU, no EMA, no extra decoder branch.
"""




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
        if not 0.0 <= q_switch_ratio <= 1.0:
            raise ValueError("q_switch_ratio must be in [0, 1].")
        if boundary_kernel % 2 != 1:
            raise ValueError("boundary_kernel must be odd.")

        self.q_switch_ratio = float(q_switch_ratio)
        self.boundary_kernel = int(boundary_kernel)
        self.boundary_gain = float(boundary_gain)
        self.eps = float(eps)

    def _weighted_bce(self, logits, target):
        pad = self.boundary_kernel // 2
        local_mean = F.avg_pool2d(
            target,
            kernel_size=self.boundary_kernel,
            stride=1,
            padding=pad,
        )
        weight = 1.0 + self.boundary_gain * torch.abs(local_mean - target)

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

    def _nc_term(self, logits, target, q):
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

    def forward(self, logits, target, iter_percentage):
        progress = float(iter_percentage)
        q = 2.0 if progress <= self.q_switch_ratio else 1.0

        wbce = self._weighted_bce(logits, target)
        nc = self._nc_term(logits, target, q=q)

        # Match the official Noisy-COD PNet implementation:
        # q=2 -> L_NC + weighted BCE
        # q=1 -> 2 * L_NC
        if q == 2.0:
            total = nc + wbce
        else:
            total = 2.0 * nc

        return total, {
            "q": q,
            "nc": nc.detach(),
            "wbce": wbce.detach(),
        }


class PvtV2B4_FPN_NC_Curriculum(PvtV2B4_FPN_Baseline):
    """Exact FPN baseline + Noisy-COD-style loss curriculum."""

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        q_switch_ratio=0.40,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.curriculum_loss = NoisyCODCurriculumLoss(
            q_switch_ratio=q_switch_ratio,
        )

    def forward(self, data, iter_percentage=1, **kwargs):
        del kwargs
        logits = self.body(data=data)

        if not self.training:
            return logits

        target = data["mask"].float()
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(
                target,
                size=logits.shape[-2:],
                mode="nearest",
            )

        total_loss, items = self.curriculum_loss(
            logits=logits,
            target=target,
            iter_percentage=iter_percentage,
        )

        q = float(items["q"])
        nc = items["nc"]
        wbce = items["wbce"]

        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid()},
            "loss": total_loss,
            "loss_items": {
                # Keep bce for compatibility with basemain_continuous.py logging.
                "bce": wbce,
                "nc": nc,
                "q": torch.as_tensor(
                    q,
                    device=logits.device,
                    dtype=logits.dtype,
                ),
                "total": total_loss.detach(),
            },
            "loss_str": (
                f"L:{total_loss.detach().item():.4f} "
                f"NC:{nc.item():.4f} "
                f"WBCE:{wbce.item():.4f} "
                f"Q:{q:.1f}"
            ),
        }

# -*- coding: utf-8 -*-
"""Reliability-Gated FPN for PVTv2-B4 binary segmentation.

The gate estimates whether an upsampled semantic feature is compatible with
the lateral feature at the current resolution.  It never consumes pseudo
labels or target reliability maps, so train and inference use the same path.

Fusion keeps the lateral feature as an identity path::

    P_i = Refine(L_i + G_i(L_i, up(P_{i+1})) * Align(up(P_{i+1})))

This makes a wrong or saturated gate less destructive than gating both input
branches.  ``target_weight`` is used only to weight the segmentation loss.
"""



import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.pvt_v2_eff import pvt_v2_eff_b4
from .zoomnext.ops import PixelNormalizer


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Choose the largest valid GroupNorm group count up to max_groups."""
    for groups in range(min(channels, max_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    """Convolution followed by batch-size-independent normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        groups: int = 1,
        activate: bool = True,
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activate:
            layers.append(nn.GELU())
        super().__init__(*layers)


class DepthwiseSeparableConv(nn.Sequential):
    """Low-cost 3x3 spatial refinement plus 1x1 channel mixing."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            ConvGNAct(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
            ),
            ConvGNAct(in_channels, out_channels, kernel_size=1),
        )


class CrossScaleReliabilityGate(nn.Module):
    """Predict a bounded spatial-channel gate from cross-scale evidence.

    Evidence consists of compressed local and top-down features, their
    absolute discrepancy, and their element-wise agreement.  The spatial and
    channel logits are added before the sigmoid, which is cheaper than
    predicting a full-resolution C-channel gate directly.

    The final layers are initialized to produce ``gate_init`` everywhere.
    Starting close to a normal FPN lets the model learn when to suppress
    semantic injection instead of forcing it to discover top-down fusion from
    scratch.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        min_gate: float = 0.05,
        gate_init: float = 0.90,
    ):
        super().__init__()
        if not 0.0 <= min_gate < 0.5:
            raise ValueError("min_gate must be in [0, 0.5).")
        if not min_gate < gate_init < 1.0 - min_gate:
            raise ValueError("gate_init must lie inside the bounded gate range.")

        hidden = max(channels // reduction, 16)
        evidence_channels = hidden * 4
        self.min_gate = float(min_gate)

        self.local_reduce = ConvGNAct(channels, hidden, kernel_size=1)
        self.top_reduce = ConvGNAct(channels, hidden, kernel_size=1)

        self.spatial_gate = nn.Sequential(
            ConvGNAct(evidence_channels, hidden, kernel_size=1),
            ConvGNAct(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
            ),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(evidence_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        # With zero last-layer weights, spatial_bias + channel_bias is a
        # constant logit.  Only the spatial branch carries the initial bias.
        bounded_probability = (gate_init - min_gate) / (1.0 - 2.0 * min_gate)
        initial_logit = math.log(bounded_probability / (1.0 - bounded_probability))
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.constant_(self.spatial_gate[-1].bias, initial_logit)
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)

    def forward(self, local: torch.Tensor, top_down: torch.Tensor) -> torch.Tensor:
        local_evidence = self.local_reduce(local)
        top_evidence = self.top_reduce(top_down)
        evidence = torch.cat(
            (
                local_evidence,
                top_evidence,
                torch.abs(local_evidence - top_evidence),
                local_evidence * top_evidence,
            ),
            dim=1,
        )
        gate_logit = self.spatial_gate(evidence) + self.channel_gate(evidence)
        gate = torch.sigmoid(gate_logit)
        return self.min_gate + (1.0 - 2.0 * self.min_gate) * gate


class ReliabilityGatedFusion(nn.Module):
    """One reliability-gated top-down FPN transition."""

    def __init__(
        self,
        channels: int,
        gate_reduction: int = 4,
        min_gate: float = 0.05,
        gate_init: float = 0.90,
    ):
        super().__init__()
        self.top_align = DepthwiseSeparableConv(channels, channels)
        self.gate = CrossScaleReliabilityGate(
            channels=channels,
            reduction=gate_reduction,
            min_gate=min_gate,
            gate_init=gate_init,
        )
        self.refine = DepthwiseSeparableConv(channels, channels)

    def forward(
        self,
        lateral: torch.Tensor,
        top_down: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_down = F.interpolate(
            top_down,
            size=lateral.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        top_down = self.top_align(top_down)
        gate = self.gate(lateral, top_down)
        fused = self.refine(lateral + gate * top_down)
        return fused, gate


class PvtV2B4_RG_FPN(nn.Module):
    """PVTv2-B4 encoder with Reliability-Gated FPN decoder.

    Input dictionary:
        image_m:       float tensor [B, 3, H, W]
        mask:          fallback hard target [B, 1, H, W]
        soft_target:   optional set-valued SAM target [B, 1, H, W]
        target_weight: optional reliability weight [B, 1, H, W]

    Evaluation returns a single full-resolution logit tensor.  Training uses
    the same weighted soft BCE contract as the curriculum implementation.
    """

    def __init__(
        self,
        pretrained: bool = True,
        input_norm: bool = True,
        fpn_dim: int = 64,
        gate_reduction: int = 4,
        min_gate: float = 0.05,
        gate_init: float = 0.90,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.encoder = pvt_v2_eff_b4(
            pretrained=pretrained,
            use_checkpoint=use_checkpoint,
        )
        self.embed_dims = list(self.encoder.embed_dims)  # [64, 128, 320, 512]
        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()

        self.laterals = nn.ModuleList(
            ConvGNAct(in_channels, fpn_dim, kernel_size=1)
            for in_channels in self.embed_dims
        )
        self.p5_refine = DepthwiseSeparableConv(fpn_dim, fpn_dim)
        self.fuse_4 = ReliabilityGatedFusion(
            fpn_dim, gate_reduction, min_gate, gate_init
        )
        self.fuse_3 = ReliabilityGatedFusion(
            fpn_dim, gate_reduction, min_gate, gate_init
        )
        self.fuse_2 = ReliabilityGatedFusion(
            fpn_dim, gate_reduction, min_gate, gate_init
        )

        self.predictor = nn.Sequential(
            DepthwiseSeparableConv(fpn_dim, fpn_dim),
            nn.Conv2d(fpn_dim, 1, kernel_size=1),
        )

    def normalize_encoder(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(self.normalizer(image))
        return (
            features["reduction_2"],  # 1/4
            features["reduction_3"],  # 1/8
            features["reduction_4"],  # 1/16
            features["reduction_5"],  # 1/32
        )

    def body(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = data["image_m"]
        encoder_features = self.normalize_encoder(image)
        l2, l3, l4, l5 = (
            lateral(feature)
            for lateral, feature in zip(self.laterals, encoder_features)
        )

        p5 = self.p5_refine(l5)
        p4, g4 = self.fuse_4(l4, p5)
        p3, g3 = self.fuse_3(l3, p4)
        p2, g2 = self.fuse_2(l2, p3)

        logits = self.predictor(p2)
        logits = F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits, {"g2": g2, "g3": g3, "g4": g4}

    @staticmethod
    def _resize_supervision(
        target: torch.Tensor,
        size: tuple[int, int],
        mode: str,
    ) -> torch.Tensor:
        if target.shape[-2:] == size:
            return target
        if mode == "nearest":
            return F.interpolate(target, size=size, mode=mode)
        return F.interpolate(target, size=size, mode=mode, align_corners=False)

    def forward(self, data, iter_percentage=1, **kwargs):
        # Keep the trainer-compatible signature. Gates do not use schedule or
        # pseudo-label metadata.
        del iter_percentage, kwargs
        logits, gates = self.body(data=data)

        if not self.training:
            return logits

        target = data.get("soft_target", data["mask"]).to(dtype=logits.dtype)
        target = self._resize_supervision(
            target,
            size=logits.shape[-2:],
            mode="bilinear",
        ).clamp_(0.0, 1.0)

        target_weight = data.get("target_weight")
        if target_weight is None:
            target_weight = torch.ones_like(target)
        else:
            target_weight = target_weight.to(dtype=logits.dtype)
            target_weight = self._resize_supervision(
                target_weight,
                size=logits.shape[-2:],
                mode="bilinear",
            ).clamp_(0.0, 1.0)

        loss_map = F.binary_cross_entropy_with_logits(
            input=logits,
            target=target,
            reduction="none",
        )
        bce_loss = (loss_map * target_weight).sum() / target_weight.sum().clamp_min(1.0)

        gate_means = {
            name: gate.detach().mean() for name, gate in gates.items()
        }
        loss_items = {
            "bce": bce_loss.detach(),
            "total": bce_loss.detach(),
            "reliable_ratio": (target_weight > 0).float().mean().detach(),
            **{f"{name}_mean": value for name, value in gate_means.items()},
        }
        gate_text = " ".join(
            f"{name.upper()}:{value.item():.3f}"
            for name, value in gate_means.items()
        )

        return {
            "vis": {"sal": logits.sigmoid()},
            "loss": bce_loss,
            "loss_items": loss_items,
            "loss_str": f"L:{bce_loss.detach().item():.4f} {gate_text}",
        }

    def get_grouped_params(self):
        """Match the optimizer grouping contract used by this repository."""
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}

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


class PvtV2B4_FPN_A6_Contrast(PvtV2B4_FPN_Baseline):

   
    def __init__(
        self,
        pretrained: bool = True,
        input_norm: bool = True,
        fpn_dim: int = 64,
        use_checkpoint: bool = False,
        contrast_dim: int = 128,
        contrast_temperature: float = 0.2,
        contrast_weight: float = 0.05,
        contrast_start: float = 1.0 / 3.0,
        contrast_warmup: float = 0.05,
        mask_threshold: float = 0.5,
        min_class_pixels: int = 4,
        detach_prototypes: bool = True,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

        if contrast_dim <= 0:
            raise ValueError(f"contrast_dim must be positive, got {contrast_dim}")
        if contrast_temperature <= 0:
            raise ValueError(
                "contrast_temperature must be positive, "
                f"got {contrast_temperature}"
            )
        if contrast_weight < 0:
            raise ValueError(f"contrast_weight must be non-negative, got {contrast_weight}")
        if not 0.0 <= contrast_start <= 1.0:
            raise ValueError(f"contrast_start must be in [0, 1], got {contrast_start}")
        if contrast_warmup < 0:
            raise ValueError(f"contrast_warmup must be non-negative, got {contrast_warmup}")
        if not 0.0 < mask_threshold < 1.0:
            raise ValueError(f"mask_threshold must be in (0, 1), got {mask_threshold}")
        if min_class_pixels < 1:
            raise ValueError(
                f"min_class_pixels must be at least 1, got {min_class_pixels}"
            )

        self.contrast_temperature = float(contrast_temperature)
        self.contrast_weight = float(contrast_weight)
        self.contrast_start = float(contrast_start)
        self.contrast_warmup = float(contrast_warmup)
        self.mask_threshold = float(mask_threshold)
        self.min_class_pixels = int(min_class_pixels)
        self.detach_prototypes = bool(detach_prototypes)

        # C4 in PVTv2-B4 has 320 channels. This head belongs to the retrained
        # optimizer group through the inherited get_grouped_params() method.
        self.contrast_proj = nn.Sequential(
            nn.Conv2d(
                self.embed_dims[2],
                contrast_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(contrast_dim),
            nn.ReLU(inplace=True),
        )

    def _decode(self, c2, c3, c4, c5, output_size):
        """Run the unchanged top-down FPN segmentation path."""
        p5 = self.smooth_5(self.lateral_5(c5))
        p4 = self.smooth_4(self.lateral_4(c4) + self._resize_like(p5, c4))
        p3 = self.smooth_3(self.lateral_3(c3) + self._resize_like(p4, c3))
        p2 = self.smooth_2(self.lateral_2(c2) + self._resize_like(p3, c2))

        logits = self.predictor(p2)
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

    def body(self, data, return_contrast_feature: bool = False):
        image = data["image_m"]
        c2, c3, c4, c5 = self.normalize_encoder(image)
        logits = self._decode(
            c2=c2,
            c3=c3,
            c4=c4,
            c5=c5,
            output_size=image.shape[-2:],
        )

        if not return_contrast_feature:
            return logits

        contrast_feature = F.normalize(self.contrast_proj(c4), dim=1)
        return logits, contrast_feature

    def _contrast_scale(self, iter_percentage) -> float:
        """Delay contrast until the clean warm-up is established."""
        progress = float(iter_percentage)
        progress = max(0.0, min(1.0, progress))
        if progress < self.contrast_start or self.contrast_weight == 0:
            return 0.0
        if self.contrast_warmup == 0:
            return self.contrast_weight

        ramp = (progress - self.contrast_start) / self.contrast_warmup
        return self.contrast_weight * max(0.0, min(1.0, ramp))

    def _ordinary_prototype_contrast(
        self,
        feature: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Contrast every foreground/background C4 pixel against two prototypes.

        The per-pixel losses are averaged separately for foreground and
        background and then combined 1:1. Thus A6 really uses all background
        pixels without allowing the usually larger background region to
        dominate the objective.
        """
        mask_small = F.interpolate(
            mask.float(),
            size=feature.shape[-2:],
            mode="nearest",
        )
        labels = (mask_small >= self.mask_threshold).long()

        image_losses = []
        for batch_idx in range(feature.shape[0]):
            feat = feature[batch_idx].flatten(1).transpose(0, 1)  # [HW, C]
            target = labels[batch_idx, 0].flatten()  # [HW]

            fg_index = target == 1
            bg_index = target == 0
            if (
                int(fg_index.sum().item()) < self.min_class_pixels
                or int(bg_index.sum().item()) < self.min_class_pixels
            ):
                continue

            fg_proto = F.normalize(feat[fg_index].mean(dim=0), dim=0)
            bg_proto = F.normalize(feat[bg_index].mean(dim=0), dim=0)
            if self.detach_prototypes:
                fg_proto = fg_proto.detach()
                bg_proto = bg_proto.detach()

            # Column 0 is background and column 1 is foreground, matching target.
            cosine_logits = torch.stack(
                [feat @ bg_proto, feat @ fg_proto],
                dim=1,
            )
            cosine_logits = cosine_logits / self.contrast_temperature
            pixel_loss = F.cross_entropy(
                cosine_logits,
                target,
                reduction="none",
            )

            loss_fg = pixel_loss[fg_index].mean()
            loss_bg = pixel_loss[bg_index].mean()
            image_losses.append(0.5 * (loss_fg + loss_bg))

        if not image_losses:
            # Keep the graph connected for rare empty/tiny pseudo masks.
            return feature.sum() * 0.0
        return torch.stack(image_losses).mean()

    def forward(self, data, iter_percentage=1, **kwargs):
        del kwargs

        if not self.training:
            return self.body(data=data, return_contrast_feature=False)

        logits, contrast_feature = self.body(
            data=data,
            return_contrast_feature=True,
        )
        mask = data["mask"].float()
        if mask.shape[-2:] != logits.shape[-2:]:
            mask = F.interpolate(mask, size=logits.shape[-2:], mode="nearest")

        bce_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=mask,
            reduction="mean",
        )
        contrast_loss = self._ordinary_prototype_contrast(
            feature=contrast_feature,
            mask=mask,
        )
        contrast_scale = self._contrast_scale(iter_percentage)
        weighted_contrast = contrast_loss * contrast_scale
        total_loss = bce_loss + weighted_contrast
        prob = logits.sigmoid()

        return {
            "vis": {"sal": prob},
            "loss": total_loss,
            "loss_items": {
                "bce": bce_loss.detach(),
                "contrast": contrast_loss.detach(),
                "contrast_weighted": weighted_contrast.detach(),
                "contrast_scale": contrast_loss.detach().new_tensor(contrast_scale),
                "total": total_loss.detach(),
            },
            "loss_str": (
                f"L:{total_loss.detach().item():.4f} "
                f"BCE:{bce_loss.detach().item():.4f} "
                f"CON:{contrast_loss.detach().item():.4f} "
                f"WCON:{weighted_contrast.detach().item():.4f}"
            ),
        }




# -*- coding: utf-8 -*-
"""A7 ablation: pseudo-prediction-consistent hard-background contrast.

A7 inherits the A6 segmentation path and projection head. The only conceptual
change is negative selection:

* A6 uses every pseudo-background pixel;
* A7 first keeps pixels where the pseudo mask and the detached current
  prediction both agree on background, then selects the most foreground-like
  negatives by prediction confidence and C4 feature similarity.

No EMA teacher, extra data field, second image forward, or inference-time branch
is required, which keeps this ablation compatible with the current repository.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F




class PvtV2B4_FPN_A7_ConsistentHardContrast(PvtV2B4_FPN_A6_Contrast):
    """A6 plus consistent hard-negative mining on projected C4 features.

    Consistent background means that both the pseudo label and the detached
    current prediction classify the location as background. Hardness is the
    weighted sum of:

    1. foreground probability (closer to 0.5 is harder among predicted BG);
    2. cosine similarity to the foreground prototype.

    Only the highest-scoring ``hard_bg_topk`` negatives per image are used.
    """

    def __init__(
        self,
        pretrained: bool = True,
        input_norm: bool = True,
        fpn_dim: int = 64,
        use_checkpoint: bool = False,
        hard_bg_topk: int = 48,
        prediction_bg_threshold: float = 0.5,
        prediction_hardness_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

        if hard_bg_topk < 1:
            raise ValueError(f"hard_bg_topk must be at least 1, got {hard_bg_topk}")
        if not 0.0 < prediction_bg_threshold < 1.0:
            raise ValueError(
                "prediction_bg_threshold must be in (0, 1), "
                f"got {prediction_bg_threshold}"
            )
        if not 0.0 <= prediction_hardness_weight <= 1.0:
            raise ValueError(
                "prediction_hardness_weight must be in [0, 1], "
                f"got {prediction_hardness_weight}"
            )

        self.hard_bg_topk = int(hard_bg_topk)
        self.prediction_bg_threshold = float(prediction_bg_threshold)
        self.prediction_hardness_weight = float(prediction_hardness_weight)

    def _prepare_small_maps(
        self,
        mask: torch.Tensor,
        prob: torch.Tensor,
        feature_size,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_small = F.interpolate(
            mask.float(),
            size=feature_size,
            mode="nearest",
        )
        prob_small = F.interpolate(
            prob.detach().float(),
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        return mask_small, prob_small

    def _semantic_hardness(
        self,
        feature_detached: torch.Tensor,
        foreground_mask: torch.Tensor,
        prob_small: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Return a detached [B,1,H,W] semantic hardness score.

        ``image`` is accepted so A8 can add a frequency term without changing
        the A7 selection/loss interface.
        """
        del image

        batch_size, _, height, width = feature_detached.shape
        score = feature_detached.new_zeros((batch_size, 1, height, width))
        alpha = self.prediction_hardness_weight

        for batch_idx in range(batch_size):
            feat = feature_detached[batch_idx].flatten(1).transpose(0, 1)
            fg_index = foreground_mask[batch_idx, 0].flatten()
            if int(fg_index.sum().item()) < self.min_class_pixels:
                continue

            fg_proto = F.normalize(feat[fg_index].mean(dim=0), dim=0)
            feature_similarity = feat @ fg_proto
            feature_similarity = (feature_similarity + 1.0) * 0.5

            prediction_score = prob_small[batch_idx, 0].flatten()
            combined = (
                alpha * prediction_score
                + (1.0 - alpha) * feature_similarity
            )
            score[batch_idx, 0] = combined.view(height, width)

        return score

    def _select_hard_background(
        self,
        feature: torch.Tensor,
        mask: torch.Tensor,
        prob: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Select top-K consistent hard negatives without gradient flow."""
        mask_small, prob_small = self._prepare_small_maps(
            mask=mask,
            prob=prob,
            feature_size=feature.shape[-2:],
        )
        foreground_mask = mask_small >= self.mask_threshold

        # "Consistent" in the current repository means agreement between the
        # pseudo mask and the detached model prediction. This avoids requiring
        # a teacher model or a second data view that the trainer does not expose.
        pseudo_background = mask_small < self.mask_threshold
        predicted_background = prob_small < self.prediction_bg_threshold
        candidate_background = pseudo_background & predicted_background

        with torch.no_grad():
            hardness = self._semantic_hardness(
                feature_detached=feature.detach(),
                foreground_mask=foreground_mask,
                prob_small=prob_small,
                image=image,
            )

            selected_background = torch.zeros_like(
                candidate_background,
                dtype=torch.bool,
            )
            selected_counts = []
            candidate_counts = []
            selected_scores = []

            for batch_idx in range(feature.shape[0]):
                candidate_flat = candidate_background[batch_idx, 0].flatten()
                candidate_index = torch.nonzero(
                    candidate_flat,
                    as_tuple=False,
                ).squeeze(1)
                candidate_counts.append(float(candidate_index.numel()))

                if candidate_index.numel() < self.min_class_pixels:
                    selected_counts.append(0.0)
                    selected_scores.append(0.0)
                    continue

                num_select = min(self.hard_bg_topk, candidate_index.numel())
                candidate_score = hardness[batch_idx, 0].flatten()[candidate_index]
                local_topk = torch.topk(
                    candidate_score,
                    k=num_select,
                    largest=True,
                    sorted=False,
                ).indices
                final_index = candidate_index[local_topk]
                selected_background[batch_idx, 0].view(-1)[final_index] = True

                selected_counts.append(float(num_select))
                selected_scores.append(float(candidate_score[local_topk].mean().item()))

            device = feature.device
            dtype = feature.dtype
            stats = {
                "selected_bg": torch.tensor(
                    selected_counts,
                    device=device,
                    dtype=dtype,
                ).mean(),
                "candidate_bg": torch.tensor(
                    candidate_counts,
                    device=device,
                    dtype=dtype,
                ).mean(),
                "selected_hardness": torch.tensor(
                    selected_scores,
                    device=device,
                    dtype=dtype,
                ).mean(),
            }

        return foreground_mask, selected_background, stats

    def _hard_background_prototype_contrast(
        self,
        feature: torch.Tensor,
        mask: torch.Tensor,
        prob: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Contrast all pseudo foreground against selected hard background."""
        foreground_mask, background_mask, stats = self._select_hard_background(
            feature=feature,
            mask=mask,
            prob=prob,
            image=image,
        )

        image_losses = []
        valid_images = 0
        for batch_idx in range(feature.shape[0]):
            feat = feature[batch_idx].flatten(1).transpose(0, 1)
            fg_index = foreground_mask[batch_idx, 0].flatten()
            bg_index = background_mask[batch_idx, 0].flatten()

            if (
                int(fg_index.sum().item()) < self.min_class_pixels
                or int(bg_index.sum().item()) < self.min_class_pixels
            ):
                continue

            fg_proto = F.normalize(feat[fg_index].mean(dim=0), dim=0)
            bg_proto = F.normalize(feat[bg_index].mean(dim=0), dim=0)
            if self.detach_prototypes:
                fg_proto = fg_proto.detach()
                bg_proto = bg_proto.detach()

            fg_feat = feat[fg_index]
            bg_feat = feat[bg_index]

            fg_logits = torch.stack(
                [fg_feat @ bg_proto, fg_feat @ fg_proto],
                dim=1,
            ) / self.contrast_temperature
            bg_logits = torch.stack(
                [bg_feat @ bg_proto, bg_feat @ fg_proto],
                dim=1,
            ) / self.contrast_temperature

            fg_target = torch.ones(
                fg_logits.shape[0],
                device=fg_logits.device,
                dtype=torch.long,
            )
            bg_target = torch.zeros(
                bg_logits.shape[0],
                device=bg_logits.device,
                dtype=torch.long,
            )

            loss_fg = F.cross_entropy(fg_logits, fg_target)
            loss_bg = F.cross_entropy(bg_logits, bg_target)
            image_losses.append(0.5 * (loss_fg + loss_bg))
            valid_images += 1

        stats["valid_contrast_images"] = feature.detach().new_tensor(
            float(valid_images)
        )
        if not image_losses:
            return feature.sum() * 0.0, stats
        return torch.stack(image_losses).mean(), stats

    def forward(self, data, iter_percentage=1, **kwargs):
        del kwargs

        if not self.training:
            return self.body(data=data, return_contrast_feature=False)

        logits, contrast_feature = self.body(
            data=data,
            return_contrast_feature=True,
        )
        mask = data["mask"].float()
        if mask.shape[-2:] != logits.shape[-2:]:
            mask = F.interpolate(mask, size=logits.shape[-2:], mode="nearest")

        bce_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=mask,
            reduction="mean",
        )
        prob = logits.sigmoid()
        contrast_loss, contrast_stats = self._hard_background_prototype_contrast(
            feature=contrast_feature,
            mask=mask,
            prob=prob,
            image=data["image_m"],
        )
        contrast_scale = self._contrast_scale(iter_percentage)
        weighted_contrast = contrast_loss * contrast_scale
        total_loss = bce_loss + weighted_contrast

        loss_items = {
            "bce": bce_loss.detach(),
            "contrast": contrast_loss.detach(),
            "contrast_weighted": weighted_contrast.detach(),
            "contrast_scale": contrast_loss.detach().new_tensor(contrast_scale),
            "total": total_loss.detach(),
        }
        loss_items.update({key: value.detach() for key, value in contrast_stats.items()})

        return {
            "vis": {"sal": prob},
            "loss": total_loss,
            "loss_items": loss_items,
            "loss_str": (
                f"L:{total_loss.detach().item():.4f} "
                f"BCE:{bce_loss.detach().item():.4f} "
                f"CON:{contrast_loss.detach().item():.4f} "
                f"WCON:{weighted_contrast.detach().item():.4f} "
                f"HBG:{contrast_stats['selected_bg'].detach().item():.1f}"
            ),
        }
# -*- coding: utf-8 -*-



def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError(f"kernel_size must be odd and >= 3, got {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    radius = kernel_size // 2
    coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, kernel_size, kernel_size)


class PvtV2B4_FPN_A8_FrequencyHardContrast(
    PvtV2B4_FPN_A7_ConsistentHardContrast
):
    """A7 plus a fixed multi-band frequency similarity term.

    The frequency descriptor is formed from three absolute residual bands:

    * fine:   |I - G1(I)|
    * medium: |G1(I) - G2(I)|
    * coarse: |G2(I) - G3(I)|

    Background pixels whose frequency descriptor resembles the pseudo-foreground
    prototype receive a larger difficulty score and are more likely to enter the
    same top-K hard-negative set used by A7.
    """

    def __init__(
        self,
        pretrained: bool = True,
        input_norm: bool = True,
        fpn_dim: int = 64,
        use_checkpoint: bool = False,
        frequency_hardness_weight: float = 0.3,
        frequency_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

        if not 0.0 <= frequency_hardness_weight <= 1.0:
            raise ValueError(
                "frequency_hardness_weight must be in [0, 1], "
                f"got {frequency_hardness_weight}"
            )
        if frequency_eps <= 0:
            raise ValueError(f"frequency_eps must be positive, got {frequency_eps}")

        self.frequency_hardness_weight = float(frequency_hardness_weight)
        self.frequency_eps = float(frequency_eps)

        # Kernels operate on the already downsampled C4-resolution grayscale
        # image, so the extra cost is small even for batch size 16.
        self.register_buffer(
            "freq_kernel_1",
            _gaussian_kernel(kernel_size=3, sigma=0.8),
            persistent=False,
        )
        self.register_buffer(
            "freq_kernel_2",
            _gaussian_kernel(kernel_size=5, sigma=1.2),
            persistent=False,
        )
        self.register_buffer(
            "freq_kernel_3",
            _gaussian_kernel(kernel_size=9, sigma=2.0),
            persistent=False,
        )

    @staticmethod
    def _blur(gray: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        padding = kernel.shape[-1] // 2
        return F.conv2d(gray, kernel, padding=padding)

    def _frequency_descriptor(
        self,
        image: torch.Tensor,
        output_size,
    ) -> torch.Tensor:
        """Build a detached, normalized [B,3,Hc4,Wc4] DoG descriptor."""
        with torch.no_grad():
            image_float = image.detach().float()
            gray = (
                0.2989 * image_float[:, 0:1]
                + 0.5870 * image_float[:, 1:2]
                + 0.1140 * image_float[:, 2:3]
            )
            gray = F.interpolate(
                gray,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

            g1 = self._blur(gray, self.freq_kernel_1.float())
            g2 = self._blur(gray, self.freq_kernel_2.float())
            g3 = self._blur(gray, self.freq_kernel_3.float())

            descriptor = torch.cat(
                [
                    (gray - g1).abs(),
                    (g1 - g2).abs(),
                    (g2 - g3).abs(),
                ],
                dim=1,
            )

            # Per-image, per-band standardization prevents one band from
            # dominating merely because of scale.
            mean = descriptor.mean(dim=(-2, -1), keepdim=True)
            std = descriptor.std(dim=(-2, -1), keepdim=True, unbiased=False)
            descriptor = (descriptor - mean) / std.clamp_min(self.frequency_eps)
            descriptor = F.normalize(
                descriptor,
                dim=1,
                eps=self.frequency_eps,
            )

        return descriptor

    def _semantic_hardness(
        self,
        feature_detached: torch.Tensor,
        foreground_mask: torch.Tensor,
        prob_small: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        semantic_score = super()._semantic_hardness(
            feature_detached=feature_detached,
            foreground_mask=foreground_mask,
            prob_small=prob_small,
            image=image,
        )
        frequency_descriptor = self._frequency_descriptor(
            image=image,
            output_size=feature_detached.shape[-2:],
        )

        batch_size, _, height, width = frequency_descriptor.shape
        frequency_score = frequency_descriptor.new_zeros(
            (batch_size, 1, height, width)
        )

        with torch.no_grad():
            for batch_idx in range(batch_size):
                descriptor = frequency_descriptor[batch_idx].flatten(1).transpose(0, 1)
                fg_index = foreground_mask[batch_idx, 0].flatten()
                if int(fg_index.sum().item()) < self.min_class_pixels:
                    continue

                fg_frequency_proto = F.normalize(
                    descriptor[fg_index].mean(dim=0),
                    dim=0,
                    eps=self.frequency_eps,
                )
                similarity = descriptor @ fg_frequency_proto
                similarity = (similarity + 1.0) * 0.5
                frequency_score[batch_idx, 0] = similarity.view(height, width)

            beta = self.frequency_hardness_weight
            final_score = (
                (1.0 - beta) * semantic_score
                + beta * frequency_score.to(
                    device=semantic_score.device,
                    dtype=semantic_score.dtype,
                )
            )

        return final_score


class ConvNeXtB384_FPN_Baseline(nn.Module):
    """
    ConvNeXt-Base (ImageNet-22K -> ImageNet-1K, 384)
    + plain top-down FPN.

    Backbone:
        convnext_base.fb_in22k_ft_in1k_384

    Feature channels:
        C2 = 128
        C3 = 256
        C4 = 512
        C5 = 1024
    """

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        # ---------------------------------------------------------
        # 1. ConvNeXt-Base backbone
        # ---------------------------------------------------------
        self.encoder = timm.create_model(
            model_name="convnext_base.fb_in22k_ft_in1k_384",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # ConvNeXt-B:
        # [128, 256, 512, 1024]
        self.embed_dims = list(
            self.encoder.feature_info.channels()
        )

        LOGGER.info(
            f"ConvNeXt-B feature channels: {self.embed_dims}"
        )

        # optional gradient checkpoint
        if use_checkpoint:
            if hasattr(
                self.encoder,
                "set_grad_checkpointing",
            ):
                self.encoder.set_grad_checkpointing(
                    enable=True
                )
                LOGGER.info(
                    "ConvNeXt gradient checkpointing enabled."
                )
            else:
                LOGGER.warning(
                    "Current timm ConvNeXt does not expose "
                    "set_grad_checkpointing()."
                )

        # ---------------------------------------------------------
        # 2. ImageNet normalization
        # ---------------------------------------------------------
        self.normalizer = (
            PixelNormalizer()
            if input_norm
            else nn.Identity()
        )

        # ---------------------------------------------------------
        # 3. FPN lateral projections
        # ---------------------------------------------------------
        self.lateral_2 = nn.Conv2d(
            self.embed_dims[0],
            fpn_dim,
            kernel_size=1,
            bias=False,
        )

        self.lateral_3 = nn.Conv2d(
            self.embed_dims[1],
            fpn_dim,
            kernel_size=1,
            bias=False,
        )

        self.lateral_4 = nn.Conv2d(
            self.embed_dims[2],
            fpn_dim,
            kernel_size=1,
            bias=False,
        )

        self.lateral_5 = nn.Conv2d(
            self.embed_dims[3],
            fpn_dim,
            kernel_size=1,
            bias=False,
        )

        # ---------------------------------------------------------
        # 4. FPN smoothing
        # ---------------------------------------------------------
        self.smooth_5 = ConvBNReLU(
            fpn_dim,
            fpn_dim,
            3,
            1,
            1,
        )

        self.smooth_4 = ConvBNReLU(
            fpn_dim,
            fpn_dim,
            3,
            1,
            1,
        )

        self.smooth_3 = ConvBNReLU(
            fpn_dim,
            fpn_dim,
            3,
            1,
            1,
        )

        self.smooth_2 = ConvBNReLU(
            fpn_dim,
            fpn_dim,
            3,
            1,
            1,
        )

        # ---------------------------------------------------------
        # 5. Segmentation head
        # ---------------------------------------------------------
        self.predictor = nn.Sequential(
            ConvBNReLU(
                fpn_dim,
                32,
                3,
                1,
                1,
            ),
            nn.Conv2d(
                32,
                1,
                kernel_size=1,
            ),
        )

    # =============================================================
    # Encoder
    # =============================================================
    def normalize_encoder(self, image):

        image = self.normalizer(image)

        features = self.encoder(image)

        # timm features_only:
        #
        # features[0]: 1/4  [128]
        # features[1]: 1/8  [256]
        # features[2]: 1/16 [512]
        # features[3]: 1/32 [1024]

        c2, c3, c4, c5 = features

        return c2, c3, c4, c5

    @staticmethod
    def _resize_like(source, target):
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    # =============================================================
    # FPN
    # =============================================================
    def body(self, data):

        # 保持与你现在 PVT-FPN 一样：
        # 只使用 image_m
        image = data["image_m"]

        c2, c3, c4, c5 = self.normalize_encoder(
            image
        )

        # C5 -> P5
        p5 = self.smooth_5(
            self.lateral_5(c5)
        )

        # C4 + P5
        p4 = self.smooth_4(
            self.lateral_4(c4)
            + self._resize_like(
                p5,
                c4,
            )
        )

        # C3 + P4
        p3 = self.smooth_3(
            self.lateral_3(c3)
            + self._resize_like(
                p4,
                c3,
            )
        )

        # C2 + P3
        p2 = self.smooth_2(
            self.lateral_2(c2)
            + self._resize_like(
                p3,
                c2,
            )
        )

        logits = self.predictor(p2)

        logits = F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits

    # =============================================================
    # Forward
    # =============================================================
    def forward(
        self,
        data,
        iter_percentage=1,
        **kwargs,
    ):

        del iter_percentage, kwargs

        logits = self.body(
            data=data
        )

        if not self.training:
            return logits

        mask = data["mask"].float()

        if mask.shape[-2:] != logits.shape[-2:]:
            mask = F.interpolate(
                mask,
                size=logits.shape[-2:],
                mode="nearest",
            )

        bce_loss = (
            F.binary_cross_entropy_with_logits(
                logits,
                mask,
                reduction="mean",
            )
        )

        return {
            "logits": logits,

            "vis": {
                "sal": logits.sigmoid(),
            },

            "loss": bce_loss,

            "loss_items": {
                "bce": bce_loss.detach(),
                "total": bce_loss.detach(),
            },

            "loss_str": (
                f"L:{bce_loss.detach().item():.4f} "
                f"BCE:{bce_loss.detach().item():.4f}"
            ),
        }

    # =============================================================
    # Optimizer parameter groups
    # =============================================================
    def get_grouped_params(self):

        param_groups = {
            "pretrained": [],
            "fixed": [],
            "retrained": [],
        }

        for name, param in self.named_parameters():

            # -----------------------------------------------------
            # 注意：
            # PVT 原来是 encoder.patch_embed1
            # ConvNeXt 要对应 encoder.stem
            # -----------------------------------------------------
            if name.startswith("encoder.stem."):

                param.requires_grad = False

                param_groups[
                    "fixed"
                ].append(param)

            elif name.startswith("encoder."):

                param_groups[
                    "pretrained"
                ].append(param)

            else:

                param_groups[
                    "retrained"
                ].append(param)

        LOGGER.info(
            "ConvNeXt-FPN Parameter Groups:{"
            f"Pretrained:"
            f"{len(param_groups['pretrained'])}, "
            f"Fixed:"
            f"{len(param_groups['fixed'])}, "
            f"ReTrained:"
            f"{len(param_groups['retrained'])}"
            "}"
        )

        return param_groups