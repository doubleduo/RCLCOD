# -*- coding: utf-8 -*-
"""
Noisy-COD ANet DWT ablation model.

A0: original Noisy-COD routing
    HH -> F1/F2, LL -> F3/F4

A1: four-band fixed mean routing
    mean(LL, LH, HL, HH) -> every stage (after original scale alignment)

A2: adaptive four-band spatial frequency router
    stage-specific softmax weights over LL/LH/HL/HH conditioned on the
    current semantic feature. The final gate is zero-initialized, therefore
    A2 starts exactly from uniform 0.25/0.25/0.25/0.25 routing (A1 behavior)
    and learns away from it.

Only the frequency-routing path changes across A0/A1/A2. Backbone, GCM3,
GPM, REU decoder, losses and output interface remain unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Backbone
# -----------------------------------------------------------------------------

def build_convnext_base_384(
    model_name: str = "convnext_base.fb_in22k_ft_in1k_384",
) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required. Install with: pip install -U timm huggingface_hub"
        ) from exc

    model = timm.create_model(
        model_name,
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    channels = list(model.feature_info.channels())
    expected = [128, 256, 512, 1024]
    if channels != expected:
        raise RuntimeError(
            f"Unexpected ConvNeXt-B feature channels: {channels}; expected {expected}"
        )
    return model


# -----------------------------------------------------------------------------
# Basic blocks
# -----------------------------------------------------------------------------

class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        need_relu=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class BasicDeConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        out_padding=0,
        need_relu=True,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=out_padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class ETM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channels, out_channels, 3, 1, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, (3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, (5, 1), padding=(2, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, (5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, (7, 1), padding=(3, 0)),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, (7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, (1, 9), padding=(0, 4)),
            BasicConv2d(out_channels, out_channels, (9, 1), padding=(4, 0)),
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        return self.relu(x_cat + self.conv_res(x))


class DWT(nn.Module):
    """Fixed Haar-like DWT from the released Noisy-COD ANet."""

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        target = (x.shape[2] // 2, x.shape[3] // 2)
        x1 = F.interpolate(x1, size=target, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=target, mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=target, mode="bilinear", align_corners=False)
        x4 = F.interpolate(x4, size=target, mode="bilinear", align_corners=False)

        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh


class GCM3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.T1 = ETM(in_channels[0], out_channels)
        self.T2 = ETM(in_channels[1], out_channels)
        self.T3 = ETM(in_channels[2], out_channels)
        self.T4 = ETM(in_channels[3], out_channels)
        self.decoder = nn.Conv2d(out_channels * 4, out_channels, 3, 1, 1)
        self.DWT = DWT()

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)
        f2 = self.T2(f2)
        f3 = self.T3(f3)
        f4 = self.T4(f4)

        camo = self.decoder(
            torch.cat(
                [
                    f1,
                    F.interpolate(f2, scale_factor=2, mode="bilinear", align_corners=True),
                    F.interpolate(f3, scale_factor=4, mode="bilinear", align_corners=True),
                    F.interpolate(f4, scale_factor=8, mode="bilinear", align_corners=True),
                ],
                dim=1,
            )
        )
        ll, lh, hl, hh = self.DWT(camo)
        return ll, lh, hl, hh, f1, f2, f3, f4


# -----------------------------------------------------------------------------
# A0 / A1 / A2 frequency routing
# -----------------------------------------------------------------------------

ABLATION_TO_FREQ_MODE = {
    "A0": "original",
    "A1": "four_band_mean",
    "A2": "adaptive_router",
    "original": "original",
    "four_band_mean": "four_band_mean",
    "adaptive_router": "adaptive_router",
}


class AdaptiveFrequencyRouter(nn.Module):
    """Stage-specific spatial softmax routing over LL/LH/HL/HH.

    The router is conditioned on the current semantic feature and all four
    frequency bands. The final 1x1 conv is zero initialized so that all logits
    are zero at initialization => uniform weights [0.25, 0.25, 0.25, 0.25].
    """

    def __init__(self, channels=64, hidden=32, temperature=1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("router temperature must be > 0")
        self.temperature = float(temperature)
        self.pre = nn.Sequential(
            nn.Conv2d(channels * 5, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.to_logits = nn.Conv2d(hidden, 4, kernel_size=1, bias=True)
        nn.init.zeros_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)

    def forward(self, context, ll, lh, hl, hh):
        # ll/lh/hl/hh received here are already independently projected.
        target_size = ll.shape[-2:]
        if context.shape[-2:] != target_size:
            context = F.interpolate(
                context,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        bands = (ll, lh, hl, hh)
        logits = self.to_logits(self.pre(torch.cat([context, *bands], dim=1)))
        weights = torch.softmax(logits / self.temperature, dim=1)
        fused = sum(weights[:, i : i + 1] * bands[i] for i in range(4))
        return fused, weights


class Branch(nn.Module):
    def __init__(
        self,
        backbone_name="convnext_base.fb_in22k_ft_in1k_384",
        channels=64,
        ablation="A0",
        router_hidden=32,
        router_temperature=1.0,
    ):
        super().__init__()
        if ablation not in ABLATION_TO_FREQ_MODE:
            raise ValueError(
                f"Unknown ablation={ablation!r}; choose from A0/A1/A2"
            )
        self.ablation = ablation
        self.freq_mode = ABLATION_TO_FREQ_MODE[ablation]

        self.shared_encoder = build_convnext_base_384(backbone_name)
        self.GCM3 = GCM3([128, 256, 512, 1024], channels)

        # Keep exactly the original deep-stage alignment modules.
        self.LL_down3 = nn.Sequential(
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1)
        )
        self.LL_down4 = nn.Sequential(
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1),
        )
        # Original implementation: nn.Upsample(scale_factor=2) -> nearest.
        self.dePixelShuffle = nn.Upsample(scale_factor=2)

        self.one_conv_f4_ll = ETM(channels * 2, channels)
        self.one_conv_f3_ll = ETM(channels * 2, channels)
        self.one_conv_f1_hh = ETM(channels * 2, channels)
        self.one_conv_f2_hh = ETM(channels * 2, channels)

        # A1/A2 use the exact same four per-band adapters. This makes A1->A2
        # a clean ablation: only the fixed 0.25 weights are replaced by a
        # context-dependent softmax router. ReLU also prevents signed Haar
        # bands from trivially cancelling under uniform averaging.
        if self.freq_mode != "original":
            self.band_proj_ll = BasicConv2d(channels, channels, 1)
            self.band_proj_lh = BasicConv2d(channels, channels, 1)
            self.band_proj_hl = BasicConv2d(channels, channels, 1)
            self.band_proj_hh = BasicConv2d(channels, channels, 1)

        if self.freq_mode == "adaptive_router":
            self.freq_router1 = AdaptiveFrequencyRouter(
                channels, hidden=router_hidden, temperature=router_temperature
            )
            self.freq_router2 = AdaptiveFrequencyRouter(
                channels, hidden=router_hidden, temperature=router_temperature
            )
            self.freq_router3 = AdaptiveFrequencyRouter(
                channels, hidden=router_hidden, temperature=router_temperature
            )
            self.freq_router4 = AdaptiveFrequencyRouter(
                channels, hidden=router_hidden, temperature=router_temperature
            )

    @staticmethod
    def _mean_four(ll, lh, hl, hh):
        return (ll + lh + hl + hh) * 0.25

    def _project_four_bands(self, ll, lh, hl, hh):
        return (
            self.band_proj_ll(ll),
            self.band_proj_lh(lh),
            self.band_proj_hl(hl),
            self.band_proj_hh(hh),
        )

    def _route_base_resolution(self, stage_idx, context, ll, lh, hl, hh):
        """Return one fused frequency map at the native DWT resolution (H/8)."""
        pll, plh, phl, phh = self._project_four_bands(ll, lh, hl, hh)
        if self.freq_mode == "four_band_mean":
            return self._mean_four(pll, plh, phl, phh), None

        router = getattr(self, f"freq_router{stage_idx}")
        return router(context, pll, plh, phl, phh)

    def forward(self, x, return_freq_weights=False):
        x1, x2, x3, x4 = self.shared_encoder(x)
        ll, lh, hl, hh, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        weights: Dict[str, torch.Tensor] = {}

        if self.freq_mode == "original":
            # A0: exact original routing.
            f1_freq = self.dePixelShuffle(hh)
            f2_freq = hh
            f3_freq = self.LL_down3(ll)
            f4_freq = self.LL_down4(ll)
        else:
            # A1/A2: route all four bands at the common DWT resolution first,
            # then reuse the exact original stage-scale alignment operators.
            r1, w1 = self._route_base_resolution(1, f1, ll, lh, hl, hh)
            r2, w2 = self._route_base_resolution(2, f2, ll, lh, hl, hh)
            r3, w3 = self._route_base_resolution(3, f3, ll, lh, hl, hh)
            r4, w4 = self._route_base_resolution(4, f4, ll, lh, hl, hh)

            f1_freq = self.dePixelShuffle(r1)
            f2_freq = r2
            f3_freq = self.LL_down3(r3)
            f4_freq = self.LL_down4(r4)

            if return_freq_weights and self.freq_mode == "adaptive_router":
                weights = {"s1": w1, "s2": w2, "s3": w3, "s4": w4}

        out1 = self.one_conv_f1_hh(torch.cat([f1_freq, f1], dim=1))
        out2 = self.one_conv_f2_hh(torch.cat([f2_freq, f2], dim=1))
        out3 = self.one_conv_f3_ll(torch.cat([f3_freq, f3], dim=1))
        out4 = self.one_conv_f4_ll(torch.cat([f4_freq, f4], dim=1))

        if return_freq_weights:
            return out1, out2, out3, out4, x4, weights
        return out1, out2, out3, out4, x4


# -----------------------------------------------------------------------------
# GPM + REU decoder (unchanged across ablations)
# -----------------------------------------------------------------------------

class GPM(nn.Module):
    def __init__(
        self,
        in_c=128,
        dilation_series=(6, 12, 18),
        padding_series=(6, 12, 18),
        depth=32,
    ):
        super().__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(in_c, depth, 1, 1),
        )
        self.branch0 = BasicConv2d(in_c, depth, 1, 1)
        self.branch1 = BasicConv2d(
            in_c,
            depth,
            3,
            1,
            padding=padding_series[0],
            dilation=dilation_series[0],
        )
        self.branch2 = BasicConv2d(
            in_c,
            depth,
            3,
            1,
            padding=padding_series[1],
            dilation=dilation_series[1],
        )
        self.branch3 = BasicConv2d(
            in_c,
            depth,
            3,
            1,
            padding=padding_series[2],
            dilation=dilation_series[2],
        )
        self.head = nn.Sequential(BasicConv2d(depth * 5, depth, 1))
        self.out = nn.Sequential(
            nn.Conv2d(depth, depth, 3, padding=1),
            nn.BatchNorm2d(depth),
            nn.PReLU(),
            nn.Conv2d(depth, 1, 3, 1, 1),
        )

        # Released GPM initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = F.interpolate(
            self.branch_main(x), size=size, mode="bilinear", align_corners=True
        )
        out = torch.cat(
            [
                branch_main,
                self.branch0(x),
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
            ],
            dim=1,
        )
        return self.out(self.head(out))


class REUBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, 3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, 3, padding=1),
        )
        self.out_B = nn.Sequential(
            BasicDeConv2d(
                in_channels,
                mid_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                out_padding=1,
            ),
            BasicConv2d(mid_channels // 2, mid_channels // 4, 3, padding=1),
            nn.Conv2d(mid_channels // 4, 1, 3, padding=1),
        )
        self.ode = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 3, padding=1),
            BasicConv2d(in_channels, in_channels, 3, padding=1),
        )

    @staticmethod
    def edge_enhance(img):
        gradient = img.clone()
        gradient[:, :, :-1, :] = torch.abs(
            gradient[:, :, :-1, :] - gradient[:, :, 1:, :]
        )
        gradient[:, :, :, :-1] = torch.abs(
            gradient[:, :, :, :-1] - gradient[:, :, :, 1:]
        )
        return torch.clamp(img - gradient, 0, 1)

    def forward(self, x, prior_cam):
        prior_cam = F.interpolate(
            prior_cam, size=x.size()[2:], mode="bilinear", align_corners=True
        )
        yt = self.conv(
            torch.cat([x, prior_cam.expand(-1, x.size(1), -1, -1)], dim=1)
        )
        ode_out = self.ode(yt)
        bound = self.edge_enhance(self.out_B(ode_out))

        reverse_prior = 1 - torch.sigmoid(prior_cam)
        y = reverse_prior.expand(-1, x.size(1), -1, -1) * x
        y = self.out_y(torch.cat([y, ode_out], dim=1))
        y = y + prior_cam
        return y, bound


class UNetDecoderWithEdges(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.REU_f1 = REUBlock(in_channels, mid_channels)
        self.REU_f2 = REUBlock(in_channels, mid_channels)
        self.REU_f3 = REUBlock(in_channels, mid_channels)
        self.REU_f4 = REUBlock(in_channels, mid_channels)

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)

    def forward(self, feats, prior_0, pic):
        f1, f2, f3, f4 = feats
        full_size = pic.size()[2:]

        f4_out, bound_f4 = self.REU_f4(f4, prior_0)
        f4_full = self._up(f4_out, full_size)
        bound_f4 = self._up(bound_f4, full_size)

        f3_out, bound_f3 = self.REU_f3(f3, f4_out)
        f3_full = self._up(f3_out, full_size)
        bound_f3 = self._up(bound_f3, full_size)

        f2_out, bound_f2 = self.REU_f2(f2, f3_out)
        f2_full = self._up(f2_out, full_size)
        bound_f2 = self._up(bound_f2, full_size)

        f1_out, bound_f1 = self.REU_f1(f1, f2_out)
        f1_full = self._up(f1_out, full_size)
        bound_f1 = self._up(bound_f1, full_size)

        return (
            f4_full,
            f3_full,
            f2_full,
            f1_full,
            bound_f4,
            bound_f3,
            bound_f2,
            bound_f1,
        )


class NoisyCODANet(nn.Module):
    def __init__(
        self,
        backbone_name="convnext_base.fb_in22k_ft_in1k_384",
        channels=64,
        ablation="A0",
        router_hidden=32,
        router_temperature=1.0,
    ):
        super().__init__()
        self.ablation = ablation
        self.net1 = Branch(
            backbone_name=backbone_name,
            channels=channels,
            ablation=ablation,
            router_hidden=router_hidden,
            router_temperature=router_temperature,
        )
        self.net2 = Branch(
            backbone_name=backbone_name,
            channels=channels,
            ablation=ablation,
            router_hidden=router_hidden,
            router_temperature=router_temperature,
        )
        self.GPM = GPM(2048)
        self.decoder = UNetDecoderWithEdges(channels * 2, channels * 2)

    def forward(self, x, x_box, return_freq_weights=False):
        if return_freq_weights:
            f1a, f2a, f3a, f4a, x41, wa = self.net1(
                x, return_freq_weights=True
            )
            f1b, f2b, f3b, f4b, x42, wb = self.net2(
                x_box, return_freq_weights=True
            )
        else:
            f1a, f2a, f3a, f4a, x41 = self.net1(x)
            f1b, f2b, f3b, f4b, x42 = self.net2(x_box)

        f1 = torch.cat([f1a, f1b], dim=1)
        f2 = torch.cat([f2a, f2b], dim=1)
        f3 = torch.cat([f3a, f3b], dim=1)
        f4 = torch.cat([f4a, f4b], dim=1)
        x4 = torch.cat([x41, x42], dim=1)

        prior_cam = self.GPM(x4)
        pred_0 = F.interpolate(
            prior_cam, size=x.size()[2:], mode="bilinear", align_corners=False
        )
        (
            out4,
            out3,
            out2,
            out1,
            bound4,
            bound3,
            bound2,
            bound1,
        ) = self.decoder([f1, f2, f3, f4], prior_cam, x)

        outputs = (
            pred_0,
            out4,
            out3,
            out2,
            out1,
            bound4,
            bound3,
            bound2,
            bound1,
        )
        if return_freq_weights:
            return outputs, {"rgb": wa, "box": wb}
        return outputs


# -----------------------------------------------------------------------------
# Losses: unchanged from Noisy-COD training code
# -----------------------------------------------------------------------------

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1.0
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(predict * target * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    return (1 - num / den).mean()


def cal_ual(seg_logits, seg_gts):
    if seg_logits.shape != seg_gts.shape:
        raise ValueError(
            f"UAL shape mismatch: logits={seg_logits.shape}, gt={seg_gts.shape}"
        )
    sigmoid_x = seg_logits.sigmoid()
    return (1 - (2 * sigmoid_x - 1).abs().pow(2)).mean()


def get_ual_coef(iter_percentage: float):
    p = min(max(float(iter_percentage), 0.0), 1.0)
    return float((1.0 - math.cos(p * math.pi)) / 2.0)
