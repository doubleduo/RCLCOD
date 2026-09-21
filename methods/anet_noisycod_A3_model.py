# -*- coding: utf-8 -*-
"""Noisy-COD ANet with A0/A1/A2/A3 frequency ablations.

A3 is a prior-preserving residual frequency router:
    stage 1/2: HH + alpha * (Adaptive - HH)
    stage 3/4: LL + alpha * (Adaptive - LL)

Each stage owns an independent spatial router and an independent alpha.
alpha is initialized to 0, therefore A3 begins exactly from A0.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Backbone
# -----------------------------------------------------------------------------

def build_convnext_base_384(
    model_name: str = "convnext_base.fb_in22k_ft_in1k_384",
):
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
            f"Unexpected ConvNeXt feature channels {channels}; expected {expected}."
        )
    return model


# -----------------------------------------------------------------------------
# Basic blocks
# -----------------------------------------------------------------------------

class BasicConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, need_relu=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.need_relu = need_relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.need_relu else x


class BasicDeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, out_padding=0, need_relu=True,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            output_padding=out_padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.need_relu = need_relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.need_relu else x


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
    """Fixed Haar-like DWT from Noisy-COD."""
    def forward(self, x):
        # 384 training input produces even feature sizes, but keep this robust.
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")

        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

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
        camo = self.decoder(torch.cat([
            f1,
            F.interpolate(f2, scale_factor=2, mode="bilinear", align_corners=True),
            F.interpolate(f3, scale_factor=4, mode="bilinear", align_corners=True),
            F.interpolate(f4, scale_factor=8, mode="bilinear", align_corners=True),
        ], dim=1))
        ll, lh, hl, hh = self.DWT(camo)
        return ll, lh, hl, hh, f1, f2, f3, f4


# -----------------------------------------------------------------------------
# A2/A3 router
# -----------------------------------------------------------------------------

class SpatialFrequencyRouter(nn.Module):
    """Predicts per-pixel softmax weights for LL/LH/HL/HH."""
    def __init__(self, channels=64, hidden=32, temperature=1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4, 1, bias=True),
        )
        # Initial routing = exactly 1/4 for all four bands.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ll, lh, hl, hh):
        bands = torch.stack([ll, lh, hl, hh], dim=1)  # [B,4,C,H,W]
        logits = self.net(torch.cat([ll, lh, hl, hh], dim=1))
        weights = torch.softmax(logits / self.temperature, dim=1)  # [B,4,H,W]
        adaptive = (bands * weights.unsqueeze(2)).sum(dim=1)
        return adaptive, weights


class Branch(nn.Module):
    def __init__(
        self,
        backbone_name="convnext_base.fb_in22k_ft_in1k_384",
        channels=64,
        ablation="A0",
        router_hidden=32,
        router_temperature=1.0,
        router_alpha_init=0.0,
    ):
        super().__init__()
        self.ablation = str(ablation).upper()
        if self.ablation not in {"A0", "A1", "A2", "A3"}:
            raise ValueError(f"Unknown ablation={ablation}; expected A0/A1/A2/A3")

        self.shared_encoder = build_convnext_base_384(backbone_name)
        self.GCM3 = GCM3([128, 256, 512, 1024], channels)

        self.LL_down3 = nn.Sequential(
            BasicConv2d(channels, channels, 3, stride=2, padding=1)
        )
        self.LL_down4 = nn.Sequential(
            BasicConv2d(channels, channels, 3, stride=2, padding=1),
            BasicConv2d(channels, channels, 3, stride=2, padding=1),
        )
        self.dePixelShuffle = nn.Upsample(scale_factor=2)  # nearest, official behavior

        self.one_conv_f4_ll = ETM(channels * 2, channels)
        self.one_conv_f3_ll = ETM(channels * 2, channels)
        self.one_conv_f1_hh = ETM(channels * 2, channels)
        self.one_conv_f2_hh = ETM(channels * 2, channels)

        if self.ablation in {"A2", "A3"}:
            # Stage-specific routers: each stage may prefer different frequencies.
            self.freq_routers = nn.ModuleList([
                SpatialFrequencyRouter(
                    channels=channels,
                    hidden=router_hidden,
                    temperature=router_temperature,
                )
                for _ in range(4)
            ])
        else:
            self.freq_routers = None

        if self.ablation == "A3":
            # alpha[0:2] correct HH shallow prior; alpha[2:4] correct LL deep prior.
            self.router_alpha = nn.Parameter(
                torch.full((4,), float(router_alpha_init), dtype=torch.float32)
            )
        else:
            self.register_parameter("router_alpha", None)

        self.last_frequency_stats = None

    def _select_frequency(self, ll, lh, hl, hh):
        if self.ablation == "A0":
            return [hh, hh, ll, ll], None

        if self.ablation == "A1":
            mean_band = (ll + lh + hl + hh) / 4.0
            return [mean_band, mean_band, mean_band, mean_band], None

        adaptive = []
        weights = []
        for router in self.freq_routers:
            a, w = router(ll, lh, hl, hh)
            adaptive.append(a)
            weights.append(w)

        if self.ablation == "A2":
            return adaptive, weights

        # A3: interpolate from original A0 prior toward A2 adaptive routing.
        # alpha=0 -> exact A0; alpha=1 -> exact A2 for that stage.
        bases = [hh, hh, ll, ll]
        routed = [
            bases[i] + self.router_alpha[i] * (adaptive[i] - bases[i])
            for i in range(4)
        ]
        return routed, weights

    @torch.no_grad()
    def get_frequency_stats(self):
        """Return latest per-stage routing diagnostics and A3 alphas.

        Stage stats contain:
            mean: [w_LL, w_LH, w_HL, w_HH]
            std:  spatial/batch std for each band
            entropy: mean categorical entropy (max ln(4)=1.386294)
            entropy_norm: entropy / ln(4)
        """
        stats = dict(self.last_frequency_stats or {})
        if self.router_alpha is not None:
            stats["alpha"] = self.router_alpha.detach().float().cpu().tolist()
        return stats

    def forward(self, x):
        x1, x2, x3, x4 = self.shared_encoder(x)
        ll, lh, hl, hh, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        routed, weights = self._select_frequency(ll, lh, hl, hh)
        r1, r2, r3, r4 = routed

        if weights is not None:
            # Diagnostics only; detached, no graph retention.
            # Entropy uses natural logarithm; maximum for 4 bands is ln(4)=1.386294.
            stats = {}
            for i, w in enumerate(weights):
                wd = w.detach().float()
                mean = wd.mean(dim=(0, 2, 3)).cpu().tolist()
                std = wd.std(dim=(0, 2, 3), unbiased=False).cpu().tolist()
                entropy = (-(wd * (wd + 1e-8).log()).sum(dim=1)).mean().cpu().item()
                stats[f"stage{i+1}"] = {
                    "mean": mean,
                    "std": std,
                    "entropy": float(entropy),
                    "entropy_norm": float(entropy / 1.38629436112),
                }
            self.last_frequency_stats = stats

        # Keep the original Noisy-COD stage-scale routing unchanged.
        r1_up = self.dePixelShuffle(r1)
        f1_out = self.one_conv_f1_hh(torch.cat([r1_up, f1], dim=1))
        f2_out = self.one_conv_f2_hh(torch.cat([r2, f2], dim=1))

        r3_down = self.LL_down3(r3)
        f3_out = self.one_conv_f3_ll(torch.cat([r3_down, f3], dim=1))

        r4_down = self.LL_down4(r4)
        f4_out = self.one_conv_f4_ll(torch.cat([r4_down, f4], dim=1))

        return f1_out, f2_out, f3_out, f4_out, x4


# -----------------------------------------------------------------------------
# Original GPM + REU decoder
# -----------------------------------------------------------------------------

class GPM(nn.Module):
    def __init__(
        self, in_c=128,
        dilation_series=(6, 12, 18),
        padding_series=(6, 12, 18),
        depth=32,
    ):
        super().__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(in_c, depth, 1, stride=1),
        )
        self.branch0 = BasicConv2d(in_c, depth, 1, stride=1)
        self.branch1 = BasicConv2d(in_c, depth, 3, stride=1, padding=padding_series[0], dilation=dilation_series[0])
        self.branch2 = BasicConv2d(in_c, depth, 3, stride=1, padding=padding_series[1], dilation=dilation_series[1])
        self.branch3 = BasicConv2d(in_c, depth, 3, stride=1, padding=padding_series[2], dilation=dilation_series[2])
        self.head = nn.Sequential(BasicConv2d(depth * 5, depth, 1, padding=0))
        self.out = nn.Sequential(
            nn.Conv2d(depth, depth, 3, padding=1),
            nn.BatchNorm2d(depth),
            nn.PReLU(),
            nn.Conv2d(depth, 1, 3, 1, 1),
        )
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
        branch_main = F.interpolate(self.branch_main(x), size=size, mode="bilinear", align_corners=True)
        out = torch.cat([
            branch_main,
            self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x),
        ], dim=1)
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
            BasicDeConv2d(in_channels, mid_channels // 2, 3, stride=2, padding=1, out_padding=1),
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
        gradient[:, :, :-1, :] = torch.abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = torch.abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        return torch.clamp(img - gradient, 0, 1)

    def forward(self, x, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=x.size()[2:], mode="bilinear", align_corners=True)
        yt = self.conv(torch.cat([x, prior_cam.expand(-1, x.size(1), -1, -1)], dim=1))
        ode_out = self.ode(yt)
        bound = self.edge_enhance(self.out_B(ode_out))
        reverse_prior = 1 - torch.sigmoid(prior_cam)
        y = reverse_prior.expand(-1, x.size(1), -1, -1) * x
        y = self.out_y(torch.cat([y, ode_out], dim=1)) + prior_cam
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
        f4_out, b4 = self.REU_f4(f4, prior_0)
        f3_out, b3 = self.REU_f3(f3, f4_out)
        f2_out, b2 = self.REU_f2(f2, f3_out)
        f1_out, b1 = self.REU_f1(f1, f2_out)
        return (
            self._up(f4_out, full_size), self._up(f3_out, full_size),
            self._up(f2_out, full_size), self._up(f1_out, full_size),
            self._up(b4, full_size), self._up(b3, full_size),
            self._up(b2, full_size), self._up(b1, full_size),
        )


class NoisyCODANet(nn.Module):
    def __init__(
        self,
        backbone_name="convnext_base.fb_in22k_ft_in1k_384",
        channels=64,
        ablation="A0",
        router_hidden=32,
        router_temperature=1.0,
        router_alpha_init=0.0,
    ):
        super().__init__()
        branch_kwargs = dict(
            backbone_name=backbone_name,
            channels=channels,
            ablation=ablation,
            router_hidden=router_hidden,
            router_temperature=router_temperature,
            router_alpha_init=router_alpha_init,
        )
        self.net1 = Branch(**branch_kwargs)
        self.net2 = Branch(**branch_kwargs)
        self.GPM = GPM(2048)
        self.decoder = UNetDecoderWithEdges(channels * 2, channels * 2)

    def forward(self, x, x_box):
        f11, f21, f31, f41, x41 = self.net1(x)
        f12, f22, f32, f42, x42 = self.net2(x_box)

        f1 = torch.cat([f11, f12], dim=1)
        f2 = torch.cat([f21, f22], dim=1)
        f3 = torch.cat([f31, f32], dim=1)
        f4 = torch.cat([f41, f42], dim=1)
        x4 = torch.cat([x41, x42], dim=1)

        prior_cam = self.GPM(x4)
        pred_0 = F.interpolate(prior_cam, size=x.size()[2:], mode="bilinear", align_corners=False)
        f4p, f3p, f2p, f1p, b4, b3, b2, b1 = self.decoder([f1, f2, f3, f4], prior_cam, x)
        return pred_0, f4p, f3p, f2p, f1p, b4, b3, b2, b1

    @torch.no_grad()
    def get_frequency_stats(self):
        return {
            "rgb_branch": self.net1.get_frequency_stats(),
            "box_branch": self.net2.get_frequency_stats(),
        }


# -----------------------------------------------------------------------------
# Noisy-COD losses
# -----------------------------------------------------------------------------

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
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
        raise ValueError(f"UAL shape mismatch: logits={seg_logits.shape}, gt={seg_gts.shape}")
    sigmoid_x = seg_logits.sigmoid()
    return (1 - (2 * sigmoid_x - 1).abs().pow(2)).mean()


def get_ual_coef(iter_percentage: float):
    return float((1 - torch.cos(torch.tensor(iter_percentage * torch.pi)).item()) / 2)
