
# -*- coding: utf-8 -*-
"""
Noisy-COD-style ANet adapted for APBOXNet.

Architecture
------------
RGB -----------------> Branch(rgb) -----\
                                         -> concat multi-level freq features
RGB * Box -----------> Branch(box) -----/          |
                                                    +--> GPM prior
                                                    +--> REU decoder
                                                    +--> mask + boundary

Each Branch:
ConvNeXt-B -> ETM x4 -> GCM3 -> DWT
                           |-> HH -> shallow F1/F2
                           |-> LL -> deep    F3/F4

Training input:
    data["image_m"] : [B,3,H,W], float in [0,1]
    data["box_mask"]: [B,1,H,W]
    data["mask"]    : [B,1,H,W] (training only)
    data["edge"]    : optional [B,1,H,W]

Inference/generation:
    image_m + box_mask
"""

import logging
import math
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger("main")


class PixelNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class BasicConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, need_relu=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.need_relu = bool(need_relu)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.need_relu else x


class BasicDeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, out_padding=0, need_relu=True
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            output_padding=out_padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.need_relu = bool(need_relu)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.need_relu else x


class ETM(nn.Module):
    """Enhanced texture/multi-receptive-field module from Noisy-COD style design."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

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

        self.conv_cat = BasicConv2d(out_channels * 4, out_channels, 1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        out = self.conv_cat(torch.cat([x0, x1, x2, x3], dim=1))
        return self.relu(out + self.conv_res(x))


class DWT(nn.Module):
    """Fixed Haar-like DWT used by the released Noisy-COD implementation."""
    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0

        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        target_hw = (x.shape[-2] // 2, x.shape[-1] // 2)
        x1 = F.interpolate(x1, size=target_hw, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=target_hw, mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=target_hw, mode="bilinear", align_corners=False)
        x4 = F.interpolate(x4, size=target_hw, mode="bilinear", align_corners=False)

        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh


class GCM3(nn.Module):
    """Hierarchy aggregation followed by DWT."""
    def __init__(self, in_channels=(128, 256, 512, 1024), out_channels=64):
        super().__init__()
        self.T1 = ETM(in_channels[0], out_channels)
        self.T2 = ETM(in_channels[1], out_channels)
        self.T3 = ETM(in_channels[2], out_channels)
        self.T4 = ETM(in_channels[3], out_channels)
        self.decoder = nn.Conv2d(out_channels * 4, out_channels, 3, padding=1)
        self.dwt = DWT()

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)
        f2 = self.T2(f2)
        f3 = self.T3(f3)
        f4 = self.T4(f4)

        target_hw = f1.shape[-2:]
        camo = self.decoder(torch.cat([
            f1,
            F.interpolate(f2, size=target_hw, mode="bilinear", align_corners=False),
            F.interpolate(f3, size=target_hw, mode="bilinear", align_corners=False),
            F.interpolate(f4, size=target_hw, mode="bilinear", align_corners=False),
        ], dim=1))
        ll, lh, hl, hh = self.dwt(camo)
        return ll, lh, hl, hh, f1, f2, f3, f4


class FrequencyBranch(nn.Module):
    """
    One complete Noisy-COD ANet branch.

    Input -> ConvNeXt-B -> GCM3 -> DWT
         HH -> shallow F1/F2
         LL -> deep    F3/F4
    """
    def __init__(self, pretrained=True, channels=64, use_checkpoint=False):
        super().__init__()

        self.encoder = timm.create_model(
            "convnext_base",
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
        )
        if use_checkpoint and hasattr(self.encoder, "set_grad_checkpointing"):
            self.encoder.set_grad_checkpointing(enable=True)

        dims = tuple(self.encoder.feature_info.channels())
        if len(dims) != 4:
            raise RuntimeError(f"ConvNeXt-B must return 4 stages, got {dims}")

        self.gcm3 = GCM3(dims, channels)

        self.ll_down3 = BasicConv2d(channels, channels, 3, stride=2, padding=1)
        self.ll_down4 = nn.Sequential(
            BasicConv2d(channels, channels, 3, stride=2, padding=1),
            BasicConv2d(channels, channels, 3, stride=2, padding=1),
        )

        self.f1_hh = ETM(channels * 2, channels)
        self.f2_hh = ETM(channels * 2, channels)
        self.f3_ll = ETM(channels * 2, channels)
        self.f4_ll = ETM(channels * 2, channels)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        ll, lh, hl, hh, f1, f2, f3, f4 = self.gcm3(x1, x2, x3, x4)

        # Keep original Noisy-COD routing: LH/HL are not injected.
        del lh, hl

        hh_up = F.interpolate(hh, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f1_hh = self.f1_hh(torch.cat([hh_up, f1], dim=1))

        hh_f2 = F.interpolate(hh, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f2_hh = self.f2_hh(torch.cat([hh_f2, f2], dim=1))

        ll_f3 = self.ll_down3(ll)
        if ll_f3.shape[-2:] != f3.shape[-2:]:
            ll_f3 = F.interpolate(ll_f3, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        f3_ll = self.f3_ll(torch.cat([ll_f3, f3], dim=1))

        ll_f4 = self.ll_down4(ll)
        if ll_f4.shape[-2:] != f4.shape[-2:]:
            ll_f4 = F.interpolate(ll_f4, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        f4_ll = self.f4_ll(torch.cat([ll_f4, f4], dim=1))

        return f1_hh, f2_hh, f3_ll, f4_ll, x4


class GPM(nn.Module):
    """Global prior module."""
    def __init__(self, in_c, depth=32, dilations=(6, 12, 18)):
        super().__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(in_c, depth, 1),
        )
        self.branch0 = BasicConv2d(in_c, depth, 1)
        self.branch1 = BasicConv2d(in_c, depth, 3, padding=dilations[0], dilation=dilations[0])
        self.branch2 = BasicConv2d(in_c, depth, 3, padding=dilations[1], dilation=dilations[1])
        self.branch3 = BasicConv2d(in_c, depth, 3, padding=dilations[2], dilation=dilations[2])

        self.head = BasicConv2d(depth * 5, depth, 1)
        self.out = nn.Sequential(
            nn.Conv2d(depth, depth, 3, padding=1),
            nn.BatchNorm2d(depth),
            nn.PReLU(),
            nn.Conv2d(depth, 1, 3, padding=1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        bm = F.interpolate(self.branch_main(x), size=size, mode="bilinear", align_corners=False)
        out = torch.cat([
            bm, self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)
        ], dim=1)
        return self.out(self.head(out))


class REUBlock(nn.Module):
    """Reverse-enhancement unit with an auxiliary boundary branch."""
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.prior_fuse = nn.Conv2d(in_channels * 2, in_channels, 1)

        self.ode = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 3, padding=1),
            BasicConv2d(in_channels, in_channels, 3, padding=1),
        )

        self.mask_head = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, 3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, 3, padding=1),
        )

        self.bound_head = nn.Sequential(
            BasicDeConv2d(
                in_channels, mid_channels // 2,
                kernel_size=3, stride=2, padding=1, out_padding=1
            ),
            BasicConv2d(mid_channels // 2, mid_channels // 4, 3, padding=1),
            nn.Conv2d(mid_channels // 4, 1, 3, padding=1),
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
        return torch.clamp(img - gradient, 0.0, 1.0)

    def forward(self, x, prior):
        prior = F.interpolate(prior, size=x.shape[-2:], mode="bilinear", align_corners=False)
        prior_expand = prior.expand(-1, x.shape[1], -1, -1)

        yt = self.prior_fuse(torch.cat([x, prior_expand], dim=1))
        ode_out = self.ode(yt)

        bound = self.edge_enhance(self.bound_head(ode_out))

        reverse = 1.0 - torch.sigmoid(prior)
        reverse_feat = reverse.expand(-1, x.shape[1], -1, -1) * x

        pred = self.mask_head(torch.cat([reverse_feat, ode_out], dim=1))
        pred = pred + prior
        return pred, bound


class REUDecoder(nn.Module):
    def __init__(self, in_channels=128, mid_channels=128):
        super().__init__()
        self.reu4 = REUBlock(in_channels, mid_channels)
        self.reu3 = REUBlock(in_channels, mid_channels)
        self.reu2 = REUBlock(in_channels, mid_channels)
        self.reu1 = REUBlock(in_channels, mid_channels)

    def forward(self, feats, prior, image):
        f1, f2, f3, f4 = feats
        target_hw = image.shape[-2:]

        p4_low, e4 = self.reu4(f4, prior)
        p3_low, e3 = self.reu3(f3, p4_low)
        p2_low, e2 = self.reu2(f2, p3_low)
        p1_low, e1 = self.reu1(f1, p2_low)

        def up(x):
            return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

        return (
            up(p4_low), up(p3_low), up(p2_low), up(p1_low),
            up(e4), up(e3), up(e2), up(e1)
        )


def structure_loss(pred, mask):
    if pred.shape[-2:] != mask.shape[-2:]:
        pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    weit = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum((2, 3)) / weit.sum((2, 3)).clamp_min(1e-6)

    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum((2, 3))
    union = ((prob + mask) * weit).sum((2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
    return (wbce + wiou).mean()


def mask_to_boundary(mask, kernel_size=5):
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, 1, pad)
    eroded = -F.max_pool2d(-mask, kernel_size, 1, pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def dice_loss_probability(pred, target):
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    pred = pred.flatten(1)
    target = target.flatten(1)
    num = 2 * (pred * target).sum(1) + 1
    den = pred.square().sum(1) + target.square().sum(1) + 1
    return (1.0 - num / den).mean()


def cosine_coef(progress):
    p = min(max(float(progress), 0.0), 1.0)
    return float((1.0 - math.cos(math.pi * p)) * 0.5)


class ConvNeXtB_NoisyCOD_ANet(nn.Module):
    """
    APBOXNet-compatible reproduction of the Noisy-COD ANet architecture.
    """
    def __init__(
        self,
        pretrained=True,
        channels=64,
        input_norm=True,
        use_checkpoint=False,
        edge_loss_weight=4.0,
        ual_loss_weight=2.0,
        copy_box_encoder_init=True,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()

        # Create RGB branch with pretrained ConvNeXt-B.
        self.rgb_branch = FrequencyBranch(
            pretrained=pretrained,
            channels=channels,
            use_checkpoint=use_checkpoint,
        )

        # Avoid downloading ImageNet weights twice.
        self.box_branch = FrequencyBranch(
            pretrained=False,
            channels=channels,
            use_checkpoint=use_checkpoint,
        )
        if copy_box_encoder_init:
            self.box_branch.encoder.load_state_dict(
                self.rgb_branch.encoder.state_dict(), strict=True
            )

        deep_dim = self.rgb_branch.encoder.feature_info.channels()[-1]
        self.gpm = GPM(in_c=deep_dim * 2, depth=32)
        self.decoder = REUDecoder(in_channels=channels * 2, mid_channels=channels * 2)

        self.edge_loss_weight = float(edge_loss_weight)
        self.ual_loss_weight = float(ual_loss_weight)

    def _prepare_inputs(self, data: Dict[str, torch.Tensor]):
        image = data["image_m"].float()

        if "box_mask" not in data:
            raise KeyError(
                "ConvNeXtB_NoisyCOD_ANet requires data['box_mask'] "
                "during both training and pseudo-label generation."
            )
        box_mask = data["box_mask"].float()
        if box_mask.ndim == 3:
            box_mask = box_mask.unsqueeze(1)
        if box_mask.shape[-2:] != image.shape[-2:]:
            box_mask = F.interpolate(
                box_mask, size=image.shape[-2:], mode="nearest"
            )
        box_mask = box_mask.clamp(0.0, 1.0)

        box_image = data.get("box_image", None)
        if box_image is None:
            box_image = image * box_mask
        elif box_image.shape[-2:] != image.shape[-2:]:
            box_image = F.interpolate(
                box_image, size=image.shape[-2:],
                mode="bilinear", align_corners=False
            )

        return image, box_image, box_mask

    def _network_forward(self, data):
        image, box_image, box_mask = self._prepare_inputs(data)

        rgb_x = self.normalizer(image)
        box_x = self.normalizer(box_image)

        r1, r2, r3, r4, rx4 = self.rgb_branch(rgb_x)
        b1, b2, b3, b4, bx4 = self.box_branch(box_x)

        # Original late fusion after frequency routing.
        f1 = torch.cat([r1, b1], dim=1)
        f2 = torch.cat([r2, b2], dim=1)
        f3 = torch.cat([r3, b3], dim=1)
        f4 = torch.cat([r4, b4], dim=1)

        deep = torch.cat([rx4, bx4], dim=1)
        prior = self.gpm(deep)
        pred0 = F.interpolate(
            prior, size=image.shape[-2:], mode="bilinear", align_corners=False
        )

        p4, p3, p2, p1, e4, e3, e2, e1 = self.decoder(
            [f1, f2, f3, f4], prior, image
        )

        return {
            "preds": (pred0, p4, p3, p2, p1, e4, e3, e2, e1),
            "logits": p1,
            "edge_logits": (e4, e3, e2, e1),
            "box_mask": box_mask,
        }

    def forward(self, data, iter_percentage=1.0, return_aux=False, **kwargs):
        del kwargs
        out = self._network_forward(data)
        preds = out["preds"]

        if not self.training:
            if return_aux:
                return out
            return out["logits"]

        if "mask" not in data:
            raise KeyError("ANet training requires data['mask'].")

        mask = data["mask"].float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        edge = data.get("edge", None)
        if edge is None:
            edge = mask_to_boundary(mask)
        else:
            edge = edge.float()
            if edge.ndim == 3:
                edge = edge.unsqueeze(1)

        pred0, p4, p3, p2, p1, e4, e3, e2, e1 = preds

        loss_init = (
            0.0625 * structure_loss(pred0, mask)
            + 0.125 * structure_loss(p4, mask)
            + 0.25 * structure_loss(p3, mask)
            + 0.5 * structure_loss(p2, mask)
        )
        loss_final = structure_loss(p1, mask)

        loss_edge_raw = (
            0.125 * dice_loss_probability(e3, edge)
            + 0.25 * dice_loss_probability(e2, edge)
            + 0.5 * dice_loss_probability(e1, edge)
        )
        loss_edge = self.edge_loss_weight * loss_edge_raw

        prob = torch.sigmoid(p1)
        ual_raw = (1.0 - (2.0 * prob - 1.0).abs().pow(2)).mean()
        ual_coef = cosine_coef(iter_percentage)
        loss_ual = self.ual_loss_weight * ual_coef * ual_raw

        total = loss_init + loss_final + loss_edge + loss_ual

        return {
            "logits": p1,
            "preds": preds,
            "loss": total,
            "loss_items": {
                "init": loss_init.detach(),
                "final": loss_final.detach(),
                "edge": loss_edge.detach(),
                "edge_raw": loss_edge_raw.detach(),
                "ual": loss_ual.detach(),
                "ual_raw": ual_raw.detach(),
                "ual_coef": float(ual_coef),
                "total": total.detach(),
            },
            "loss_str": (
                f"L:{total.detach().item():.4f} "
                f"INIT:{loss_init.detach().item():.4f} "
                f"FINAL:{loss_final.detach().item():.4f} "
                f"EDGE:{loss_edge.detach().item():.4f} "
                f"UAL:{loss_ual.detach().item():.4f}"
            ),
            "vis": {
                "sal": prob,
                "box": out["box_mask"],
                "boundary": e1.clamp(0.0, 1.0),
            },
        }

    def get_grouped_params(self):
        groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, p in self.named_parameters():
            if name.startswith("rgb_branch.encoder.") or name.startswith("box_branch.encoder."):
                groups["pretrained"].append(p)
            else:
                groups["retrained"].append(p)
        return groups
