# -*- coding: utf-8 -*-
"""Noisy-COD PNet adapted to the WS_SAM_CL training/eval interface."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.pvt_v2_eff import pvt_v2_eff_b4
from .zoomnext.ops import PixelNormalizer

LOGGER = logging.getLogger("main")


class CBA(nn.Module):
    def __init__(self, ci, co, k, s=1, p=0, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, s, p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(co)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ETM(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.b0 = CBA(ci, co, 3, p=1)
        self.b1 = nn.Sequential(
            CBA(co, co, 1), CBA(co, co, (1, 3), p=(0, 1)),
            CBA(co, co, (3, 1), p=(1, 0)), CBA(co, co, (1, 5), p=(0, 2)),
            CBA(co, co, (5, 1), p=(2, 0)),
        )
        self.b2 = nn.Sequential(
            CBA(co, co, 1), CBA(co, co, (1, 5), p=(0, 2)),
            CBA(co, co, (5, 1), p=(2, 0)), CBA(co, co, (1, 7), p=(0, 3)),
            CBA(co, co, (7, 1), p=(3, 0)),
        )
        self.b3 = nn.Sequential(
            CBA(co, co, 1), CBA(co, co, (1, 7), p=(0, 3)),
            CBA(co, co, (7, 1), p=(3, 0)), CBA(co, co, (1, 9), p=(0, 4)),
            CBA(co, co, (9, 1), p=(4, 0)),
        )
        self.cat = CBA(4 * co, co, 1)
        self.res = CBA(ci, co, 3, p=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        return self.relu(self.cat(torch.cat((x0, x1, x2, x3), 1)) + self.res(x))


class DWT(nn.Module):
    def forward(self, x):
        a = x[:, :, 0::2, :] / 2
        b = x[:, :, 1::2, :] / 2
        x1, x3 = a[:, :, :, 0::2], a[:, :, :, 1::2]
        x2, x4 = b[:, :, :, 0::2], b[:, :, :, 1::2]
        size = (x.shape[-2] // 2, x.shape[-1] // 2)
        xs = [F.interpolate(v, size=size, mode="bilinear", align_corners=False)
              for v in (x1, x2, x3, x4)]
        x1, x2, x3, x4 = xs
        return (
            x1 + x2 + x3 + x4,
            -x1 + x2 - x3 + x4,
            -x1 - x2 + x3 + x4,
            x1 - x2 - x3 + x4,
        )


class GCM3(nn.Module):
    def __init__(self, in_channels, co=64):
        super().__init__()
        self.ts = nn.ModuleList([ETM(c, co) for c in in_channels])
        self.fuse = nn.Conv2d(4 * co, co, 3, 1, 1)
        self.dwt = DWT()

    def forward(self, x1, x2, x3, x4):
        fs = [m(x) for m, x in zip(self.ts, (x1, x2, x3, x4))]
        size = fs[0].shape[-2:]
        camo = self.fuse(torch.cat([
            fs[0],
            *[F.interpolate(f, size=size, mode="bilinear", align_corners=False)
              for f in fs[1:]],
        ], 1))
        ll, lh, hl, hh = self.dwt(camo)
        return ll, lh, hl, hh, *fs


class GPM(nn.Module):
    def __init__(self, ci=512, depth=32):
        super().__init__()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), CBA(ci, depth, 1))
        self.b0 = CBA(ci, depth, 1)
        self.b1 = CBA(ci, depth, 3, p=6, d=6)
        self.b2 = CBA(ci, depth, 3, p=12, d=12)
        self.b3 = CBA(ci, depth, 3, p=18, d=18)
        self.head = CBA(depth * 5, depth, 1)
        self.out = nn.Sequential(
            nn.Conv2d(depth, depth, 3, padding=1), nn.BatchNorm2d(depth),
            nn.PReLU(), nn.Conv2d(depth, 1, 3, padding=1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[-2:]
        pooled = F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=False)
        return self.out(self.head(torch.cat((pooled, self.b0(x), self.b1(x), self.b2(x), self.b3(x)), 1)))


class REU(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.prior_fuse = nn.Conv2d(2 * c, c, 1)
        self.ode = nn.Sequential(CBA(c, c, 3, p=1), CBA(c, c, 3, p=1))
        self.mask_head = nn.Sequential(
            CBA(2 * c, c, 3, p=1), CBA(c, c // 2, 3, p=1), nn.Conv2d(c // 2, 1, 3, padding=1)
        )
        self.edge_head = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(c // 2), nn.ReLU(inplace=True),
            CBA(c // 2, c // 4, 3, p=1), nn.Conv2d(c // 4, 1, 3, padding=1),
        )

    @staticmethod
    def edge_enhance(x):
        grad = x.clone()
        grad[:, :, :-1] = torch.abs(grad[:, :, :-1] - grad[:, :, 1:])
        grad[:, :, :, :-1] = torch.abs(grad[:, :, :, :-1] - grad[:, :, :, 1:])
        return torch.clamp(x - grad, 0, 1)

    def forward(self, x, prior):
        prior = F.interpolate(prior, size=x.shape[-2:], mode="bilinear", align_corners=False)
        prior_c = prior.expand(-1, x.shape[1], -1, -1)
        ode = self.ode(self.prior_fuse(torch.cat((x, prior_c), 1)))
        edge = self.edge_enhance(self.edge_head(ode))
        reverse = (1 - torch.sigmoid(prior)).expand_as(x) * x
        mask = self.mask_head(torch.cat((reverse, ode), 1)) + prior
        return mask, edge


class PNetDecoder(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.r1, self.r2, self.r3, self.r4 = REU(c), REU(c), REU(c), REU(c)

    @staticmethod
    def up(x, image):
        return F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, fs, prior, image):
        f1, f2, f3, f4 = fs
        m4, e4 = self.r4(f4, prior)
        m3, e3 = self.r3(f3, m4)
        m2, e2 = self.r2(f2, m3)
        m1, e1 = self.r1(f1, m2)
        return tuple(self.up(x, image) for x in (m4, m3, m2, m1, e4, e3, e2, e1))


class NCLoss(nn.Module):
    @staticmethod
    def wbce(logits, target):
        w = 1 + 5 * torch.abs(F.avg_pool2d(target, 31, 1, 15) - target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return ((w * bce).sum((2, 3)) / w.sum((2, 3)).clamp_min(1e-6)).mean()

    def forward(self, logits, target, q):
        wbce = self.wbce(logits, target)
        p, y = logits.sigmoid().flatten(1), target.flatten(1)
        num = torch.abs(p - y).pow(q).sum(1)
        inter = (p * y).sum(1)
        den = p.sum(1) + y.sum(1) - inter + 1e-6
        err = (num / den).mean()
        return err + wbce if q == 2 else 2 * err


class PvtV2B4_PNet(nn.Module):
    """Drop-in replacement for PvtV2B4_FPN_Baseline."""

    def __init__(self, pretrained=True, input_norm=True, fpn_dim=64, channels=None,
                 use_checkpoint=False, q_switch=0.60, edge_weight=1.0,
                 ual_weight=2.0, **kwargs):
        super().__init__()
        del kwargs
        c = int(fpn_dim if channels is None else channels)
        self.q_switch = float(q_switch)
        self.edge_weight = float(edge_weight)
        self.ual_weight = float(ual_weight)
        self.encoder = pvt_v2_eff_b4(pretrained=pretrained, use_checkpoint=use_checkpoint)
        self.embed_dims = list(self.encoder.embed_dims)
        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()

        self.gcm3 = GCM3(self.embed_dims, c)
        self.f1_hh, self.f2_hh = ETM(2 * c, c), ETM(2 * c, c)
        self.ll3 = CBA(c, c, 3, s=2, p=1)
        self.ll4 = nn.Sequential(CBA(c, c, 3, s=2, p=1), CBA(c, c, 3, s=2, p=1))
        self.f3_ll, self.f4_ll = ETM(2 * c, c), ETM(2 * c, c)
        self.gpm = GPM(self.embed_dims[-1])
        self.decoder = PNetDecoder(c)
        self.nc = NCLoss()

    def normalize_encoder(self, image):
        fs = self.encoder(self.normalizer(image))
        return fs["reduction_2"], fs["reduction_3"], fs["reduction_4"], fs["reduction_5"]

    @staticmethod
    def resize(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def body(self, data):
        image = data["image_m"]
        x1, x2, x3, x4 = self.normalize_encoder(image)
        ll, _, _, hh, f1, f2, f3, f4 = self.gcm3(x1, x2, x3, x4)
        f1 = self.f1_hh(torch.cat((self.resize(hh, f1), f1), 1))
        f2 = self.f2_hh(torch.cat((self.resize(hh, f2), f2), 1))
        ll3, ll4 = self.ll3(ll), self.ll4(ll)
        f3 = self.f3_ll(torch.cat((self.resize(ll3, f3), f3), 1))
        f4 = self.f4_ll(torch.cat((self.resize(ll4, f4), f4), 1))
        prior = self.gpm(x4)
        p0 = F.interpolate(prior, size=image.shape[-2:], mode="bilinear", align_corners=False)
        m4, m3, m2, m1, e4, e3, e2, e1 = self.decoder((f1, f2, f3, f4), prior, image)
        return p0, m4, m3, m2, m1, e4, e3, e2, e1

    @staticmethod
    def edge_from_mask(mask):
        d = F.max_pool2d(mask, 3, 1, 1)
        e = -F.max_pool2d(-mask, 3, 1, 1)
        return (d - e).clamp(0, 1)

    @staticmethod
    def dice(pred, target):
        p, y = pred.flatten(1), target.flatten(1)
        return (1 - (2 * (p * y).sum(1) + 1) / (p.pow(2).sum(1) + y.pow(2).sum(1) + 1)).mean()

    @staticmethod
    def ual(logits):
        p = logits.sigmoid()
        return (1 - torch.abs(2 * p - 1).pow(2)).mean()

    def forward(self, data, iter_percentage=1.0, **kwargs):
        del kwargs
        pred = self.body(data)
        logits = pred[4]
        if not self.training:
            return logits

        target = (data["soft_target"] if "soft_target" in data else data["mask"]).float().clamp(0, 1)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        edge = data.get("edge")
        if edge is None:
            edge = self.edge_from_mask(target)
        else:
            edge = edge.float().clamp(0, 1)
            if edge.shape[-2:] != logits.shape[-2:]:
                edge = F.interpolate(edge, size=logits.shape[-2:], mode="bilinear", align_corners=False)

        progress = float(max(0.0, min(1.0, iter_percentage)))
        q = 1 if progress > self.q_switch else 2
        loss_init = (self.nc(pred[0], target, q) * 0.0625 +
                     self.nc(pred[1], target, q) * 0.125 +
                     self.nc(pred[2], target, q) * 0.25 +
                     self.nc(pred[3], target, q) * 0.5)
        loss_final = self.nc(pred[4], target, q)
        loss_edge = (self.dice(pred[6], edge) * 0.125 +
                     self.dice(pred[7], edge) * 0.25 +
                     self.dice(pred[8], edge) * 0.5)
        ual_coef = (1 - math.cos(math.pi * progress)) / 2
        loss_ual = self.ual(pred[4]) * ual_coef
        loss = loss_init + loss_final + self.edge_weight * loss_edge + self.ual_weight * loss_ual

        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid(), "edge": pred[8]},
            "loss": loss,
            "loss_items": {
                "bce": loss_final.detach(),
                "pnet_init": loss_init.detach(), "pnet_final": loss_final.detach(),
                "pnet_edge": loss_edge.detach(), "pnet_ual": loss_ual.detach(),
                "q": logits.new_tensor(float(q)), "total": loss.detach(),
            },
            "loss_str": (f"L:{loss.detach().item():.4f} Init:{loss_init.detach().item():.4f} "
                         f"Final:{loss_final.detach().item():.4f} Edge:{loss_edge.detach().item():.4f} "
                         f"UAL:{loss_ual.detach().item():.4f} q:{q}"),
        }

    def get_grouped_params(self):
        groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, p in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                p.requires_grad = False
                groups["fixed"].append(p)
            elif name.startswith("encoder."):
                groups["pretrained"].append(p)
            else:
                groups["retrained"].append(p)
        LOGGER.info(
            "Parameter Groups:{Pretrained: %d, Fixed: %d, ReTrained: %d}",
            len(groups["pretrained"]), len(groups["fixed"]), len(groups["retrained"]),
        )
        return groups



    """PNet + multi-scale NCLoss + edge supervision, without UAL."""

    def __init__(
        self,
        q_switch=0.60,
        edge_weight=1.0,
        **kwargs,
    ):
        super().__init__(
            q_switch=q_switch,
            edge_weight=edge_weight,
            ual_weight=0.0,
            **kwargs,
        )

    def forward(self, data, iter_percentage=1.0, **kwargs):
        del kwargs

        pred = self.body(data)
        logits = pred[4]  # final P1

        if not self.training:
            return logits

        target = (
            data["soft_target"]
            if "soft_target" in data
            else data["mask"]
        ).float().clamp(0, 1)

        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(
                target,
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Prefer explicit edge GT/pseudo-edge if dataset supplies one.
        # Otherwise derive a morphological boundary from the mask online.
        edge = data.get("edge")
        if edge is None:
            edge = self.edge_from_mask(target)
        else:
            edge = edge.float().clamp(0, 1)
            if edge.shape[-2:] != logits.shape[-2:]:
                edge = F.interpolate(
                    edge,
                    size=logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        progress = float(max(0.0, min(1.0, iter_percentage)))
        q = 1 if progress > self.q_switch else 2

        loss_init = (
            self.nc(pred[0], target, q) * 0.0625
            + self.nc(pred[1], target, q) * 0.125
            + self.nc(pred[2], target, q) * 0.25
            + self.nc(pred[3], target, q) * 0.5
        )
        loss_final = self.nc(pred[4], target, q)

        # Match the Noisy-COD PNet edge supervision levels:
        # e3 1/8, e2 1/4, e1 1/2.
        # pred[5] = e4 is intentionally not supervised.
        loss_edge = (
            self.dice(pred[6], edge) * 0.125
            + self.dice(pred[7], edge) * 0.25
            + self.dice(pred[8], edge) * 0.5
        )

        loss = (
            loss_init
            + loss_final
            + self.edge_weight * loss_edge
        )

        return {
            "logits": logits,
            "vis": {
                "sal": logits.sigmoid(),
                "edge": pred[8],
                "edge_gt": edge,
            },
            "loss": loss,
            "loss_items": {
                "bce": loss_final.detach(),
                "pnet_init": loss_init.detach(),
                "pnet_final": loss_final.detach(),
                "pnet_edge": loss_edge.detach(),
                "q": logits.new_tensor(float(q)),
                "total": loss.detach(),
            },
            "loss_str": (
                f"L:{loss.detach().item():.4f} "
                f"NCinit:{loss_init.detach().item():.4f} "
                f"NCfinal:{loss_final.detach().item():.4f} "
                f"Edge:{loss_edge.detach().item():.4f} "
                f"q:{q}"
            ),
        }


# -*- coding: utf-8 -*-
"""
Clean loss ablations on the original PVTv2-B4 + plain FPN baseline.

B1: plain FPN + multi-scale Noise-Correction Loss (NC)
B2: B1 + auxiliary edge Dice loss
B3: B2 + uncertainty-aware loss (UAL)

No GCM3 / DWT / GPM / REU is used here.
"""


from .fpn_baseline import PvtV2B4_FPN_Baseline
from .zoomnext.ops import ConvBNReLU


def _resize_to_input(x, image):
    return F.interpolate(
        x,
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _prepare_mask(mask, logits):
    mask = mask.float()
    if mask.shape[-2:] != logits.shape[-2:]:
        mask = F.interpolate(mask, size=logits.shape[-2:], mode="nearest")
    return mask


def _weighted_bce(preds, targets):
    weit = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets
    )
    wbce = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-6)
    return wbce.mean()


def nc_loss(preds, targets, q):
    """Noise Correction Loss used by Noisy-COD PNet."""
    wbce = _weighted_bce(preds, targets)

    probs = torch.sigmoid(preds).flatten(1)
    targets_flat = targets.flatten(1)

    numerator = torch.sum(torch.abs(probs - targets_flat).pow(q), dim=1)
    intersection = torch.sum(probs * targets_flat, dim=1)
    denominator = (
        torch.sum(probs, dim=1)
        + torch.sum(targets_flat, dim=1)
        - intersection
        + 1e-6
    )
    normalized_error = (numerator / denominator).mean()

    if int(q) == 2:
        return normalized_error + wbce
    return 2.0 * normalized_error


def dice_loss(preds, targets):
    probs = torch.sigmoid(preds).flatten(1)
    targets = targets.flatten(1)
    smooth = 1.0
    numerator = 2.0 * torch.sum(probs * targets, dim=1) + smooth
    denominator = (
        torch.sum(probs.pow(2), dim=1)
        + torch.sum(targets.pow(2), dim=1)
        + smooth
    )
    return (1.0 - numerator / denominator).mean()


def mask_to_edge(mask, kernel_size=3):
    """Fallback edge target when the dataloader has no explicit edge map."""
    mask = mask.float().clamp(0.0, 1.0)
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def ual_loss(seg_logits):
    probs = torch.sigmoid(seg_logits)
    return (1.0 - torch.abs(2.0 * probs - 1.0).pow(2)).mean()


def cosine_ual_coef(iter_percentage):
    t = float(max(0.0, min(1.0, iter_percentage)))
    return (1.0 - math.cos(math.pi * t)) / 2.0


class _PlainFPNMultiScaleBase(PvtV2B4_FPN_Baseline):
    """Original plain FPN trunk + auxiliary prediction heads only."""

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        q_switch=0.60,
        edge_weight=1.0,
        ual_weight=2.0,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.q_switch = float(q_switch)
        self.edge_weight = float(edge_weight)
        self.ual_weight = float(ual_weight)

        self.aux_seg_5 = nn.Conv2d(fpn_dim, 1, kernel_size=1)
        self.aux_seg_4 = nn.Conv2d(fpn_dim, 1, kernel_size=1)
        self.aux_seg_3 = nn.Conv2d(fpn_dim, 1, kernel_size=1)
        self.aux_seg_2 = nn.Conv2d(fpn_dim, 1, kernel_size=1)

    def _fpn_features(self, image):
        c2, c3, c4, c5 = self.normalize_encoder(image)

        p5 = self.smooth_5(self.lateral_5(c5))
        p4 = self.smooth_4(self.lateral_4(c4) + self._resize_like(p5, c4))
        p3 = self.smooth_3(self.lateral_3(c3) + self._resize_like(p4, c3))
        p2 = self.smooth_2(self.lateral_2(c2) + self._resize_like(p3, c2))
        return p2, p3, p4, p5

    def _seg_outputs(self, image):
        p2, p3, p4, p5 = self._fpn_features(image)

        pred0 = _resize_to_input(self.aux_seg_5(p5), image)
        pred4 = _resize_to_input(self.aux_seg_4(p4), image)
        pred3 = _resize_to_input(self.aux_seg_3(p3), image)
        pred2 = _resize_to_input(self.aux_seg_2(p2), image)
        pred1 = _resize_to_input(self.predictor(p2), image)

        return {
            "final": pred1,
            "seg_preds": (pred0, pred4, pred3, pred2, pred1),
            "fpn": (p2, p3, p4, p5),
        }

    def _q_value(self, iter_percentage):
        return 2 if float(iter_percentage) <= self.q_switch else 1

    def _multi_scale_nc(self, seg_preds, mask, q):
        weights = (0.0625, 0.125, 0.25, 0.5, 1.0)
        losses = [nc_loss(pred, mask, q) for pred in seg_preds]
        total = sum(w * l for w, l in zip(weights, losses))
        return total, losses

    @staticmethod
    def _monitor_bce(final_logits, mask):
        return F.binary_cross_entropy_with_logits(final_logits, mask, reduction="mean")


# -*- coding: utf-8 -*-
"""
P1 Source-Aware Noisy-COD PNet.

Architecture is kept identical to the current PASAM Noisy-COD PNet:
    RGB -> PVTv2-B4 -> GCM3 -> DWT -> GPM -> four-stage REU decoder
        -> 5 mask predictions + 4 edge predictions

P1 changes ONLY the mask-supervision wrapper:
    fully annotated GT -> StructureLoss (weighted BCE + weighted IoU)
    ANet pseudo label  -> NCLoss
    source fusion      -> normalized source-level weighting (default 1:1)

Edge loss and UAL remain unchanged for a clean P0 -> P1 ablation.
Evaluation still returns only the final mask logits and is compatible with
basemain.test().
"""

import logging
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.pvt_v2_eff import pvt_v2_eff_b4
from .zoomnext.ops import PixelNormalizer
from .zoomnext.layers import MHSIU

LOGGER = logging.getLogger("main")


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
        bn=nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.need_relu = bool(need_relu)

    def forward(self, x):
        x = self.bn(self.conv(x))
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
        bn=nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=out_padding,
            bias=False,
        )
        self.bn = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.need_relu = bool(need_relu)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.need_relu:
            x = self.relu(x)
        return x


class ETM(nn.Module):
    """Multi-receptive-field enhancement module used by Noisy-COD."""

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
    """Haar-like DWT from the released PNet."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0

        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        target_hw = (x.shape[2] // 2, x.shape[3] // 2)
        x1 = F.interpolate(x1, size=target_hw, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=target_hw, mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=target_hw, mode="bilinear", align_corners=False)
        x4 = F.interpolate(x4, size=target_hw, mode="bilinear", align_corners=False)

        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh


class MHSIU_GCM3(nn.Module):
    """
    ZoomNeXt-style same-stage multi-scale fusion + original PNet GCM3/DWT.

    The three branches must be the SAME PVT semantic stage extracted from
    different input zoom factors, not adjacent PVT hierarchy levels.

    Per stage:
        large / medium / small feature
             -> shared ETM projection to `out_channels`
             -> MHSIU scale selection on the medium grid

    Then the original GCM3 operation is preserved:
        f1/f2/f3/f4 -> resize to f1 -> concat -> 3x3 -> DWT.
    """

    def __init__(
        self,
        in_channels=(64, 128, 320, 512),
        out_channels=64,
        siu_groups=4,
    ):
        super().__init__()

        # Shared stage transforms. The same transform is applied to the
        # large/medium/small feature of a given semantic stage.
        self.T1 = ETM(in_channels[0], out_channels)
        self.T2 = ETM(in_channels[1], out_channels)
        self.T3 = ETM(in_channels[2], out_channels)
        self.T4 = ETM(in_channels[3], out_channels)

        # Stage-wise multi-head scale integration units.
        self.siu1 = MHSIU(out_channels, num_groups=siu_groups)
        self.siu2 = MHSIU(out_channels, num_groups=siu_groups)
        self.siu3 = MHSIU(out_channels, num_groups=siu_groups)
        self.siu4 = MHSIU(out_channels, num_groups=siu_groups)

        # Original GCM3 hierarchy aggregation and DWT.
        self.decoder = nn.Conv2d(
            out_channels * 4,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.DWT = DWT()

    @staticmethod
    def _check_feats(feats, name):
        if not isinstance(feats, (tuple, list)) or len(feats) != 4:
            raise ValueError(
                f"{name} must be a 4-element tuple/list of PVT features."
            )

    @staticmethod
    def _fuse_stage(transform, siu, feat_l, feat_m, feat_s):
        l = transform(feat_l)
        m = transform(feat_m)
        s = transform(feat_s)
        return siu(l=l, m=m, s=s)

    def forward(self, feats_l, feats_m, feats_s):
        self._check_feats(feats_l, "feats_l")
        self._check_feats(feats_m, "feats_m")
        self._check_feats(feats_s, "feats_s")

        l1, l2, l3, l4 = feats_l
        m1, m2, m3, m4 = feats_m
        s1, s2, s3, s4 = feats_s

        # First fuse different zoom views at the SAME semantic stage.
        f1 = self._fuse_stage(self.T1, self.siu1, l1, m1, s1)
        f2 = self._fuse_stage(self.T2, self.siu2, l2, m2, s2)
        f3 = self._fuse_stage(self.T3, self.siu3, l3, m3, s3)
        f4 = self._fuse_stage(self.T4, self.siu4, l4, m4, s4)

        # Then preserve the original GCM3 hierarchy fusion.
        target_hw = f1.shape[-2:]
        camo = self.decoder(
            torch.cat(
                [
                    f1,
                    F.interpolate(
                        f2,
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    ),
                    F.interpolate(
                        f3,
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    ),
                    F.interpolate(
                        f4,
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    ),
                ],
                dim=1,
            )
        )

        ll, lh, hl, hh = self.DWT(camo)
        return ll, lh, hl, hh, f1, f2, f3, f4


class GPM(nn.Module):
    """Global prior module from the released PNet."""

    def __init__(
        self,
        in_c=512,
        dilation_series=(6, 12, 18),
        padding_series=(6, 12, 18),
        depth=32,
    ):
        super().__init__()

        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(in_c, depth, kernel_size=1, stride=1),
        )
        self.branch0 = BasicConv2d(in_c, depth, kernel_size=1, stride=1)
        self.branch1 = BasicConv2d(
            in_c,
            depth,
            kernel_size=3,
            stride=1,
            padding=padding_series[0],
            dilation=dilation_series[0],
        )
        self.branch2 = BasicConv2d(
            in_c,
            depth,
            kernel_size=3,
            stride=1,
            padding=padding_series[1],
            dilation=dilation_series[1],
        )
        self.branch3 = BasicConv2d(
            in_c,
            depth,
            kernel_size=3,
            stride=1,
            padding=padding_series[2],
            dilation=dilation_series[2],
        )

        self.head = nn.Sequential(
            BasicConv2d(depth * 5, depth, kernel_size=1),
        )
        self.out = nn.Sequential(
            nn.Conv2d(depth, depth, 3, padding=1),
            nn.BatchNorm2d(depth),
            nn.PReLU(),
            nn.Conv2d(depth, 1, 3, 1, 1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(
            branch_main,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        out = torch.cat(
            [branch_main, branch0, branch1, branch2, branch3],
            dim=1,
        )
        return self.out(self.head(out))


class UNetDecoderWithEdgesBlock(nn.Module):
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
        return torch.clamp(img - gradient, 0.0, 1.0)

    def forward(self, x, prior_cam):
        prior_cam = F.interpolate(
            prior_cam,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        yt = self.conv(
            torch.cat(
                [x, prior_cam.expand(-1, x.size(1), -1, -1)],
                dim=1,
            )
        )

        ode_out = self.ode(yt)
        bound = self.edge_enhance(self.out_B(ode_out))

        reverse_prior = 1.0 - torch.sigmoid(prior_cam)
        y = reverse_prior.expand(-1, x.size(1), -1, -1) * x

        y = self.out_y(torch.cat([y, ode_out], dim=1))
        y = y + prior_cam
        return y, bound


class UNetDecoderWithEdges(nn.Module):
    def __init__(self, in_channels=64, mid_channels=64):
        super().__init__()
        self.REU_f1 = UNetDecoderWithEdgesBlock(in_channels, mid_channels)
        self.REU_f2 = UNetDecoderWithEdgesBlock(in_channels, mid_channels)
        self.REU_f3 = UNetDecoderWithEdgesBlock(in_channels, mid_channels)
        self.REU_f4 = UNetDecoderWithEdgesBlock(in_channels, mid_channels)

    def forward(self, xs, prior_0, image):
        f1, f2, f3, f4 = xs
        target_hw = image.shape[-2:]

        f4_out, bound_f4 = self.REU_f4(f4, prior_0)
        f4_pred = F.interpolate(
            f4_out, size=target_hw, mode="bilinear", align_corners=False
        )
        bound_f4 = F.interpolate(
            bound_f4, size=target_hw, mode="bilinear", align_corners=False
        )

        f3_out, bound_f3 = self.REU_f3(f3, f4_out)
        f3_pred = F.interpolate(
            f3_out, size=target_hw, mode="bilinear", align_corners=False
        )
        bound_f3 = F.interpolate(
            bound_f3, size=target_hw, mode="bilinear", align_corners=False
        )

        f2_out, bound_f2 = self.REU_f2(f2, f3_out)
        f2_pred = F.interpolate(
            f2_out, size=target_hw, mode="bilinear", align_corners=False
        )
        bound_f2 = F.interpolate(
            bound_f2, size=target_hw, mode="bilinear", align_corners=False
        )

        f1_out, bound_f1 = self.REU_f1(f1, f2_out)
        f1_pred = F.interpolate(
            f1_out, size=target_hw, mode="bilinear", align_corners=False
        )
        bound_f1 = F.interpolate(
            bound_f1, size=target_hw, mode="bilinear", align_corners=False
        )

        return (
            f4_pred,
            f3_pred,
            f2_pred,
            f1_pred,
            bound_f4,
            bound_f3,
            bound_f2,
            bound_f1,
        )


class NCLoss(nn.Module):
    """Noise correction loss from the released PNet trainer."""

    @staticmethod
    def wbce_loss(preds, targets):
        weit = 1.0 + 5.0 * torch.abs(
            F.avg_pool2d(
                targets,
                kernel_size=31,
                stride=1,
                padding=15,
            )
            - targets
        )
        wbce = F.binary_cross_entropy_with_logits(
            preds,
            targets,
            reduction="none",
        )
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(
            dim=(2, 3)
        ).clamp_min(1e-6)
        return wbce.mean()

    def forward(self, preds, targets, q):
        if preds.shape[-2:] != targets.shape[-2:]:
            preds = F.interpolate(
                preds,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        wbce = self.wbce_loss(preds, targets)

        probs = torch.sigmoid(preds).flatten(1)
        targets_flat = targets.flatten(1)

        numerator = torch.sum(
            torch.abs(probs - targets_flat).pow(float(q)),
            dim=1,
        )
        intersection = torch.sum(probs * targets_flat, dim=1)
        denominator = (
            torch.sum(probs, dim=1)
            + torch.sum(targets_flat, dim=1)
            - intersection
            + 1e-6
        )

        loss = numerator / denominator

        if int(q) == 2:
            return loss.mean() + wbce
        return loss.mean() * 2.0


class StructureLoss(nn.Module):
    """Clean-label structure loss: weighted BCE + weighted IoU."""

    def forward(self, pred, mask):
        if pred.shape[-2:] != mask.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        weit = 1.0 + 5.0 * torch.abs(
            F.avg_pool2d(
                mask,
                kernel_size=31,
                stride=1,
                padding=15,
            )
            - mask
        )

        wbce = F.binary_cross_entropy_with_logits(
            pred,
            mask,
            reduction="none",
        )
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(
            dim=(2, 3)
        ).clamp_min(1e-6)

        prob = torch.sigmoid(pred)
        inter = ((prob * mask) * weit).sum(dim=(2, 3))
        union = ((prob + mask) * weit).sum(dim=(2, 3))
        wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)

        return (wbce + wiou).mean()


def dice_loss_probability(predict, target):
    """Boundary branch outputs are probabilities; do not sigmoid again."""
    if predict.shape[-2:] != target.shape[-2:]:
        predict = F.interpolate(
            predict,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    predict = predict.flatten(1)
    target = target.flatten(1)
    num = 2.0 * torch.sum(predict * target, dim=1) + 1.0
    den = (
        torch.sum(predict.pow(2), dim=1)
        + torch.sum(target.pow(2), dim=1)
        + 1.0
    )
    return (1.0 - num / den).mean()


def mask_to_boundary(mask, kernel_size=5):
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, 1, pad)
    eroded = -F.max_pool2d(-mask, kernel_size, 1, pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def cosine_coef(iter_percentage):
    p = min(max(float(iter_percentage), 0.0), 1.0)
    return float((1.0 - math.cos(math.pi * p)) * 0.5)


class PvtV2B4_NoisyPNet(nn.Module):
    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        channels=64,
        use_checkpoint=False,
        ual_weight=2.0,
        # P1-MHSIU multi-scale settings
        siu_groups=4,
        large_scale=1.5,
        small_scale=0.5,
        source_aware_loss=False,
        full_loss_type="structure",
        weak_loss_type="nc",
        full_loss_weight=1.0,
        weak_loss_weight=1.0,
        mask_level_weights=(0.0625, 0.125, 0.25, 0.5, 1.0),
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.encoder = pvt_v2_eff_b4(
            pretrained=pretrained,
            use_checkpoint=use_checkpoint,
        )
        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()

        self.GCM3 = MHSIU_GCM3(
            in_channels=(64, 128, 320, 512),
            out_channels=channels,
            siu_groups=int(siu_groups),
        )

        self.siu_groups = int(siu_groups)
        self.large_scale = float(large_scale)
        self.small_scale = float(small_scale)
        if self.large_scale <= 1.0:
            raise ValueError("large_scale must be > 1.0 for MHSIU.")
        if not (0.0 < self.small_scale < 1.0):
            raise ValueError("small_scale must be in (0, 1) for MHSIU.")

        self.LL_down3 = nn.Sequential(
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1)
        )
        self.LL_down4 = nn.Sequential(
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1),
        )

        self.hh_up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        self.one_conv_f4_ll = ETM(channels * 2, channels)
        self.one_conv_f3_ll = ETM(channels * 2, channels)
        self.one_conv_f1_hh = ETM(channels * 2, channels)
        self.one_conv_f2_hh = ETM(channels * 2, channels)

        self.GPM = GPM(512)
        self.decoder = UNetDecoderWithEdges(channels, channels)

        self.nc_loss = NCLoss()
        self.structure_loss = StructureLoss()
        self.ual_weight = float(ual_weight)

        self.source_aware_loss = bool(source_aware_loss)
        self.full_loss_type = str(full_loss_type).lower()
        self.weak_loss_type = str(weak_loss_type).lower()
        self.full_loss_weight = float(full_loss_weight)
        self.weak_loss_weight = float(weak_loss_weight)
        self.mask_level_weights = tuple(float(x) for x in mask_level_weights)

        if self.full_loss_type != "structure":
            raise ValueError("P1 full_loss_type must be 'structure'.")
        if self.weak_loss_type != "nc":
            raise ValueError("P1 weak_loss_type must be 'nc'.")
        if len(self.mask_level_weights) != 5:
            raise ValueError("mask_level_weights must contain exactly 5 values.")
        if self.full_loss_weight < 0 or self.weak_loss_weight < 0:
            raise ValueError("Loss weights must be >= 0.")
        if self.full_loss_weight + self.weak_loss_weight <= 0:
            raise ValueError("full_loss_weight + weak_loss_weight must be > 0.")

        LOGGER.info(
            "PNet MHSIU | large=%.3fx | medium=1.000x | small=%.3fx | groups=%d",
            self.large_scale,
            self.small_scale,
            self.siu_groups,
        )

        LOGGER.info(
            "PNet loss mode | source_aware=%s | FULL=%s w=%.3f | "
            "WEAK=%s w=%.3f | levels=%s",
            self.source_aware_loss,
            self.full_loss_type,
            self.full_loss_weight,
            self.weak_loss_type,
            self.weak_loss_weight,
            self.mask_level_weights,
        )

    @staticmethod
    def _scaled_image(image, scale):
        """Resize a normalized-range RGB tensor without changing its values."""
        h, w = image.shape[-2:]
        target_h = max(32, int(round(h * float(scale))))
        target_w = max(32, int(round(w * float(scale))))
        return F.interpolate(
            image,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

    def _encode_image(self, image):
        """Shared PVTv2-B4 encoder for one zoom view."""
        x = self.normalizer(image)
        feats = self.encoder(x)
        return (
            feats["reduction_2"],
            feats["reduction_3"],
            feats["reduction_4"],
            feats["reduction_5"],
        )

    def _network_forward(self, image):
        # ==============================================================
        # 1) Build three zoom views inside the model.
        #    The dataloader/main stays unchanged.
        # ==============================================================
        image_m = image
        image_l = self._scaled_image(image, self.large_scale)
        image_s = self._scaled_image(image, self.small_scale)

        # ==============================================================
        # 2) Shared PVT-B4. Parameters are shared; this is NOT 3 encoders.
        # ==============================================================
        feats_l = self._encode_image(image_l)
        feats_m = self._encode_image(image_m)
        feats_s = self._encode_image(image_s)

        # Keep the base-scale deepest PVT feature for the original GPM.
        # This preserves GPM(512) and makes P1 -> P1-MHSIU ablation clean.
        x4_m = feats_m[3]

        # ==============================================================
        # 3) Stage-wise MHSIU -> original GCM3 hierarchy -> DWT.
        # ==============================================================
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(
            feats_l=feats_l,
            feats_m=feats_m,
            feats_s=feats_s,
        )
        del LH, HL

        # ==============================================================
        # 4) Original PNet DWT injection is unchanged.
        # ==============================================================
        HH_up = self.hh_up(HH)
        f1_HH = self.one_conv_f1_hh(
            torch.cat([HH_up, f1], dim=1)
        )
        f2_HH = self.one_conv_f2_hh(
            torch.cat([HH, f2], dim=1)
        )

        LL_down3 = self.LL_down3(LL)
        f3_LL = self.one_conv_f3_ll(
            torch.cat([LL_down3, f3], dim=1)
        )

        LL_down4 = self.LL_down4(LL)
        f4_LL = self.one_conv_f4_ll(
            torch.cat([LL_down4, f4], dim=1)
        )

        # ==============================================================
        # 5) Original GPM stays on the 1.0x deepest PVT feature.
        # ==============================================================
        prior_cam = self.GPM(x4_m)
        pred_0 = F.interpolate(
            prior_cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # ==============================================================
        # 6) Original four-stage REU decoder is unchanged.
        # ==============================================================
        p4, p3, p2, p1, e4, e3, e2, e1 = self.decoder(
            [f1_HH, f2_HH, f3_LL, f4_LL],
            prior_cam,
            image,
        )

        return pred_0, p4, p3, p2, p1, e4, e3, e2, e1

    def _source_mask_loss(
        self,
        preds,
        target,
        sample_idx,
        loss_type,
        q_value,
    ):
        """Five-level deep supervision for one supervision source."""
        num_samples = int(sample_idx.sum().item())
        if num_samples == 0:
            zero = preds[0].sum() * 0.0
            return zero, zero, zero

        w0, w1, w2, w3, w4 = self.mask_level_weights

        def one_loss(pred):
            pred_branch = pred[sample_idx]
            if loss_type == "structure":
                return self.structure_loss(pred_branch, target)
            if loss_type == "nc":
                return self.nc_loss(pred_branch, target, q_value)
            raise ValueError(f"Unknown loss type: {loss_type}")

        loss_init = (
            w0 * one_loss(preds[0])
            + w1 * one_loss(preds[1])
            + w2 * one_loss(preds[2])
            + w3 * one_loss(preds[3])
        )
        loss_final = w4 * one_loss(preds[4])
        return loss_init, loss_final, loss_init + loss_final

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        iter_percentage=1.0,
        q_value=2,
        **kwargs,
    ):
        del kwargs

        preds = self._network_forward(data["image_m"])

        # Compatible with basemain evaluator: final mask logits only.
        if not self.training:
            return preds[4]

        mask = data["mask"].float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        if "edge" in data:
            edge = data["edge"].float()
            if edge.ndim == 3:
                edge = edge.unsqueeze(1)
        else:
            edge = mask_to_boundary(mask, kernel_size=5)

        q_value = int(q_value)

        if self.source_aware_loss:
            if "is_pseudo" not in data:
                raise KeyError(
                    "source_aware_loss=True but data['is_pseudo'] is missing. "
                    "Use the P1 _concat_batch() in pnet_basemain.py."
                )

            is_pseudo = data["is_pseudo"].to(mask.device).bool()
            if is_pseudo.ndim != 1 or is_pseudo.shape[0] != mask.shape[0]:
                raise ValueError(
                    "is_pseudo must have shape [B]. "
                    f"Got {tuple(is_pseudo.shape)}, batch={mask.shape[0]}."
                )

            full_idx = ~is_pseudo
            weak_idx = is_pseudo
            if not bool(full_idx.any()):
                raise RuntimeError("P1 batch contains no fully annotated samples.")
            if not bool(weak_idx.any()):
                raise RuntimeError("P1 batch contains no weak pseudo samples.")

            mask_full = mask[full_idx]
            mask_weak = mask[weak_idx]

            loss_full_init, loss_full_final, loss_full_mask = self._source_mask_loss(
                preds=preds,
                target=mask_full,
                sample_idx=full_idx,
                loss_type=self.full_loss_type,
                q_value=q_value,
            )
            loss_weak_init, loss_weak_final, loss_weak_mask = self._source_mask_loss(
                preds=preds,
                target=mask_weak,
                sample_idx=weak_idx,
                loss_type=self.weak_loss_type,
                q_value=q_value,
            )

            source_weight_sum = self.full_loss_weight + self.weak_loss_weight
            loss_mask = (
                self.full_loss_weight * loss_full_mask
                + self.weak_loss_weight * loss_weak_mask
            ) / source_weight_sum
        else:
            # Exact P0-compatible unified NCLoss path.
            w0, w1, w2, w3, w4 = self.mask_level_weights
            loss_weak_init = (
                self.nc_loss(preds[0], mask, q_value) * w0
                + self.nc_loss(preds[1], mask, q_value) * w1
                + self.nc_loss(preds[2], mask, q_value) * w2
                + self.nc_loss(preds[3], mask, q_value) * w3
            )
            loss_weak_final = self.nc_loss(preds[4], mask, q_value) * w4
            loss_weak_mask = loss_weak_init + loss_weak_final

            zero = preds[4].sum() * 0.0
            loss_full_init = zero
            loss_full_final = zero
            loss_full_mask = zero
            loss_mask = loss_weak_mask

        # P1 keeps the original mixed-source edge loss unchanged.
        loss_edge = (
            dice_loss_probability(preds[6], edge) * 0.125
            + dice_loss_probability(preds[7], edge) * 0.25
            + dice_loss_probability(preds[8], edge) * 0.5
        )

        # P1 keeps the original mixed-source UAL unchanged.
        prob = torch.sigmoid(preds[4])
        ual_raw = (1.0 - (2.0 * prob - 1.0).abs().pow(2)).mean()
        ual_coef = cosine_coef(iter_percentage)
        loss_ual = self.ual_weight * float(ual_coef) * ual_raw

        total_loss = loss_mask + loss_edge + loss_ual

        return {
            "logits": preds[4],
            "preds": preds,
            "loss": total_loss,
            "loss_items": {
                "full_init": loss_full_init.detach(),
                "full_final": loss_full_final.detach(),
                "full_mask": loss_full_mask.detach(),
                "weak_init": loss_weak_init.detach(),
                "weak_final": loss_weak_final.detach(),
                "weak_mask": loss_weak_mask.detach(),
                "mask": loss_mask.detach(),
                "edge": loss_edge.detach(),
                "ual": loss_ual.detach(),
                "ual_raw": ual_raw.detach(),
                "ual_coef": float(ual_coef),
                "q": float(q_value),
                "total": total_loss.detach(),
            },
            "loss_str": (
                f"L:{total_loss.detach().item():.4f} "
                f"M:{loss_mask.detach().item():.4f} "
                f"FULL:{loss_full_mask.detach().item():.4f} "
                f"WEAK:{loss_weak_mask.detach().item():.4f} "
                f"EDGE:{loss_edge.detach().item():.4f} "
                f"UAL:{loss_ual.detach().item():.4f} "
                f"Q:{q_value}"
            ),
            "vis": {
                "sal": torch.sigmoid(preds[4]),
                "boundary": preds[8].clamp(0.0, 1.0),
            },
        }

    def get_grouped_params(self):
        groups = {
            "pretrained": [],
            "fixed": [],
            "retrained": [],
        }
        for name, param in self.named_parameters():
            if name.startswith("encoder."):
                groups["pretrained"].append(param)
            else:
                groups["retrained"].append(param)

        LOGGER.info(
            "PNet Parameter Groups:{"
            f"Pretrained:{len(groups['pretrained'])}, "
            f"Fixed:{len(groups['fixed'])}, "
            f"ReTrained:{len(groups['retrained'])}"
            "}"
        )
        return groups


# Explicit experiment-name alias. Both names construct the same model.
PvtV2B4_NoisyPNet_MHSIU = PvtV2B4_NoisyPNet
