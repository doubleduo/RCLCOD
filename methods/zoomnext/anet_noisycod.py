# -*- coding: utf-8 -*-
"""
Clean Noisy-COD style ANet + controlled ablations for APBOXNet.

A0 baseline:
    RGB --------------------> ConvNeXt-B -> GCM3 -> DWT -> freq injection --\
                                                                           +-> fusion -> REU decoder
    RGB * Box --------------> ConvNeXt-B -> GCM3 -> DWT -> freq injection --/
    deepest RGB/Box feature --------------------------------------> GPM prior

Input:
    data["image_m"] : [B,3,H,W], float in [0,1]
    data["box_mask"]: [B,1,H,W], 0/1
    data["mask"]    : [B,1,H,W], train only

Eval:
    return final logits [B,1,H,W]

Train:
    return dict(logits, loss, loss_items, loss_str, vis)

Ablations:
    A0: Noisy-COD base
    A1: no box prompt (second branch gets RGB; capacity unchanged)
    A2: no DWT/frequency injection
    A3: no GPM (1x1 simple prior)
    A4: no edge loss
    A5: no UAL
    A6: use LH+HL+HH instead of HH only
    A7: relation fusion [R,B,|R-B|,R*B]
    A8: A6 + A7
"""

import logging
import math
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger("main")


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def structure_loss(logits, mask):
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    weit = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    wbce = (weit * wbce).sum((2, 3)) / weit.sum((2, 3)).clamp_min(1e-6)
    prob = torch.sigmoid(logits)
    inter = ((prob * mask) * weit).sum((2, 3))
    union = ((prob + mask) * weit).sum((2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
    return (wbce + wiou).mean()


def dice_loss_logits(logits, target):
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    pred = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    num = 2.0 * (pred * target).sum(1) + 1.0
    den = pred.square().sum(1) + target.square().sum(1) + 1.0
    return (1.0 - num / den).mean()


def mask_to_boundary(mask, kernel_size=5):
    pad = kernel_size // 2
    dil = F.max_pool2d(mask, kernel_size, 1, pad)
    ero = -F.max_pool2d(-mask, kernel_size, 1, pad)
    return (dil - ero).clamp(0, 1)


def cosine_coef(p, start=0.0, end=1.0):
    p = float(p)
    if p <= start:
        return 0.0
    if p >= end:
        return 1.0
    q = (p - start) / max(end - start, 1e-8)
    return 0.5 * (1.0 - math.cos(math.pi * q))


# ---------------------------------------------------------------------------
# blocks from / aligned with Noisy-COD
# ---------------------------------------------------------------------------

class BasicConv2d(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0, d=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BasicDeConv2d(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=2, p=1, out_p=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, k, s, p, output_padding=out_p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ETM(nn.Module):
    """Noisy-COD ETM: progressive asymmetric kernels + residual."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.b0 = BasicConv2d(in_c, out_c, 3, p=1)
        self.b1 = nn.Sequential(
            BasicConv2d(out_c, out_c, 1),
            BasicConv2d(out_c, out_c, (1, 3), p=(0, 1)),
            BasicConv2d(out_c, out_c, (3, 1), p=(1, 0)),
            BasicConv2d(out_c, out_c, (1, 5), p=(0, 2)),
            BasicConv2d(out_c, out_c, (5, 1), p=(2, 0)),
        )
        self.b2 = nn.Sequential(
            BasicConv2d(out_c, out_c, 1),
            BasicConv2d(out_c, out_c, (1, 5), p=(0, 2)),
            BasicConv2d(out_c, out_c, (5, 1), p=(2, 0)),
            BasicConv2d(out_c, out_c, (1, 7), p=(0, 3)),
            BasicConv2d(out_c, out_c, (7, 1), p=(3, 0)),
        )
        self.b3 = nn.Sequential(
            BasicConv2d(out_c, out_c, 1),
            BasicConv2d(out_c, out_c, (1, 7), p=(0, 3)),
            BasicConv2d(out_c, out_c, (7, 1), p=(3, 0)),
            BasicConv2d(out_c, out_c, (1, 9), p=(0, 4)),
            BasicConv2d(out_c, out_c, (9, 1), p=(4, 0)),
        )
        self.cat = BasicConv2d(out_c * 4, out_c, 1)
        self.res = BasicConv2d(in_c, out_c, 3, p=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        return self.relu(self.cat(torch.cat([x0, x1, x2, x3], 1)) + self.res(x))


class DWT(nn.Module):
    """Fixed Haar-like DWT used by Noisy-COD."""
    def forward(self, x):
        # pad odd spatial size if needed
        ph = x.shape[-2] % 2
        pw = x.shape[-1] % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
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
    """
    Four ConvNeXt stages -> ETM projections -> same-resolution camo feature -> DWT.
    """
    def __init__(self, in_dims, ch=64):
        super().__init__()
        self.t1 = ETM(in_dims[0], ch)
        self.t2 = ETM(in_dims[1], ch)
        self.t3 = ETM(in_dims[2], ch)
        self.t4 = ETM(in_dims[3], ch)
        self.decoder = nn.Conv2d(ch * 4, ch, 3, 1, 1)
        self.dwt = DWT()

    def forward(self, x1, x2, x3, x4):
        f1 = self.t1(x1)
        f2 = self.t2(x2)
        f3 = self.t3(x3)
        f4 = self.t4(x4)
        size = f1.shape[-2:]
        camo = self.decoder(torch.cat([
            f1,
            F.interpolate(f2, size=size, mode="bilinear", align_corners=False),
            F.interpolate(f3, size=size, mode="bilinear", align_corners=False),
            F.interpolate(f4, size=size, mode="bilinear", align_corners=False),
        ], 1))
        ll, lh, hl, hh = self.dwt(camo)
        return ll, lh, hl, hh, f1, f2, f3, f4


class FrequencyBranch(nn.Module):
    """
    One complete Noisy-COD branch.

    baseline:
       HH -> shallow F1/F2
       LL -> deep    F3/F4
    """
    def __init__(self, encoder, dims, ch=64, use_dwt=True, use_all_hf=False):
        super().__init__()
        self.encoder = encoder
        self.gcm3 = GCM3(dims, ch)
        self.use_dwt = use_dwt
        self.use_all_hf = use_all_hf

        self.ll_down3 = BasicConv2d(ch, ch, 3, s=2, p=1)
        self.ll_down4 = nn.Sequential(
            BasicConv2d(ch, ch, 3, s=2, p=1),
            BasicConv2d(ch, ch, 3, s=2, p=1),
        )
        self.f1_hf = ETM(ch * 2, ch)
        self.f2_hf = ETM(ch * 2, ch)
        self.f3_ll = ETM(ch * 2, ch)
        self.f4_ll = ETM(ch * 2, ch)

        # A6: directional high-frequency aggregation.
        self.hf_all = BasicConv2d(ch * 3, ch, 1) if use_all_hf else None

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        ll, lh, hl, hh, f1, f2, f3, f4 = self.gcm3(x1, x2, x3, x4)

        if not self.use_dwt:  # A2: same ETM features, no wavelet injection
            return (f1, f2, f3, f4), x4

        hf = self.hf_all(torch.cat([lh, hl, hh], 1)) if self.use_all_hf else hh

        hf_up = F.interpolate(hf, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        o1 = self.f1_hf(torch.cat([hf_up, f1], 1))

        hf2 = F.interpolate(hf, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        o2 = self.f2_hf(torch.cat([hf2, f2], 1))

        ll3 = self.ll_down3(ll)
        ll3 = F.interpolate(ll3, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        o3 = self.f3_ll(torch.cat([ll3, f3], 1))

        ll4 = self.ll_down4(ll)
        ll4 = F.interpolate(ll4, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        o4 = self.f4_ll(torch.cat([ll4, f4], 1))

        return (o1, o2, o3, o4), x4


class GPM(nn.Module):
    def __init__(self, in_c=2048, depth=32):
        super().__init__()
        self.g = nn.Sequential(nn.AdaptiveAvgPool2d(1), BasicConv2d(in_c, depth, 1))
        self.b0 = BasicConv2d(in_c, depth, 1)
        self.b1 = BasicConv2d(in_c, depth, 3, p=6, d=6)
        self.b2 = BasicConv2d(in_c, depth, 3, p=12, d=12)
        self.b3 = BasicConv2d(in_c, depth, 3, p=18, d=18)
        self.head = BasicConv2d(depth * 5, depth, 1)
        self.out = nn.Sequential(
            nn.Conv2d(depth, depth, 3, padding=1),
            nn.BatchNorm2d(depth),
            nn.PReLU(),
            nn.Conv2d(depth, 1, 3, padding=1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        g = F.interpolate(self.g(x), size=size, mode="bilinear", align_corners=False)
        x = torch.cat([g, self.b0(x), self.b1(x), self.b2(x), self.b3(x)], 1)
        return self.out(self.head(x))


class RelationFusion(nn.Module):
    """A7: explicit agreement/disagreement relation; output stays 2*ch channels."""
    def __init__(self, ch):
        super().__init__()
        self.proj = BasicConv2d(ch * 4, ch * 2, 1)

    def forward(self, r, b):
        return self.proj(torch.cat([r, b, torch.abs(r - b), r * b], 1))


class REUBlock(nn.Module):
    """Reverse-guided refinement + boundary head, aligned with Noisy-COD."""
    def __init__(self, in_c):
        super().__init__()
        self.prior_fuse = nn.Conv2d(in_c * 2, in_c, 1)
        self.ode = nn.Sequential(
            BasicConv2d(in_c, in_c, 3, p=1),
            BasicConv2d(in_c, in_c, 3, p=1),
        )
        self.out_mask = nn.Sequential(
            BasicConv2d(in_c * 2, in_c, 3, p=1),
            BasicConv2d(in_c, in_c // 2, 3, p=1),
            nn.Conv2d(in_c // 2, 1, 3, padding=1),
        )
        self.out_edge = nn.Sequential(
            BasicDeConv2d(in_c, in_c // 2),
            BasicConv2d(in_c // 2, in_c // 4, 3, p=1),
            nn.Conv2d(in_c // 4, 1, 3, padding=1),
        )

    def forward(self, x, prior):
        prior = F.interpolate(prior, size=x.shape[-2:], mode="bilinear", align_corners=False)
        prior_c = prior.expand(-1, x.shape[1], -1, -1)
        ode = self.ode(self.prior_fuse(torch.cat([x, prior_c], 1)))
        edge = self.out_edge(ode)
        reverse = 1.0 - torch.sigmoid(prior)
        reverse_feat = reverse.expand_as(x) * x
        mask = self.out_mask(torch.cat([reverse_feat, ode], 1)) + prior
        return mask, edge


class REUDecoder(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.r4 = REUBlock(in_c)
        self.r3 = REUBlock(in_c)
        self.r2 = REUBlock(in_c)
        self.r1 = REUBlock(in_c)

    def forward(self, feats, prior, target_hw):
        f1, f2, f3, f4 = feats
        p4, e4 = self.r4(f4, prior)
        p3, e3 = self.r3(f3, p4)
        p2, e2 = self.r2(f2, p3)
        p1, e1 = self.r1(f1, p2)

        ps = [prior, p4, p3, p2, p1]
        es = [e4, e3, e2, e1]
        ps = [F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False) for x in ps]
        es = [F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False) for x in es]
        return ps, es


# ---------------------------------------------------------------------------
# ANet
# ---------------------------------------------------------------------------

ABLATIONS = {
    "A0": dict(use_box=True,  use_dwt=True,  use_gpm=True, edge=True,  ual=True,  all_hf=False, relation=False),
    "A1": dict(use_box=False, use_dwt=True,  use_gpm=True, edge=True,  ual=True,  all_hf=False, relation=False),
    "A2": dict(use_box=True,  use_dwt=False, use_gpm=True, edge=True,  ual=True,  all_hf=False, relation=False),
    "A3": dict(use_box=True,  use_dwt=True,  use_gpm=False,edge=True,  ual=True,  all_hf=False, relation=False),
    "A4": dict(use_box=True,  use_dwt=True,  use_gpm=True, edge=False, ual=True,  all_hf=False, relation=False),
    "A5": dict(use_box=True,  use_dwt=True,  use_gpm=True, edge=True,  ual=False, all_hf=False, relation=False),
    "A6": dict(use_box=True,  use_dwt=True,  use_gpm=True, edge=True,  ual=True,  all_hf=True,  relation=False),
    "A7": dict(use_box=True,  use_dwt=True,  use_gpm=True, edge=True,  ual=True,  all_hf=False, relation=True),
    "A8": dict(use_box=True,  use_dwt=True,  use_gpm=True, edge=True,  ual=True,  all_hf=True,  relation=True),
}


class ConvNeXtB_NoisyCOD_ANet(nn.Module):
    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        channels=64,
        ablation="A0",
        edge_loss_weight=4.0,
        ual_loss_weight=2.0,
        ual_start=0.0,
        ual_full=1.0,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        ablation = str(ablation).upper()
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation={ablation}; choose {list(ABLATIONS)}")
        self.ablation = ablation
        self.flags = ABLATIONS[ablation]
        self.edge_loss_weight = float(edge_loss_weight) if self.flags["edge"] else 0.0
        self.ual_loss_weight = float(ual_loss_weight) if self.flags["ual"] else 0.0
        self.ual_start = float(ual_start)
        self.ual_full = float(ual_full)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), persistent=False)
        self.input_norm = bool(input_norm)

        enc_rgb = timm.create_model(
            "convnext_base", pretrained=pretrained, features_only=True, out_indices=(0,1,2,3)
        )
        enc_box = timm.create_model(
            "convnext_base", pretrained=False, features_only=True, out_indices=(0,1,2,3)
        )
        enc_box.load_state_dict(enc_rgb.state_dict(), strict=True)
        dims = list(enc_rgb.feature_info.channels())

        if use_checkpoint:
            for enc in (enc_rgb, enc_box):
                if hasattr(enc, "set_grad_checkpointing"):
                    enc.set_grad_checkpointing(True)

        self.rgb_branch = FrequencyBranch(
            enc_rgb, dims, channels, self.flags["use_dwt"], self.flags["all_hf"]
        )
        self.box_branch = FrequencyBranch(
            enc_box, dims, channels, self.flags["use_dwt"], self.flags["all_hf"]
        )

        if self.flags["relation"]:
            self.fusers = nn.ModuleList([RelationFusion(channels) for _ in range(4)])
        else:
            self.fusers = None

        self.gpm = GPM(dims[-1] * 2, depth=32) if self.flags["use_gpm"] else None
        self.simple_prior = nn.Conv2d(channels * 2, 1, 1) if not self.flags["use_gpm"] else None
        self.decoder = REUDecoder(channels * 2)

        LOGGER.info("NoisyCOD ANet %s flags=%s", self.ablation, self.flags)

    def norm(self, x):
        return (x - self.mean) / self.std if self.input_norm else x

    def prepare_inputs(self, data):
        image = data["image_m"].float()
        box_mask = data["box_mask"].float()
        if box_mask.ndim == 3:
            box_mask = box_mask.unsqueeze(1)
        if box_mask.shape[-2:] != image.shape[-2:]:
            box_mask = F.interpolate(box_mask, size=image.shape[-2:], mode="nearest")
        box_mask = box_mask.clamp(0, 1)
        # A1 keeps second branch and parameters but removes the box prompt.
        box_image = image * box_mask if self.flags["use_box"] else image
        return image, box_image, box_mask

    def fuse(self, rgb_feats, box_feats):
        if self.fusers is None:
            return [torch.cat([r, b], 1) for r, b in zip(rgb_feats, box_feats)]
        return [m(r, b) for m, r, b in zip(self.fusers, rgb_feats, box_feats)]

    def _forward_features(self, data):
        image, box_image, box_mask = self.prepare_inputs(data)
        rgb_feats, rgb_x4 = self.rgb_branch(self.norm(image))
        box_feats, box_x4 = self.box_branch(self.norm(box_image))
        fused = self.fuse(rgb_feats, box_feats)

        if self.gpm is not None:
            prior = self.gpm(torch.cat([rgb_x4, box_x4], 1))
        else:
            prior = self.simple_prior(fused[-1])

        mask_logits, edge_logits = self.decoder(fused, prior, image.shape[-2:])
        return {
            "logits": mask_logits[-1],
            "mask_logits": mask_logits,
            "edge_logits": edge_logits,
            "box_mask": box_mask,
        }

    def forward(self, data: Dict[str, torch.Tensor], iter_percentage=1.0, return_aux=False, **kwargs):
        del kwargs
        out = self._forward_features(data)
        if not self.training:
            return out if return_aux else out["logits"]

        if "mask" not in data:
            raise KeyError("ANet training requires data['mask']")
        mask = data["mask"].float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != out["logits"].shape[-2:]:
            mask = F.interpolate(mask, size=out["logits"].shape[-2:], mode="nearest")

        p0, p4, p3, p2, p1 = out["mask_logits"]
        s_init = (
            0.0625 * structure_loss(p0, mask)
            + 0.125 * structure_loss(p4, mask)
            + 0.25 * structure_loss(p3, mask)
            + 0.5 * structure_loss(p2, mask)
        )
        s_final = structure_loss(p1, mask)
        structure = s_init + s_final

        edge_target = mask_to_boundary(mask)
        _, e3, e2, e1 = out["edge_logits"]
        edge_raw = (
            0.125 * dice_loss_logits(e3, edge_target)
            + 0.25 * dice_loss_logits(e2, edge_target)
            + 0.5 * dice_loss_logits(e1, edge_target)
        )
        edge_loss = self.edge_loss_weight * edge_raw

        prob = torch.sigmoid(p1)
        ual_raw = (1.0 - (2.0 * prob - 1.0).abs().pow(2)).mean()
        coef = cosine_coef(iter_percentage, self.ual_start, self.ual_full)
        ual_loss = self.ual_loss_weight * coef * ual_raw

        total = structure + edge_loss + ual_loss
        return {
            "logits": p1,
            "loss": total,
            "loss_items": {
                "structure": structure.detach(),
                "structure_init": s_init.detach(),
                "structure_final": s_final.detach(),
                "edge": edge_loss.detach(),
                "edge_raw": edge_raw.detach(),
                "ual": ual_loss.detach(),
                "ual_raw": ual_raw.detach(),
                "ual_coef": float(coef),
                "total": total.detach(),
            },
            "loss_str": (
                f"{self.ablation} L:{total.detach().item():.4f} "
                f"STR:{structure.detach().item():.4f} "
                f"EDGE:{edge_loss.detach().item():.4f} "
                f"UAL:{ual_loss.detach().item():.4f}"
            ),
            "vis": {
                "sal": prob.detach(),
                "box": out["box_mask"].detach(),
                "boundary": torch.sigmoid(e1).detach(),
            },
        }

    def get_grouped_params(self):
        groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, p in self.named_parameters():
            if ".encoder." in name:
                groups["pretrained"].append(p)
            else:
                groups["retrained"].append(p)
        return groups


def build_anet(**kwargs):
    return ConvNeXtB_NoisyCOD_ANet(**kwargs)
