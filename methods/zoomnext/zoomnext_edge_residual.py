# -*- coding: utf-8 -*-
"""PVTv2-B4 ZoomNeXt + RGB edge residual (instead of a 0.5x PVT branch)
+ boundary-localized uncertainty-aware loss (Edge-UAL).

Based on the repository's real `PvtV2B4_ZoomNeXt`:
  * Shared pretrained PVTv2-B4 on 1.0x/1.5x images
  * MHSIU2 at P5/P4/P3/P2 and original RGPU top-down decoder
  * NO PVT forward on image_s, and NO NC loss.

Important: the repository's PVT ZoomNeXt ALREADY uses two image scales.
The original 3-scale ResNet/Zoom_DeepNC variants are different models.
Here a cheap edge residual replaces the *role* of the missing small-scale
branch, while the two pretrained semantic streams remain untouched.

Ablations:
  PvtV2B4_Zoom_BCE_Control          BCE only
  PvtV2B4_Zoom_FullUAL_Control      full-map UAL only
  PvtV2B4_Zoom_EdgeUAL              edge UAL only
  PvtV2B4_Zoom_EdgeRes_BCE          edge residual + BCE, no UAL
  PvtV2B4_Zoom_EdgeRes_FullUAL      edge residual + full-map UAL
  PvtV2B4_Zoom_EdgeRes_EdgeUAL      edge residual + edge UAL (target)
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .zoomnext import PvtV2B4_ZoomNeXt
from .ops import resize_to


class SobelEdge(nn.Module):
    """Parameter-free luminance Sobel edges; independent of pseudo-labels."""

    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
        gy = gx.T.contiguous()
        self.register_buffer("kx", gx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ky", gy.view(1, 1, 3, 3), persistent=False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # Caller supplies unnormalized RGB in [0,1] from the dataset.
        if rgb.shape[1] != 3:
            raise ValueError("SobelEdge expects [B,3,H,W]")
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            x = rgb.float()
            grey = (0.2989*x[:, 0:1] + 0.5870*x[:, 1:2] + 0.1140*x[:, 2:3])
            # Replicate padding avoids artificial edges on image boundaries.
            grey = F.pad(grey, (1, 1, 1, 1), mode="replicate")
            dx = F.conv2d(grey, self.kx.float())
            dy = F.conv2d(grey, self.ky.float())
            magnitude = (dx.abs() + dy.abs()).clamp(0.0, 1.0)
        return magnitude.to(dtype=rgb.dtype)


class ShallowEdgeStem(nn.Module):
    """384 RGB+Sobel -> /4 edge features. No extra transformer encoder."""

    def __init__(self, channels: int = 64, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden, 5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        # RGB stays in its original range. Edge contains no GT information.
        return self.net(torch.cat((image, edge), dim=1))


class SemanticGuidedEdgeResidual(nn.Module):
    """Edge correction gated by same-level semantic features, with ZeroConv."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.edge_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.zero = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.zero.weight)
        nn.init.zeros_(self.zero.bias)

    def forward(self, semantic: torch.Tensor, edge_feat: torch.Tensor,
                edge_magnitude: torch.Tensor, strength: float = 1.0):
        target_hw = semantic.shape[-2:]
        edge_feat = F.adaptive_avg_pool2d(edge_feat, target_hw)
        edge_magnitude = F.adaptive_max_pool2d(edge_magnitude.float(), target_hw)
        edge_magnitude = edge_magnitude.to(dtype=semantic.dtype)
        edge_feat = self.edge_refine(edge_feat.to(dtype=semantic.dtype))
        gate = self.gate(torch.cat([semantic, edge_feat], dim=1))
        # The additive 0.5 ensures faint yet meaningful COD edges aren't erased.
        candidate = edge_feat * gate * (0.5 + 0.5 * edge_magnitude)
        delta = float(strength) * self.zero(candidate)
        out = semantic + delta
        with torch.no_grad():
            ratio = delta.detach().float().abs().mean() / (
                semantic.detach().float().abs().mean() + 1e-6
            )
            gate_mean = gate.detach().float().mean()
        return out, ratio, gate_mean


class _ZoomEdgeBase(PvtV2B4_ZoomNeXt):
    """Standard dual-scale ZoomNeXt; replace 0.5-scale role by edge residual."""

    def __init__(self, pretrained=True, num_frames=1, input_norm=True,
                 mid_dim=64, siu_groups=4, hmu_groups=6,
                 use_checkpoint=False, *, edge_residual=True,
                 edge_levels: Sequence[int] = (3, 2),
                 edge_strength=1.0, ual_scope="edge", ual_weight=0.20,
                 edge_kernel=7, ual_start_ratio=0.40, ual_full_ratio=0.70,
                 edge_ual_clean_only=False, **kwargs):
        # Match the parent's signature; harmless trainer extras must not leak.
        kwargs.pop("num_frames", None)
        kwargs.pop("use_checkpoint", None)
        super().__init__(pretrained=pretrained, num_frames=num_frames,
                         input_norm=input_norm, mid_dim=mid_dim,
                         siu_groups=siu_groups, hmu_groups=hmu_groups,
                         use_checkpoint=use_checkpoint)
        if ual_scope not in ("none", "full", "edge"):
            raise ValueError("ual_scope must be 'none', 'full', or 'edge'")
        if edge_kernel < 3 or edge_kernel % 2 == 0:
            raise ValueError("edge_kernel must be odd and >=3")
        if not (0 <= ual_start_ratio < ual_full_ratio <= 1):
            raise ValueError("UAL schedule requires 0<=start<full<=1")
        if ual_weight < 0 or edge_strength < 0:
            raise ValueError("loss/edge strength must be nonnegative")
        if any(int(x) not in (2,3,4,5) for x in edge_levels):
            raise ValueError("edge_levels only supports stage 2,3,4,5")

        self.edge_residual = bool(edge_residual)
        self.edge_levels = tuple(sorted(set(int(v) for v in edge_levels)))
        self.edge_strength = float(edge_strength)
        self.ual_scope = str(ual_scope)
        self.ual_weight = float(ual_weight)
        self.edge_kernel = int(edge_kernel)
        self.ual_start_ratio = float(ual_start_ratio)
        self.ual_full_ratio = float(ual_full_ratio)
        self.edge_ual_clean_only = bool(edge_ual_clean_only)

        if self.edge_residual:
            self.sobel = SobelEdge()
            self.edge_stem = ShallowEdgeStem(mid_dim)
            self.edge_blocks = nn.ModuleDict({
                str(l): SemanticGuidedEdgeResidual(mid_dim)
                for l in self.edge_levels
            })
        # Nonpersistent scalars for CSV logging.
        self._edge_diag = {}

    @staticmethod
    def _prepare_mask(mask: torch.Tensor, logits: torch.Tensor):
        mask = mask.float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != logits.shape[-2:]:
            mask = F.interpolate(mask, size=logits.shape[-2:], mode="nearest")
        if mask.shape != logits.shape:
            raise ValueError(f"Mask {mask.shape} and logits {logits.shape} mismatch")
        return mask.clamp(0,1)

    def _boundary_band(self, mask: torch.Tensor):
        # Pseudo-label-derived band. Detached: can't be changed by the model.
        with torch.no_grad():
            binary = (mask >= 0.5).to(dtype=mask.dtype)
            k = self.edge_kernel
            dilate = F.max_pool2d(binary,kernel_size=k,stride=1,padding=k//2)
            erode = 1.0 - F.max_pool2d(1.0-binary,kernel_size=k,stride=1,padding=k//2)
            return (dilate-erode).clamp(0.0,1.0).detach()

    def _ual_coef(self, iter_percentage):
        p = min(max(float(iter_percentage),0.),1.)
        if p <= self.ual_start_ratio:
            return 0.0
        if p >= self.ual_full_ratio:
            return 1.0
        t = (p-self.ual_start_ratio)/(self.ual_full_ratio-self.ual_start_ratio)
        return 0.5 * (1-math.cos(math.pi*t))

    def _edge_inject(self, idx, z, edge_feats, edge_mag):
        # ModuleDict does not implement Mapping.get on every PyTorch version.
        if not self.edge_residual or str(idx) not in self.edge_blocks:
            return z
        y, r, g = self.edge_blocks[str(idx)](
            z,edge_feats,edge_mag,strength=self.edge_strength
        )
        self._edge_diag[f"edge_res_p{idx}"] = r
        self._edge_diag[f"edge_gate_p{idx}"] = g
        return y

    def body(self, data):
        img = data["image_m"]
        # Existing pretrained two-scale PVTv2-B4 + MHSIU2 path is unchanged.
        large = data["image_l"]  # identical input interface to original ZoomNeXt
        l_feats = self.normalize_encoder(large)
        m_feats = self.normalize_encoder(img)

        if self.edge_residual:
            edge_mag = self.sobel(img)
            edge_feat = self.edge_stem(img,edge_mag)
        else:
            edge_mag = edge_feat = None
        self._edge_diag = {}

        l5,m5=self.tra_5(l_feats[3]),self.tra_5(m_feats[3])
        z5=self.siu_5(l=l5,m=m5)
        z5=self._edge_inject(5,z5,edge_feat,edge_mag)
        x=self.hmu_5(z5)

        l4,m4=self.tra_4(l_feats[2]),self.tra_4(m_feats[2])
        z4=self.siu_4(l=l4,m=m4)
        z4=self._edge_inject(4,z4,edge_feat,edge_mag)
        x=self.hmu_4(z4 + resize_to(x,tgt_hw=z4.shape[-2:]))

        l3,m3=self.tra_3(l_feats[1]),self.tra_3(m_feats[1])
        z3=self.siu_3(l=l3,m=m3)
        z3=self._edge_inject(3,z3,edge_feat,edge_mag)
        x=self.hmu_3(z3 + resize_to(x,tgt_hw=z3.shape[-2:]))

        l2,m2=self.tra_2(l_feats[0]),self.tra_2(m_feats[0])
        z2=self.siu_2(l=l2,m=m2)
        z2=self._edge_inject(2,z2,edge_feat,edge_mag)
        x=self.hmu_2(z2 + resize_to(x,tgt_hw=z2.shape[-2:]))

        x=self.tra_1(x)
        logits=self.predictor(x)
        if logits.shape[-2:]!=img.shape[-2:]:
            logits=F.interpolate(logits,size=img.shape[-2:],mode="bilinear",align_corners=False)
        return logits

    def forward(self,data,iter_percentage=1,**kwargs):
        del kwargs
        logits=self.body(data)
        if not self.training:
            return logits
        mask=self._prepare_mask(data["mask"],logits)
        bce=F.binary_cross_entropy_with_logits(logits.float(),mask.float())

        # UAL encourages confident decisions. Limit to boundary of pseudo mask.
        probs=torch.sigmoid(logits.float())
        uncertainty=4.0*probs*(1-probs)
        band = self._boundary_band(mask) if self.ual_scope=="edge" else None
        if self.ual_scope=="none":
            raw_ual=bce.detach().new_zeros(())
            ual_coef=0.0
        elif self.ual_scope=="full":
            raw_ual=uncertainty.flatten(1).mean(dim=1).mean()
            ual_coef=self._ual_coef(iter_percentage)
        else:
            coverage=band.flatten(1).sum(dim=1)
            per_sample=(uncertainty*band).flatten(1).sum(dim=1)/coverage.clamp_min(1.0)
            valid=(coverage>0).float()
            if self.edge_ual_clean_only:
                pool_id=data.get("pool_id")
                if pool_id is None:
                    raise KeyError("edge_ual_clean_only requires pool_id (0 clean, 1 noisy)")
                pool_id=pool_id.reshape(-1).to(device=valid.device)
                if pool_id.numel()!=valid.numel():
                    raise ValueError("pool_id batch size mismatch")
                valid=valid*(pool_id==0).float()
            raw_ual=(per_sample*valid).sum()/valid.sum().clamp_min(1.0)
            ual_coef=self._ual_coef(iter_percentage)
        loss_ual=self.ual_weight*ual_coef*raw_ual
        total=bce+loss_ual
        items={
            "total":total.detach(),"bce":bce.detach(),
            "ual":loss_ual.detach(),"ual_raw":raw_ual.detach(),
            "ual_coef":logits.new_tensor(ual_coef).detach(),
            "edge_band_ratio":(band.detach().float().mean() if band is not None else logits.new_zeros(())),
        }
        items.update(self._edge_diag)
        return {
            "logits":logits,"vis":{"sal":logits.sigmoid()},
            "loss":total,"loss_items":items,
            "loss_str":f"L:{total.detach().item():.4f} BCE:{bce.detach().item():.4f} E-UAL:{loss_ual.detach().item():.4f}",
        }


# Identical semantic feature extractor in all controls; only two toggles differ.
class PvtV2B4_Zoom_BCE_Control(_ZoomEdgeBase):
    def __init__(self,**kwargs):
        kwargs.pop("edge_residual",None); kwargs.pop("ual_scope",None)
        super().__init__(edge_residual=False,ual_scope="none",**kwargs)

class PvtV2B4_Zoom_FullUAL_Control(_ZoomEdgeBase):
    def __init__(self,**kwargs):
        kwargs.pop("edge_residual",None); kwargs.pop("ual_scope",None)
        super().__init__(edge_residual=False,ual_scope="full",**kwargs)

class PvtV2B4_Zoom_EdgeUAL(_ZoomEdgeBase):
    def __init__(self,**kwargs):
        kwargs.pop("edge_residual",None); kwargs.pop("ual_scope",None)
        super().__init__(edge_residual=False,ual_scope="edge",**kwargs)

class PvtV2B4_Zoom_EdgeRes_BCE(_ZoomEdgeBase):
    def __init__(self,**kwargs):
        kwargs.pop("edge_residual",None); kwargs.pop("ual_scope",None)
        super().__init__(edge_residual=True,ual_scope="none",**kwargs)

class PvtV2B4_Zoom_EdgeRes_FullUAL(_ZoomEdgeBase):
    def __init__(self,**kwargs):
        kwargs.pop("edge_residual",None); kwargs.pop("ual_scope",None)
        super().__init__(edge_residual=True,ual_scope="full",**kwargs)

class PvtV2B4_Zoom_EdgeRes_EdgeUAL(_ZoomEdgeBase):
    def __init__(self,**kwargs):
        kwargs.pop("edge_residual",None); kwargs.pop("ual_scope",None)
        super().__init__(edge_residual=True,ual_scope="edge",**kwargs)

__all__=[
    "PvtV2B4_Zoom_BCE_Control",
    "PvtV2B4_Zoom_FullUAL_Control",
    "PvtV2B4_Zoom_EdgeUAL",
    "PvtV2B4_Zoom_EdgeRes_BCE",
    "PvtV2B4_Zoom_EdgeRes_FullUAL",
    "PvtV2B4_Zoom_EdgeRes_EdgeUAL",
]
