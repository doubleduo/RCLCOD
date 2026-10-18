# -*- coding: utf-8 -*-
"""Loss-only UAL ablations for the plain PVTv2-B4 FPN.

The encoder, FPN decoder and inference path are exactly inherited from
``PvtV2B4_FPN_Baseline``.  Only the training objective changes:

    L = L_bce + lambda(t) * L_ual

Two scopes are provided for a fair ablation:

* ``PvtV2B4_FPN_UAL``: UAL over the complete prediction map.
* ``PvtV2B4_FPN_EdgeRegionUAL``: UAL only in a target-derived boundary band.

The boundary selector is detached.  UAL therefore cannot move or enlarge its
own valid region.  It only sharpens predictions inside the fixed edge band.
"""

import math

import torch
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline


class PvtV2B4_FPN_UALBase(PvtV2B4_FPN_Baseline):
    """Plain FPN with uncertainty-aware regularization during training."""

    def __init__(
        self,
        ual_scope="edge",          # full | edge
        ual_weight=0.20,
        edge_kernel=7,
        ual_start_ratio=0.40,
        ual_full_ratio=0.70,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if ual_scope not in ("full", "edge"):
            raise ValueError("ual_scope must be 'full' or 'edge'")
        if ual_weight < 0:
            raise ValueError("ual_weight must be non-negative")
        if edge_kernel < 3 or edge_kernel % 2 == 0:
            raise ValueError("edge_kernel must be an odd integer >= 3")
        if not 0.0 <= ual_start_ratio < ual_full_ratio <= 1.0:
            raise ValueError(
                "require 0 <= ual_start_ratio < ual_full_ratio <= 1"
            )

        self.ual_scope = str(ual_scope)
        self.ual_weight = float(ual_weight)
        self.edge_kernel = int(edge_kernel)
        self.ual_start_ratio = float(ual_start_ratio)
        self.ual_full_ratio = float(ual_full_ratio)

    @staticmethod
    def _prepare_target(target, logits):
        target = target.float().to(device=logits.device)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(
                target,
                size=logits.shape[-2:],
                mode="nearest",
            )
        target = target.clamp(0.0, 1.0)
        if target.shape != logits.shape:
            raise ValueError(
                f"mask/logits shape mismatch: {target.shape} vs {logits.shape}"
            )
        return target

    def _edge_band(self, target):
        """Symmetric inside/outside band around the pseudo-mask boundary."""
        binary = (target >= 0.5).to(target.dtype)
        k = self.edge_kernel
        dilated = F.max_pool2d(
            binary, kernel_size=k, stride=1, padding=k // 2
        )
        eroded = 1.0 - F.max_pool2d(
            1.0 - binary, kernel_size=k, stride=1, padding=k // 2
        )
        return (dilated - eroded).clamp(0.0, 1.0).detach()

    def _schedule(self, iter_percentage):
        """Delayed cosine ramp: zero early, then smoothly reach one."""
        progress = max(0.0, min(1.0, float(iter_percentage)))
        start = self.ual_start_ratio
        end = self.ual_full_ratio
        if progress <= start:
            return 0.0
        if progress >= end:
            return 1.0
        ratio = (progress - start) / (end - start)
        return 0.5 * (1.0 - math.cos(math.pi * ratio))

    @staticmethod
    def _uncertainty_map(logits):
        """UAL = 1-|2p-1|^2 = 4p(1-p); minimum at confident 0/1."""
        probability = logits.float().sigmoid()
        return 1.0 - (2.0 * probability - 1.0).square()

    @staticmethod
    def _masked_mean_per_image(values, valid):
        numerator = (values * valid).flatten(1).sum(dim=1)
        count = valid.flatten(1).sum(dim=1)
        mean = numerator / count.clamp_min(1.0)
        # Empty/full pseudo masks have no boundary and get exactly zero edge UAL.
        return torch.where(count > 0, mean, torch.zeros_like(mean)), count

    def forward(self, data, iter_percentage=1.0, **kwargs):
        del kwargs
        logits = self.body(data=data)
        if not self.training:
            return logits

        target = self._prepare_target(data["mask"], logits)

        # Compute losses in fp32 for stable AMP training.
        logits32 = logits.float()
        target32 = target.float()
        per_image_bce = F.binary_cross_entropy_with_logits(
            logits32, target32, reduction="none"
        ).flatten(1).mean(dim=1)

        uncertainty = self._uncertainty_map(logits32)
        edge = self._edge_band(target32)
        edge_ual, edge_count = self._masked_mean_per_image(uncertainty, edge)
        full_ual = uncertainty.flatten(1).mean(dim=1)

        if self.ual_scope == "edge":
            raw_ual = edge_ual
        else:
            raw_ual = full_ual

        schedule = self._schedule(iter_percentage)
        weighted_ual = self.ual_weight * schedule * raw_ual
        total = (per_image_bce + weighted_ual).mean()

        bce = per_image_bce.mean().detach()
        ual_raw = raw_ual.mean().detach()
        ual_term = weighted_ual.mean().detach()
        edge_ratio = edge.flatten(1).mean(dim=1).mean().detach()
        edge_valid = (edge_count > 0).float().mean().detach()

        return {
            "logits": logits,
            "vis": {
                "sal": logits.sigmoid(),
                "ual_edge": edge,
                "uncertainty": uncertainty.detach(),
            },
            "loss": total,
            "loss_items": {
                "bce": bce,
                "ual_raw": ual_raw,
                "ual": ual_term,
                "ual_coef": total.detach().new_tensor(schedule),
                "edge_ratio": edge_ratio,
                "edge_valid": edge_valid,
                "total": total.detach(),
            },
            "loss_str": (
                f"L:{total.detach().item():.4f} "
                f"BCE:{bce.item():.4f} "
                f"UAL:{ual_term.item():.4f} "
                f"RAW:{ual_raw.item():.4f} "
                f"UC:{schedule:.3f} "
                f"ER:{edge_ratio.item():.3f}"
            ),
        }


class PvtV2B4_FPN_UAL(PvtV2B4_FPN_UALBase):
    """A1: plain FPN + full-map UAL."""

    def __init__(self, **kwargs):
        kwargs.pop("ual_scope", None)
        super().__init__(ual_scope="full", **kwargs)


class PvtV2B4_FPN_EdgeRegionUAL(PvtV2B4_FPN_UALBase):
    """A2: plain FPN + UAL only inside the pseudo-mask edge band."""

    def __init__(self, **kwargs):
        kwargs.pop("ual_scope", None)
        super().__init__(ual_scope="edge", **kwargs)


class PvtV2B4_FPN_RegionUAL(PvtV2B4_FPN_EdgeRegionUAL):
    """Short model-name alias: Region-UAL means edge-band-only UAL here."""


__all__ = [
    "PvtV2B4_FPN_UALBase",
    "PvtV2B4_FPN_UAL",
    "PvtV2B4_FPN_EdgeRegionUAL",
    "PvtV2B4_FPN_RegionUAL",
]
