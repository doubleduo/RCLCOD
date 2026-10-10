# -*- coding: utf-8 -*-
"""High/Middle UAL ablation on top of the UNCHANGED PVTv2-B4 FPN architecture.

Pool IDs supplied by RCLCOD basemain.ImageTrainDataset:
  clean/high = 0, noisy/middle = 1.

The binary cross entropy is FULL-MASK BCE for both pools in every variant.
Only the uncertainty-aware regularizer changes.  There is NO UAL on Unvalue
because Unvalue is not included in this experiment.

No trainable parameters are added, and inference is identical to the base FPN.
"""
import math

import torch
import torch.nn.functional as F
from .fpn_baseline import PvtV2B4_FPN_Baseline


class PvtV2B4_FPN_PoolUAL(PvtV2B4_FPN_Baseline):
    """Ablate high full UAL and middle none/full/low/edge UAL."""

    def __init__(
        self,
        high_ual_weight=0.20,
        middle_ual_mode="none",        # none | full | low | edge
        middle_ual_weight=0.05,
        edge_kernel=7,
        ual_schedule="cosine",        # cosine | fixed
        **kwargs,
    ):
        super().__init__(**kwargs)
        if middle_ual_mode not in ("none", "full", "low", "edge"):
            raise ValueError("middle_ual_mode must be none/full/low/edge")
        if ual_schedule not in ("cosine", "fixed"):
            raise ValueError("ual_schedule must be cosine or fixed")
        if edge_kernel < 3 or edge_kernel % 2 != 1:
            raise ValueError("edge_kernel must be odd and >=3")
        if high_ual_weight < 0 or middle_ual_weight < 0:
            raise ValueError("UAL weights must be >= 0")
        self.high_ual_weight = float(high_ual_weight)
        self.middle_ual_weight = float(middle_ual_weight)
        self.middle_ual_mode = str(middle_ual_mode)
        self.edge_kernel = int(edge_kernel)
        self.ual_schedule = str(ual_schedule)

    def _schedule_coef(self, iter_percentage):
        if self.ual_schedule == "fixed":
            return 1.0
        progress = max(0.0, min(1.0, float(iter_percentage)))
        return 0.5 * (1.0 - math.cos(math.pi * progress))

    def _edge_band(self, target):
        """Inside + outside of each pseudo-mask boundary. No GT used."""
        k = self.edge_kernel
        dilated = F.max_pool2d(target, kernel_size=k, stride=1, padding=k // 2)
        eroded = 1.0 - F.max_pool2d(1.0 - target, kernel_size=k, stride=1, padding=k // 2)
        return (dilated - eroded).clamp_(0.0, 1.0).detach()

    @staticmethod
    def _subset_mean(values, selector):
        count = selector.to(values.dtype).sum()
        return (values * selector.to(values.dtype)).sum() / count.clamp_min(1.0)

    def forward(self, data, iter_percentage=1, **kwargs):
        if not self.training:
            return self.body(data=data)

        z = self.body(data=data)
        y = data["mask"].float().to(device=z.device)
        if y.ndim == 3:
            y = y.unsqueeze(1)
        if y.shape[-2:] != z.shape[-2:]:
            y = F.interpolate(y, size=z.shape[-2:], mode="nearest")
        y = y.clamp(0.0, 1.0)
        if y.shape != z.shape:
            raise ValueError(f"mask/logits shape mismatch: {y.shape}, {z.shape}")

        pool = data.get("pool_id", None)
        if pool is None:
            raise KeyError("pool_id absent; use clean/noisy datasets via basemain_continuous.py")
        pool = pool.reshape(-1).to(z.device)
        if pool.shape[0] != z.shape[0] or not torch.all((pool == 0) | (pool == 1)):
            raise ValueError("Only clean=0 / noisy=1 supported; no Unvalue in this ablation")
        is_high = pool == 0
        is_middle = pool == 1

        # Compute BCE and UAL in fp32, robust under autocast.
        z32 = z.float()
        per_image_bce = F.binary_cross_entropy_with_logits(
            z32, y.float(), reduction="none"
        ).flatten(1).mean(dim=1)
        p = z32.sigmoid()
        u_map = 4.0 * p * (1.0 - p)
        full_ual = u_map.flatten(1).mean(dim=1)
        edge = self._edge_band(y.float())
        edge_pixels = edge.flatten(1).sum(dim=1)
        edge_ual = (u_map * edge).flatten(1).sum(dim=1) / edge_pixels.clamp_min(1.0)
        # Empty/full masks should not receive fake boundary UAL.
        edge_ual = torch.where(edge_pixels > 0, edge_ual, torch.zeros_like(edge_ual))
        edge_coverage = edge.flatten(1).mean(dim=1)

        progress_coef = self._schedule_coef(iter_percentage)
        high_term = progress_coef * self.high_ual_weight * full_ual
        if self.middle_ual_mode == "none":
            mid_raw = torch.zeros_like(full_ual)
            mid_coef = 0.0
        elif self.middle_ual_mode == "edge":
            mid_raw = edge_ual
            mid_coef = self.middle_ual_weight
        elif self.middle_ual_mode == "full":
            mid_raw = full_ual
            mid_coef = self.middle_ual_weight
        else:  # low: also full image, but with a smaller configured weight
            mid_raw = full_ual
            mid_coef = self.middle_ual_weight
        mid_term = progress_coef * mid_coef * mid_raw
        selected_term = torch.where(is_high, high_term, mid_term)
        total = (per_image_bce + selected_term).mean()

        info = {
            "bce": per_image_bce.mean().detach(),
            "high_bce": self._subset_mean(per_image_bce.detach(), is_high),
            "middle_bce": self._subset_mean(per_image_bce.detach(), is_middle),
            "high_ual_raw": self._subset_mean(full_ual.detach(), is_high),
            "middle_ual_raw": self._subset_mean(mid_raw.detach(), is_middle),
            "high_ual_term": self._subset_mean(high_term.detach(), is_high),
            "middle_ual_term": self._subset_mean(mid_term.detach(), is_middle),
            "edge_coverage": self._subset_mean(edge_coverage.detach(), is_middle),
            "ual_coef": total.detach().new_tensor(progress_coef),
            "total": total.detach(),
        }
        return {
            "logits": z,
            "vis": {"sal": p},
            "loss": total,
            "loss_items": info,
            "loss_str": (
                f"L:{total.detach().item():.4f} "
                f"BCE:{info['bce'].item():.4f} "
                f"HUAL:{info['high_ual_term'].item():.4f} "
                f"MUAL:{info['middle_ual_term'].item():.4f} "
                f"edge_cov:{info['edge_coverage'].item():.3f}"
            ),
        }
