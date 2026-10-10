# -*- coding: utf-8 -*-
"""
Reliability-conditioned denoising losses for PvtV2B4_FPN.

Design goals
------------
1) pool_id (Clean/Noisy) controls sample reliability;
2) reliability controls Noise-Correction loss behavior;
3) unreliable pseudo-mask boundaries are automatically weakened;
4) optional reliability-aware Spatial Coherence loss (RSC);
5) optional reliability-aware Boundary UAL (BUAL).

This file intentionally DOES NOT change the FPN architecture. Only the
training objective is changed, so A2/A3/A4 are strict loss ablations.

pool_id convention
------------------
0: Clean
1: Noisy
-1: unknown/all -> treated as Clean for backward compatibility
"""

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline


class ReliabilityConditionedLoss(nn.Module):
    """Reliable NC + reliable spatial coherence + boundary UAL.

    The default ``q_mode='binary'`` is the cleanest first ablation:
        Clean -> q=2,  L = NC(q=2) + WBCE
        Noisy -> q=1,  L = 2 * NC(q=1)

    Sample contribution is additionally scaled by reliability:
        Clean reliability = 1.0
        Noisy reliability = 0.5 (default)

    Boundary reliability is weaker than interior reliability. Internally we use
        pixel_gate = 1 - boundary * (1 - sample_reliability)
    and then use sample_reliability again when reducing the batch. Therefore,
    for noisy reliability r=0.5:
        interior effective weight ~= 0.5
        boundary effective weight ~= 0.25

    ``q_mode='continuous'`` is the V2 extension:
        q_i = 1 + r_i
        base_i = (2-r_i) * NC(q_i) + r_i * WBCE
    which continuously interpolates between MAE-like robust learning and
    IoU-form/strong fitting.
    """

    def __init__(
        self,
        clean_reliability: float = 1.0,
        noisy_reliability: float = 0.5,
        q_mode: str = "binary",
        wbce_kernel: int = 31,
        wbce_boundary_gain: float = 5.0,
        boundary_kernel: int = 7,
        sc_weight: float = 0.0,
        sc_alpha: float = 1.0,
        sc_max_level: int = 1,
        sc_loss_clip: float = 10.0,
        bual_weight: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        if not 0.0 <= clean_reliability <= 1.0:
            raise ValueError("clean_reliability must be in [0,1]")
        if not 0.0 <= noisy_reliability <= 1.0:
            raise ValueError("noisy_reliability must be in [0,1]")
        if q_mode not in ("binary", "continuous"):
            raise ValueError("q_mode must be 'binary' or 'continuous'")
        if wbce_kernel < 3 or wbce_kernel % 2 == 0:
            raise ValueError("wbce_kernel must be odd and >= 3")
        if boundary_kernel < 3 or boundary_kernel % 2 == 0:
            raise ValueError("boundary_kernel must be odd and >= 3")
        if sc_max_level < 1:
            raise ValueError("sc_max_level must be >= 1")
        if sc_weight < 0 or bual_weight < 0:
            raise ValueError("loss weights must be non-negative")

        self.clean_reliability = float(clean_reliability)
        self.noisy_reliability = float(noisy_reliability)
        self.q_mode = str(q_mode)

        self.wbce_kernel = int(wbce_kernel)
        self.wbce_boundary_gain = float(wbce_boundary_gain)
        self.boundary_kernel = int(boundary_kernel)

        self.sc_weight = float(sc_weight)
        self.sc_alpha = float(sc_alpha)
        self.sc_max_level = int(sc_max_level)
        self.sc_loss_clip = float(sc_loss_clip)

        self.bual_weight = float(bual_weight)
        self.eps = float(eps)

    # ------------------------------------------------------------------
    # Reliability
    # ------------------------------------------------------------------
    def _sample_reliability(self, pool_id: torch.Tensor, batch_size: int, device) -> torch.Tensor:
        if pool_id is None:
            return torch.full(
                (batch_size,),
                self.clean_reliability,
                device=device,
                dtype=torch.float32,
            )

        pool_id = pool_id.reshape(-1).to(device=device)
        if pool_id.numel() != batch_size:
            raise ValueError(
                f"pool_id length {pool_id.numel()} != batch size {batch_size}"
            )

        rel = torch.full(
            (batch_size,),
            self.clean_reliability,
            device=device,
            dtype=torch.float32,
        )
        rel = torch.where(
            pool_id == 1,
            torch.full_like(rel, self.noisy_reliability),
            rel,
        )
        # pool_id == -1 / other values remain Clean-like for compatibility.
        return rel.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Boundary uncertainty mask
    # ------------------------------------------------------------------
    def _boundary_band(self, target: torch.Tensor) -> torch.Tensor:
        """Morphological boundary band from the pseudo mask."""
        k = self.boundary_kernel
        pad = k // 2

        dilated = F.max_pool2d(target, kernel_size=k, stride=1, padding=pad)
        eroded = -F.max_pool2d(-target, kernel_size=k, stride=1, padding=pad)
        return (dilated - eroded).clamp(0.0, 1.0).detach()

    def _pixel_gate(self, boundary: torch.Tensor, sample_rel: torch.Tensor) -> torch.Tensor:
        """Interior=1, boundary=r. Batch reduction later adds another factor r.

        Thus noisy boundary is automatically weaker than noisy interior.
        """
        r = sample_rel[:, None, None, None]
        return (1.0 - boundary * (1.0 - r)).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Reliable NC
    # ------------------------------------------------------------------
    def _weighted_bce_per_sample(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pixel_gate: torch.Tensor,
    ) -> torch.Tensor:
        pad = self.wbce_kernel // 2
        local_mean = F.avg_pool2d(
            target,
            kernel_size=self.wbce_kernel,
            stride=1,
            padding=pad,
        )
        structure_weight = (
            1.0
            + self.wbce_boundary_gain * torch.abs(local_mean - target)
        )
        weight = structure_weight * pixel_gate

        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        num = (weight * bce_map).sum(dim=(1, 2, 3))
        den = weight.sum(dim=(1, 2, 3)).clamp_min(self.eps)
        return num / den

    def _nc_per_sample(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        q: torch.Tensor,
        pixel_gate: torch.Tensor,
    ) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        q_map = q[:, None, None, None]

        error = torch.abs(prob - target).clamp_min(self.eps)
        numerator = (
            pixel_gate * error.pow(q_map)
        ).sum(dim=(1, 2, 3))

        # Weighted soft-union preserves the original NC region normalization,
        # but weakens unreliable boundary pixels.
        union_map = prob + target - prob * target
        denominator = (
            pixel_gate * union_map
        ).sum(dim=(1, 2, 3)).clamp_min(self.eps)

        return numerator / denominator

    def _reliable_nc(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        sample_rel: torch.Tensor,
        pixel_gate: torch.Tensor,
        pool_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.q_mode == "binary":
            if pool_id is None:
                is_noisy = torch.zeros_like(sample_rel, dtype=torch.bool)
            else:
                is_noisy = pool_id.reshape(-1).to(logits.device) == 1

            q = torch.where(
                is_noisy,
                torch.ones_like(sample_rel),
                torch.full_like(sample_rel, 2.0),
            )
            nc = self._nc_per_sample(logits, target, q, pixel_gate)
            wbce = self._weighted_bce_per_sample(logits, target, pixel_gate)

            # Faithful Noisy-COD endpoints, selected per sample rather than
            # switched globally by epoch.
            base = torch.where(
                is_noisy,
                2.0 * nc,
                nc + wbce,
            )
            return base, nc, wbce, q

        # Continuous reliability-conditioned extension.
        q = 1.0 + sample_rel
        nc = self._nc_per_sample(logits, target, q, pixel_gate)
        wbce = self._weighted_bce_per_sample(logits, target, pixel_gate)

        # Endpoints:
        # r=1 -> 1*NC(q=2) + 1*WBCE
        # r=0 -> 2*NC(q=1) + 0*WBCE
        base = (2.0 - sample_rel) * nc + sample_rel * wbce
        return base, nc, wbce, q

    # ------------------------------------------------------------------
    # Reliable Spatial Coherence (SCLoss-inspired)
    # ------------------------------------------------------------------
    @staticmethod
    def _ring_offsets(level: int) -> Iterable[Tuple[int, int]]:
        """Chebyshev ring offsets for kth-level adjacency."""
        for dy in range(-level, level + 1):
            for dx in range(-level, level + 1):
                if dy == 0 and dx == 0:
                    continue
                if max(abs(dy), abs(dx)) == level:
                    yield dy, dx

    @staticmethod
    def _paired_views(x: torch.Tensor, dy: int, dx: int):
        """Return valid center/neighbor crops without circular wrap."""
        h, w = x.shape[-2:]

        if dy >= 0:
            y0_i, y1_i = 0, h - dy
            y0_j, y1_j = dy, h
        else:
            y0_i, y1_i = -dy, h
            y0_j, y1_j = 0, h + dy

        if dx >= 0:
            x0_i, x1_i = 0, w - dx
            x0_j, x1_j = dx, w
        else:
            x0_i, x1_i = -dx, w
            x0_j, x1_j = 0, w + dx

        return (
            x[..., y0_i:y1_i, x0_i:x1_i],
            x[..., y0_j:y1_j, x0_j:x1_j],
        )

    def _spatial_coherence_per_sample(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pixel_gate: torch.Tensor,
    ) -> torch.Tensor:
        if self.sc_weight <= 0:
            return logits.new_zeros((logits.shape[0],), dtype=torch.float32)

        prob = torch.sigmoid(logits)
        single = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        batch = logits.shape[0]
        total_num = logits.new_zeros((batch,), dtype=torch.float32)
        total_den = logits.new_zeros((batch,), dtype=torch.float32)

        for level in range(1, self.sc_max_level + 1):
            level_weight = float(0.5 ** (level - 1))

            for dy, dx in self._ring_offsets(level):
                pi, pj = self._paired_views(prob, dy, dx)
                yi, yj = self._paired_views(target, dy, dx)
                si, _ = self._paired_views(single, dy, dx)
                gi, gj = self._paired_views(pixel_gate, dy, dx)

                # SCLoss Eq.(2)-style mutual response:
                # only foreground-foreground pairs are positive connectivity;
                # all other pairs are treated as non-foreground connectivity.
                # AMP-safe mutual-response BCE.  ``pi`` and ``pj`` are
                # probabilities, so do NOT feed them directly to
                # binary_cross_entropy under autocast.  Convert the pair
                # probability to a logit in fp32, then use the numerically
                # stable BCE-with-logits implementation.  This is
                # mathematically equivalent to BCE(pair_prob, pair_label).
                pair_prob = (pi.float() * pj.float()).clamp(
                    self.eps,
                    1.0 - self.eps,
                )
                pair_label = (yi.float() * yj.float()).clamp(0.0, 1.0)
                pair_logit = torch.logit(pair_prob)
                mutual = F.binary_cross_entropy_with_logits(
                    pair_logit,
                    pair_label,
                    reduction="none",
                )

                # Default pairwise regularizer from the paper: exp(-p_i p_j).
                # Keep this branch in fp32 as well; pair_loss is promoted to
                # fp32 and safely back-propagates through the AMP graph.
                pair_reg = torch.exp(-pair_prob)

                pair_loss = si / (
                    mutual + self.sc_alpha * pair_reg + self.eps
                )
                if self.sc_loss_clip > 0:
                    pair_loss = pair_loss.clamp_max(self.sc_loss_clip)

                # Reliability-aware adaptation: a pair is only as reliable as
                # its weaker endpoint.
                pair_gate = torch.minimum(gi, gj)

                num = (pair_gate * pair_loss).sum(dim=(1, 2, 3))
                den = pair_gate.sum(dim=(1, 2, 3)).clamp_min(self.eps)

                total_num = total_num + level_weight * num
                total_den = total_den + level_weight * den

        return total_num / total_den.clamp_min(self.eps)

    # ------------------------------------------------------------------
    # Reliability-aware Boundary UAL
    # ------------------------------------------------------------------
    def _boundary_ual_per_sample(
        self,
        logits: torch.Tensor,
        boundary: torch.Tensor,
        pixel_gate: torch.Tensor,
    ) -> torch.Tensor:
        if self.bual_weight <= 0:
            return logits.new_zeros((logits.shape[0],), dtype=torch.float32)

        prob = torch.sigmoid(logits)
        uncertainty = 1.0 - (2.0 * prob - 1.0).abs().pow(2)

        # Only reliable boundary pixels are encouraged to leave p~=0.5.
        weight = boundary * pixel_gate
        num = (weight * uncertainty).sum(dim=(1, 2, 3))
        den = weight.sum(dim=(1, 2, 3)).clamp_min(self.eps)
        return num / den

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pool_id: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Force loss arithmetic to fp32 under AMP for stable log/power terms.
        logits_f = logits.float()
        target_f = target.float().clamp(0.0, 1.0)

        batch = logits_f.shape[0]
        sample_rel = self._sample_reliability(
            pool_id=pool_id,
            batch_size=batch,
            device=logits_f.device,
        )

        boundary = self._boundary_band(target_f)
        pixel_gate = self._pixel_gate(boundary, sample_rel)

        base, nc, wbce, q = self._reliable_nc(
            logits=logits_f,
            target=target_f,
            sample_rel=sample_rel,
            pixel_gate=pixel_gate,
            pool_id=pool_id,
        )

        rsc = self._spatial_coherence_per_sample(
            logits=logits_f,
            target=target_f,
            pixel_gate=pixel_gate,
        )

        bual = self._boundary_ual_per_sample(
            logits=logits_f,
            boundary=boundary,
            pixel_gate=pixel_gate,
        )

        per_sample_total = (
            base
            + self.sc_weight * rsc
            + self.bual_weight * bual
        )

        # Clean/Noisy ID controls the final sample contribution.
        # This is independent from the curriculum sampler mass.
        total = (
            (sample_rel * per_sample_total).sum()
            / sample_rel.sum().clamp_min(self.eps)
        )

        clean_mask = (
            torch.ones_like(sample_rel, dtype=torch.bool)
            if pool_id is None
            else pool_id.reshape(-1).to(logits_f.device) != 1
        )
        noisy_mask = ~clean_mask

        def safe_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            if mask.any():
                return x[mask].mean()
            return x.new_zeros(())

        items = {
            "total": total.detach(),
            "rnc": (
                (sample_rel * base).sum()
                / sample_rel.sum().clamp_min(self.eps)
            ).detach(),
            "nc": (
                (sample_rel * nc).sum()
                / sample_rel.sum().clamp_min(self.eps)
            ).detach(),
            "wbce": safe_mean(wbce, clean_mask).detach(),
            "rsc": safe_mean(rsc, torch.ones_like(clean_mask)).detach(),
            "bual": safe_mean(bual, torch.ones_like(clean_mask)).detach(),
            "q": q.mean().detach(),
            "reliability": sample_rel.mean().detach(),
            "clean_frac": clean_mask.float().mean().detach(),
            "noisy_frac": noisy_mask.float().mean().detach(),
            "boundary_ratio": boundary.mean().detach(),
            "boundary_gate": pixel_gate.mean().detach(),
            # visual-only maps
            "boundary": boundary.detach(),
            "pixel_gate": pixel_gate.detach(),
        }
        return total, items


class PvtV2B4_FPN_Reliable(PvtV2B4_FPN_Baseline):
    """PVTv2-B4 FPN with reliability-conditioned training only."""

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        clean_reliability=1.0,
        noisy_reliability=0.5,
        q_mode="binary",
        wbce_kernel=31,
        wbce_boundary_gain=5.0,
        boundary_kernel=7,
        sc_weight=0.0,
        sc_alpha=1.0,
        sc_max_level=1,
        sc_loss_clip=10.0,
        bual_weight=0.0,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

        self.reliable_loss = ReliabilityConditionedLoss(
            clean_reliability=clean_reliability,
            noisy_reliability=noisy_reliability,
            q_mode=q_mode,
            wbce_kernel=wbce_kernel,
            wbce_boundary_gain=wbce_boundary_gain,
            boundary_kernel=boundary_kernel,
            sc_weight=sc_weight,
            sc_alpha=sc_alpha,
            sc_max_level=sc_max_level,
            sc_loss_clip=sc_loss_clip,
            bual_weight=bual_weight,
        )

    def forward(self, data, iter_percentage=1.0, **kwargs):
        # No epoch/stage switch is used by the loss. Keep argument for trainer compatibility.
        del iter_percentage, kwargs

        logits = self.body(data=data)
        if not self.training:
            return logits

        if "mask" not in data:
            raise KeyError("Training requires data['mask']")

        target = data["mask"].float()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(
                target,
                size=logits.shape[-2:],
                mode="nearest",
            )

        pool_id = data.get("pool_id", None)
        total, items = self.reliable_loss(
            logits=logits,
            target=target,
            pool_id=pool_id,
        )

        return {
            "logits": logits,
            "vis": {
                "sal": logits.sigmoid(),
                "boundary": items["boundary"],
                "rel_gate": items["pixel_gate"],
            },
            "loss": total,
            "loss_items": {
                # compatibility key expected by basemain_continuous.py
                "bce": items["wbce"],
                "total": items["total"],
                "rnc": items["rnc"],
                "nc": items["nc"],
                "wbce": items["wbce"],
                "rsc": items["rsc"],
                "bual": items["bual"],
                "q": items["q"],
                "reliability": items["reliability"],
                "clean_frac": items["clean_frac"],
                "noisy_frac": items["noisy_frac"],
                "boundary_ratio": items["boundary_ratio"],
                "boundary_gate": items["boundary_gate"],
            },
            "loss_str": (
                f"L:{total.detach().item():.4f} "
                f"RNC:{items['rnc'].item():.4f} "
                f"RSC:{items['rsc'].item():.4f} "
                f"BUAL:{items['bual'].item():.4f} "
                f"R:{items['reliability'].item():.3f} "
                f"Q:{items['q'].item():.2f}"
            ),
        }


__all__ = [
    "ReliabilityConditionedLoss",
    "PvtV2B4_FPN_Reliable",
]
