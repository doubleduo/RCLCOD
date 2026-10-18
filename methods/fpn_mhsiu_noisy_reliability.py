"""FPN-MHSIU with reliability weighting on the Noisy pool only.

Network:
    PvtV2B4_FPN_ZoomNeXt_MHSIU (four-level, dual-scale MHSIU2; no RGPU)

Supervision:
    Clean: original full-pixel BCE.
    Noisy: q-weighted BCE with optional pixel reliability U.

The network body is inherited unchanged from ``methods.fpn_zoomnext``.  This
file only replaces the training loss so the comparison with the plain-FPN
reliability experiment changes one architectural variable: MHSIU.
"""

import torch
import torch.nn.functional as F

from .fpn_zoomnext import PvtV2B4_FPN_ZoomNeXt_MHSIU


MODES = ("none", "image", "pixel", "both")


def noisy_reliable_bce(
    logits,
    target,
    pool_id,
    q=None,
    probability=None,
    mode="both",
    eps=1e-6,
):
    """Batch-mean BCE: Clean unchanged, Noisy weighted by q and/or U.

    Args:
        logits: ``[B, 1, H, W]`` model logits.
        target: hard pseudo mask, broadcast-compatible with logits.
        pool_id: ``[B]`` where Clean=0 and Noisy=1.
        q: image reliability in ``[0, 1]``. Used only for Noisy rows.
        probability: aligned SAM probability map ``[B, 1, h, w]``.
        mode: ``none``, ``image``, ``pixel`` or ``both``.

    ``U=(2P-1)^2`` reweights pixels and is normalized inside each image.
    ``q`` is not normalized across the batch, so unreliable Noisy images
    genuinely contribute a smaller gradient while Clean keeps coefficient 1.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode}")
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape[0] == 0:
        raise ValueError("logits must have shape [B,1,H,W] with B>0")

    z = logits.float()
    y = target.to(device=z.device, dtype=torch.float32)
    if y.ndim == 3:
        y = y.unsqueeze(1)
    if y.ndim != 4 or y.shape[:2] != z.shape[:2]:
        raise ValueError("target must match logits batch and channel dimensions")
    if y.shape[-2:] != z.shape[-2:]:
        y = F.interpolate(y, size=z.shape[-2:], mode="nearest")

    ids = pool_id.to(z.device).reshape(-1)
    if ids.numel() != z.shape[0] or not bool(((ids == 0) | (ids == 1)).all()):
        raise ValueError("pool_id must be [B], Clean=0, Noisy=1")
    noisy = ids == 1

    bce_map = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    plain = bce_map.flatten(1).mean(1)
    q_used = torch.ones_like(plain)
    u = torch.ones_like(z)

    if bool(noisy.any()) and mode in ("image", "both"):
        if q is None:
            raise KeyError("sam_q is required for Noisy image weighting")
        q_input = q.detach().to(device=z.device, dtype=torch.float32).reshape(-1)
        if q_input.numel() != z.shape[0]:
            raise ValueError("q must have shape [B]")
        q_noisy = q_input[noisy]
        valid_q = torch.isfinite(q_noisy) & (q_noisy >= 0) & (q_noisy <= 1)
        if not bool(valid_q.all()):
            raise ValueError("Noisy q must be finite and in [0,1]")
        q_used = torch.where(noisy, q_input, q_used)

    if bool(noisy.any()) and mode in ("pixel", "both"):
        if probability is None:
            raise KeyError("sam_probability is required for Noisy pixel weighting")
        p = probability.detach().to(device=z.device, dtype=torch.float32)
        if p.ndim != 4 or p.shape[:2] != z.shape[:2]:
            raise ValueError("probability must be [B,1,H,W]")
        pn = p[noisy]
        valid_p = torch.isfinite(pn) & (pn >= 0) & (pn <= 1)
        if not bool(valid_p.all()):
            raise ValueError("Noisy probability must be finite and in [0,1]")
        if pn.shape[-2:] != z.shape[-2:]:
            pn = F.interpolate(
                pn,
                size=z.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        u[noisy] = (2.0 * pn - 1.0).square()

    weighted = (bce_map * u).flatten(1).sum(1)
    weighted = weighted / u.flatten(1).sum(1).clamp_min(eps)
    per_image = torch.where(noisy, q_used * weighted, plain)
    loss = per_image.mean()

    def subset_mean(value, selector):
        return value.detach()[selector].sum() / selector.sum().clamp_min(1)

    stats = {
        "total": loss.detach(),
        "bce": plain.mean().detach(),
        "clean_bce": subset_mean(plain, ~noisy),
        "noisy_bce": subset_mean(plain, noisy),
        "noisy_weighted": subset_mean(per_image, noisy),
        "noisy_q": subset_mean(q_used, noisy),
        "noisy_u": subset_mean(u.flatten(1).mean(1), noisy),
    }
    return loss, stats


class PvtV2B4_FPN_MHSIU_NoisyReliable(PvtV2B4_FPN_ZoomNeXt_MHSIU):
    """Original FPN-MHSIU body plus Noisy-only reliability BCE."""

    def __init__(self, reliability_mode="both", **kwargs):
        super().__init__(**kwargs)
        if reliability_mode not in MODES:
            raise ValueError(f"reliability_mode must be one of {MODES}")
        self.reliability_mode = reliability_mode

    def forward(self, data, iter_percentage=1, **kwargs):
        del iter_percentage, kwargs
        logits = self.body(data=data)
        if not self.training:
            return logits

        loss, stats = noisy_reliable_bce(
            logits=logits,
            target=data["mask"],
            pool_id=data["pool_id"],
            q=data.get("sam_q"),
            probability=data.get("sam_probability"),
            mode=self.reliability_mode,
        )
        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid()},
            "loss": loss,
            "loss_items": stats,
            "loss_str": (
                f"L:{loss.detach().item():.4f} "
                f"C:{stats['clean_bce'].item():.4f} "
                f"N:{stats['noisy_weighted'].item():.4f} "
                f"Nq:{stats['noisy_q'].item():.3f} "
                f"Nu:{stats['noisy_u'].item():.3f}"
            ),
        }


__all__ = [
    "MODES",
    "noisy_reliable_bce",
    "PvtV2B4_FPN_MHSIU_NoisyReliable",
]
