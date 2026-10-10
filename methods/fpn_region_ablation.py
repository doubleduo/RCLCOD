# -*- coding: utf-8 -*-
"""Minimal FPN region-supervision ablation for RCLCOD.

Add this file at ``methods/fpn_region_ablation.py`` and export the class from
``methods/__init__.py``. This module does not modify the FPN architecture.

Modes:
  baseline: exactly reuse the FPN baseline's BCE-only forward.
  seed:     High full BCE, Middle reliable-FG/reliable-BG BCE only.
  adaptive: seed + stop-gradient student-confidence gated uncertain boundary BCE.

IMPORTANT: The original PvtV2B4_FPN_Baseline does not have UAL, so Middle
receives NO UAL in all three modes, and High receives no UAL either.

For ``seed``/``adaptive``, set ``train.data.use_box=True`` in the config.
"""

import torch
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline


class PvtV2B4_FPN_RegionAblation(PvtV2B4_FPN_Baseline):
    """Identical FPN inference path; change loss on Middle only."""

    def __init__(
        self,
        region_mode="seed",
        region_kernel=5,
        box_margin_kernel=7,
        adaptive_scale=0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if region_mode not in ("baseline", "seed", "adaptive"):
            raise ValueError("region_mode must be baseline, seed, or adaptive")
        if region_kernel < 1 or region_kernel % 2 != 1:
            raise ValueError("region_kernel must be a positive odd integer")
        if box_margin_kernel < 1 or box_margin_kernel % 2 != 1:
            raise ValueError("box_margin_kernel must be a positive odd integer")
        if adaptive_scale < 0:
            raise ValueError("adaptive_scale must be non-negative")
        self.region_mode = str(region_mode)
        self.region_kernel = int(region_kernel)
        self.box_margin_kernel = int(box_margin_kernel)
        self.adaptive_scale = float(adaptive_scale)

    @staticmethod
    def _resize_binary(x, size):
        if x.shape[-2:] != size:
            x = F.interpolate(x.float(), size=size, mode="nearest")
        return (x > 0.5).float()

    @staticmethod
    def _masked_mean_per_image(values, valid):
        count = valid.flatten(1).sum(dim=1)
        mean = (values * valid).flatten(1).sum(dim=1) / count.clamp_min(1.0)
        return mean, count

    def forward(self, data, iter_percentage=1, **kwargs):
        if not self.training or self.region_mode == "baseline":
            # Keep original baseline loss and inference EXACTLY as implemented.
            return super().forward(data=data, iter_percentage=iter_percentage, **kwargs)

        logits = self.body(data=data)
        mask = self._resize_binary(data["mask"].to(logits.device), logits.shape[-2:])
        pool = data.get("pool_id")
        if pool is None:
            raise KeyError("Missing pool_id. Use basemain_continuous.py clean/noisy datasets.")
        pool = pool.to(logits.device).reshape(-1)
        if pool.shape[0] != logits.shape[0]:
            raise ValueError("pool_id batch size mismatch")
        if not torch.all((pool == 0) | (pool == 1)):
            raise ValueError("This minimal ablation supports High(pool_id=0) / Middle(pool_id=1) only")
        is_middle = pool == 1

        pixel_bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
        full_loss = pixel_bce.flatten(1).mean(1)
        bce_high = full_loss
        k = self.region_kernel
        # Robust binary-mask morphology on the network's spatial resolution.
        fg_core = (1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=k // 2)).clamp(0, 1)
        # Preserve extremely thin structures if they entirely disappear after erosion.
        has_fg = mask.flatten(1).sum(1) > 0
        no_fg_core = (fg_core.flatten(1).sum(1) == 0) & has_fg
        fg_core = torch.where(no_fg_core[:, None, None, None], mask, fg_core)

        box = data.get("box_mask")
        if box is None:
            raise KeyError("Middle needs box_mask: set train.data.use_box=True")
        box = self._resize_binary(box.to(logits.device), logits.shape[-2:])
        kb = self.box_margin_kernel
        # A slightly enlarged box, outside of which pixels are reliable BG.
        expanded_box = F.max_pool2d(box, kernel_size=kb, stride=1, padding=kb // 2)
        bg_reliable = (1.0 - expanded_box).clamp(0, 1)
        bg_reliable = bg_reliable * (1.0 - fg_core)

        # Box geometry is an immutable supervision constraint.
        fg_core = fg_core * box
        fg_loss, fg_n = self._masked_mean_per_image(pixel_bce, fg_core)
        bg_loss, bg_n = self._masked_mean_per_image(pixel_bce, bg_reliable)
        fg_ok, bg_ok = (fg_n > 0).float(), (bg_n > 0).float()
        # Balanced FG/BG losses; missing side does not create a fake zero penalty.
        seed_loss = (fg_loss * fg_ok + bg_loss * bg_ok) / (fg_ok + bg_ok).clamp_min(1)
        # An empty pseudo-mask has no reliable foreground. Do NOT silently promote
        # it to full-mask training; report its seed coverage so it can be rejected upstream.

        dynamic_loss = torch.zeros_like(seed_loss)
        mean_gate = logits.new_zeros(())
        band_ratio = logits.new_zeros(())
        if self.region_mode == "adaptive":
            # Supervise ONLY the uncertain boundary band within the annotated box.
            # The surrounding uncertain *interior* remains ignored.
            dilated_mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
            boundary_band = ((dilated_mask - fg_core).clamp(0, 1) * box)
            boundary_band = boundary_band * (1.0 - bg_reliable)
            with torch.no_grad():
                p = logits.detach().sigmoid()
                confidence = (2.0 * p - 1.0).abs()
                agreement = 1.0 - (p - mask).abs()
                gate = (confidence * agreement).square()
            # Crucially normalize by TOTAL eligible band size, not sum(gate),
            # so higher confidence genuinely increases the dynamic loss weight.
            # Normalize by all candidate boundary pixels, not sum of gates.
            dynamic_loss = (
                (pixel_bce * boundary_band * gate).flatten(1).sum(1)
                / boundary_band.flatten(1).sum(1).clamp_min(1.0)
            )
            mean_gate = (
                (boundary_band * gate).flatten(1).sum(1)
                / boundary_band.flatten(1).sum(1).clamp_min(1.0)
            )[is_middle].mean() if bool(is_middle.any()) else logits.new_zeros(())
            band_ratio = boundary_band.flatten(1).mean(1)[is_middle].mean() if bool(is_middle.any()) else logits.new_zeros(())

        middle_loss = seed_loss + self.adaptive_scale * dynamic_loss
        selected_loss = torch.where(is_middle, middle_loss, bce_high)
        total = selected_loss.mean()

        # NO UAL anywhere in the base FPN; specifically Middle never uses UAL.
        # Return scalar diagnostics understood by basemain_continuous.py logger.
        mid_present = is_middle.float().sum().clamp_min(1.0)
        avg_seed = (seed_loss.detach() * is_middle.float()).sum() / mid_present
        avg_dyn = (dynamic_loss.detach() * is_middle.float()).sum() / mid_present
        fg_coverage = (fg_core.flatten(1).mean(1) * is_middle.float()).sum() / mid_present
        bg_coverage = (bg_reliable.flatten(1).mean(1) * is_middle.float()).sum() / mid_present
        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid()},
            "loss": total,
            "loss_items": {
                "bce": total.detach(),
                "total": total.detach(),
                "high_bce": ((full_loss.detach() * (~is_middle).float()).sum() / (~is_middle).float().sum().clamp_min(1)),
                "middle_seed": avg_seed,
                "middle_dynamic": avg_dyn,
                "fg_coverage": fg_coverage.detach(),
                "bg_coverage": bg_coverage.detach(),
                "band_ratio": band_ratio.detach(),
                "band_gate": mean_gate.detach(),
                "ual": total.detach().new_zeros(()),
            },
            "loss_str": (
                f"L:{total.detach().item():.4f} "
                f"seed:{avg_seed.item():.4f} dyn:{avg_dyn.item():.4f} "
                f"fg_cov:{fg_coverage.item():.3f} bg_cov:{bg_coverage.item():.3f} "
                f"UAL:0"
            ),
        }
