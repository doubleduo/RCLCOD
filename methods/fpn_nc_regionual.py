# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline, NoisyCODCurriculumLoss


class RegionUALNoisyCODLoss(NoisyCODCurriculumLoss):
    """Noisy-COD staged NC loss + late Region-UAL."""

    def __init__(
        self,
        q_switch_ratio=0.40,
        boundary_kernel=31,
        boundary_gain=5.0,
        region_kernel=15,
        region_ual_weight=0.20,
        region_ual_start_ratio=0.40,
        region_ual_full_ratio=0.70,
        eps=1e-6,
    ):
        super().__init__(
            q_switch_ratio=q_switch_ratio,
            boundary_kernel=boundary_kernel,
            boundary_gain=boundary_gain,
            eps=eps,
        )
        if region_kernel < 3 or region_kernel % 2 == 0:
            raise ValueError("region_kernel must be odd and >= 3")
        if not 0.0 <= region_ual_start_ratio < region_ual_full_ratio <= 1.0:
            raise ValueError("invalid Region-UAL schedule")
        self.region_kernel = int(region_kernel)
        self.region_ual_weight = float(region_ual_weight)
        self.region_ual_start_ratio = float(region_ual_start_ratio)
        self.region_ual_full_ratio = float(region_ual_full_ratio)

    def _make_region(self, target):
        pad = self.region_kernel // 2
        return F.max_pool2d(
            target,
            kernel_size=self.region_kernel,
            stride=1,
            padding=pad,
        ).clamp(0.0, 1.0).detach()

    def _region_ual(self, logits, region):
        prob = torch.sigmoid(logits)
        uncertainty = 1.0 - (2.0 * prob - 1.0).abs().pow(2)
        num = (uncertainty * region).sum(dim=(1, 2, 3))
        den = region.sum(dim=(1, 2, 3)).clamp_min(self.eps)
        return (num / den).mean()

    def _region_coef(self, progress):
        progress = float(progress)
        s = self.region_ual_start_ratio
        e = self.region_ual_full_ratio
        if progress <= s:
            return 0.0
        if progress >= e:
            return 1.0
        r = (progress - s) / max(e - s, 1e-6)
        return float(0.5 * (1.0 - math.cos(math.pi * r)))

    def forward(self, logits, target, iter_percentage):
        progress = float(iter_percentage)
        q = 2.0 if progress <= self.q_switch_ratio else 1.0

        wbce = self._weighted_bce(logits, target)
        nc = self._nc_term(logits, target, q=q)

        if q == 2.0:
            base_loss = nc + wbce
        else:
            base_loss = 2.0 * nc

        region = self._make_region(target)
        region_ual_raw = self._region_ual(logits, region)
        region_coef = self._region_coef(progress)
        region_ual = self.region_ual_weight * region_coef * region_ual_raw

        total = base_loss + region_ual
        return total, {
            "q": q,
            "nc": nc.detach(),
            "wbce": wbce.detach(),
            "base_loss": base_loss.detach(),
            "region_ual": region_ual.detach(),
            "region_ual_raw": region_ual_raw.detach(),
            "region_coef": logits.new_tensor(region_coef),
            "region_ratio": region.mean().detach(),
            "region": region,
        }


class PvtV2B4_FPN_NC_RegionUAL(PvtV2B4_FPN_Baseline):
    """Same FPN architecture, only the training objective changes."""

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        q_switch_ratio=0.40,
        boundary_kernel=31,
        boundary_gain=5.0,
        region_kernel=15,
        region_ual_weight=0.20,
        region_ual_start_ratio=0.40,
        region_ual_full_ratio=0.70,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.curriculum_loss = RegionUALNoisyCODLoss(
            q_switch_ratio=q_switch_ratio,
            boundary_kernel=boundary_kernel,
            boundary_gain=boundary_gain,
            region_kernel=region_kernel,
            region_ual_weight=region_ual_weight,
            region_ual_start_ratio=region_ual_start_ratio,
            region_ual_full_ratio=region_ual_full_ratio,
        )

    def forward(self, data, iter_percentage=1.0, **kwargs):
        del kwargs
        logits = self.body(data=data)
        if not self.training:
            return logits

        target = data["mask"].float()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
        target = target.clamp(0.0, 1.0)

        total, items = self.curriculum_loss(
            logits=logits,
            target=target,
            iter_percentage=iter_percentage,
        )

        q = float(items["q"])
        return {
            "logits": logits,
            "vis": {
                "sal": logits.sigmoid(),
                "region": items["region"],
            },
            "loss": total,
            "loss_items": {
                "total": total.detach(),
                "bce": items["wbce"],
                "wbce": items["wbce"],
                "nc": items["nc"],
                "base_loss": items["base_loss"],
                "q": logits.new_tensor(q),
                "region_ual": items["region_ual"],
                "region_ual_raw": items["region_ual_raw"],
                "region_coef": items["region_coef"],
                "region_ratio": items["region_ratio"],
            },
            "loss_str": (
                f"L:{total.detach().item():.4f} "
                f"NC:{items['nc'].item():.4f} "
                f"WBCE:{items['wbce'].item():.4f} "
                f"RUAL:{items['region_ual'].item():.4f} "
                f"RAW:{items['region_ual_raw'].item():.4f} "
                f"RC:{items['region_coef'].item():.3f} "
                f"RR:{items['region_ratio'].item():.3f} "
                f"Q:{q:.1f}"
            ),
        }


__all__ = ["RegionUALNoisyCODLoss", "PvtV2B4_FPN_NC_RegionUAL"]
