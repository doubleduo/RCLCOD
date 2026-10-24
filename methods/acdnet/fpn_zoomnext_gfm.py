# -*- coding: utf-8 -*-
"""FPN + full C2--C5 MHSIU + ACDNet-style gradient feedback refinement.

This file keeps the original dual-scale MHSIU path in fpn_zoomnext.py:
the 1.0x and 1.5x images both run through C2--C5 and MHSIU2 fuses all four
levels.  RGPU is disabled.  A coarse FPN prediction is then used to obtain
foreground/background gradients for P2/P3, which modulate a lightweight
residual refinement head.

Three controlled variants are exported:
  * Refine: the extra refinement head without gradient modulation.
  * GFM_P2: identical head, with gradient modulation at P2 only.
  * GFM_P23: identical head, with gradient modulation at P2 and P3.

The final residual predictor and every GFM output scale are zero-initialized,
so all variants start from exactly the original coarse FPN+MHSIU prediction.
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_zoomnext import PvtV2B4_FPN_ZoomNeXt
from .zoomnext.ops import ConvBNReLU


class GradientFeatureModulator(nn.Module):
    """Channel modulation from foreground/background prediction gradients.

    ``grad_fg`` and ``grad_bg`` are treated as detached attribution signals;
    the model therefore avoids expensive and unstable second-order gradients.
    The learnable residual scale starts at zero, preserving the input feature.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(int(channels) // int(reduction), 8)

        self.channel_logits = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        # alpha controls the contribution of gradient contrast.
        # residual_scale=0 makes the whole module an exact identity at init.
        self.alpha = nn.Parameter(torch.ones(1))
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, feature, grad_fg, grad_bg):
        grad_delta = (
            grad_fg.detach().float() - grad_bg.detach().float()
        ).mean(dim=(2, 3), keepdim=True)

        # Per-sample normalization prevents a few large gradients from
        # saturating the channel gate under AMP.
        denominator = grad_delta.abs().mean(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        grad_delta = torch.tanh(grad_delta / denominator)
        grad_delta = grad_delta.to(dtype=feature.dtype)

        attention_logits = self.channel_logits(feature)
        gate = 2.0 * torch.sigmoid(
            attention_logits + self.alpha * grad_delta
        )

        scale = torch.tanh(self.residual_scale)
        return feature + scale * feature * (gate - 1.0)


class PvtV2B4_FPN_ZoomNeXt_MHSIU_GradientRefine(
    PvtV2B4_FPN_ZoomNeXt
):
    """Common implementation for controlled GFM ablations.

    Args:
        gfm_levels: Any subset of ("p2", "p3").  An empty tuple keeps the
            same refinement decoder but disables gradient modulation.
        coarse_aux_weight: Weight of the coarse prediction loss.  Final and
            coarse losses are normalized by ``1 + coarse_aux_weight`` to keep
            the initial loss scale comparable with the original BCE baseline.
        confidence_power: Soft confidence sharpening used to construct the
            foreground/background scores.  This avoids brittle hard thresholds
            and empty confident regions at the beginning of training.
    """

    _VALID_GFM_LEVELS = {"p2", "p3"}

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        siu_groups=4,
        clean_only_highres=False,
        gfm_levels=("p2", "p3"),
        gfm_reduction=4,
        coarse_aux_weight=0.4,
        confidence_power=2.0,
        **kwargs,
    ):
        # These variants are specifically FPN + MHSIU; RGPU stays disabled.
        kwargs.pop("use_mhsiu", None)
        kwargs.pop("use_rgpu", None)
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            siu_groups=siu_groups,
            use_mhsiu=True,
            use_rgpu=False,
            clean_only_highres=clean_only_highres,
            **kwargs,
        )

        self.gfm_levels = tuple(str(x).lower() for x in gfm_levels)
        invalid = set(self.gfm_levels) - self._VALID_GFM_LEVELS
        if invalid:
            raise ValueError(
                f"Unsupported gfm_levels={sorted(invalid)}; "
                "only 'p2' and 'p3' are supported."
            )
        if len(set(self.gfm_levels)) != len(self.gfm_levels):
            raise ValueError("gfm_levels must not contain duplicates")
        if coarse_aux_weight < 0:
            raise ValueError("coarse_aux_weight must be >= 0")
        if confidence_power <= 0:
            raise ValueError("confidence_power must be > 0")

        self.coarse_aux_weight = float(coarse_aux_weight)
        self.confidence_power = float(confidence_power)

        self.gfms = nn.ModuleDict(
            {
                level: GradientFeatureModulator(
                    channels=fpn_dim,
                    reduction=gfm_reduction,
                )
                for level in self.gfm_levels
            }
        )

        # A small second-stage decoder shared by Refine/P2/P23 variants.
        # Keeping it identical is essential for a fair GFM ablation.
        self.refine_3 = ConvBNReLU(
            fpn_dim, fpn_dim, 3, 1, 1
        )
        self.refine_2 = ConvBNReLU(
            fpn_dim, fpn_dim, 3, 1, 1
        )
        self.residual_predictor = nn.Sequential(
            ConvBNReLU(fpn_dim, 32, 3, 1, 1),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # final_logits == coarse_logits at initialization.
        nn.init.zeros_(self.residual_predictor[-1].weight)
        if self.residual_predictor[-1].bias is not None:
            nn.init.zeros_(self.residual_predictor[-1].bias)

    def _encode_mhsiu(self, data):
        """Original full C2--C5 1.0x/1.5x MHSIU path, without RGPU."""
        image_m = data["image_m"]
        m2, m3, m4, m5 = self.normalize_encoder(image_m)

        z2 = self.lateral_2(m2)
        z3 = self.lateral_3(m3)
        z4 = self.lateral_4(m4)
        z5 = self.lateral_5(m5)

        if self.training and self.clean_only_highres:
            pool_id = data.get("pool_id")
            if pool_id is None:
                raise KeyError(
                    "clean_only_highres=True requires data['pool_id'] "
                    "during training (0=Clean, 1=Noisy)."
                )
            pool_id = pool_id.reshape(-1).to(device=image_m.device)
            if pool_id.numel() != image_m.shape[0]:
                raise ValueError(
                    "pool_id must have one value per batch sample"
                )
            if not torch.all((pool_id == 0) | (pool_id == 1)):
                raise ValueError(
                    "Training pool_id must contain only 0 (Clean) "
                    "or 1 (Noisy)"
                )

            clean_indices = torch.nonzero(
                pool_id == 0, as_tuple=True
            )[0]
            if clean_indices.numel() > 0:
                image_l = self._large_image(data, image_m)
                z2, z3, z4, z5 = self._fuse_large_subset(
                    (z2, z3, z4, z5),
                    image_l,
                    clean_indices,
                )
        else:
            image_l = self._large_image(data, image_m)
            l2, l3, l4, l5 = self.normalize_encoder(image_l)
            z2 = self.siu_2(l=self.lateral_2(l2), m=z2)
            z3 = self.siu_3(l=self.lateral_3(l3), m=z3)
            z4 = self.siu_4(l=self.lateral_4(l4), m=z4)
            z5 = self.siu_5(l=self.lateral_5(l5), m=z5)

        return (z2, z3, z4, z5), (m2, m3, m4, m5)

    def _confidence_scores(self, coarse_logits):
        """Differentiable fg/bg scores with detached soft confidence masks."""
        probability = torch.sigmoid(coarse_logits.float())

        fg_weight = probability.detach().pow(self.confidence_power)
        bg_weight = (1.0 - probability.detach()).pow(
            self.confidence_power
        )

        fg_score = (
            (probability * fg_weight).flatten(1).sum(dim=1)
            / fg_weight.flatten(1).sum(dim=1).clamp_min(1e-6)
        ).sum()
        bg_score = (
            ((1.0 - probability) * bg_weight)
            .flatten(1)
            .sum(dim=1)
            / bg_weight.flatten(1).sum(dim=1).clamp_min(1e-6)
        ).sum()

        return fg_score, bg_score

    def _prediction_gradients(self, coarse_logits, features):
        """Return detached foreground/background gradients for features."""
        fg_score, bg_score = self._confidence_scores(coarse_logits)

        grad_fg = torch.autograd.grad(
            outputs=fg_score,
            inputs=features,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        grad_bg = torch.autograd.grad(
            outputs=bg_score,
            inputs=features,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

        return (
            tuple(grad.detach() for grad in grad_fg),
            tuple(grad.detach() for grad in grad_bg),
        )

    def _decode_with_gradient_refinement(self, z_feats, m_feats):
        """Coarse FPN followed by optional P2/P3 gradient modulation."""
        z2, z3, z4, z5 = z_feats
        m2, m3, m4, _ = m_feats

        # Evaluation is normally wrapped in torch.no_grad().  Actual
        # prediction gradients still require a graph, so rebuild only the
        # lightweight FPN/refinement part with gradient tracking.  The encoder
        # and MHSIU stay graph-free during inference.
        outer_grad_enabled = torch.is_grad_enabled()
        context = nullcontext() if outer_grad_enabled else torch.enable_grad()

        with context:
            if not outer_grad_enabled:
                z2 = z2.detach().requires_grad_(True)
                z3 = z3.detach().requires_grad_(True)
                z4 = z4.detach().requires_grad_(True)
                z5 = z5.detach().requires_grad_(True)

            p5 = self.smooth_5(z5)
            p4 = self.smooth_4(
                z4 + self._resize_like(p5, m4)
            )
            p3 = self.smooth_3(
                z3 + self._resize_like(p4, m3)
            )
            p2 = self.smooth_2(
                z2 + self._resize_like(p3, m2)
            )

            coarse_logits = self.predictor(p2)

            refined_p2 = p2
            refined_p3 = p3

            if self.gfm_levels:
                level_features = {"p2": p2, "p3": p3}
                selected = [
                    level_features[level]
                    for level in self.gfm_levels
                ]
                grad_fg, grad_bg = self._prediction_gradients(
                    coarse_logits,
                    selected,
                )

                for index, level in enumerate(self.gfm_levels):
                    modulated = self.gfms[level](
                        level_features[level],
                        grad_fg[index],
                        grad_bg[index],
                    )
                    if level == "p2":
                        refined_p2 = modulated
                    elif level == "p3":
                        refined_p3 = modulated

            refined_p3 = self.refine_3(refined_p3)
            refined_p2 = self.refine_2(
                refined_p2
                + self._resize_like(refined_p3, refined_p2)
            )

            residual_logits = self.residual_predictor(refined_p2)
            final_logits = coarse_logits + residual_logits

        if not outer_grad_enabled:
            final_logits = final_logits.detach()
            coarse_logits = coarse_logits.detach()

        return final_logits, coarse_logits

    def _forward_logits(self, data):
        image_m = data["image_m"]
        z_feats, m_feats = self._encode_mhsiu(data)
        final_logits, coarse_logits = (
            self._decode_with_gradient_refinement(z_feats, m_feats)
        )

        output_size = image_m.shape[-2:]
        final_logits = F.interpolate(
            final_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        coarse_logits = F.interpolate(
            coarse_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return final_logits, coarse_logits

    def body(self, data):
        final_logits, _ = self._forward_logits(data)
        return final_logits

    def forward(self, data, iter_percentage=1, **kwargs):
        del iter_percentage, kwargs
        final_logits, coarse_logits = self._forward_logits(data)

        if not self.training:
            return final_logits

        target = data["mask"].float()
        if target.shape[-2:] != final_logits.shape[-2:]:
            target = F.interpolate(
                target,
                size=final_logits.shape[-2:],
                mode="nearest",
            )

        final_bce = F.binary_cross_entropy_with_logits(
            final_logits,
            target,
            reduction="mean",
        )
        coarse_bce = F.binary_cross_entropy_with_logits(
            coarse_logits,
            target,
            reduction="mean",
        )

        # Normalization keeps the initial gradient/loss scale close to the
        # original single-head BCE model.
        total_loss = (
            final_bce + self.coarse_aux_weight * coarse_bce
        ) / (1.0 + self.coarse_aux_weight)

        return {
            "logits": final_logits,
            "vis": {
                "sal": final_logits.sigmoid(),
                "coarse": coarse_logits.sigmoid(),
            },
            "loss": total_loss,
            "loss_items": {
                # Preserve the trainer's expected ``bce`` field.
                "bce": final_bce.detach(),
                "coarse_bce": coarse_bce.detach(),
                "total": total_loss.detach(),
            },
            "loss_str": (
                f"L:{total_loss.detach().item():.4f} "
                f"BCE:{final_bce.detach().item():.4f} "
                f"C:{coarse_bce.detach().item():.4f}"
            ),
        }


class PvtV2B4_FPN_ZoomNeXt_MHSIU_Refine(
    PvtV2B4_FPN_ZoomNeXt_MHSIU_GradientRefine
):
    """Control: extra residual refinement head, without GFM."""

    def __init__(self, **kwargs):
        kwargs.pop("gfm_levels", None)
        super().__init__(gfm_levels=(), **kwargs)


class PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_P2(
    PvtV2B4_FPN_ZoomNeXt_MHSIU_GradientRefine
):
    """Gradient feature modulation at P2 only."""

    def __init__(self, **kwargs):
        kwargs.pop("gfm_levels", None)
        super().__init__(gfm_levels=("p2",), **kwargs)


class PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_P23(
    PvtV2B4_FPN_ZoomNeXt_MHSIU_GradientRefine
):
    """Recommended variant: gradient feature modulation at P2 and P3."""

    def __init__(self, **kwargs):
        kwargs.pop("gfm_levels", None)
        super().__init__(gfm_levels=("p2", "p3"), **kwargs)


__all__ = [
    "GradientFeatureModulator",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU_GradientRefine",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU_Refine",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_P2",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_P23",
]
