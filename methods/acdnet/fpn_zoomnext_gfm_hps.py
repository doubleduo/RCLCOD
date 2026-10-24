# -*- coding: utf-8 -*-
"""Full GFM + HPS ablations on the original FPN + C2--C5 MHSIU model.

The original 1.0x/1.5x PVT and four-level MHSIU path is unchanged.  The first
FPN makes the coarse prediction and propagates foreground/background gradients
back to the four MHSIU features.  GFM optionally modulates those Z2--Z5
features, after which a second FPN produces the refined pyramid.  HPS builds
foreground/background prototypes from refined P5/P4/P3, updates them by
hierarchical cross-attention, and classifies the high-resolution P2 embedding.

Exported controlled variants:
  * FullRefine: second full FPN decoder only (extra-compute control).
  * GFM_All: FullRefine + GFM on Z2--Z5.
  * HPS: FullRefine + gradient-guided HPS, without GFM.
  * GFM_HPS: FullRefine + GFM on Z2--Z5 + gradient-guided HPS.

This is an implementation consistent with the published architecture figure;
it is not claimed to be the authors' unavailable line-by-line implementation.
"""

from contextlib import nullcontext
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fpn_zoomnext import PvtV2B4_FPN_ZoomNeXt
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


from ..zoomnext.ops import ConvBNReLU


def _masked_average_pool(feature, mask, eps=1e-6):
    """Soft masked average pooling: BCHW + B1HW -> BC."""
    mask = F.interpolate(
        mask,
        size=feature.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    numerator = (feature * mask).sum(dim=(2, 3))
    denominator = mask.sum(dim=(2, 3)).clamp_min(eps)
    return numerator / denominator


class PrototypeCrossAttentionLayer(nn.Module):
    """Two prototype queries attend to one feature-pyramid level."""

    def __init__(self, dim, num_heads=4, ffn_ratio=4.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        hidden = int(dim * ffn_ratio)
        self.query_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, queries, feature):
        tokens = feature.flatten(2).transpose(1, 2)
        query_update, _ = self.cross_attention(
            query=self.query_norm(queries),
            key=self.token_norm(tokens),
            value=self.token_norm(tokens),
            need_weights=False,
        )
        queries = queries + query_update
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class HierarchicalPrototypeDecoder(nn.Module):
    """Gradient-guided initialization + P5->P4->P3 prototype prompting."""

    def __init__(
        self,
        dim,
        num_heads=4,
        levels=("p5", "p4", "p3"),
    ):
        super().__init__()
        self.levels = tuple(levels)
        if self.levels != ("p5", "p4", "p3"):
            raise ValueError(
                "The controlled HPS implementation expects levels "
                "('p5', 'p4', 'p3')."
            )

        # Separate foreground/background weights over P5/P4/P3.
        # Zero initialization starts from an equal average over scales.
        self.level_logits = nn.Parameter(
            torch.zeros(2, len(self.levels))
        )

        self.layers = nn.ModuleList(
            [
                PrototypeCrossAttentionLayer(
                    dim=dim,
                    num_heads=num_heads,
                )
                for _ in self.levels
            ]
        )

        self.pixel_embedding = ConvBNReLU(
            dim, dim, 3, 1, 1
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(10.0), dtype=torch.float32)
        )

    @staticmethod
    def _gradient_channel_gate(gradient):
        """Convert a BCHW attribution gradient into a stable BC gate."""
        importance = gradient.detach().float().abs().mean(dim=(2, 3))
        importance = importance / importance.mean(
            dim=1, keepdim=True
        ).clamp_min(1e-6)

        # Uniform gradients produce a gate of exactly one.
        return 2.0 * torch.sigmoid(importance - 1.0)

    def _initial_queries(
        self,
        features,
        coarse_probability,
        grad_fg=None,
        grad_bg=None,
    ):
        fg_prototypes = []
        bg_prototypes = []

        for level in self.levels:
            feature = features[level]
            foreground = _masked_average_pool(
                feature,
                coarse_probability.detach(),
            )
            background = _masked_average_pool(
                feature,
                1.0 - coarse_probability.detach(),
            )

            if grad_fg is not None and grad_bg is not None:
                foreground = foreground * self._gradient_channel_gate(
                    grad_fg[level]
                ).to(dtype=foreground.dtype)
                background = background * self._gradient_channel_gate(
                    grad_bg[level]
                ).to(dtype=background.dtype)

            fg_prototypes.append(foreground)
            bg_prototypes.append(background)

        fg_stack = torch.stack(fg_prototypes, dim=1)  # B,L,C
        bg_stack = torch.stack(bg_prototypes, dim=1)  # B,L,C
        level_weights = torch.softmax(self.level_logits, dim=1)

        q_fg = (
            fg_stack
            * level_weights[0].view(1, -1, 1).to(fg_stack.dtype)
        ).sum(dim=1)
        q_bg = (
            bg_stack
            * level_weights[1].view(1, -1, 1).to(bg_stack.dtype)
        ).sum(dim=1)

        return torch.stack([q_fg, q_bg], dim=1)  # B,2,C

    def forward(
        self,
        features,
        coarse_probability,
        grad_fg=None,
        grad_bg=None,
    ):
        queries = self._initial_queries(
            features=features,
            coarse_probability=coarse_probability,
            grad_fg=grad_fg,
            grad_bg=grad_bg,
        )

        # Hierarchical prompting: deep semantics -> middle -> shallow detail.
        for level, layer in zip(self.levels, self.layers):
            queries = layer(queries, features[level])

        pixel = F.normalize(
            self.pixel_embedding(features["p2"]),
            dim=1,
        )
        queries = F.normalize(queries, dim=-1)

        similarities = torch.einsum(
            "bchw,bkc->bkhw",
            pixel,
            queries,
        )
        scale = self.logit_scale.exp().clamp(max=100.0)

        # Binary foreground-vs-background prototype classifier.
        prototype_logits = scale * (
            similarities[:, 0:1] - similarities[:, 1:2]
        )
        return prototype_logits, queries


class PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_HPS_Base(
    PvtV2B4_FPN_ZoomNeXt
):
    """Configurable base for full GFM/HPS component ablations."""

    _FEATURE_LEVELS = ("z2", "z3", "z4", "z5")
    _PROTOTYPE_LEVELS = ("p5", "p4", "p3")
    _PROTOTYPE_GRAD_LEVELS = ("z5", "z4", "z3")

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        siu_groups=4,
        clean_only_highres=False,
        use_gfm=True,
        use_hps=True,
        gradient_guided_prototypes=True,
        gfm_reduction=4,
        prototype_heads=4,
        confidence_power=2.0,
        coarse_aux_weight=0.4,
        prototype_aux_weight=0.2,
        **kwargs,
    ):
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

        if confidence_power <= 0:
            raise ValueError("confidence_power must be > 0")
        if coarse_aux_weight < 0 or prototype_aux_weight < 0:
            raise ValueError("Auxiliary loss weights must be >= 0")

        self.use_gfm = bool(use_gfm)
        self.use_hps = bool(use_hps)
        self.gradient_guided_prototypes = bool(
            gradient_guided_prototypes
        )
        self.confidence_power = float(confidence_power)
        self.coarse_aux_weight = float(coarse_aux_weight)
        self.prototype_aux_weight = (
            float(prototype_aux_weight) if self.use_hps else 0.0
        )

        if self.use_gfm:
            self.gfms = nn.ModuleDict(
                {
                    level: GradientFeatureModulator(
                        channels=fpn_dim,
                        reduction=gfm_reduction,
                    )
                    for level in self._FEATURE_LEVELS
                }
            )

        # All four controlled variants use this same second FPN and residual
        # prediction head.  Therefore GFM/HPS gains cannot be attributed only
        # to adding a decoder.  The zero-initialized last layer also makes each
        # model start from exactly the original coarse prediction.
        self.refine_5 = ConvBNReLU(fpn_dim, fpn_dim, 3, 1, 1)
        self.refine_4 = ConvBNReLU(fpn_dim, fpn_dim, 3, 1, 1)
        self.refine_3 = ConvBNReLU(fpn_dim, fpn_dim, 3, 1, 1)
        self.refine_2 = ConvBNReLU(fpn_dim, fpn_dim, 3, 1, 1)
        self.residual_predictor = nn.Sequential(
            ConvBNReLU(fpn_dim, 32, 3, 1, 1),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        nn.init.zeros_(self.residual_predictor[-1].weight)
        if self.residual_predictor[-1].bias is not None:
            nn.init.zeros_(self.residual_predictor[-1].bias)

        if self.use_hps:
            self.hps_decoder = HierarchicalPrototypeDecoder(
                dim=fpn_dim,
                num_heads=prototype_heads,
                levels=self._PROTOTYPE_LEVELS,
            )
            # Zero starts from the exact coarse prediction, while the
            # prototype auxiliary loss trains HPS from the first iteration.
            self.prototype_fusion_scale = nn.Parameter(torch.zeros(1))

    def _encode_mhsiu(self, data):
        """Unchanged original full C2--C5 dual-scale MHSIU encoder."""
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
                    "Training pool_id must contain only 0 or 1"
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

    def _coarse_fpn(self, z_feats, m_feats):
        z2, z3, z4, z5 = z_feats
        m2, m3, m4, _ = m_feats

        p5 = self.smooth_5(z5)
        p4 = self.smooth_4(z4 + self._resize_like(p5, m4))
        p3 = self.smooth_3(z3 + self._resize_like(p4, m3))
        p2 = self.smooth_2(z2 + self._resize_like(p3, m2))
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}

    def _refined_fpn(self, z_map):
        """Second FPN shared by every controlled ablation variant."""
        r5 = self.refine_5(z_map["z5"])
        r4 = self.refine_4(
            z_map["z4"] + self._resize_like(r5, z_map["z4"])
        )
        r3 = self.refine_3(
            z_map["z3"] + self._resize_like(r4, z_map["z3"])
        )
        r2 = self.refine_2(
            z_map["z2"] + self._resize_like(r3, z_map["z2"])
        )
        return {"p2": r2, "p3": r3, "p4": r4, "p5": r5}

    def _confidence_scores(self, coarse_logits):
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

    def _prediction_gradients(self, coarse_logits, feature_map, levels):
        selected = [feature_map[level] for level in levels]
        fg_score, bg_score = self._confidence_scores(coarse_logits)

        fg_values = torch.autograd.grad(
            fg_score,
            selected,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        bg_values = torch.autograd.grad(
            bg_score,
            selected,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

        grad_fg = {
            level: value.detach()
            for level, value in zip(levels, fg_values)
        }
        grad_bg = {
            level: value.detach()
            for level, value in zip(levels, bg_values)
        }
        return grad_fg, grad_bg

    def _required_gradient_levels(self):
        levels = set()
        if self.use_gfm:
            levels.update(self._FEATURE_LEVELS)
        if self.use_hps and self.gradient_guided_prototypes:
            levels.update(self._PROTOTYPE_GRAD_LEVELS)
        return tuple(
            level
            for level in self._FEATURE_LEVELS
            if level in levels
        )

    def _decode(self, z_feats, m_feats):
        required_levels = self._required_gradient_levels()
        outer_grad_enabled = torch.is_grad_enabled()
        need_internal_graph = bool(required_levels)

        context = (
            torch.enable_grad()
            if need_internal_graph and not outer_grad_enabled
            else nullcontext()
        )

        with context:
            if need_internal_graph and not outer_grad_enabled:
                z_feats = tuple(
                    z.detach().requires_grad_(True) for z in z_feats
                )

            z_map = dict(zip(self._FEATURE_LEVELS, z_feats))
            coarse_features = self._coarse_fpn(z_feats, m_feats)
            coarse_logits = self.predictor(coarse_features["p2"])
            coarse_probability = torch.sigmoid(coarse_logits.float())

            grad_fg = {}
            grad_bg = {}
            if required_levels:
                grad_fg, grad_bg = self._prediction_gradients(
                    coarse_logits=coarse_logits,
                    feature_map=z_map,
                    levels=required_levels,
                )

            modulated_z = dict(z_map)
            if self.use_gfm:
                for level in self._FEATURE_LEVELS:
                    modulated_z[level] = self.gfms[level](
                        z_map[level],
                        grad_fg[level],
                        grad_bg[level],
                    )

            # Figure-consistent path: gradient modulation precedes the second
            # pixel decoder, rather than operating on its already-decoded P2.
            refined = self._refined_fpn(modulated_z)
            residual_logits = self.residual_predictor(refined["p2"])
            final_logits = coarse_logits + residual_logits

            prototype_logits = None
            if self.use_hps:
                if self.gradient_guided_prototypes:
                    hps_fg = {
                        "p5": grad_fg["z5"],
                        "p4": grad_fg["z4"],
                        "p3": grad_fg["z3"],
                    }
                    hps_bg = {
                        "p5": grad_bg["z5"],
                        "p4": grad_bg["z4"],
                        "p3": grad_bg["z3"],
                    }
                else:
                    hps_fg = None
                    hps_bg = None
                prototype_logits, _ = self.hps_decoder(
                    features=refined,
                    coarse_probability=coarse_probability,
                    grad_fg=hps_fg,
                    grad_bg=hps_bg,
                )
                final_logits = final_logits + (
                    torch.tanh(self.prototype_fusion_scale)
                    * prototype_logits
                )

        if not outer_grad_enabled:
            final_logits = final_logits.detach()
            coarse_logits = coarse_logits.detach()
            if prototype_logits is not None:
                prototype_logits = prototype_logits.detach()

        return final_logits, coarse_logits, prototype_logits

    def _forward_logits(self, data):
        image_m = data["image_m"]
        z_feats, m_feats = self._encode_mhsiu(data)
        final_logits, coarse_logits, prototype_logits = self._decode(
            z_feats,
            m_feats,
        )

        size = image_m.shape[-2:]
        final_logits = F.interpolate(
            final_logits,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        coarse_logits = F.interpolate(
            coarse_logits,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        if prototype_logits is not None:
            prototype_logits = F.interpolate(
                prototype_logits,
                size=size,
                mode="bilinear",
                align_corners=False,
            )

        return final_logits, coarse_logits, prototype_logits

    def body(self, data):
        final_logits, _, _ = self._forward_logits(data)
        return final_logits

    def forward(self, data, iter_percentage=1, **kwargs):
        del iter_percentage, kwargs
        final_logits, coarse_logits, prototype_logits = (
            self._forward_logits(data)
        )

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
            final_logits, target, reduction="mean"
        )
        coarse_bce = F.binary_cross_entropy_with_logits(
            coarse_logits, target, reduction="mean"
        )

        numerator = final_bce + self.coarse_aux_weight * coarse_bce
        denominator = 1.0 + self.coarse_aux_weight
        prototype_bce = None

        if prototype_logits is not None and self.prototype_aux_weight > 0:
            prototype_bce = F.binary_cross_entropy_with_logits(
                prototype_logits,
                target,
                reduction="mean",
            )
            numerator = (
                numerator
                + self.prototype_aux_weight * prototype_bce
            )
            denominator += self.prototype_aux_weight

        total_loss = numerator / denominator

        loss_items = {
            "bce": final_bce.detach(),
            "coarse_bce": coarse_bce.detach(),
            "total": total_loss.detach(),
        }
        loss_text = (
            f"L:{total_loss.detach().item():.4f} "
            f"BCE:{final_bce.detach().item():.4f} "
            f"C:{coarse_bce.detach().item():.4f}"
        )

        vis = {
            "sal": final_logits.sigmoid(),
            "coarse": coarse_logits.sigmoid(),
        }
        if prototype_bce is not None:
            loss_items["prototype_bce"] = prototype_bce.detach()
            loss_text += f" P:{prototype_bce.detach().item():.4f}"
            vis["prototype"] = prototype_logits.sigmoid()

        return {
            "logits": final_logits,
            "vis": vis,
            "loss": total_loss,
            "loss_items": loss_items,
            "loss_str": loss_text,
        }


class PvtV2B4_FPN_ZoomNeXt_MHSIU_FullRefine(
    PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_HPS_Base
):
    """Compute control: full second FPN decoder, no GFM and no HPS."""

    def __init__(self, **kwargs):
        kwargs.pop("use_gfm", None)
        kwargs.pop("use_hps", None)
        kwargs.pop("gradient_guided_prototypes", None)
        super().__init__(
            use_gfm=False,
            use_hps=False,
            gradient_guided_prototypes=False,
            **kwargs,
        )


class PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_All(
    PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_HPS_Base
):
    """Full second FPN decoder + GFM on Z2/Z3/Z4/Z5, without HPS."""

    def __init__(self, **kwargs):
        kwargs.pop("use_gfm", None)
        kwargs.pop("use_hps", None)
        kwargs.pop("gradient_guided_prototypes", None)
        super().__init__(
            use_gfm=True,
            use_hps=False,
            gradient_guided_prototypes=False,
            **kwargs,
        )


class PvtV2B4_FPN_ZoomNeXt_MHSIU_HPS(
    PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_HPS_Base
):
    """Second FPN + gradient-guided HPS, without GFM."""

    def __init__(self, **kwargs):
        kwargs.pop("use_gfm", None)
        kwargs.pop("use_hps", None)
        kwargs.pop("gradient_guided_prototypes", None)
        super().__init__(
            use_gfm=False,
            use_hps=True,
            gradient_guided_prototypes=True,
            **kwargs,
        )


class PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_HPS(
    PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_HPS_Base
):
    """Full model: Z2--Z5 GFM + second FPN + hierarchical prototypes."""

    def __init__(self, **kwargs):
        kwargs.pop("use_gfm", None)
        kwargs.pop("use_hps", None)
        kwargs.pop("gradient_guided_prototypes", None)
        super().__init__(
            use_gfm=True,
            use_hps=True,
            gradient_guided_prototypes=True,
            **kwargs,
        )


__all__ = [
    "PrototypeCrossAttentionLayer",
    "HierarchicalPrototypeDecoder",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_HPS_Base",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU_FullRefine",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_All",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU_HPS",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU_GFM_HPS",
]
