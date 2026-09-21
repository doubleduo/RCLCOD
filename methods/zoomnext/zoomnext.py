import abc
import logging

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.efficientnet import EfficientNet
from ..backbone.pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b3, pvt_v2_eff_b4, pvt_v2_eff_b5
from .layers import MHSIU, RGPU, SimpleASPP,MHSIU2
from .ops import ConvBNReLU, PixelNormalizer, resize_to

LOGGER = logging.getLogger("main")


class _ZoomNeXt_Base(nn.Module):
    @staticmethod
    def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = 0, 1

        ual_coef = 1.0
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            if method == "linear":
                ratio = (max_coef - min_coef) / (max_point - min_point)
                ual_coef = ratio * (iter_percentage - min_point)
            elif method == "cos":
                perc = (iter_percentage - min_point) / (max_point - min_point)
                normalized_coef = (1 - np.cos(perc * np.pi)) / 2
                ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
        return ual_coef

    @abc.abstractmethod
    def body(self):
        pass

    def forward(self, data, iter_percentage=1, **kwargs):
        logits = self.body(data=data)

        if self.training:
            mask = data["mask"]
            prob = logits.sigmoid()

            losses = []
            loss_str = []

            sod_loss = F.binary_cross_entropy_with_logits(input=logits, target=mask, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"bce: {sod_loss.item():.5f}")

            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")
            return dict(vis=dict(sal=prob), loss=sum(losses), loss_str=" ".join(loss_str))
        else:
            return logits

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "clip." in name:
                    param.requires_grad = False
                    param_groups["fixed"].append(param)
                else:
                    param_groups["retrained"].append(param)
        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups



    def __init__(
        self, pretrained=True, num_frames=1, input_norm=True, mid_dim=64, siu_groups=4, hmu_groups=6, **kwargs
    ):
        super().__init__()
        self.encoder = timm.create_model(
            model_name="resnet50", features_only=True, out_indices=range(5), pretrained=False
        )
        if pretrained:
            params = torch.hub.load_state_dict_from_url(
                url="https://github.com/lartpang/Archieve/releases/download/pretrained-model/resnet50-timm.pth",
                map_location="cpu",
            )
            self.encoder.load_state_dict(params, strict=False)

        self.tra_5 = SimpleASPP(in_dim=2048, out_dim=mid_dim)
        self.siu_5 = MHSIU(mid_dim, siu_groups)
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_4 = ConvBNReLU(1024, mid_dim, 3, 1, 1)
        self.siu_4 = MHSIU(mid_dim, siu_groups)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_3 = ConvBNReLU(512, mid_dim, 3, 1, 1)
        self.siu_3 = MHSIU(mid_dim, siu_groups)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_2 = ConvBNReLU(256, mid_dim, 3, 1, 1)
        self.siu_2 = MHSIU(mid_dim, siu_groups)
        self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_1 = ConvBNReLU(64, mid_dim, 3, 1, 1)
        self.siu_1 = MHSIU(mid_dim, siu_groups)
        self.hmu_1 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        c1, c2, c3, c4, c5 = self.encoder(x)
        return c1, c2, c3, c4, c5

    def body(self, data):
        l_trans_feats = self.normalize_encoder(data["image_l"])
        m_trans_feats = self.normalize_encoder(data["image_m"])
        s_trans_feats = self.normalize_encoder(data["image_s"])

        l, m, s = (
            self.tra_5(l_trans_feats[4]),
            self.tra_5(m_trans_feats[4]),
            self.tra_5(s_trans_feats[4]),
        )
        lms = self.siu_5(l=l, m=m, s=s)
        x = self.hmu_5(lms)

        l, m, s = (
            self.tra_4(l_trans_feats[3]),
            self.tra_4(m_trans_feats[3]),
            self.tra_4(s_trans_feats[3]),
        )
        lms = self.siu_4(l=l, m=m, s=s)
        x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = (
            self.tra_3(l_trans_feats[2]),
            self.tra_3(m_trans_feats[2]),
            self.tra_3(s_trans_feats[2]),
        )
        lms = self.siu_3(l=l, m=m, s=s)
        x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = (
            self.tra_2(l_trans_feats[1]),
            self.tra_2(m_trans_feats[1]),
            self.tra_2(s_trans_feats[1]),
        )
        lms = self.siu_2(l=l, m=m, s=s)
        x = self.hmu_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = (
            self.tra_1(l_trans_feats[0]),
            self.tra_1(m_trans_feats[0]),
            self.tra_1(s_trans_feats[0]),
        )
        lms = self.siu_1(l=l, m=m, s=s)
        x = self.hmu_1(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        return self.predictor(x)
class PvtV2B2_ZoomNeXt(_ZoomNeXt_Base):
    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
    ):
        super().__init__()
        self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)

        self.embed_dims = self.encoder.embed_dims
        self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.siu_5 = MHSIU2(mid_dim, siu_groups)
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.siu_4 = MHSIU2(mid_dim, siu_groups)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.siu_3 = MHSIU2(mid_dim, siu_groups)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.siu_2 = MHSIU2(mid_dim, siu_groups)
        self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvBNReLU(64, mid_dim, 3, 1, 1)
        )

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint)

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder(x)
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]
        return c2, c3, c4, c5

    def body(self, data):
        l_trans_feats = self.normalize_encoder(data["image_l"])
        m_trans_feats = self.normalize_encoder(data["image_m"])


        l, m,= self.tra_5(l_trans_feats[3]), self.tra_5(m_trans_feats[3])
        lms = self.siu_5(l=l, m=m)
        x = self.hmu_5(lms)

        l, m= self.tra_4(l_trans_feats[2]), self.tra_4(m_trans_feats[2])
        lms = self.siu_4(l=l, m=m)
        x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m= self.tra_3(l_trans_feats[1]), self.tra_3(m_trans_feats[1])
        lms = self.siu_3(l=l, m=m)
        x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m,= self.tra_2(l_trans_feats[0]), self.tra_2(m_trans_feats[0])
        lms = self.siu_2(l=l, m=m)
        x = self.hmu_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        x = self.tra_1(x)
        return self.predictor(x)  








class PvtV2B3_ZoomNeXt(PvtV2B2_ZoomNeXt):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b3(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B4_ZoomNeXt(PvtV2B2_ZoomNeXt):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b4(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B5_ZoomNeXt(PvtV2B2_ZoomNeXt):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)
 
class ConvNeXtB_ZoomNeXt(PvtV2B2_ZoomNeXt):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = timm.create_model(
            model_name="convnext_base",
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
        )

    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
    ):
        _ZoomNeXt_Base.__init__(self)

        self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)

        # ConvNeXt-B 四个stage通道一般是 [128, 256, 512, 1024]
        self.embed_dims = self.encoder.feature_info.channels()

        self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.siu_5 = MHSIU(mid_dim, siu_groups)
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.siu_4 = MHSIU(mid_dim, siu_groups)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.siu_3 = MHSIU(mid_dim, siu_groups)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.siu_2 = MHSIU(mid_dim, siu_groups)
        self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_dim, mid_dim, 3, 1, 1),
        )

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_dim, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        c2, c3, c4, c5 = self.encoder(x)
        return c2, c3, c4, c5

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}

        for name, param in self.named_parameters():
            if name.startswith("encoder.stem."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                param_groups["retrained"].append(param)

        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups







# -*- coding: utf-8 -*-
"""Loss-only NC curriculum for the repository's current PVT-B4 ZoomNeXt.

This module intentionally keeps ``PvtV2B4_ZoomNeXt`` unchanged and replaces
only its training objective:

    first 40% of training: q = 2, L = L_NC(q=2) + L_WBCE
    remaining training:    q = 1, L = 2 * L_NC(q=1)

The current APBOXNet commit uses the 1.5x/1.0x two-scale MHSIU2 path inside
``PvtV2B4_ZoomNeXt``.  Inheriting from that exact class makes the comparison
against ``PvtV2B4_ZoomNeXt`` a strict loss-only ablation: encoder, MHSIU2,
RGPU, predictor and inference path all remain identical.

No CSR, box supervision, deep supervision, UAL, EMA or extra decoder branch
is introduced here.
"""

import torch
import torch.nn.functional as F



from ..fpn_baseline import NoisyCODCurriculumLoss

class PvtV2B4_ZoomNeXt_NC_Curriculum(PvtV2B4_ZoomNeXt):
    """Current PVTv2-B4 ZoomNeXt with the Noisy-COD loss curriculum."""

    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
        q_switch_ratio=0.40,
        **kwargs,
    ):
        # Absorb framework-level optional arguments without changing the
        # constructor of the original ZoomNeXt implementation.
        del kwargs
        super().__init__(
            pretrained=pretrained,
            num_frames=num_frames,
            input_norm=input_norm,
            mid_dim=mid_dim,
            siu_groups=siu_groups,
            hmu_groups=hmu_groups,
            use_checkpoint=use_checkpoint,
        )
        self.curriculum_loss = NoisyCODCurriculumLoss(
            q_switch_ratio=q_switch_ratio,
        )

    def forward(self, data, iter_percentage=1.0, **kwargs):
        del kwargs
        logits = self.body(data=data)

        # Keep the evaluator contract identical to PvtV2B4_ZoomNeXt.
        if not self.training:
            return logits

        if "mask" not in data:
            raise KeyError("Training requires data['mask'].")

        target = data["mask"].float()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(
                target,
                size=logits.shape[-2:],
                mode="nearest",
            )
        target = target.clamp(0.0, 1.0)

        total_loss, items = self.curriculum_loss(
            logits=logits,
            target=target,
            iter_percentage=iter_percentage,
        )

        q = float(items["q"])
        nc = items["nc"]
        wbce = items["wbce"]

        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid()},
            "loss": total_loss,
            "loss_items": {
                # basemain_continuous.py expects the compatibility key "bce".
                "bce": wbce,
                "wbce": wbce,
                "nc": nc,
                "q": torch.as_tensor(
                    q,
                    device=logits.device,
                    dtype=logits.dtype,
                ),
                "total": total_loss.detach(),
            },
            "loss_str": (
                f"L:{total_loss.detach().item():.4f} "
                f"NC:{nc.item():.4f} "
                f"WBCE:{wbce.item():.4f} "
                f"Q:{q:.1f}"
            ),
        }





# -*- coding: utf-8 -*-
"""Z3: dual-scale PVTv2-B4 ZoomNeXt + NC + box-guided training losses.

The RGB inference graph remains the repository's current dual-scale ZoomNeXt:

    image_l (1.5x) + image_m (1.0x)
        -> shared PVTv2-B4 -> MHSIU2 -> RGPU -> mask

Box masks are consumed only during training for:

1. soft supervision of the two-way MHSIU2 routing attention;
2. a box-out background-prototype separation loss.

No box feature is concatenated to the RGB path. Evaluation therefore needs no
box and has no train-only branch left in the inference graph.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange




def _cosine_lerp(start, end, progress):
    progress = min(max(float(progress), 0.0), 1.0)
    ratio = 0.5 * (1.0 - math.cos(math.pi * progress))
    return float(start + (end - start) * ratio)


class MHSIU2WithAttention(MHSIU2):
    """Exact MHSIU2 computation with its two-scale attention also returned."""

    def forward(self, l, m):
        batch_size = m.shape[0]
        target_size = m.shape[2:]

        l = self.conv_l_pre(l)
        l = (
            F.adaptive_max_pool2d(l, target_size)
            + F.adaptive_avg_pool2d(l, target_size)
        )
        l = self.conv_l(l)
        m = self.conv_m(m)

        lm = torch.cat([l, m], dim=1)
        attn_features = self.conv_lm(lm)
        attn_features = rearrange(
            attn_features,
            "bt (nb ng d) h w -> (bt ng) (nb d) h w",
            nb=2,
            ng=self.num_groups,
        )
        attention = self.trans(attn_features)

        values = self.initial_merge(lm)
        values = rearrange(
            values,
            "bt (nb ng d) h w -> (bt ng) nb d h w",
            nb=2,
            ng=self.num_groups,
        )
        fused = (attention.unsqueeze(dim=2) * values).sum(dim=1)
        fused = rearrange(
            fused,
            "(bt ng) d h w -> bt (ng d) h w",
            bt=batch_size,
            ng=self.num_groups,
        )
        attention = rearrange(
            attention,
            "(bt ng) nb h w -> bt ng nb h w",
            bt=batch_size,
            ng=self.num_groups,
        )
        return fused, attention


class PvtV2B4_ZoomNeXt_Z3(PvtV2B4_ZoomNeXt_NC_Curriculum):
    """Z3 model with training-only box routing/background constraints."""

    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
        q_switch_ratio=0.40,
        scale_weight_start=0.20,
        scale_weight_end=0.05,
        background_weight=0.10,
        background_start_ratio=0.10,
        background_full_ratio=0.40,
        small_box_ratio=0.10,
        large_box_ratio=0.50,
        large_branch_prior_min=0.20,
        large_branch_prior_max=0.80,
        uncertain_box_weight=0.25,
        background_margin=0.10,
        background_temperature=0.20,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            num_frames=num_frames,
            input_norm=input_norm,
            mid_dim=mid_dim,
            siu_groups=siu_groups,
            hmu_groups=hmu_groups,
            use_checkpoint=use_checkpoint,
            q_switch_ratio=q_switch_ratio,
            **kwargs,
        )

        # Same parameterization as MHSIU2; only the attention return changes.
        self.siu_5 = MHSIU2WithAttention(mid_dim, siu_groups)
        self.siu_4 = MHSIU2WithAttention(mid_dim, siu_groups)
        self.siu_3 = MHSIU2WithAttention(mid_dim, siu_groups)
        self.siu_2 = MHSIU2WithAttention(mid_dim, siu_groups)

        if not 0.0 <= small_box_ratio < large_box_ratio <= 1.0:
            raise ValueError("Require 0 <= small_box_ratio < large_box_ratio <= 1.")
        if not (
            0.0 <= large_branch_prior_min
            <= large_branch_prior_max <= 1.0
        ):
            raise ValueError(
                "Require 0 <= large_branch_prior_min <= "
                "large_branch_prior_max <= 1."
            )
        if not 0.0 <= uncertain_box_weight <= 1.0:
            raise ValueError("uncertain_box_weight must be in [0, 1].")
        if not 0.0 < background_temperature:
            raise ValueError("background_temperature must be positive.")

        self.scale_weight_start = float(scale_weight_start)
        self.scale_weight_end = float(scale_weight_end)
        self.background_weight = float(background_weight)
        self.background_start_ratio = float(background_start_ratio)
        self.background_full_ratio = float(background_full_ratio)
        self.small_box_ratio = float(small_box_ratio)
        self.large_box_ratio = float(large_box_ratio)
        self.large_branch_prior_min = float(large_branch_prior_min)
        self.large_branch_prior_max = float(large_branch_prior_max)
        self.uncertain_box_weight = float(uncertain_box_weight)
        self.background_margin = float(background_margin)
        self.background_temperature = float(background_temperature)

    def body_with_aux(self, data):
        large_features = self.normalize_encoder(data["image_l"])
        medium_features = self.normalize_encoder(data["image_m"])

        l5 = self.tra_5(large_features[3])
        m5 = self.tra_5(medium_features[3])
        z5, a5 = self.siu_5(l=l5, m=m5)
        x5 = self.hmu_5(z5)

        l4 = self.tra_4(large_features[2])
        m4 = self.tra_4(medium_features[2])
        z4, a4 = self.siu_4(l=l4, m=m4)
        x4 = self.hmu_4(z4 + resize_to(x5, tgt_hw=z4.shape[-2:]))

        l3 = self.tra_3(large_features[1])
        m3 = self.tra_3(medium_features[1])
        z3, a3 = self.siu_3(l=l3, m=m3)
        x3 = self.hmu_3(z3 + resize_to(x4, tgt_hw=z3.shape[-2:]))

        l2 = self.tra_2(large_features[0])
        m2 = self.tra_2(medium_features[0])
        z2, a2 = self.siu_2(l=l2, m=m2)
        x2 = self.hmu_2(z2 + resize_to(x3, tgt_hw=z2.shape[-2:]))

        logits = self.predictor(self.tra_1(x2))
        return logits, {
            "attention": (a5, a4, a3, a2),
            "decoder_feature": x2,
        }

    @staticmethod
    def _prepare_box_mask(data, target_size):
        if "box_mask" not in data:
            raise KeyError(
                "Z3 training requires data['box_mask']. Set "
                "train.data.use_box=True in the config."
            )
        box_mask = data["box_mask"].float()
        if box_mask.ndim == 3:
            box_mask = box_mask.unsqueeze(1)
        if box_mask.shape[-2:] != tuple(target_size):
            box_mask = F.interpolate(
                box_mask,
                size=target_size,
                mode="nearest",
            )
        box_mask = box_mask.clamp(0.0, 1.0)
        if bool((box_mask.flatten(1).sum(dim=1) <= 0).any()):
            raise ValueError("Every Z3 training sample must contain a non-empty box.")
        return box_mask

    def _box_scale_prior(self, box_mask):
        area_ratio = box_mask.mean(dim=(1, 2, 3))
        normalized = (
            (area_ratio - self.small_box_ratio)
            / (self.large_box_ratio - self.small_box_ratio)
        ).clamp(0.0, 1.0)

        # Branch order follows MHSIU2: [large-input 1.5x, medium-input 1.0x].
        large_prior = self.large_branch_prior_max + normalized * (
            self.large_branch_prior_min - self.large_branch_prior_max
        )
        prior = torch.stack([large_prior, 1.0 - large_prior], dim=1)

        # BO and boundary-touching boxes have less reliable size semantics.
        touches_border = (
            (box_mask[:, :, 0, :].amax(dim=(1, 2)) > 0.5)
            | (box_mask[:, :, -1, :].amax(dim=(1, 2)) > 0.5)
            | (box_mask[:, :, :, 0].amax(dim=(1, 2)) > 0.5)
            | (box_mask[:, :, :, -1].amax(dim=(1, 2)) > 0.5)
        )
        uncertain = (area_ratio >= self.large_box_ratio) | touches_border
        reliability = torch.where(
            uncertain,
            torch.full_like(area_ratio, self.uncertain_box_weight),
            torch.ones_like(area_ratio),
        )
        return prior, reliability, area_ratio

    def _scale_routing_loss(self, attentions, box_mask):
        prior, reliability, area_ratio = self._box_scale_prior(box_mask)
        stage_losses = []
        route_means = []

        for attention in attentions:
            stage_box = F.interpolate(
                box_mask,
                size=attention.shape[-2:],
                mode="nearest",
            ).unsqueeze(1)
            denominator = (
                stage_box.sum(dim=(1, 3, 4))
                * attention.shape[1]
            ).clamp_min(1.0)
            route = (attention * stage_box).sum(dim=(1, 3, 4)) / denominator
            route = route.clamp_min(1e-6)
            route = route / route.sum(dim=1, keepdim=True)

            sample_kl = (
                prior * (prior.clamp_min(1e-6).log() - route.log())
            ).sum(dim=1)
            stage_losses.append(
                (sample_kl * reliability).sum()
                / reliability.sum().clamp_min(1e-6)
            )
            route_means.append(route.detach().mean(dim=0))

        return torch.stack(stage_losses).mean(), route_means, area_ratio

    def _background_prototype_loss(self, feature, box_mask, target):
        feature_box = F.interpolate(
            box_mask,
            size=feature.shape[-2:],
            mode="nearest",
        )
        target_small = F.interpolate(
            target,
            size=feature.shape[-2:],
            mode="nearest",
        )
        outside = (1.0 - feature_box).clamp(0.0, 1.0)
        foreground = (target_small * feature_box).clamp(0.0, 1.0)

        outside_count = outside.sum(dim=(2, 3), keepdim=True)
        foreground_count = foreground.sum(dim=(2, 3), keepdim=True)
        valid = (
            (outside_count.flatten(1).squeeze(1) > 0)
            & (foreground_count.flatten(1).squeeze(1) > 0)
        )
        if not bool(valid.any()):
            return feature.sum() * 0.0

        normalized_feature = F.normalize(feature, dim=1, eps=1e-6)
        prototype = (
            normalized_feature * outside
        ).sum(dim=(2, 3), keepdim=True) / outside_count.clamp_min(1.0)
        prototype = F.normalize(prototype, dim=1, eps=1e-6).detach()
        similarity = (normalized_feature * prototype).sum(dim=1, keepdim=True)

        background_reference = (
            (similarity * outside).sum(dim=(2, 3), keepdim=True)
            / outside_count.clamp_min(1.0)
        ).detach()
        ranking = F.softplus(
            (
                similarity
                - background_reference
                + self.background_margin
            )
            / self.background_temperature
        )
        per_sample = (
            (ranking * foreground).sum(dim=(1, 2, 3))
            / foreground.sum(dim=(1, 2, 3)).clamp_min(1.0)
        )
        return per_sample[valid].mean()

    def _loss_weights(self, iter_percentage):
        progress = min(max(float(iter_percentage), 0.0), 1.0)
        scale_weight = _cosine_lerp(
            self.scale_weight_start,
            self.scale_weight_end,
            progress,
        )
        if progress <= self.background_start_ratio:
            background_weight = 0.0
        elif progress >= self.background_full_ratio:
            background_weight = self.background_weight
        else:
            local = (
                (progress - self.background_start_ratio)
                / max(
                    self.background_full_ratio - self.background_start_ratio,
                    1e-6,
                )
            )
            background_weight = _cosine_lerp(0.0, self.background_weight, local)
        return scale_weight, background_weight

    def forward(self, data, iter_percentage=1.0, **kwargs):
        del kwargs
        logits, aux = self.body_with_aux(data)

        if not self.training:
            return logits

        if "mask" not in data:
            raise KeyError("Z3 training requires data['mask'].")
        target = data["mask"].float()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
        target = target.clamp(0.0, 1.0)
        box_mask = self._prepare_box_mask(data, target.shape[-2:])

        nc_loss, nc_items = self.curriculum_loss(
            logits=logits,
            target=target,
            iter_percentage=iter_percentage,
        )
        scale_raw, route_means, area_ratio = self._scale_routing_loss(
            aux["attention"],
            box_mask,
        )
        background_raw = self._background_prototype_loss(
            aux["decoder_feature"],
            box_mask,
            target,
        )
        scale_weight, background_weight = self._loss_weights(iter_percentage)
        scale_loss = float(scale_weight) * scale_raw
        background_loss = float(background_weight) * background_raw
        total_loss = nc_loss + scale_loss + background_loss

        q = float(nc_items["q"])
        nc = nc_items["nc"]
        wbce = nc_items["wbce"]
        loss_items = {
            "bce": wbce,
            "wbce": wbce,
            "nc": nc,
            "scale": scale_loss.detach(),
            "scale_raw": scale_raw.detach(),
            "background": background_loss.detach(),
            "background_raw": background_raw.detach(),
            "q": logits.new_tensor(q),
            "scale_weight": logits.new_tensor(scale_weight),
            "background_weight": logits.new_tensor(background_weight),
            "box_area": area_ratio.detach().mean(),
            "total": total_loss.detach(),
        }
        for level, route in zip((5, 4, 3, 2), route_means):
            loss_items[f"m{level}_large"] = route[0]
            loss_items[f"m{level}_medium"] = route[1]

        return {
            "logits": logits,
            "loss": total_loss,
            "loss_items": loss_items,
            "loss_str": (
                f"L:{total_loss.detach().item():.4f} "
                f"NC:{nc.item():.4f} WBCE:{wbce.item():.4f} Q:{q:.1f} "
                f"SC:{scale_loss.detach().item():.4f} "
                f"BG:{background_loss.detach().item():.4f} "
                f"Wsc:{scale_weight:.3f} Wbg:{background_weight:.3f}"
            ),
            "vis": {
                "sal": logits.sigmoid(),
                "box": box_mask,
            },
        }



# -*- coding: utf-8 -*-
"""
ZoomNeXt + Three-Scale MHSIU + Deep NCLoss
===========================================

Recommended path in PASAM:
    methods/zoomnext/zoomnext_deepnc.py

Ablation purpose
----------------
Keep the original ZoomNeXt three-scale fusion mechanism (MHSIU) unchanged,
and modify ONLY the supervision:

    image_s = 0.5x
    image_m = 1.0x
    image_l = 1.5x
          |
     shared PVTv2-B4
          |
   stage-wise MHSIU
          |
   RGPU5 -> P5
     |
   RGPU4 -> P4
     |
   RGPU3 -> P3
     |
   RGPU2 -> P2
     |
    tra_1
     |
   predictor -> P1 (final)

Deep NCLoss:
    L_deep_nc =
        1/16 * NC(P5, Y)
      + 1/8  * NC(P4, Y)
      + 1/4  * NC(P3, Y)
      + 1/2  * NC(P2, Y)
      + 1     * NC(P1, Y)

Final-only UAL:
    L = L_deep_nc + lambda_ual(t) * UAL(P1)

Notes
-----
1. This file intentionally does NOT add DWT / PNet decoder / edge branch.
   It is designed as a clean ablation of Deep NCLoss on three-scale ZoomNeXt.
2. During evaluation, ONLY final P1 logits are returned, so it stays compatible
   with PASAM's current evaluator.
3. q_value defaults to 2, matching the current PNet ablation implementation.
"""



import logging
import math
from typing import Dict, List, Sequence





class NCLoss(nn.Module):
    """Noise Correction Loss adapted from PASAM's current Noisy-COD PNet."""

    @staticmethod
    def wbce_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weit = 1.0 + 5.0 * torch.abs(
            F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets
        )
        wbce = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp_min(1e-6)
        return wbce.mean()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, q: int = 2) -> torch.Tensor:
        if preds.shape[-2:] != targets.shape[-2:]:
            preds = F.interpolate(
                preds,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        targets = targets.float().clamp(0.0, 1.0)
        wbce = self.wbce_loss(preds, targets)

        probs = torch.sigmoid(preds).flatten(1)
        targets_flat = targets.flatten(1)
        q = int(q)

        numerator = torch.sum(
            torch.abs(probs - targets_flat).pow(float(q)),
            dim=1,
        )
        intersection = torch.sum(probs * targets_flat, dim=1)
        denominator = (
            torch.sum(probs, dim=1)
            + torch.sum(targets_flat, dim=1)
            - intersection
            + 1e-6
        )
        nc_region = numerator / denominator

        if q == 2:
            return nc_region.mean() + wbce
        return nc_region.mean() * 2.0


class Zoom_DeepNC(nn.Module):
    """
    PVTv2-B4 ZoomNeXt with original three-scale MHSIU fusion,
    five-level deep NCLoss supervision, and final-only UAL.
    """

    def __init__(
        self,
        pretrained: bool = True,
        input_norm: bool = True,
        mid_dim: int = 64,
        siu_groups: int = 4,
        hmu_groups: int = 6,
        num_frames: int = 1,
        use_checkpoint: bool = False,
        deep_weights: Sequence[float] = (0.0625, 0.125, 0.25, 0.5, 1.0),
        q_value: int = 2,
        use_ual: bool = True,
        ual_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        if len(deep_weights) != 5:
            raise ValueError(
                f"deep_weights must contain 5 values for P5..P1, got {deep_weights}"
            )

        self.deep_weights = tuple(float(v) for v in deep_weights)
        self.default_q_value = int(q_value)
        self.use_ual = bool(use_ual)
        self.ual_weight = float(ual_weight)

        # Shared PVTv2-B4 backbone.
        self.encoder = pvt_v2_eff_b4(
            pretrained=pretrained,
            use_checkpoint=use_checkpoint,
        )
        self.embed_dims = self.encoder.embed_dims
        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()

        # Lateral projections.
        self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)

        # Original ZoomNeXt three-scale fusion.
        self.siu_5 = MHSIU(mid_dim, siu_groups)
        self.siu_4 = MHSIU(mid_dim, siu_groups)
        self.siu_3 = MHSIU(mid_dim, siu_groups)
        self.siu_2 = MHSIU(mid_dim, siu_groups)

        # Original ZoomNeXt progressive RGPU decoder.
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
        self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_dim, mid_dim, 3, 1, 1),
        )

        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_dim, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

        # Deep-supervision heads after RGPU5/4/3/2.
        self.aux_head_5 = nn.Conv2d(mid_dim, 1, 1)
        self.aux_head_4 = nn.Conv2d(mid_dim, 1, 1)
        self.aux_head_3 = nn.Conv2d(mid_dim, 1, 1)
        self.aux_head_2 = nn.Conv2d(mid_dim, 1, 1)

        self.nc_loss = NCLoss()

    def normalize_encoder(self, x: torch.Tensor):
        x = self.normalizer(x)
        features = self.encoder(x)
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]
        return c2, c3, c4, c5

    @staticmethod
    def cosine_coef(iter_percentage: float) -> float:
        p = min(max(float(iter_percentage), 0.0), 1.0)
        return float((1.0 - math.cos(math.pi * p)) * 0.5)

    @staticmethod
    def uncertainty_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        return (1.0 - (2.0 * prob - 1.0).abs().pow(2)).mean()

    def _extract_three_scale_features(self, data: Dict[str, torch.Tensor]):
        required = ("image_l", "image_m", "image_s")
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(
                f"Three-scale DeepNC ZoomNeXt requires {required}, missing={missing}"
            )

        # One shared backbone, three resolutions.
        l_feats = self.normalize_encoder(data["image_l"])
        m_feats = self.normalize_encoder(data["image_m"])
        s_feats = self.normalize_encoder(data["image_s"])
        return l_feats, m_feats, s_feats

    def body(self, data: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        l_feats, m_feats, s_feats = self._extract_three_scale_features(data)

        # Stage 5: deepest.
        l5 = self.tra_5(l_feats[3])
        m5 = self.tra_5(m_feats[3])
        s5 = self.tra_5(s_feats[3])
        z5 = self.siu_5(l=l5, m=m5, s=s5)
        x5 = self.hmu_5(z5)
        p5 = self.aux_head_5(x5)

        # Stage 4.
        l4 = self.tra_4(l_feats[2])
        m4 = self.tra_4(m_feats[2])
        s4 = self.tra_4(s_feats[2])
        z4 = self.siu_4(l=l4, m=m4, s=s4)
        x4 = self.hmu_4(z4 + resize_to(x5, tgt_hw=z4.shape[-2:]))
        p4 = self.aux_head_4(x4)

        # Stage 3.
        l3 = self.tra_3(l_feats[1])
        m3 = self.tra_3(m_feats[1])
        s3 = self.tra_3(s_feats[1])
        z3 = self.siu_3(l=l3, m=m3, s=s3)
        x3 = self.hmu_3(z3 + resize_to(x4, tgt_hw=z3.shape[-2:]))
        p3 = self.aux_head_3(x3)

        # Stage 2: shallowest PVT feature.
        l2 = self.tra_2(l_feats[0])
        m2 = self.tra_2(m_feats[0])
        s2 = self.tra_2(s_feats[0])
        z2 = self.siu_2(l=l2, m=m2, s=s2)
        x2 = self.hmu_2(z2 + resize_to(x3, tgt_hw=z2.shape[-2:]))
        p2 = self.aux_head_2(x2)

        # Final prediction.
        x1 = self.tra_1(x2)
        p1 = self.predictor(x1)

        return [p5, p4, p3, p2, p1]

    def _deep_nc_loss(
        self,
        preds: Sequence[torch.Tensor],
        mask: torch.Tensor,
        q_value: int,
    ):
        if len(preds) != 5:
            raise ValueError(f"Expected 5 predictions, got {len(preds)}")

        level_losses = []
        weighted_losses = []

        for pred, weight in zip(preds, self.deep_weights):
            level_loss = self.nc_loss(pred, mask, q=q_value)
            level_losses.append(level_loss)
            weighted_losses.append(weight * level_loss)

        return sum(weighted_losses), level_losses

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        iter_percentage: float = 1.0,
        q_value: int | None = None,
        **kwargs,
    ):
        del kwargs

        preds = self.body(data)
        p5, p4, p3, p2, p1 = preds

        # Keep current PASAM evaluator compatibility.
        if not self.training:
            return p1

        if "mask" not in data:
            raise KeyError("Training requires data['mask'].")

        mask = data["mask"].float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.clamp(0.0, 1.0)

        q = self.default_q_value if q_value is None else int(q_value)

        loss_deep_nc, nc_levels = self._deep_nc_loss(
            preds=preds,
            mask=mask,
            q_value=q,
        )

        # UAL on final P1 ONLY.
        if self.use_ual:
            ual_raw = self.uncertainty_loss_from_logits(p1)
            ual_coef = self.cosine_coef(iter_percentage)
            loss_ual = self.ual_weight * float(ual_coef) * ual_raw
        else:
            ual_raw = p1.new_zeros(())
            ual_coef = 0.0
            loss_ual = p1.new_zeros(())

        total_loss = loss_deep_nc + loss_ual

        loss_items = {
            "total": total_loss.detach(),
            "deep_nc": loss_deep_nc.detach(),
            "nc_p5": nc_levels[0].detach(),
            "nc_p4": nc_levels[1].detach(),
            "nc_p3": nc_levels[2].detach(),
            "nc_p2": nc_levels[3].detach(),
            "nc_p1": nc_levels[4].detach(),
            "ual": loss_ual.detach(),
            "ual_raw": ual_raw.detach(),
            "ual_coef": float(ual_coef),
            "q": float(q),
        }

        loss_str = (
            f"L:{total_loss.detach().item():.4f} "
            f"DeepNC:{loss_deep_nc.detach().item():.4f} "
            f"P5:{nc_levels[0].detach().item():.4f} "
            f"P4:{nc_levels[1].detach().item():.4f} "
            f"P3:{nc_levels[2].detach().item():.4f} "
            f"P2:{nc_levels[3].detach().item():.4f} "
            f"P1:{nc_levels[4].detach().item():.4f} "
            f"UAL:{loss_ual.detach().item():.4f} "
            f"Q:{q}"
        )

        return {
            "logits": p1,
            "preds": preds,
            "loss": total_loss,
            "loss_items": loss_items,
            "loss_str": loss_str,
            "vis": {"sal": torch.sigmoid(p1)},
        }

    def get_grouped_params(self):
        param_groups = {
            "pretrained": [],
            "fixed": [],
            "retrained": [],
        }

        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                param_groups["retrained"].append(param)

        LOGGER.info(
            "ZoomNeXt-DeepNC Parameter Groups:{"
            f"Pretrained:{len(param_groups['pretrained'])}, "
            f"Fixed:{len(param_groups['fixed'])}, "
            f"ReTrained:{len(param_groups['retrained'])}"
            "}"
        )
        return param_groups


class ConvNeXtB384_ZoomNeXt(PvtV2B2_ZoomNeXt):
    """
    ConvNeXt-Base 22K -> 1K 384
    + SAME ZoomNeXt decoder as PvtV2B4_ZoomNeXt

    Only replace backbone:
        PVTv2-B4
            ->
        convnext_base.fb_in22k_ft_in1k_384
    """

    def set_backbone(
        self,
        pretrained: bool,
        use_checkpoint: bool,
    ):
        self.encoder = timm.create_model(
            model_name="convnext_base.fb_in22k_ft_in1k_384",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # 让父类 PvtV2B2_ZoomNeXt.__init__()
        # 可以像 PVT 一样读取 encoder.embed_dims
        self.encoder.embed_dims = list(
            self.encoder.feature_info.channels()
        )

        # should be:
        # [128, 256, 512, 1024]
        LOGGER.info(
            f"ConvNeXt-B384 feature channels: "
            f"{self.encoder.embed_dims}"
        )

        if use_checkpoint:
            if hasattr(
                self.encoder,
                "set_grad_checkpointing",
            ):
                self.encoder.set_grad_checkpointing(
                    enable=True
                )
                LOGGER.info(
                    "ConvNeXt-B384 gradient checkpointing enabled."
                )
            else:
                LOGGER.warning(
                    "ConvNeXt backbone does not support "
                    "set_grad_checkpointing()."
                )

    def normalize_encoder(self, x):
        """
        timm features_only output:

        c2: 1/4   128 channels
        c3: 1/8   256 channels
        c4: 1/16  512 channels
        c5: 1/32 1024 channels
        """

        x = self.normalizer(x)

        c2, c3, c4, c5 = self.encoder(x)

        return c2, c3, c4, c5

    def get_grouped_params(self):
        """
        Keep optimizer grouping consistent with PVT version.

        PVT:
            patch_embed1 -> fixed

        ConvNeXt:
            stem -> fixed
        """

        param_groups = {
            "pretrained": [],
            "fixed": [],
            "retrained": [],
        }

        for name, param in self.named_parameters():

            if name.startswith("encoder.stem."):
                param.requires_grad = False
                param_groups["fixed"].append(param)

            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)

            else:
                param_groups["retrained"].append(param)

        LOGGER.info(
            "ConvNeXtB384-ZoomNeXt Parameter Groups:{"
            f"Pretrained: "
            f"{len(param_groups['pretrained'])}, "
            f"Fixed: "
            f"{len(param_groups['fixed'])}, "
            f"ReTrained: "
            f"{len(param_groups['retrained'])}"
            "}"
        )

        return param_groups