# -*- coding: utf-8 -*-
"""
ZoomNeXt backbone ablation for weakly-supervised COD.

Kept models:
    1) PvtV2B4_ZoomNeXt
    2) SwinB_ZoomNeXt
    3) SMTB_ZoomNeXt
    4) DINOv2B_ZoomNeXt
    5) ConvNeXtB_ZoomNeXt

All five models share exactly the same:
    - three-scale inputs: image_l / image_m / image_s
    - MHSIU fusion blocks
    - RGPU progressive decoder
    - prediction head
    - BCE + UAL training objective

Only the backbone feature extractor is changed.

Notes:
    * SMT-B uses the official AFeng-x/SMT implementation. Put its `models`
      directory on PYTHONPATH, or pass `smt_repo_root` to the constructor.
      A pretrained SMT-B checkpoint can be passed via `smt_pretrained_path`.
    * DINOv2-B uses the official torch.hub model
      `facebookresearch/dinov2: dinov2_vitb14`.
    * timm must contain the selected Swin/ConvNeXt model definitions.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.pvt_v2_eff import pvt_v2_eff_b4
from .layers import MHSIU, RGPU, SimpleASPP
from .ops import ConvBNReLU, PixelNormalizer, resize_to

LOGGER = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Common ZoomNeXt model
# ---------------------------------------------------------------------------

class _ZoomNeXtBackboneAblation(nn.Module):
    """A common three-scale ZoomNeXt decoder with a replaceable backbone."""

    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        del use_checkpoint, kwargs

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.encoder = None
        self.embed_dims = None

        # Subclasses must initialize self.encoder and self.embed_dims before
        # calling _build_decoder().
        self._build_decoder(
            mid_dim=mid_dim,
            siu_groups=siu_groups,
            hmu_groups=hmu_groups,
            num_frames=num_frames,
        )

    def _build_decoder(self, mid_dim, siu_groups, hmu_groups, num_frames):
        if self.embed_dims is None or len(self.embed_dims) != 4:
            raise RuntimeError(
                f"Backbone must expose four feature levels, got {self.embed_dims}"
            )

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

        # The original ZoomNeXt decoder upsamples the last decoder feature
        # before the final prediction.
        self.tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_dim, mid_dim, 3, 1, 1),
        )
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_dim, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def normalize_encoder(self, x):
        """Return four NCHW feature maps ordered from stride 4 to stride 32."""
        x = self.normalizer(x)
        feats = self.encoder(x)
        if isinstance(feats, dict):
            feats = [
                feats["reduction_2"],
                feats["reduction_3"],
                feats["reduction_4"],
                feats["reduction_5"],
            ]
        if not isinstance(feats, (list, tuple)) or len(feats) != 4:
            raise RuntimeError(
                "Backbone must return exactly four feature maps "
                "(stride 4/8/16/32)."
            )
        return tuple(self._to_nchw(f, c) for f, c in zip(feats, self.embed_dims))

    @staticmethod
    def _to_nchw(feature, channels):
        # Some timm Swin implementations return NHWC feature maps.
        if feature.ndim != 4:
            raise RuntimeError(f"Expected 4D feature map, got {feature.shape}")
        if feature.shape[1] == channels:
            return feature.contiguous()
        if feature.shape[-1] == channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise RuntimeError(
            f"Cannot infer feature layout for shape={tuple(feature.shape)}, "
            f"expected channel count={channels}"
        )

    def body(self, data):
        required = ("image_l", "image_m", "image_s")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(
                f"ZoomNeXt requires three-scale inputs {required}; missing={missing}"
            )

        l_feats = self.normalize_encoder(data["image_l"])
        m_feats = self.normalize_encoder(data["image_m"])
        s_feats = self.normalize_encoder(data["image_s"])

        l, m, s = (
            self.tra_5(l_feats[3]),
            self.tra_5(m_feats[3]),
            self.tra_5(s_feats[3]),
        )
        fused = self.siu_5(l=l, m=m, s=s)
        x = self.hmu_5(fused)

        l, m, s = (
            self.tra_4(l_feats[2]),
            self.tra_4(m_feats[2]),
            self.tra_4(s_feats[2]),
        )
        fused = self.siu_4(l=l, m=m, s=s)
        x = self.hmu_4(fused + resize_to(x, tgt_hw=fused.shape[-2:]))

        l, m, s = (
            self.tra_3(l_feats[1]),
            self.tra_3(m_feats[1]),
            self.tra_3(s_feats[1]),
        )
        fused = self.siu_3(l=l, m=m, s=s)
        x = self.hmu_3(fused + resize_to(x, tgt_hw=fused.shape[-2:]))

        l, m, s = (
            self.tra_2(l_feats[0]),
            self.tra_2(m_feats[0]),
            self.tra_2(s_feats[0]),
        )
        fused = self.siu_2(l=l, m=m, s=s)
        x = self.hmu_2(fused + resize_to(x, tgt_hw=fused.shape[-2:]))

        return self.predictor(self.tra_1(x))

    @staticmethod
    def get_coef(iter_percentage=1.0, method="cos", milestones=(0.0, 1.0)):
        low, high = min(milestones), max(milestones)
        if high <= low:
            return 1.0
        if iter_percentage <= low:
            return 0.0
        if iter_percentage >= high:
            return 1.0
        progress = (iter_percentage - low) / (high - low)
        if method == "linear":
            return float(progress)
        return float((1.0 - np.cos(progress * np.pi)) * 0.5)

    def forward(self, data, iter_percentage=1.0, **kwargs):
        del kwargs
        logits = self.body(data)
        if not self.training:
            return logits

        if "mask" not in data:
            raise KeyError("Training requires data['mask'].")

        mask = data["mask"].float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != logits.shape[-2:]:
            mask = F.interpolate(mask, size=logits.shape[-2:], mode="nearest")
        mask = mask.clamp(0.0, 1.0)

        prob = logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="mean")
        ual_coef = self.get_coef(iter_percentage, method="cos", milestones=(0.0, 1.0))
        ual = ual_coef * (1.0 - (2.0 * prob - 1.0).abs().pow(2)).mean()
        total = bce + ual

        return {
            "vis": {"sal": prob},
            "loss": total,
            "loss_str": f"bce:{bce.item():.5f} powual_{ual_coef:.5f}:{ual.item():.5f}",
        }

    def get_grouped_params(self):
        # Keep the optimizer interface expected by basemain_continuous.py.
        # All pretrained backbones are fine-tuned; no backbone-specific stem
        # is frozen, so the fine-tuning protocol is consistent across models.
        groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder."):
                groups["pretrained"].append(param)
            else:
                groups["retrained"].append(param)
        LOGGER.info(
            "Backbone ablation parameter groups: "
            f"Pretrained={len(groups['pretrained'])}, "
            f"Fixed={len(groups['fixed'])}, "
            f"ReTrained={len(groups['retrained'])}"
        )
        return groups


# ---------------------------------------------------------------------------
# PVTv2-B4
# ---------------------------------------------------------------------------

class PvtV2B4_ZoomNeXt(_ZoomNeXtBackboneAblation):
    def __init__(
        self, pretrained=True, num_frames=1, input_norm=True, mid_dim=64,
        siu_groups=4, hmu_groups=6, use_checkpoint=False, **kwargs
    ):
        # Initialize nn.Module directly; build the backbone before decoder.
        nn.Module.__init__(self)
        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.encoder = pvt_v2_eff_b4(
            pretrained=pretrained, use_checkpoint=use_checkpoint
        )
        self.embed_dims = list(self.encoder.embed_dims)
        self._build_decoder(mid_dim, siu_groups, hmu_groups, num_frames)

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        feats = self.encoder(x)
        return (
            feats["reduction_2"],
            feats["reduction_3"],
            feats["reduction_4"],
            feats["reduction_5"],
        )


# ---------------------------------------------------------------------------
# Swin-Base
# ---------------------------------------------------------------------------

class SwinB_ZoomNeXt(_ZoomNeXtBackboneAblation):
    """Swin-B 384 + original three-scale ZoomNeXt decoder."""

    BACKBONE = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"

    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
        swin_input_size=384,
        auto_resize_swin=True,
        **kwargs,
    ):
        del use_checkpoint, kwargs

        nn.Module.__init__(self)

        # 1. Input normalization
        self.normalizer = (
            PixelNormalizer() if input_norm else nn.Identity()
        )

        # 2. Swin input configuration
        self.swin_input_size = int(swin_input_size)
        self.auto_resize_swin = bool(auto_resize_swin)

        # 3. Swin-Base backbone
        self.encoder = timm.create_model(
            self.BACKBONE,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # 4. Four-level feature channels
        self.embed_dims = list(
            self.encoder.feature_info.channels()
        )

        assert self.embed_dims == [128, 256, 512, 1024], (
            f"Unexpected Swin-B channels: {self.embed_dims}"
        )

        # 5. Original ZoomNeXt decoder
        self._build_decoder(
            mid_dim=mid_dim,
            siu_groups=siu_groups,
            hmu_groups=hmu_groups,
            num_frames=num_frames,
        )

        LOGGER.info(
            f"Swin-B ZoomNeXt initialized: "
            f"input={self.swin_input_size}, "
            f"channels={self.embed_dims}"
        )

    def _prepare_encoder_input(self, image):
        target_hw = (
            self.swin_input_size,
            self.swin_input_size,
        )

        if image.shape[-2:] != target_hw:

            if not self.auto_resize_swin:
                raise ValueError(
                    f"Swin expects {target_hw}, "
                    f"got {tuple(image.shape[-2:])}"
                )

            image = F.interpolate(
                image,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        return self.normalizer(image)

    def normalize_encoder(self, image):
        image = self._prepare_encoder_input(image)

        features = self.encoder(image)

        return tuple(
            self._to_nchw(feature, channels)
            for feature, channels in zip(
                features, self.embed_dims
            )
        )

    def body(self, data):
        # Original three-scale ZoomNeXt forward
        logits = super().body(data)

        # Restore to original image_m resolution
        target_hw = data["image_m"].shape[-2:]

        if logits.shape[-2:] != target_hw:
            logits = F.interpolate(
                logits,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        return logits

# ---------------------------------------------------------------------------
# SMT-Base (official AFeng-x/SMT implementation)
# ---------------------------------------------------------------------------


from methods.backbone.smt import smt_b


class SMTB_ZoomNeXt(_ZoomNeXtBackboneAblation):
    """SMT-Base + original three-scale ZoomNeXt decoder."""

    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
        smt_pretrained_path="./weights/smt_base.pth",
        **kwargs,
    ):
        del use_checkpoint, kwargs

        nn.Module.__init__(self)

        self.normalizer = (
            PixelNormalizer() if input_norm else nn.Identity()
        )

        # Use the project's existing SMT implementation.
        self.encoder = smt_b()

        # Load SMT-B pretrained weights.
        if pretrained:
            if not os.path.isfile(smt_pretrained_path):
                raise FileNotFoundError(
                    f"SMT-B checkpoint not found: "
                    f"{smt_pretrained_path}"
                )

            checkpoint = torch.load(
                smt_pretrained_path,
                map_location="cpu",
                weights_only=False,
            )

            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            state_dict = {
                k.removeprefix("module."): v
                for k, v in state_dict.items()
            }

            missing, unexpected = self.encoder.load_state_dict(
                state_dict,
                strict=False,
            )

            LOGGER.info(
                f"SMT-B pretrained loaded: {smt_pretrained_path}, "
                f"missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )

        # SMT-B feature channels.
        self.embed_dims = [64, 128, 256, 512]

        self._build_decoder(
            mid_dim=mid_dim,
            siu_groups=siu_groups,
            hmu_groups=hmu_groups,
            num_frames=num_frames,
        )

    def normalize_encoder(self, x):
        x = self.normalizer(x)

        # Same feature extraction interface as CFFKDNet_smt.
        features = self.encoder(x)

        return tuple(features)


# ---------------------------------------------------------------------------
# DINOv2-Base/14
# ---------------------------------------------------------------------------

class _DINOv2Pyramid(nn.Module):
    """
    DINOv2 ViT-B/14 intermediate-layer adapter.

    ViT produces same-resolution patch-token maps rather than a native feature
    pyramid. Four intermediate transformer layers are projected to the four
    spatial scales expected by the unchanged ZoomNeXt decoder.
    """

    def __init__(self, pretrained=True, layers=(2, 5, 8, 11)):
        super().__init__()
        self.layers = tuple(layers)
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=pretrained,
        )
        self.embed_dims = [768, 768, 768, 768]

    def forward(self, x):
        height, width = x.shape[-2:]
        # DINOv2/14 requires spatial dimensions divisible by 14.
        pad_h = (14 - height % 14) % 14
        pad_w = (14 - width % 14) % 14
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        patch_maps = self.model.get_intermediate_layers(
            x, n=self.layers, reshape=True, return_class_token=False
        )
        outputs = []
        target_sizes = [
            (max(1, height // 4), max(1, width // 4)),
            (max(1, height // 8), max(1, width // 8)),
            (max(1, height // 16), max(1, width // 16)),
            (max(1, height // 32), max(1, width // 32)),
        ]
        for feature, target_size in zip(patch_maps, target_sizes):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            outputs.append(feature.contiguous())
        return outputs


class DINOv2B_ZoomNeXt(_ZoomNeXtBackboneAblation):
    def __init__(
        self, pretrained=True, num_frames=1, input_norm=True, mid_dim=64,
        siu_groups=4, hmu_groups=6, use_checkpoint=False, **kwargs
    ):
        del use_checkpoint, kwargs
        nn.Module.__init__(self)
        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.encoder = _DINOv2Pyramid(pretrained=pretrained)
        self.embed_dims = self.encoder.embed_dims
        self._build_decoder(mid_dim, siu_groups, hmu_groups, num_frames)


# ---------------------------------------------------------------------------
# ConvNeXt-Base
# ---------------------------------------------------------------------------

class ConvNeXtB_ZoomNeXt(_ZoomNeXtBackboneAblation):
    def __init__(
        self, pretrained=True, num_frames=1, input_norm=True, mid_dim=64,
        siu_groups=4, hmu_groups=6, use_checkpoint=False,
        model_name="convnext_base", **kwargs
    ):
        nn.Module.__init__(self)
        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.embed_dims = list(self.encoder.feature_info.channels())
        self._build_decoder(mid_dim, siu_groups, hmu_groups, num_frames)
