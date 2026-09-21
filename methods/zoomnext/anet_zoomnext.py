# -*- coding: utf-8 -*-
"""
ConvNeXt-B ZoomNeXt ANet.

ANet-style two-branch input:
    branch-1: RGB image
    branch-2: RGB * bounding-box mask

Both branches use ConvNeXt-B encoders. Same-level features are independently
projected to `mid_dim`, concatenated, fused, then decoded by ONE ZoomNeXt
top-down RGPU decoder.

Training supervision follows the useful parts of Noisy-COD ANet:
    - multi-level structure loss
    - boundary Dice loss
    - cosine-ramped UAL

No SAM/SAM2 is used.

Expected training data:
    data["image_m"] : [B,3,H,W], float in [0,1]
    data["box_mask"]: [B,1,H,W], binary filled boxes
    data["mask"]    : [B,1,H,W], dense GT for ANet subset

Expected pseudo-label generation data:
    image_m + box_mask
"""



import logging
import math
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RGPU, SimpleASPP
from .ops import ConvBNReLU, PixelNormalizer, resize_to

LOGGER = logging.getLogger("main")


def _structure_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Weighted BCE + weighted IoU used by many COD/SOD models."""
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    weit = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    wbce = F.binary_cross_entropy_with_logits(
        logits, mask, reduction="none"
    )
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp_min(1e-6)

    prob = torch.sigmoid(logits)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)

    return (wbce + wiou).mean()


def _dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    pred = torch.sigmoid(logits)
    pred = pred.flatten(1)
    target = target.flatten(1)
    inter = (pred * target).sum(dim=1)
    denom = pred.square().sum(dim=1) + target.square().sum(dim=1)
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def _mask_to_boundary(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Generate a thin boundary target online from a binary mask."""
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp_(0.0, 1.0)


def _cosine_coef(
    iter_percentage: float,
    start: float = 0.0,
    end: float = 1.0,
) -> float:
    p = float(iter_percentage)
    if p <= start:
        return 0.0
    if p >= end:
        return 1.0
    q = (p - start) / max(end - start, 1e-8)
    return float((1.0 - math.cos(math.pi * q)) * 0.5)


class ConvNeXtB_ZoomNeXt_ANet(nn.Module):
    """
    ANet variant built from the repository's ConvNeXtB_ZoomNeXt.

    Design:
      RGB ------------> ConvNeXt-B ----> tra_rgb_i --\
                                                       concat -> fuse_i -> ZoomNeXt RGPU
      RGB * Box ------> ConvNeXt-B ----> tra_box_i --/

    The decoder is shared after feature fusion, so this is much cheaper than
    running two complete ZoomNeXt networks.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_frames: int = 1,
        input_norm: bool = True,
        mid_dim: int = 64,
        siu_groups: int = 4,  # kept for config compatibility
        hmu_groups: int = 6,
        use_checkpoint: bool = False,
        edge_loss_weight: float = 4.0,
        ual_loss_weight: float = 2.0,
        ual_start: float = 0.0,
        ual_full: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        del siu_groups, kwargs

        self.edge_loss_weight = float(edge_loss_weight)
        self.ual_loss_weight = float(ual_loss_weight)
        self.ual_start = float(ual_start)
        self.ual_full = float(ual_full)

        # Load ImageNet weights once, then clone to the BoxPrompt encoder.
        self.encoder_rgb = timm.create_model(
            model_name="convnext_base.fb_in22k_ft_in1k_384",
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
        )
        self.encoder_box = timm.create_model(
            model_name="convnext_base.fb_in22k_ft_in1k_384",
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=False,
        )
        self.encoder_box.load_state_dict(self.encoder_rgb.state_dict(), strict=True)

        if use_checkpoint:
            for encoder in (self.encoder_rgb, self.encoder_box):
                if hasattr(encoder, "set_grad_checkpointing"):
                    encoder.set_grad_checkpointing(enable=True)

        self.embed_dims = list(self.encoder_rgb.feature_info.channels())
        if len(self.embed_dims) != 4:
            raise RuntimeError(
                f"ConvNeXt-B must return 4 stages, got {self.embed_dims}"
            )

        # Each branch gets its own transition modules.
        self.rgb_tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.box_tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)

        self.rgb_tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.box_tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)

        self.rgb_tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.box_tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)

        self.rgb_tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.box_tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)

        # Same-level RGB / BoxPrompt feature fusion.
        self.fuse_5 = ConvBNReLU(mid_dim * 2, mid_dim, 1, 1, 0)
        self.fuse_4 = ConvBNReLU(mid_dim * 2, mid_dim, 1, 1, 0)
        self.fuse_3 = ConvBNReLU(mid_dim * 2, mid_dim, 1, 1, 0)
        self.fuse_2 = ConvBNReLU(mid_dim * 2, mid_dim, 1, 1, 0)

        # Keep the ZoomNeXt top-down RGPU decoder.
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
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

        # Noisy-COD-like deep mask supervision.
        self.aux_head_5 = nn.Conv2d(mid_dim, 1, 1)
        self.aux_head_4 = nn.Conv2d(mid_dim, 1, 1)
        self.aux_head_3 = nn.Conv2d(mid_dim, 1, 1)
        self.aux_head_2 = nn.Conv2d(mid_dim, 1, 1)

        # Boundary heads. We produce four; training supervises the last three
        # exactly like Noisy-COD's progressively stronger edge supervision.
        self.edge_head_4 = nn.Conv2d(mid_dim, 1, 1)
        self.edge_head_3 = nn.Conv2d(mid_dim, 1, 1)
        self.edge_head_2 = nn.Conv2d(mid_dim, 1, 1)
        self.edge_head_1 = nn.Conv2d(mid_dim, 1, 1)

    def _encode(
        self,
        encoder: nn.Module,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.normalizer(image)
        c2, c3, c4, c5 = encoder(image)
        return c2, c3, c4, c5

    @staticmethod
    def _prepare_box_image(
        image: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "box_mask" in data:
            box_mask = data["box_mask"].float()
            if box_mask.ndim == 3:
                box_mask = box_mask.unsqueeze(1)
            if box_mask.shape[-2:] != image.shape[-2:]:
                box_mask = F.interpolate(
                    box_mask,
                    size=image.shape[-2:],
                    mode="nearest",
                )
            box_mask = box_mask.clamp(0.0, 1.0)
        else:
            raise KeyError(
                "ConvNeXtB_ZoomNeXt_ANet requires data['box_mask']. "
                "ANet inference/generation also needs a bounding-box prompt."
            )

        if "box_image" in data:
            box_image = data["box_image"]
            if box_image.shape[-2:] != image.shape[-2:]:
                box_image = F.interpolate(
                    box_image,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            box_image = image * box_mask + mean * (1.0 - box_mask)
            
            

        return box_image, box_mask

    def _forward_features(self, data: Dict[str, torch.Tensor]):
        image = data["image_m"]
        box_image, box_mask = self._prepare_box_image(image, data)

        rgb = self._encode(self.encoder_rgb, image)
        box = self._encode(self.encoder_box, box_image)

        f5 = self.fuse_5(
            torch.cat(
                [self.rgb_tra_5(rgb[3]), self.box_tra_5(box[3])],
                dim=1,
            )
        )
        d5 = self.hmu_5(f5)

        f4 = self.fuse_4(
            torch.cat(
                [self.rgb_tra_4(rgb[2]), self.box_tra_4(box[2])],
                dim=1,
            )
        )
        d4 = self.hmu_4(f4 + resize_to(d5, tgt_hw=f4.shape[-2:]))

        f3 = self.fuse_3(
            torch.cat(
                [self.rgb_tra_3(rgb[1]), self.box_tra_3(box[1])],
                dim=1,
            )
        )
        d3 = self.hmu_3(f3 + resize_to(d4, tgt_hw=f3.shape[-2:]))

        f2 = self.fuse_2(
            torch.cat(
                [self.rgb_tra_2(rgb[0]), self.box_tra_2(box[0])],
                dim=1,
            )
        )
        d2 = self.hmu_2(f2 + resize_to(d3, tgt_hw=f2.shape[-2:]))

        d1 = self.tra_1(d2)
        final_logits = self.predictor(d1)

        target_hw = image.shape[-2:]
        mask_logits = [
            F.interpolate(
                self.aux_head_5(d5),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                self.aux_head_4(d4),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                self.aux_head_3(d3),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                self.aux_head_2(d2),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ),
            final_logits,
        ]

        edge_logits = [
            F.interpolate(
                self.edge_head_4(d4),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                self.edge_head_3(d3),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                self.edge_head_2(d2),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                self.edge_head_1(d1),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ),
        ]

        return {
            "logits": final_logits,
            "mask_logits": mask_logits,
            "edge_logits": edge_logits,
            "box_mask": box_mask,
        }

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        iter_percentage: float = 1.0,
        return_aux: bool = False,
        **kwargs,
    ):
        del kwargs
        out = self._forward_features(data)
        final_logits = out["logits"]

        if not self.training:
            if return_aux:
                return out
            return final_logits

        if "mask" not in data:
            raise KeyError(
                "ANet training requires dense data['mask'] on the fully "
                "annotated subset. Box-only samples are used later for "
                "pseudo-label generation, not for this ANet training loss."
            )

        mask = data["mask"].float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != final_logits.shape[-2:]:
            mask = F.interpolate(
                mask,
                size=final_logits.shape[-2:],
                mode="nearest",
            )

        p0, p4, p3, p2, p1 = out["mask_logits"]
        structure = (
            (1.0 / 16.0) * _structure_loss(p0, mask)
            + (1.0 / 8.0) * _structure_loss(p4, mask)
            + (1.0 / 4.0) * _structure_loss(p3, mask)
            + (1.0 / 2.0) * _structure_loss(p2, mask)
            + _structure_loss(p1, mask)
        )

        boundary_target = _mask_to_boundary(mask, kernel_size=5)
        _, e3, e2, e1 = out["edge_logits"]
        edge_raw = (
            (1.0 / 8.0) * _dice_loss_from_logits(e3, boundary_target)
            + (1.0 / 4.0) * _dice_loss_from_logits(e2, boundary_target)
            + (1.0 / 2.0) * _dice_loss_from_logits(e1, boundary_target)
        )
        edge_loss = self.edge_loss_weight * edge_raw

        prob = torch.sigmoid(final_logits)
        ual_raw = (1.0 - (2.0 * prob - 1.0).abs().pow(2)).mean()
        ual_coef = _cosine_coef(
            iter_percentage,
            start=self.ual_start,
            end=self.ual_full,
        )
        ual_loss = self.ual_loss_weight * float(ual_coef) * ual_raw

        total_loss = structure + edge_loss + ual_loss

        return {
            "logits": final_logits,
            "loss": total_loss,
            "loss_items": {
                "structure": structure.detach(),
                "edge": edge_loss.detach(),
                "edge_raw": edge_raw.detach(),
                "ual": ual_loss.detach(),
                "ual_raw": ual_raw.detach(),
                "ual_coef": float(ual_coef),
                "total": total_loss.detach(),
            },
            "loss_str": (
                f"L:{total_loss.detach().item():.4f} "
                f"STR:{structure.detach().item():.4f} "
                f"EDGE:{edge_loss.detach().item():.4f} "
                f"UAL:{ual_loss.detach().item():.4f}"
            ),
            "vis": {
                "sal": prob,
                "box": out["box_mask"],
                "boundary": torch.sigmoid(e1),
            },
        }

    def get_grouped_params(self):
        """Compatible with the repository's finetune-style optimizers."""
        groups = {"pretrained": [], "fixed": [], "retrained": []}

        for name, param in self.named_parameters():
            if name.startswith("encoder_rgb.") or name.startswith("encoder_box."):
                groups["pretrained"].append(param)
            else:
                groups["retrained"].append(param)

        LOGGER.info(
            "ANet Parameter Groups:{"
            f"Pretrained: {len(groups['pretrained'])}, "
            f"Fixed: {len(groups['fixed'])}, "
            f"ReTrained: {len(groups['retrained'])}"
            "}"
        )
        return groups


class ConvNeXtB_ZoomNeXt_APNet(nn.Module):
    """
    One-stage APNet for 10% dense GT + box supervision.

    Training:
        RGB --------------------------> shared ConvNeXt-B ----> RGB-only student decoder
          \\                                                       |
           \\                                                      +--> student mask
            \\
             + Box --> RGB*Box --> Box ConvNeXt-B --> fused ANet teacher decoder
                                                            |
                                                            +--> teacher mask

        Both student and teacher are supervised by the same dense GT.
        The student additionally receives prediction and localization
        distillation from the box-guided teacher.

    Inference:
        RGB --> shared ConvNeXt-B --> RGB-only student decoder --> mask

    IMPORTANT:
        `box_mask` is required only while `self.training == True`.
        Evaluation/inference never reads box_mask.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_frames: int = 1,
        input_norm: bool = True,
        mid_dim: int = 64,
        siu_groups: int = 4,  # config compatibility
        hmu_groups: int = 6,
        use_checkpoint: bool = False,
        edge_loss_weight: float = 4.0,
        ual_loss_weight: float = 2.0,
        ual_start: float = 0.0,
        ual_full: float = 1.0,
        # ---- APNet one-stage transfer weights ----
        teacher_loss_weight: float = 0.5,
        pred_kd_weight: float = 0.5,
        loc_kd_weight: float = 0.0,
        weak_kd_weight: float = 1.0,
        outside_weight: float = 0.25,
        weak_conf_thresh: float = 0.70,
        kd_start: float = 0.20,
        kd_full: float = 0.50,
        kd_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        del siu_groups, kwargs

        self.edge_loss_weight = float(edge_loss_weight)
        self.ual_loss_weight = float(ual_loss_weight)
        self.ual_start = float(ual_start)
        self.ual_full = float(ual_full)

        self.teacher_loss_weight = float(teacher_loss_weight)
        self.pred_kd_weight = float(pred_kd_weight)
        self.loc_kd_weight = float(loc_kd_weight)
        self.weak_kd_weight = float(weak_kd_weight)
        self.outside_weight = float(outside_weight)
        self.weak_conf_thresh = float(weak_conf_thresh)
        self.kd_start = float(kd_start)
        self.kd_full = float(kd_full)
        self.kd_temperature = float(kd_temperature)

        # ------------------------------------------------------------------
        # Shared RGB backbone: this is the ONLY backbone used at inference.
        # ------------------------------------------------------------------
        self.encoder_rgb = timm.create_model(
            model_name="convnext_base",
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
        )

        # ------------------------------------------------------------------
        # Training-only box teacher backbone.
        # Kept from the original ANet so the first one-stage version has a
        # strong box-guided teacher and minimal architectural risk.
        # ------------------------------------------------------------------
        self.encoder_box = timm.create_model(
            model_name="convnext_base",
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=False,
        )
        self.encoder_box.load_state_dict(self.encoder_rgb.state_dict(), strict=True)

        if use_checkpoint:
            for encoder in (self.encoder_rgb, self.encoder_box):
                if hasattr(encoder, "set_grad_checkpointing"):
                    encoder.set_grad_checkpointing(enable=True)

        self.embed_dims = list(self.encoder_rgb.feature_info.channels())
        if len(self.embed_dims) != 4:
            raise RuntimeError(
                f"ConvNeXt-B must return 4 stages, got {self.embed_dims}"
            )

        # ------------------------------------------------------------------
        # Shared RGB transitions. Both teacher and student start from these
        # RGB features, which helps transfer box-guided knowledge into the
        # RGB representation itself.
        # ------------------------------------------------------------------
        self.rgb_tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.rgb_tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.rgb_tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.rgb_tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)

        # Training-only box transitions for the ANet teacher.
        self.box_tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.box_tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.box_tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.box_tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)

        # Original ANet same-level fusion: teacher only.
        self.fuse_5 = ConvBNReLU(mid_dim * 2, mid_dim, 1, 1, 0)
        self.fuse_4 = ConvBNReLU(mid_dim * 2, mid_dim, 1, 1, 0)
        self.fuse_3 = ConvBNReLU(mid_dim * 2, mid_dim, 1, 1, 0)
        self.fuse_2 = ConvBNReLU(mid_dim * 2, mid_dim, 1, 1, 0)

        # ------------------------------------------------------------------
        # Teacher decoder = original ANet decoder.
        # ------------------------------------------------------------------
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

        self.aux_head_5 = nn.Conv2d(mid_dim, 1, 1)
        self.aux_head_4 = nn.Conv2d(mid_dim, 1, 1)
        self.aux_head_3 = nn.Conv2d(mid_dim, 1, 1)
        self.aux_head_2 = nn.Conv2d(mid_dim, 1, 1)

        self.edge_head_4 = nn.Conv2d(mid_dim, 1, 1)
        self.edge_head_3 = nn.Conv2d(mid_dim, 1, 1)
        self.edge_head_2 = nn.Conv2d(mid_dim, 1, 1)
        self.edge_head_1 = nn.Conv2d(mid_dim, 1, 1)

        # ------------------------------------------------------------------
        # RGB-only student decoder. This entire path is box-free.
        # It mirrors ZoomNeXt so comparison with the two-stage baseline stays
        # controlled and the first version focuses on training paradigm.
        # ------------------------------------------------------------------
        self.student_hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
        self.student_hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
        self.student_hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
        self.student_hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.student_tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_dim, mid_dim, 3, 1, 1),
        )
        self.student_predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_dim, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

        self.student_aux_head_5 = nn.Conv2d(mid_dim, 1, 1)
        self.student_aux_head_4 = nn.Conv2d(mid_dim, 1, 1)
        self.student_aux_head_3 = nn.Conv2d(mid_dim, 1, 1)
        self.student_aux_head_2 = nn.Conv2d(mid_dim, 1, 1)

        self.student_edge_head_4 = nn.Conv2d(mid_dim, 1, 1)
        self.student_edge_head_3 = nn.Conv2d(mid_dim, 1, 1)
        self.student_edge_head_2 = nn.Conv2d(mid_dim, 1, 1)
        self.student_edge_head_1 = nn.Conv2d(mid_dim, 1, 1)

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()

        LOGGER.info(
            "APNet initialized: one-stage 10%%GT+Box training, RGB-only inference. "
            "TeacherLoss=%.3f PredKD=%.3f LocKD=%.3f KD[%.2f->%.2f]",
            self.teacher_loss_weight,
            self.pred_kd_weight,
            self.loc_kd_weight,
            self.kd_start,
            self.kd_full,
        )

    def _encode(
        self,
        encoder: nn.Module,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.normalizer(image)
        c2, c3, c4, c5 = encoder(image)
        return c2, c3, c4, c5

    @staticmethod
    def _prepare_box_image(
        image: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "box_mask" not in data:
            raise KeyError(
                "APNet training requires data['box_mask']; "
                "box_mask is NOT required during eval/inference."
            )

        box_mask = data["box_mask"].float()
        if box_mask.ndim == 3:
            box_mask = box_mask.unsqueeze(1)
        if box_mask.shape[-2:] != image.shape[-2:]:
            box_mask = F.interpolate(
                box_mask,
                size=image.shape[-2:],
                mode="nearest",
            )
        box_mask = box_mask.clamp(0.0, 1.0)

        if "box_image" in data:
            box_image = data["box_image"]
            if box_image.shape[-2:] != image.shape[-2:]:
                box_image = F.interpolate(
                    box_image,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            box_image = image * box_mask

        return box_image, box_mask

    @staticmethod
    def _resize_logits(logits: torch.Tensor, target_hw) -> torch.Tensor:
        if logits.shape[-2:] == tuple(target_hw):
            return logits
        return F.interpolate(
            logits,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

    def _rgb_transition_features(
        self,
        rgb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """Shared RGB feature projections used by both teacher and student."""
        r5 = self.rgb_tra_5(rgb[3])
        r4 = self.rgb_tra_4(rgb[2])
        r3 = self.rgb_tra_3(rgb[1])
        r2 = self.rgb_tra_2(rgb[0])
        return r2, r3, r4, r5

    def _forward_student_from_rgb_features(
        self,
        rgb_feats,
        target_hw,
    ):
        """
        RGB-only student path.
        No box feature is consumed anywhere in this function.
        """
        r2, r3, r4, r5 = rgb_feats

        d5 = self.student_hmu_5(r5)
        d4 = self.student_hmu_4(
            r4 + resize_to(d5, tgt_hw=r4.shape[-2:])
        )
        d3 = self.student_hmu_3(
            r3 + resize_to(d4, tgt_hw=r3.shape[-2:])
        )
        d2 = self.student_hmu_2(
            r2 + resize_to(d3, tgt_hw=r2.shape[-2:])
        )
        d1 = self.student_tra_1(d2)
        final_logits = self.student_predictor(d1)

        mask_logits = [
            self._resize_logits(self.student_aux_head_5(d5), target_hw),
            self._resize_logits(self.student_aux_head_4(d4), target_hw),
            self._resize_logits(self.student_aux_head_3(d3), target_hw),
            self._resize_logits(self.student_aux_head_2(d2), target_hw),
            self._resize_logits(final_logits, target_hw),
        ]

        edge_logits = [
            self._resize_logits(self.student_edge_head_4(d4), target_hw),
            self._resize_logits(self.student_edge_head_3(d3), target_hw),
            self._resize_logits(self.student_edge_head_2(d2), target_hw),
            self._resize_logits(self.student_edge_head_1(d1), target_hw),
        ]

        return {
            "logits": self._resize_logits(final_logits, target_hw),
            "mask_logits": mask_logits,
            "edge_logits": edge_logits,
            # multi-scale decoder features for localization transfer
            "decoder_features": (d2, d3, d4, d5),
        }

    def _forward_teacher_from_features(
        self,
        rgb_feats,
        box_feats,
        target_hw,
        box_mask,
    ):
        """
        Original RGB+Box ANet path, used only during training.
        """
        r2, r3, r4, r5 = rgb_feats
        b2, b3, b4, b5 = box_feats

        f5 = self.fuse_5(torch.cat([r5, b5], dim=1))
        d5 = self.hmu_5(f5)

        f4 = self.fuse_4(torch.cat([r4, b4], dim=1))
        d4 = self.hmu_4(f4 + resize_to(d5, tgt_hw=f4.shape[-2:]))

        f3 = self.fuse_3(torch.cat([r3, b3], dim=1))
        d3 = self.hmu_3(f3 + resize_to(d4, tgt_hw=f3.shape[-2:]))

        f2 = self.fuse_2(torch.cat([r2, b2], dim=1))
        d2 = self.hmu_2(f2 + resize_to(d3, tgt_hw=f2.shape[-2:]))

        d1 = self.tra_1(d2)
        final_logits = self.predictor(d1)

        mask_logits = [
            self._resize_logits(self.aux_head_5(d5), target_hw),
            self._resize_logits(self.aux_head_4(d4), target_hw),
            self._resize_logits(self.aux_head_3(d3), target_hw),
            self._resize_logits(self.aux_head_2(d2), target_hw),
            self._resize_logits(final_logits, target_hw),
        ]

        edge_logits = [
            self._resize_logits(self.edge_head_4(d4), target_hw),
            self._resize_logits(self.edge_head_3(d3), target_hw),
            self._resize_logits(self.edge_head_2(d2), target_hw),
            self._resize_logits(self.edge_head_1(d1), target_hw),
        ]

        return {
            "logits": self._resize_logits(final_logits, target_hw),
            "mask_logits": mask_logits,
            "edge_logits": edge_logits,
            "box_mask": box_mask,
            "decoder_features": (d2, d3, d4, d5),
        }

    def _forward_student(self, image: torch.Tensor):
        """Standalone RGB-only inference path."""
        rgb = self._encode(self.encoder_rgb, image)
        rgb_feats = self._rgb_transition_features(rgb)
        return self._forward_student_from_rgb_features(
            rgb_feats=rgb_feats,
            target_hw=image.shape[-2:],
        )

    @staticmethod
    def _spatial_attention(feature: torch.Tensor) -> torch.Tensor:
        """
        Convert CxHxW feature into a normalized spatial localization map.
        We distill *where to look*, not raw feature vectors.
        """
        att = feature.pow(2).mean(dim=1, keepdim=True)
        flat = att.flatten(1)
        flat = flat / (flat.norm(p=2, dim=1, keepdim=True) + 1e-6)
        return flat.view_as(att)

    def _localization_distill_loss(
        self,
        student_features,
        teacher_features,
    ) -> torch.Tensor:
        losses = []
        for fs, ft in zip(student_features, teacher_features):
            a_s = self._spatial_attention(fs)
            a_t = self._spatial_attention(ft.detach())
            if a_s.shape[-2:] != a_t.shape[-2:]:
                a_t = F.interpolate(
                    a_t,
                    size=a_s.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            losses.append(F.smooth_l1_loss(a_s, a_t))
        if not losses:
            return torch.zeros((), device=student_features[0].device)
        return torch.stack(losses).mean()

    def _prediction_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary soft-target distillation.
        Teacher is stop-gradient so KD cannot collapse teacher toward student.
        """
        t = max(self.kd_temperature, 1e-6)
        teacher_prob = torch.sigmoid(teacher_logits.detach() / t)
        loss = F.binary_cross_entropy_with_logits(
            student_logits / t,
            teacher_prob,
        )
        return loss * (t * t)

    def _masked_prediction_distill_loss(self, student_logits, teacher_logits, box_mask=None):
        """Confidence-masked online KD for box-only samples."""
        t = max(self.kd_temperature, 1e-6)
        with torch.no_grad():
            tp = torch.sigmoid(teacher_logits.detach() / t)
            conf = torch.maximum(tp, 1.0 - tp)
            valid = (conf >= self.weak_conf_thresh).float()
            if box_mask is not None:
                # Keep all confident pixels; outside box will additionally receive a hard background constraint.
                if box_mask.shape[-2:] != valid.shape[-2:]:
                    box_mask = F.interpolate(box_mask.float(), size=valid.shape[-2:], mode="nearest")
        per = F.binary_cross_entropy_with_logits(student_logits / t, tp, reduction="none")
        denom = valid.sum().clamp_min(1.0)
        return (per * valid).sum() / denom * (t * t)

    @staticmethod
    def _outside_box_loss(student_logits, box_mask):
        if box_mask.shape[-2:] != student_logits.shape[-2:]:
            box_mask = F.interpolate(box_mask.float(), size=student_logits.shape[-2:], mode="nearest")
        outside = (1.0 - box_mask.float()).clamp(0.0, 1.0)
        target = torch.zeros_like(student_logits)
        per = F.binary_cross_entropy_with_logits(student_logits, target, reduction="none")
        return (per * outside).sum() / outside.sum().clamp_min(1.0)

    def _segmentation_loss(
        self,
        out: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        iter_percentage: float,
    ):
        """Original ANet/Noisy-COD-style GT supervision, reused for both paths."""
        final_logits = out["logits"]

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        if mask.shape[-2:] != final_logits.shape[-2:]:
            mask = F.interpolate(
                mask,
                size=final_logits.shape[-2:],
                mode="nearest",
            )

        p0, p4, p3, p2, p1 = out["mask_logits"]
        structure = (
            (1.0 / 16.0) * _structure_loss(p0, mask)
            + (1.0 / 8.0) * _structure_loss(p4, mask)
            + (1.0 / 4.0) * _structure_loss(p3, mask)
            + (1.0 / 2.0) * _structure_loss(p2, mask)
            + _structure_loss(p1, mask)
        )

        boundary_target = _mask_to_boundary(mask, kernel_size=5)
        _, e3, e2, e1 = out["edge_logits"]
        edge_raw = (
            (1.0 / 8.0) * _dice_loss_from_logits(e3, boundary_target)
            + (1.0 / 4.0) * _dice_loss_from_logits(e2, boundary_target)
            + (1.0 / 2.0) * _dice_loss_from_logits(e1, boundary_target)
        )
        edge_loss = self.edge_loss_weight * edge_raw

        prob = torch.sigmoid(final_logits)
        ual_raw = (1.0 - (2.0 * prob - 1.0).abs().pow(2)).mean()
        ual_coef = _cosine_coef(
            iter_percentage,
            start=self.ual_start,
            end=self.ual_full,
        )
        ual_loss = self.ual_loss_weight * float(ual_coef) * ual_raw

        total = structure + edge_loss + ual_loss
        return {
            "total": total,
            "structure": structure,
            "edge": edge_loss,
            "edge_raw": edge_raw,
            "ual": ual_loss,
            "ual_raw": ual_raw,
            "ual_coef": float(ual_coef),
            "prob": prob,
            "boundary": torch.sigmoid(e1),
        }

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        iter_percentage: float = 1.0,
        return_aux: bool = False,
        **kwargs,
    ):
        del kwargs
        if "image_m" not in data:
            raise KeyError("APNet requires data['image_m'].")
        image = data["image_m"]

        # Inference is strictly RGB-only.
        if not self.training:
            student_out = self._forward_student(image)
            if return_aux:
                return {**student_out, "student_logits": student_out["logits"]}
            return student_out["logits"]

        if "box_mask" not in data:
            raise KeyError("APNet training requires data['box_mask'].")

        # Shared RGB encoder is evaluated once for both student and teacher.
        rgb = self._encode(self.encoder_rgb, image)
        rgb_feats = self._rgb_transition_features(rgb)
        student_out = self._forward_student_from_rgb_features(rgb_feats, image.shape[-2:])

        box_image, box_mask = self._prepare_box_image(image, data)
        box = self._encode(self.encoder_box, box_image)
        box_feats = (
            self.box_tra_2(box[0]), self.box_tra_3(box[1]),
            self.box_tra_4(box[2]), self.box_tra_5(box[3]),
        )
        teacher_out = self._forward_teacher_from_features(
            rgb_feats, box_feats, image.shape[-2:], box_mask
        )

        kd_coef = _cosine_coef(iter_percentage, start=self.kd_start, end=self.kd_full)
        is_fully = "mask" in data

        if is_fully:
            mask = data["mask"].float()
            student_loss = self._segmentation_loss(student_out, mask, iter_percentage)
            teacher_loss = self._segmentation_loss(teacher_out, mask, iter_percentage)
            pred_kd_raw = self._prediction_distill_loss(student_out["logits"], teacher_out["logits"])
            pred_kd = self.pred_kd_weight * float(kd_coef) * pred_kd_raw
            loc_kd_raw = self._localization_distill_loss(student_out["decoder_features"], teacher_out["decoder_features"])
            loc_kd = self.loc_kd_weight * float(kd_coef) * loc_kd_raw
            total_loss = student_loss["total"] + self.teacher_loss_weight * teacher_loss["total"] + pred_kd + loc_kd
            return {
                "logits": student_out["logits"], "student_logits": student_out["logits"],
                "teacher_logits": teacher_out["logits"], "loss": total_loss,
                "loss_items": {
                    "structure": student_loss["structure"].detach(), "edge": student_loss["edge"].detach(),
                    "ual": student_loss["ual"].detach(), "student_total": student_loss["total"].detach(),
                    "teacher_total": teacher_loss["total"].detach(), "pred_kd": pred_kd.detach(),
                    "loc_kd": loc_kd.detach(), "weak_kd": torch.zeros_like(total_loss.detach()),
                    "outside": torch.zeros_like(total_loss.detach()), "kd_coef": float(kd_coef), "total": total_loss.detach(),
                },
                "loss_str": f"FULL L:{total_loss.detach().item():.4f} S:{student_loss['total'].detach().item():.4f} T:{teacher_loss['total'].detach().item():.4f} PKD:{pred_kd.detach().item():.4f} K:{kd_coef:.3f}",
                "vis": {"sal": student_loss["prob"], "teacher_sal": teacher_loss["prob"], "box": box_mask, "boundary": student_loss["boundary"]},
            }

        # Weak box-only sample: teacher is an online target; no pseudo mask is stored.
        weak_kd_raw = self._masked_prediction_distill_loss(student_out["logits"], teacher_out["logits"], box_mask)
        outside_raw = self._outside_box_loss(student_out["logits"], box_mask)
        weak_kd = self.weak_kd_weight * float(kd_coef) * weak_kd_raw
        outside = self.outside_weight * outside_raw
        total_loss = weak_kd + outside
        zero = total_loss.detach() * 0.0
        return {
            "logits": student_out["logits"], "student_logits": student_out["logits"],
            "teacher_logits": teacher_out["logits"], "loss": total_loss,
            "loss_items": {
                "structure": zero, "edge": zero, "ual": zero, "pred_kd": zero, "loc_kd": zero,
                "weak_kd": weak_kd.detach(), "weak_kd_raw": weak_kd_raw.detach(),
                "outside": outside.detach(), "outside_raw": outside_raw.detach(),
                "kd_coef": float(kd_coef), "total": total_loss.detach(),
            },
            "loss_str": f"WEAK L:{total_loss.detach().item():.4f} WKD:{weak_kd.detach().item():.4f} OUT:{outside.detach().item():.4f} K:{kd_coef:.3f}",
            "vis": {"sal": torch.sigmoid(student_out["logits"]), "teacher_sal": torch.sigmoid(teacher_out["logits"]), "box": box_mask},
        }

    def get_grouped_params(self):
        """Compatible with the repository's finetune-style optimizers."""
        groups = {"pretrained": [], "fixed": [], "retrained": []}

        for name, param in self.named_parameters():
            if name.startswith("encoder_rgb.") or name.startswith("encoder_box."):
                groups["pretrained"].append(param)
            else:
                groups["retrained"].append(param)

        LOGGER.info(
            "APNet Parameter Groups:{"
            f"Pretrained: {len(groups['pretrained'])}, "
            f"Fixed: {len(groups['fixed'])}, "
            f"ReTrained: {len(groups['retrained'])}"
            "}"
        )
        return groups

# ============================================================================
# Unified model factory
# ============================================================================
MODEL_REGISTRY = {
    "anet": ConvNeXtB_ZoomNeXt_ANet,
    "apnet": ConvNeXtB_ZoomNeXt_APNet,
}


def build_anet_or_apnet(
    model_type: str = "anet",
    **kwargs,
):
    """
    Build either the original ANet or one-stage APNet.

    Args:
        model_type:
            "anet"  -> ConvNeXtB_ZoomNeXt_ANet
            "apnet" -> ConvNeXtB_ZoomNeXt_APNet
        **kwargs:
            Forwarded to the selected model constructor.
    """
    key = str(model_type).lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type={model_type!r}. "
            f"Supported: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](**kwargs)
