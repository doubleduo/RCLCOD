# -*- coding: utf-8 -*-
"""Improved PVTv2-B4 FPN for clean/noisy/unvalue curriculum training.

Drop-in replacement for ``methods/fpn_unvalue_nc.py``.  The public class name
is unchanged, so the existing ``basemain_unvalue.py`` registration continues
to work.

Main changes:
1. Decoder normalization can use GroupNorm (default) or the original BatchNorm.
2. Clean masks receive a small Soft-IoU term in addition to BCE.
3. Reliable unvalue pixels can learn the EMA teacher target directly instead
   of mixing it with a mask already declared unreliable.
4. An optional P3 auxiliary head is available for a separate ablation and is
   disabled by default.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline
from .zoomnext.ops import ConvGNReLU


def _cosine_value(progress, start, end, low, high):
    progress = float(min(max(progress, 0.0), 1.0))
    if progress <= start:
        return float(low)
    if progress >= end:
        return float(high)
    local = (progress - start) / max(end - start, 1e-8)
    coefficient = 0.5 * (1.0 - math.cos(math.pi * local))
    return float(low + (high - low) * coefficient)


def _resize(value, size, mode="nearest"):
    if value.shape[-2:] == size:
        return value
    if mode == "nearest":
        return F.interpolate(value.float(), size=size, mode=mode)
    return F.interpolate(
        value.float(), size=size, mode=mode, align_corners=False
    )


def _dilate(mask, kernel_size):
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    return F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )


def _sample_mean(value, pixel_weight, eps=1e-6):
    numerator = (value * pixel_weight).flatten(1).sum(dim=1)
    denominator = pixel_weight.flatten(1).sum(dim=1).clamp_min(eps)
    return numerator / denominator


def _selected_mean(value, selector):
    selector = selector.to(device=value.device, dtype=value.dtype)
    return (value * selector).sum() / selector.sum().clamp_min(1.0)


def _soft_iou_each(logits, target, pixel_weight=None, eps=1e-6):
    probability = logits.sigmoid()
    if pixel_weight is None:
        pixel_weight = torch.ones_like(probability)
    intersection = (
        pixel_weight * probability * target
    ).flatten(1).sum(dim=1)
    union = (
        pixel_weight
        * (probability + target - probability * target)
    ).flatten(1).sum(dim=1)
    return 1.0 - (intersection + eps) / (union + eps)


class MaskAnchoredUnvalueLoss(nn.Module):
    """Dense mask anchors plus box-constrained teacher supervision."""

    def __init__(
        self,
        clean_iou_weight=0.20,
        noisy_mask_weight=1.0,
        box_dilate_kernel=9,
        teacher_confidence=0.90,
        teacher_disagreement=0.05,
        min_teacher_foreground=16,
        outside_weight=0.20,
        dynamic_weight=0.25,
        consistency_weight=0.05,
        teacher_target_mode="teacher",
        dynamic_mix_max=0.50,
        eps=1e-6,
        **unused_kwargs,
    ):
        super().__init__()
        del unused_kwargs
        if teacher_target_mode not in {"teacher", "blend"}:
            raise ValueError(
                "teacher_target_mode must be 'teacher' or 'blend'."
            )
        self.clean_iou_weight = float(clean_iou_weight)
        self.noisy_mask_weight = float(noisy_mask_weight)
        self.box_dilate_kernel = int(box_dilate_kernel)
        self.teacher_confidence = float(teacher_confidence)
        self.teacher_disagreement = float(teacher_disagreement)
        self.min_teacher_foreground = int(min_teacher_foreground)
        self.outside_weight = float(outside_weight)
        self.dynamic_weight = float(dynamic_weight)
        self.consistency_weight = float(consistency_weight)
        self.teacher_target_mode = str(teacher_target_mode)
        self.dynamic_mix_max = float(dynamic_mix_max)
        self.eps = float(eps)

    def _mask_pool_loss(self, logits, target, pool_id):
        clean_selector = pool_id == 0
        noisy_selector = pool_id == 1
        mask_selector = clean_selector | noisy_selector

        bce_each = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        ).flatten(1).mean(dim=1)
        iou_each = _soft_iou_each(
            logits, target, eps=self.eps
        )

        sample_weight = (
            clean_selector.to(logits.dtype)
            + self.noisy_mask_weight * noisy_selector.to(logits.dtype)
        )
        per_sample = bce_each + (
            self.clean_iou_weight
            * clean_selector.to(logits.dtype)
            * iou_each
        )
        mask_loss = (
            per_sample * sample_weight
        ).sum() / sample_weight.sum().clamp_min(1.0)

        return (
            mask_loss,
            _selected_mean(bce_each, mask_selector),
            _selected_mean(iou_each, clean_selector),
        )

    def _unvalue_loss(
        self,
        logits,
        original_target,
        box_mask,
        teacher_prob,
        teacher_disagreement,
        teacher_valid,
        unvalue_selector,
        progress,
    ):
        zero = logits.sum() * 0.0
        if box_mask is None:
            return zero, zero, zero, zero, zero

        box = (box_mask >= 0.5).to(logits.dtype)

        # A dilated box provides a neutral ring around the annotation border.
        # Only pixels safely outside it are supervised as background.
        outside = 1.0 - _dilate(box, self.box_dilate_kernel)
        outside_each = _sample_mean(
            F.softplus(logits), outside, self.eps
        )
        outside_loss = _selected_mean(
            outside_each, unvalue_selector
        )

        if teacher_prob is None:
            return outside_loss, zero, zero, zero, zero

        teacher = teacher_prob.detach().clamp(0.0, 1.0)
        if teacher_disagreement is None:
            stable = torch.ones_like(teacher)
        else:
            stable = (
                teacher_disagreement.detach()
                <= self.teacher_disagreement
            ).to(logits.dtype)

        confident = (
            (teacher >= self.teacher_confidence)
            | (teacher <= 1.0 - self.teacher_confidence)
        ).to(logits.dtype)
        reliable = box * stable * confident

        reliable_foreground = (
            box
            * stable
            * (teacher >= self.teacher_confidence).to(logits.dtype)
        )
        foreground_count = reliable_foreground.flatten(1).sum(1)
        has_foreground = (
            foreground_count >= self.min_teacher_foreground
        ).to(logits.dtype)
        if teacher_valid is not None:
            has_foreground = (
                has_foreground * teacher_valid.to(logits.dtype)
            )
        valid_unvalue = (
            unvalue_selector.to(logits.dtype) * has_foreground
        )

        if self.teacher_target_mode == "teacher":
            dynamic_target = teacher
        else:
            mix = _cosine_value(
                progress,
                start=40.0 / 150.0,
                end=90.0 / 150.0,
                low=0.20,
                high=self.dynamic_mix_max,
            )
            dynamic_target = (
                (1.0 - mix) * original_target + mix * teacher
            )
        dynamic_target = dynamic_target.detach()

        dynamic_each = _soft_iou_each(
            logits,
            dynamic_target,
            pixel_weight=reliable,
            eps=self.eps,
        )
        dynamic_loss = _selected_mean(
            dynamic_each, valid_unvalue
        )

        probability = logits.sigmoid()
        consistency_each = _sample_mean(
            (probability - teacher).square(),
            reliable,
            self.eps,
        )
        consistency_loss = _selected_mean(
            consistency_each, valid_unvalue
        )

        reliable_count = reliable.flatten(1).sum(1)
        box_count = box.flatten(1).sum(1).clamp_min(1.0)
        reliable_ratio_each = reliable_count / box_count
        reliable_ratio = _selected_mean(
            reliable_ratio_each, unvalue_selector
        )
        accepted_ratio = _selected_mean(
            has_foreground, unvalue_selector
        )
        return (
            outside_loss,
            dynamic_loss,
            consistency_loss,
            reliable_ratio,
            accepted_ratio,
        )

    def forward(self, logits, target, data, iter_percentage):
        progress = float(iter_percentage)
        batch_size = logits.shape[0]
        device, dtype = logits.device, logits.dtype

        pool_id = data.get("pool_id")
        if pool_id is None:
            pool_id = torch.zeros(
                batch_size, device=device, dtype=torch.long
            )
        pool_id = pool_id.to(device=device).reshape(batch_size, -1)[:, 0]
        unvalue_selector = pool_id == 2

        mask_loss, bce, clean_iou = self._mask_pool_loss(
            logits, target, pool_id
        )

        box_mask = data.get("box_mask")
        if box_mask is not None:
            box_mask = _resize(
                box_mask.to(device=device, dtype=dtype),
                logits.shape[-2:],
            ).clamp(0.0, 1.0)

        teacher_prob = data.get("teacher_prob")
        if teacher_prob is not None:
            teacher_prob = _resize(
                teacher_prob.to(device=device, dtype=dtype),
                logits.shape[-2:],
                mode="bilinear",
            )
        disagreement = data.get("teacher_disagreement")
        if disagreement is not None:
            disagreement = _resize(
                disagreement.to(device=device, dtype=dtype),
                logits.shape[-2:],
                mode="bilinear",
            )
        teacher_valid = data.get("teacher_valid")
        if teacher_valid is not None:
            teacher_valid = teacher_valid.to(
                device=device, dtype=dtype
            ).reshape(-1)

        (
            outside,
            dynamic,
            consistency,
            reliable_ratio,
            accepted_ratio,
        ) = self._unvalue_loss(
            logits=logits,
            original_target=target,
            box_mask=box_mask,
            teacher_prob=teacher_prob,
            teacher_disagreement=disagreement,
            teacher_valid=teacher_valid,
            unvalue_selector=unvalue_selector,
            progress=progress,
        )

        box_gate = _cosine_value(
            progress,
            40.0 / 150.0,
            60.0 / 150.0,
            0.0,
            self.outside_weight,
        )
        dynamic_gate = _cosine_value(
            progress,
            60.0 / 150.0,
            90.0 / 150.0,
            0.0,
            self.dynamic_weight,
        )
        consistency_gate = _cosine_value(
            progress,
            60.0 / 150.0,
            90.0 / 150.0,
            0.0,
            self.consistency_weight,
        )

        total = (
            mask_loss
            + box_gate * outside
            + dynamic_gate * dynamic
            + consistency_gate * consistency
        )
        return total, {
            "bce": bce.detach(),
            "clean_iou": clean_iou.detach(),
            "unvalue_out": outside.detach(),
            "unvalue_out_weight": logits.new_tensor(box_gate),
            "unvalue_dynamic": dynamic.detach(),
            "unvalue_dynamic_weight": logits.new_tensor(dynamic_gate),
            "unvalue_consistency": consistency.detach(),
            "unvalue_reliable_pixel": reliable_ratio.detach(),
            "unvalue_accepted": accepted_ratio.detach(),
            "unvalue_batch_count": (
                unvalue_selector.sum().detach().to(dtype)
            ),
        }


class PvtV2B4_FPN_Unvalue(PvtV2B4_FPN_Baseline):
    """PVTv2-B4 FPN with robust curriculum supervision."""

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        decoder_norm="gn",
        gn_groups=8,
        aux_p3_weight=0.0,
        **kwargs,
    ):
        loss_keys = {
            "clean_iou_weight",
            "noisy_mask_weight",
            "box_dilate_kernel",
            "teacher_confidence",
            "teacher_disagreement",
            "min_teacher_foreground",
            "outside_weight",
            "dynamic_weight",
            "consistency_weight",
            "teacher_target_mode",
            "dynamic_mix_max",
            # Accepted only for backward-compatible old configs.
            "mask_loss_mode",
            "q_switch_ratio",
            "boundary_kernel",
            "boundary_gain",
        }
        loss_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if key in loss_keys
        }
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

        decoder_norm = str(decoder_norm).lower()
        if decoder_norm not in {"bn", "gn"}:
            raise ValueError("decoder_norm must be 'bn' or 'gn'.")
        self.decoder_norm = decoder_norm
        self.aux_p3_weight = float(aux_p3_weight)

        if decoder_norm == "gn":
            if fpn_dim % gn_groups != 0 or 32 % gn_groups != 0:
                raise ValueError(
                    "gn_groups must divide both fpn_dim and 32."
                )
            self.smooth_5 = ConvGNReLU(
                fpn_dim, fpn_dim, 3, 1, 1,
                gn_groups=gn_groups,
            )
            self.smooth_4 = ConvGNReLU(
                fpn_dim, fpn_dim, 3, 1, 1,
                gn_groups=gn_groups,
            )
            self.smooth_3 = ConvGNReLU(
                fpn_dim, fpn_dim, 3, 1, 1,
                gn_groups=gn_groups,
            )
            self.smooth_2 = ConvGNReLU(
                fpn_dim, fpn_dim, 3, 1, 1,
                gn_groups=gn_groups,
            )
            self.predictor = nn.Sequential(
                ConvGNReLU(
                    fpn_dim, 32, 3, 1, 1,
                    gn_groups=gn_groups,
                ),
                nn.Conv2d(32, 1, kernel_size=1),
            )

        self.aux_p3 = (
            nn.Conv2d(fpn_dim, 1, kernel_size=1)
            if self.aux_p3_weight > 0.0
            else None
        )
        self.unvalue_loss = MaskAnchoredUnvalueLoss(**loss_kwargs)

    def body(self, data, return_p3=False):
        image = data["image_m"]
        c2, c3, c4, c5 = self.normalize_encoder(image)

        p5 = self.smooth_5(self.lateral_5(c5))
        p4 = self.smooth_4(
            self.lateral_4(c4) + self._resize_like(p5, c4)
        )
        p3 = self.smooth_3(
            self.lateral_3(c3) + self._resize_like(p4, c3)
        )
        p2 = self.smooth_2(
            self.lateral_2(c2) + self._resize_like(p3, c2)
        )

        logits = self.predictor(p2)
        logits = F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if return_p3:
            return logits, p3
        return logits

    def _auxiliary_p3_loss(self, p3, target, pool_id):
        if self.aux_p3 is None:
            return p3.sum() * 0.0
        aux_logits = self.aux_p3(p3)
        # Area interpolation preserves foreground occupancy for small objects.
        aux_target = F.interpolate(
            target.float(),
            size=aux_logits.shape[-2:],
            mode="area",
        )
        aux_each = F.binary_cross_entropy_with_logits(
            aux_logits, aux_target, reduction="none"
        ).flatten(1).mean(dim=1)
        return _selected_mean(aux_each, pool_id == 0)

    def forward(self, data, iter_percentage=1.0, **kwargs):
        del kwargs
        logits, p3 = self.body(data=data, return_p3=True)
        if not self.training:
            return logits

        target = data["mask"].to(
            device=logits.device, dtype=logits.dtype
        )
        target = _resize(
            target, logits.shape[-2:]
        ).clamp(0.0, 1.0)

        total, items = self.unvalue_loss(
            logits=logits,
            target=target,
            data=data,
            iter_percentage=iter_percentage,
        )

        pool_id = data.get("pool_id")
        if pool_id is None:
            pool_id = torch.zeros(
                logits.shape[0],
                device=logits.device,
                dtype=torch.long,
            )
        pool_id = pool_id.to(logits.device).reshape(logits.shape[0], -1)[:, 0]
        aux_p3 = self._auxiliary_p3_loss(p3, target, pool_id)
        total = total + self.aux_p3_weight * aux_p3

        items["aux_p3"] = aux_p3.detach()
        items["aux_p3_weight"] = logits.new_tensor(
            self.aux_p3_weight
        )
        items["total"] = total.detach()
        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid()},
            "loss": total,
            "loss_items": items,
            "loss_str": (
                f"L:{total.detach().item():.4f} "
                f"BCE:{items['bce'].item():.4f} "
                f"CIOU:{items['clean_iou'].item():.4f} "
                f"OUT:{items['unvalue_out'].item():.4f} "
                f"DYN:{items['unvalue_dynamic'].item():.4f} "
                f"ACC:{items['unvalue_accepted'].item():.2f}"
            ),
        }

