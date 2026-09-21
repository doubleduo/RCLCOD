# -*- coding: utf-8 -*-
"""
APBOXNet: PVTv2-B4 FPN + CSR-V2 + NC-V2

主要改进
--------
NC-V2
1. q 从硬切换改成连续 cosine 变化，消除 epoch 60/61 的 loss 跳变。
2. 损失公式全程固定：
       L = L_anchor + lambda_nc * L_rnc + lambda_cons * L_cons
3. 如果存在 box_mask：
       可靠前景 = erode(pseudo_mask) ∩ box
       可靠背景 = outside(dilate(box))
       其余区域降低权重。
4. 支持 pool_reliability / is_noisy / pool_id，按样本调整 q。
5. 支持可选 teacher_prob / ema_prob，一致性监督不会直接翻转伪标签。

CSR-V2
1. 三个尺度分支使用独立 value projection，避免 value_mix 串扰。
2. 删除 ZeroConv，避免 CSR 前期几乎收不到梯度。
3. 使用小门控 + RMS 相对归一化，限制 residual 对 FPN 的扰动。
4. CSR gate 默认最高 0.25，不再释放到 1.0。
5. P3 的 CSR context 使用 p4_base，避免 P4 residual 重复注入。
6. 输出 route entropy、spatial std、residual ratio 等诊断量。

可选尺度监督
------------
如果 dataloader 提供：
    scale_prior_p3: [B, 3] 或 [B, 3, H, W]
    scale_prior_p4: [B, 3] 或 [B, 3, H, W]
则自动加入 KL scale-routing loss。

三路顺序：
    0 = fine
    1 = current
    2 = context
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.pvt_v2_eff import pvt_v2_eff_b4
from .zoomnext.ops import ConvBNReLU, PixelNormalizer


LOGGER = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def _clip_progress(progress: float) -> float:
    return min(max(float(progress), 0.0), 1.0)


def _cosine_lerp(start: float, end: float, x: float) -> float:
    x = min(max(float(x), 0.0), 1.0)
    coefficient = 0.5 * (1.0 - math.cos(math.pi * x))
    return float(start + (end - start) * coefficient)


def _cosine_transition(
    progress: float,
    start_ratio: float,
    end_ratio: float,
    start_value: float,
    end_value: float,
) -> float:
    progress = _clip_progress(progress)

    if progress <= start_ratio:
        return float(start_value)

    if progress >= end_ratio:
        return float(end_value)

    local_progress = (
        (progress - start_ratio)
        / max(end_ratio - start_ratio, 1e-8)
    )
    return _cosine_lerp(
        start_value,
        end_value,
        local_progress,
    )


def csr_gate_from_progress(
    progress: float,
    gate_max: float = 0.25,
) -> float:
    """CSR residual schedule for a nominal 150-epoch run.

    Epoch 1-50:
        0.01 -> 0.10

    Epoch 51-95:
        0.10 -> gate_max

    Epoch 96-150:
        gate_max
    """
    gate_max = float(gate_max)
    if not 0.10 <= gate_max <= 0.50:
        raise ValueError(
            "csr_gate_max must be in [0.10, 0.50], "
            f"but got {gate_max}."
        )

    progress = _clip_progress(progress)
    stage1_end = 50.0 / 150.0
    stage2_end = 95.0 / 150.0

    if progress <= stage1_end:
        return _cosine_transition(
            progress,
            0.0,
            stage1_end,
            0.01,
            0.10,
        )

    if progress <= stage2_end:
        return _cosine_transition(
            progress,
            stage1_end,
            stage2_end,
            0.10,
            gate_max,
        )

    return gate_max


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _resize_mask(
    tensor: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    if tensor.shape[-2:] == size:
        return tensor

    return F.interpolate(
        tensor.float(),
        size=size,
        mode="nearest",
    )


def _dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
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


def _erode(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask

    if kernel_size % 2 == 0:
        kernel_size += 1

    return 1.0 - F.max_pool2d(
        1.0 - mask,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )


def _prepare_single_channel(
    value,
    batch_size: int,
    size: Tuple[int, int],
    device,
    dtype,
) -> Optional[torch.Tensor]:
    if value is None:
        return None

    try:
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=dtype)
        else:
            tensor = torch.as_tensor(
                value,
                device=device,
                dtype=dtype,
            )
    except (TypeError, ValueError):
        return None

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)

    if tensor.ndim != 4:
        return None

    if tensor.shape[1] > 1:
        tensor = tensor.amax(dim=1, keepdim=True)

    if tensor.shape[0] == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size, -1, -1, -1)

    if tensor.shape[0] != batch_size:
        return None

    tensor = _resize_mask(tensor, size)
    return tensor.clamp(0.0, 1.0)


def _prepare_sample_scalar(
    value,
    batch_size: int,
    device,
    dtype,
) -> Optional[torch.Tensor]:
    if value is None:
        return None

    try:
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=dtype)
        else:
            tensor = torch.as_tensor(
                value,
                device=device,
                dtype=dtype,
            )
    except (TypeError, ValueError):
        return None

    if tensor.numel() == 1:
        return tensor.reshape(1).expand(batch_size)

    if tensor.shape[0] != batch_size:
        return None

    return tensor.reshape(batch_size, -1).mean(dim=1)


def _pool_reliability(
    data: Dict,
    batch_size: int,
    device,
    dtype,
) -> torch.Tensor:
    """Return [B] reliability: 0=noisy, 1=high-quality.

    Recommended dataloader field:
        pool_reliability: float in [0, 1]

    Also accepts:
        is_noisy: 1 means noisy
        pool_id: 0 means clean, non-zero means noisy
        pool_name: list[str] containing clean/noisy
    """
    reliability = _prepare_sample_scalar(
        data.get("pool_reliability"),
        batch_size,
        device,
        dtype,
    )

    if reliability is not None:
        return reliability.clamp(0.0, 1.0)

    is_noisy = _prepare_sample_scalar(
        data.get("is_noisy"),
        batch_size,
        device,
        dtype,
    )
    if is_noisy is not None:
        return (1.0 - is_noisy).clamp(0.0, 1.0)

    pool_id = _prepare_sample_scalar(
        data.get("pool_id"),
        batch_size,
        device,
        dtype,
    )
    if pool_id is not None:
        return (pool_id <= 0).to(dtype=dtype)

    pool_names = data.get("pool_name")
    if isinstance(pool_names, (list, tuple)):
        values = []
        for name in pool_names:
            name = str(name).lower()
            if "clean" in name:
                values.append(1.0)
            elif "noisy" in name:
                values.append(0.0)
            else:
                values.append(0.5)

        if len(values) == batch_size:
            return torch.tensor(
                values,
                device=device,
                dtype=dtype,
            )

    # Compatible fallback for the current APBOXNet dataloader.
    return torch.full(
        (batch_size,),
        0.5,
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# CSR-V2
# ---------------------------------------------------------------------------

class BranchPureScaleRouter(nn.Module):
    """Grouped three-scale spatial router.

    Attention can observe all branches, but the value projection of each
    branch is independent. This prevents a common 3C value mixer from
    contaminating fine/current/context semantics before routing.

    Attention shape:
        [B, groups, 3, H, W]

    Branch order:
        fine/current/context
    """

    def __init__(
        self,
        channels: int = 64,
        num_groups: int = 4,
    ):
        super().__init__()

        if channels % num_groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by "
                f"num_groups={num_groups}."
            )

        self.channels = int(channels)
        self.num_groups = int(num_groups)
        group_dim = channels // num_groups

        self.fine_pre = ConvBNReLU(
            channels, channels, 3, 1, 1
        )
        self.current_pre = ConvBNReLU(
            channels, channels, 3, 1, 1
        )
        self.context_pre = ConvBNReLU(
            channels, channels, 3, 1, 1
        )

        self.fine_refine = ConvBNReLU(
            channels, channels, 3, 1, 1
        )
        self.current_refine = ConvBNReLU(
            channels, channels, 3, 1, 1
        )
        self.context_refine = ConvBNReLU(
            channels, channels, 3, 1, 1
        )

        # Joint evidence is allowed only in the routing path.
        self.attention_mixer = ConvBNReLU(
            3 * channels,
            3 * channels,
            1,
        )

        self.route = nn.Sequential(
            ConvBNReLU(3 * group_dim, group_dim, 1),
            ConvBNReLU(group_dim, group_dim, 3, 1, 1),
            nn.Conv2d(group_dim, 3, kernel_size=1, bias=True),
        )

        # Branch-pure values.
        self.fine_value = ConvBNReLU(
            channels, channels, 1
        )
        self.current_value = ConvBNReLU(
            channels, channels, 1
        )
        self.context_value = ConvBNReLU(
            channels, channels, 1
        )

    @staticmethod
    def _align_fine(
        tensor: torch.Tensor,
        size: Tuple[int, int],
    ) -> torch.Tensor:
        if tensor.shape[-2:] == size:
            return tensor

        maximum = F.adaptive_max_pool2d(
            tensor,
            output_size=size,
        )
        average = F.adaptive_avg_pool2d(
            tensor,
            output_size=size,
        )

        # Average instead of the previous max+avg sum to avoid a 2x
        # magnitude change before routing.
        return 0.5 * (maximum + average)

    @staticmethod
    def _align_context(
        tensor: torch.Tensor,
        size: Tuple[int, int],
    ) -> torch.Tensor:
        if tensor.shape[-2:] == size:
            return tensor

        return F.interpolate(
            tensor,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        fine: torch.Tensor,
        current: torch.Tensor,
        context: torch.Tensor,
    ):
        target_size = current.shape[-2:]

        fine = self._align_fine(
            self.fine_pre(fine),
            target_size,
        )
        current = self.current_pre(current)
        context = self._align_context(
            self.context_pre(context),
            target_size,
        )

        fine = self.fine_refine(fine)
        current = self.current_refine(current)
        context = self.context_refine(context)

        batch_size, channels, height, width = current.shape
        groups = self.num_groups
        group_dim = channels // groups

        mixed_evidence = self.attention_mixer(
            torch.cat(
                [fine, current, context],
                dim=1,
            )
        )

        mixed_evidence = (
            mixed_evidence
            .view(
                batch_size,
                3,
                groups,
                group_dim,
                height,
                width,
            )
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(
                batch_size * groups,
                3 * group_dim,
                height,
                width,
            )
        )

        attention = torch.softmax(
            self.route(mixed_evidence),
            dim=1,
        )
        attention = attention.view(
            batch_size,
            groups,
            3,
            height,
            width,
        )

        fine_value = self.fine_value(fine).view(
            batch_size,
            groups,
            group_dim,
            height,
            width,
        )
        current_value = self.current_value(current).view(
            batch_size,
            groups,
            group_dim,
            height,
            width,
        )
        context_value = self.context_value(context).view(
            batch_size,
            groups,
            group_dim,
            height,
            width,
        )

        values = torch.stack(
            [fine_value, current_value, context_value],
            dim=2,
        )

        fused = (
            attention.unsqueeze(3) * values
        ).sum(dim=2)

        fused = fused.reshape(
            batch_size,
            channels,
            height,
            width,
        )

        return fused, attention


class RelativeResidualInjection(nn.Module):
    """Small, bounded CSR correction relative to the FPN feature RMS."""

    def __init__(
        self,
        channels: int,
        scale_min: float = 0.25,
        scale_max: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        # Unlike ZeroConv, Dirac initialization keeps a gradient path
        # from the first iteration. The small external gate controls risk.
        nn.init.dirac_(self.projection.weight)

        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.eps = float(eps)

    def forward(
        self,
        base: torch.Tensor,
        residual: torch.Tensor,
        gate: float,
    ):
        delta = self.projection(residual)

        base_rms = (
            base.detach()
            .square()
            .mean(dim=(1, 2, 3), keepdim=True)
            .add(self.eps)
            .sqrt()
        )
        delta_rms = (
            delta.detach()
            .square()
            .mean(dim=(1, 2, 3), keepdim=True)
            .add(self.eps)
            .sqrt()
        )

        relative_scale = (
            base_rms / delta_rms
        ).clamp(
            min=self.scale_min,
            max=self.scale_max,
        )

        normalized_delta = delta * relative_scale
        injected_delta = float(gate) * normalized_delta
        output = base + injected_delta

        residual_ratio = (
            injected_delta.detach()
            .square()
            .mean(dim=(1, 2, 3))
            .add(self.eps)
            .sqrt()
            / base_rms.flatten()
        ).mean()

        return output, residual_ratio


def _route_statistics(
    attention: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    detached = attention.detach()

    branch_mean = detached.mean(dim=(0, 1, 3, 4))

    entropy = -(
        detached.clamp_min(1e-8)
        * detached.clamp_min(1e-8).log()
    ).sum(dim=2)
    entropy = entropy.mean() / math.log(3.0)

    # First average groups, then measure whether routing varies spatially.
    spatial_route = detached.mean(dim=1)
    spatial_std = spatial_route.std(
        dim=(-2, -1),
        unbiased=False,
    ).mean()

    return {
        "mean": branch_mean,
        "entropy": entropy,
        "spatial_std": spatial_std,
    }


# ---------------------------------------------------------------------------
# NC-V2
# ---------------------------------------------------------------------------

class ReliabilityAwareNCLoss(nn.Module):
    """Continuous, reliability-aware noisy-label loss."""

    def __init__(
        self,
        nc_weight: float = 1.0,
        consistency_weight: float = 0.20,
        uncertain_weight: float = 0.25,
        boundary_weight: float = 0.50,
        foreground_erode_kernel: int = 5,
        box_dilate_kernel: int = 9,
        boundary_kernel: int = 31,
        teacher_confidence: float = 0.80,
        corrected_negative_weight: float = 0.20,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.nc_weight = float(nc_weight)
        self.consistency_weight = float(consistency_weight)
        self.uncertain_weight = float(uncertain_weight)
        self.boundary_weight = float(boundary_weight)

        self.foreground_erode_kernel = int(
            foreground_erode_kernel
        )
        self.box_dilate_kernel = int(box_dilate_kernel)
        self.boundary_kernel = int(boundary_kernel)

        self.teacher_confidence = float(teacher_confidence)
        self.corrected_negative_weight = float(
            corrected_negative_weight
        )
        self.eps = float(eps)

    def _scheduled_q(
        self,
        progress: float,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        # Noisy samples robustify earlier and end at q=1.
        noisy_q = _cosine_transition(
            progress,
            start_ratio=0.15,
            end_ratio=0.55,
            start_value=2.0,
            end_value=1.0,
        )

        # High-quality samples retain more of the q=2 structural behavior.
        clean_q = _cosine_transition(
            progress,
            start_ratio=0.45,
            end_ratio=0.80,
            start_value=2.0,
            end_value=1.5,
        )

        return (
            (1.0 - reliability) * noisy_q
            + reliability * clean_q
        )

    def _teacher_probability(
        self,
        data: Dict,
        batch_size: int,
        size: Tuple[int, int],
        device,
        dtype,
    ) -> Optional[torch.Tensor]:
        teacher = data.get("teacher_prob")
        if teacher is None:
            teacher = data.get("ema_prob")
        if teacher is None:
            teacher = data.get("teacher_logits")

        teacher = _prepare_single_channel(
            teacher,
            batch_size,
            size,
            device,
            dtype,
        )
        if teacher is None:
            return None

        # _prepare_single_channel clips to [0,1], so teacher_logits should
        # preferably be converted to probability in the trainer.
        return teacher.detach()

    def _build_weights(
        self,
        target: torch.Tensor,
        box_mask: Optional[torch.Tensor],
        teacher_prob: Optional[torch.Tensor],
        reliability: torch.Tensor,
    ):
        batch_size = target.shape[0]
        target_binary = (target >= 0.5).to(target.dtype)

        if box_mask is None:
            padding = self.boundary_kernel // 2
            local_mean = F.avg_pool2d(
                target,
                kernel_size=self.boundary_kernel,
                stride=1,
                padding=padding,
            )

            boundary_strength = (
                2.0 * torch.abs(local_mean - target)
            ).clamp(0.0, 1.0)

            pixel_weight = (
                1.0
                - (1.0 - self.boundary_weight)
                * boundary_strength
            ).detach()

            return pixel_weight, None, None

        box_binary = (box_mask >= 0.5).to(target.dtype)

        reliable_foreground = (
            _erode(
                target_binary,
                self.foreground_erode_kernel,
            )
            * box_binary
        )

        reliable_background = (
            1.0
            - _dilate(
                box_binary,
                self.box_dilate_kernel,
            )
        ).clamp(0.0, 1.0)

        uncertain_region = (
            1.0
            - reliable_foreground
            - reliable_background
        ).clamp(0.0, 1.0)

        sample_reliability = reliability.view(
            batch_size, 1, 1, 1
        )

        # Noisy samples receive less supervision in uncertain regions.
        uncertain_strength = self.uncertain_weight * (
            0.5 + 0.5 * sample_reliability
        )

        pixel_weight = (
            reliable_foreground
            + reliable_background
            + uncertain_strength * uncertain_region
        )

        # One-sided correction:
        # a persistent teacher foreground prediction inside a box only
        # reduces the negative pseudo-label weight; it never flips target.
        if teacher_prob is not None:
            suspected_missing_foreground = (
                (box_binary > 0.5)
                & (target_binary < 0.5)
                & (teacher_prob >= self.teacher_confidence)
            ).to(target.dtype)

            correction = (
                1.0
                - suspected_missing_foreground
                * (1.0 - self.corrected_negative_weight)
            )
            pixel_weight = pixel_weight * correction.detach()

        return (
            pixel_weight.detach(),
            reliable_foreground.detach(),
            reliable_background.detach(),
        )

    def _anchor_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pixel_weight: torch.Tensor,
        reliable_foreground: Optional[torch.Tensor],
        reliable_background: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        fallback = (
            (bce_map * pixel_weight).flatten(1).sum(dim=1)
            / pixel_weight.flatten(1).sum(dim=1).clamp_min(
                self.eps
            )
        )

        if (
            reliable_foreground is None
            or reliable_background is None
        ):
            return fallback.mean()

        foreground_count = (
            reliable_foreground.flatten(1).sum(dim=1)
        )
        background_count = (
            reliable_background.flatten(1).sum(dim=1)
        )

        foreground_loss = (
            (bce_map * reliable_foreground)
            .flatten(1)
            .sum(dim=1)
            / foreground_count.clamp_min(self.eps)
        )
        background_loss = (
            (bce_map * reliable_background)
            .flatten(1)
            .sum(dim=1)
            / background_count.clamp_min(self.eps)
        )

        has_foreground = (
            foreground_count > 0
        ).to(logits.dtype)
        has_background = (
            background_count > 0
        ).to(logits.dtype)

        valid_regions = has_foreground + has_background

        balanced = (
            foreground_loss * has_foreground
            + background_loss * has_background
        ) / valid_regions.clamp_min(1.0)

        balanced = torch.where(
            valid_regions > 0,
            balanced,
            fallback,
        )

        return balanced.mean()

    def _robust_nc(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pixel_weight: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        probability = torch.sigmoid(logits)

        probability = probability.flatten(1)
        target = target.flatten(1)
        pixel_weight = pixel_weight.flatten(1)

        q = q.view(-1, 1)

        numerator = (
            pixel_weight
            * torch.abs(probability - target).pow(q)
        ).sum(dim=1)

        union = (
            probability
            + target
            - probability * target
        )

        denominator = (
            pixel_weight * union
        ).sum(dim=1).clamp_min(self.eps)

        return (numerator / denominator).mean()

    def _consistency_loss(
        self,
        logits: torch.Tensor,
        teacher_prob: Optional[torch.Tensor],
        progress: float,
    ):
        zero = logits.new_zeros(())

        if teacher_prob is None:
            return zero, 0.0

        teacher_confident = (
            (teacher_prob >= self.teacher_confidence)
            | (teacher_prob <= 1.0 - self.teacher_confidence)
        ).to(logits.dtype)

        consistency_map = (
            torch.sigmoid(logits) - teacher_prob
        ).square()

        consistency = (
            (consistency_map * teacher_confident)
            .flatten(1)
            .sum(dim=1)
            / teacher_confident
            .flatten(1)
            .sum(dim=1)
            .clamp_min(1.0)
        ).mean()

        consistency_gate = _cosine_transition(
            progress,
            start_ratio=0.25,
            end_ratio=0.65,
            start_value=0.0,
            end_value=self.consistency_weight,
        )

        return consistency, consistency_gate

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        data: Dict,
        iter_percentage: float,
    ):
        batch_size = logits.shape[0]
        progress = _clip_progress(iter_percentage)

        reliability = _pool_reliability(
            data,
            batch_size,
            logits.device,
            logits.dtype,
        )

        q = self._scheduled_q(
            progress,
            reliability,
        )

        box_mask = _prepare_single_channel(
            data.get("box_mask"),
            batch_size,
            logits.shape[-2:],
            logits.device,
            logits.dtype,
        )

        teacher_prob = self._teacher_probability(
            data,
            batch_size,
            logits.shape[-2:],
            logits.device,
            logits.dtype,
        )

        (
            pixel_weight,
            reliable_foreground,
            reliable_background,
        ) = self._build_weights(
            target,
            box_mask,
            teacher_prob,
            reliability,
        )

        anchor = self._anchor_loss(
            logits,
            target,
            pixel_weight,
            reliable_foreground,
            reliable_background,
        )

        robust_nc = self._robust_nc(
            logits,
            target,
            pixel_weight,
            q,
        )

        consistency, consistency_gate = (
            self._consistency_loss(
                logits,
                teacher_prob,
                progress,
            )
        )

        # The formula never changes between stages.
        total = (
            anchor
            + self.nc_weight * robust_nc
            + consistency_gate * consistency
        )

        diagnostics = {
            "anchor": anchor.detach(),
            "nc": robust_nc.detach(),
            "consistency": consistency.detach(),
            "consistency_weight": logits.new_tensor(
                consistency_gate
            ),
            "q": q.detach().mean(),
            "q_min": q.detach().amin(),
            "q_max": q.detach().amax(),
            "sample_reliability": reliability.detach().mean(),
            "pixel_reliability": pixel_weight.detach().mean(),
        }

        return total, diagnostics


# ---------------------------------------------------------------------------
# Optional scale-routing supervision
# ---------------------------------------------------------------------------

def _scale_prior_kl(
    attention: torch.Tensor,
    prior,
    valid=None,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """KL(prior || route), supporting [B,3] or [B,3,H,W]."""
    if prior is None:
        return None

    batch_size = attention.shape[0]
    prediction = attention.mean(dim=1)
    height, width = prediction.shape[-2:]

    try:
        if torch.is_tensor(prior):
            prior = prior.to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
        else:
            prior = torch.as_tensor(
                prior,
                device=prediction.device,
                dtype=prediction.dtype,
            )
    except (TypeError, ValueError):
        return None

    if prior.ndim == 1 and prior.numel() == 3:
        prior = prior.view(1, 3, 1, 1)
    elif prior.ndim == 2 and prior.shape[1] == 3:
        prior = prior[:, :, None, None]
    elif prior.ndim != 4 or prior.shape[1] != 3:
        return None

    if prior.shape[0] == 1 and batch_size > 1:
        prior = prior.expand(batch_size, -1, -1, -1)

    if prior.shape[0] != batch_size:
        return None

    if prior.shape[-2:] != (height, width):
        prior = F.interpolate(
            prior,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    prior = prior.clamp_min(0.0)
    prior = prior / prior.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(eps)

    prediction = prediction.clamp_min(eps)
    prediction = prediction / prediction.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(eps)

    kl_map = (
        prior.clamp_min(eps)
        * (
            prior.clamp_min(eps).log()
            - prediction.log()
        )
    ).sum(dim=1, keepdim=True)

    if valid is None:
        return kl_map.mean()

    valid = _prepare_single_channel(
        valid,
        batch_size,
        (height, width),
        prediction.device,
        prediction.dtype,
    )
    if valid is None:
        return kl_map.mean()

    return (
        (kl_map * valid).sum()
        / valid.sum().clamp_min(1.0)
    )


# ---------------------------------------------------------------------------
# Full FPN model
# ---------------------------------------------------------------------------

class _PvtV2B4_FPN_V2(nn.Module):
    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        csr_groups=4,
        csr_gate_max=0.25,
        scale_loss_weight=0.20,
        use_csr=True,
        loss_mode="nc",
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        # Do not inherit the old alpha=1.0 behavior accidentally.
        if "csr_alpha_max" in kwargs:
            LOGGER.warning(
                "csr_alpha_max is ignored by CSR-V2; "
                "use csr_gate_max instead."
            )

        self.use_csr = bool(use_csr)
        self.loss_mode = str(loss_mode)
        self.csr_gate_max = float(csr_gate_max)
        self.scale_loss_weight = float(scale_loss_weight)

        if self.loss_mode not in {"nc", "bce"}:
            raise ValueError(
                f"Unsupported loss_mode={self.loss_mode}."
            )

        self.encoder = pvt_v2_eff_b4(
            pretrained=pretrained,
            use_checkpoint=use_checkpoint,
        )
        self.embed_dims = list(self.encoder.embed_dims)

        self.normalizer = (
            PixelNormalizer()
            if input_norm
            else nn.Identity()
        )

        self.lateral_2 = nn.Conv2d(
            self.embed_dims[0], fpn_dim, 1, bias=False
        )
        self.lateral_3 = nn.Conv2d(
            self.embed_dims[1], fpn_dim, 1, bias=False
        )
        self.lateral_4 = nn.Conv2d(
            self.embed_dims[2], fpn_dim, 1, bias=False
        )
        self.lateral_5 = nn.Conv2d(
            self.embed_dims[3], fpn_dim, 1, bias=False
        )

        self.smooth_5 = ConvBNReLU(
            fpn_dim, fpn_dim, 3, 1, 1
        )
        self.smooth_4 = ConvBNReLU(
            fpn_dim, fpn_dim, 3, 1, 1
        )
        self.smooth_3 = ConvBNReLU(
            fpn_dim, fpn_dim, 3, 1, 1
        )
        self.smooth_2 = ConvBNReLU(
            fpn_dim, fpn_dim, 3, 1, 1
        )

        if self.use_csr:
            self.csr_4 = BranchPureScaleRouter(
                channels=fpn_dim,
                num_groups=csr_groups,
            )
            self.csr_3 = BranchPureScaleRouter(
                channels=fpn_dim,
                num_groups=csr_groups,
            )

            self.inject_4 = RelativeResidualInjection(fpn_dim)
            self.inject_3 = RelativeResidualInjection(fpn_dim)

        self.predictor = nn.Sequential(
            ConvBNReLU(fpn_dim, 32, 3, 1, 1),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self.nc_loss = ReliabilityAwareNCLoss()

    def normalize_encoder(self, image):
        image = self.normalizer(image)
        features = self.encoder(image)

        return (
            features["reduction_2"],
            features["reduction_3"],
            features["reduction_4"],
            features["reduction_5"],
        )

    @staticmethod
    def _resize_like(source, target):
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def body(
        self,
        data: Dict,
        iter_percentage: float = 1.0,
    ):
        image = data["image_m"]
        c2, c3, c4, c5 = self.normalize_encoder(image)

        l2 = self.lateral_2(c2)
        l3 = self.lateral_3(c3)
        l4 = self.lateral_4(c4)
        l5 = self.lateral_5(c5)

        p5 = self.smooth_5(l5)

        top4 = self._resize_like(p5, l4)
        p4_base = self.smooth_4(l4 + top4)

        gate = 0.0
        attention_4 = None
        attention_3 = None

        residual_ratio_4 = image.new_zeros(())
        residual_ratio_3 = image.new_zeros(())

        if self.use_csr:
            gate = csr_gate_from_progress(
                iter_percentage,
                gate_max=self.csr_gate_max,
            )

            residual_4, attention_4 = self.csr_4(
                fine=l3,
                current=l4,
                context=p5,
            )
            p4, residual_ratio_4 = self.inject_4(
                p4_base,
                residual_4,
                gate,
            )
        else:
            p4 = p4_base

        # The corrected P4 is propagated once through the normal FPN path.
        top3 = self._resize_like(p4, l3)
        p3_base = self.smooth_3(l3 + top3)

        if self.use_csr:
            # Use p4_base here, not p4, to avoid routing the same P4
            # correction through the second residual path again.
            residual_3, attention_3 = self.csr_3(
                fine=l2,
                current=l3,
                context=p4_base,
            )
            p3, residual_ratio_3 = self.inject_3(
                p3_base,
                residual_3,
                gate,
            )
        else:
            p3 = p3_base

        p2 = self.smooth_2(
            l2 + self._resize_like(p3, l2)
        )

        logits = self.predictor(p2)
        logits = F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        auxiliary = {
            "gate": float(gate),
            "a3": attention_3,
            "a4": attention_4,
            "residual_ratio_p3": residual_ratio_3,
            "residual_ratio_p4": residual_ratio_4,
        }

        return logits, auxiliary

    def _scale_loss(
        self,
        data: Dict,
        auxiliary: Dict,
        logits: torch.Tensor,
    ):
        zero = logits.new_zeros(())

        if not self.use_csr:
            return zero

        losses = []

        prior_p3 = data.get("scale_prior_p3")
        prior_p4 = data.get("scale_prior_p4")

        # A shared prior is accepted for compatibility, although
        # per-level priors are preferred.
        if prior_p3 is None:
            prior_p3 = data.get("box_scale_prior")
        if prior_p4 is None:
            prior_p4 = data.get("box_scale_prior")

        valid_p3 = data.get("scale_valid_p3")
        valid_p4 = data.get("scale_valid_p4")

        if valid_p3 is None:
            valid_p3 = data.get("scale_valid")
        if valid_p4 is None:
            valid_p4 = data.get("scale_valid")

        loss_p3 = _scale_prior_kl(
            auxiliary["a3"],
            prior_p3,
            valid_p3,
        )
        loss_p4 = _scale_prior_kl(
            auxiliary["a4"],
            prior_p4,
            valid_p4,
        )

        if loss_p3 is not None:
            losses.append(loss_p3)
        if loss_p4 is not None:
            losses.append(loss_p4)

        if not losses:
            return zero

        return torch.stack(losses).mean()

    def forward(
        self,
        data,
        iter_percentage=1.0,
        **kwargs,
    ):
        del kwargs

        logits, auxiliary = self.body(
            data,
            iter_percentage=iter_percentage,
        )

        if not self.training:
            return logits

        target = data["mask"].to(
            device=logits.device,
            dtype=logits.dtype,
        )
        target = _resize_mask(
            target,
            logits.shape[-2:],
        ).clamp(0.0, 1.0)

        if self.loss_mode == "nc":
            segmentation_loss, nc_items = self.nc_loss(
                logits=logits,
                target=target,
                data=data,
                iter_percentage=iter_percentage,
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="mean",
            )
            segmentation_loss = bce
            zero = logits.new_zeros(())

            nc_items = {
                "anchor": bce.detach(),
                "nc": zero,
                "consistency": zero,
                "consistency_weight": zero,
                "q": zero,
                "q_min": zero,
                "q_max": zero,
                "sample_reliability": zero,
                "pixel_reliability": zero,
            }

        scale_loss = self._scale_loss(
            data,
            auxiliary,
            logits,
        )

        total = (
            segmentation_loss
            + self.scale_loss_weight * scale_loss
        )

        loss_items = {
            # Keep compatibility with existing CSV/trainer code.
            "bce": nc_items["anchor"],
            "anchor": nc_items["anchor"],
            "nc": nc_items["nc"],
            "q": nc_items["q"],
            "q_min": nc_items["q_min"],
            "q_max": nc_items["q_max"],
            "consistency": nc_items["consistency"],
            "consistency_weight": nc_items[
                "consistency_weight"
            ],
            "sample_reliability": nc_items[
                "sample_reliability"
            ],
            "pixel_reliability": nc_items[
                "pixel_reliability"
            ],
            "scale_loss": scale_loss.detach(),
            "csr_alpha": logits.new_tensor(
                auxiliary["gate"]
            ),
            "csr_gate": logits.new_tensor(
                auxiliary["gate"]
            ),
            "p3_residual_ratio": auxiliary[
                "residual_ratio_p3"
            ].detach(),
            "p4_residual_ratio": auxiliary[
                "residual_ratio_p4"
            ].detach(),
            "total": total.detach(),
        }

        if self.use_csr:
            stats_3 = _route_statistics(auxiliary["a3"])
            stats_4 = _route_statistics(auxiliary["a4"])

            loss_items.update({
                "p3_fine": stats_3["mean"][0],
                "p3_current": stats_3["mean"][1],
                "p3_context": stats_3["mean"][2],
                "p3_route_entropy": stats_3["entropy"],
                "p3_route_spatial_std": stats_3["spatial_std"],

                "p4_fine": stats_4["mean"][0],
                "p4_current": stats_4["mean"][1],
                "p4_context": stats_4["mean"][2],
                "p4_route_entropy": stats_4["entropy"],
                "p4_route_spatial_std": stats_4["spatial_std"],
            })

        loss_string = (
            f"L:{total.detach().item():.4f} "
            f"A:{nc_items['anchor'].item():.4f} "
            f"NC:{nc_items['nc'].item():.4f} "
            f"Q:{nc_items['q'].item():.2f} "
            f"G:{auxiliary['gate']:.3f} "
            f"SL:{scale_loss.detach().item():.4f}"
        )

        return {
            "logits": logits,
            "vis": {
                "sal": logits.sigmoid(),
            },
            "loss": total,
            "loss_items": loss_items,
            "loss_str": loss_string,
        }

    def get_grouped_params(self):
        param_groups = {
            "pretrained": [],
            "fixed": [],
            "retrained": [],
        }

        for name, parameter in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                parameter.requires_grad = False
                param_groups["fixed"].append(parameter)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(parameter)
            else:
                param_groups["retrained"].append(parameter)

        LOGGER.info(
            "Parameter Groups:{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}"
            "}"
        )

        return param_groups


# ---------------------------------------------------------------------------
# Public experiment classes
# ---------------------------------------------------------------------------

class PvtV2B4_FPN_NC_V2(_PvtV2B4_FPN_V2):
    """FPN + NC-V2: isolates the denoising-loss contribution."""

    def __init__(self, **kwargs):
        kwargs.pop("use_csr", None)
        kwargs.pop("loss_mode", None)
        super().__init__(
            use_csr=False,
            loss_mode="nc",
            **kwargs,
        )


class PvtV2B4_FPN_CSR_V2_BCE(_PvtV2B4_FPN_V2):
    """FPN + CSR-V2 + BCE: isolates the routing contribution."""

    def __init__(self, **kwargs):
        kwargs.pop("use_csr", None)
        kwargs.pop("loss_mode", None)
        super().__init__(
            use_csr=True,
            loss_mode="bce",
            **kwargs,
        )


class PvtV2B4_FPN_CSR_NC_V2(_PvtV2B4_FPN_V2):
    """FPN + CSR-V2 + NC-V2: proposed complete model."""

    def __init__(self, **kwargs):
        kwargs.pop("use_csr", None)
        kwargs.pop("loss_mode", None)
        super().__init__(
            use_csr=True,
            loss_mode="nc",
            **kwargs,
        )