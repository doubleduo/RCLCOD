# -*- coding: utf-8 -*-
"""Same-Class Neighbor Penalization for binary segmentation.

SCNP is used only while training.  It transforms logits before an auxiliary
loss; it never changes the logits returned by the model at inference time.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCNP2D(nn.Module):
    """Binary 2-D SCNP implemented with min/max pooling.

    Args:
        kernel_size: Odd neighborhood width.  Use 3 first for COD.
        large_value: Finite masking constant.  1e4 is safe for AMP fp16.
    """

    def __init__(self, kernel_size: int = 3, large_value: float = 1e4):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("SCNP kernel_size must be a positive odd integer.")
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.large_value = float(large_value)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(
                "Binary SCNP expects logits shaped [B, 1, H, W], "
                f"but got {tuple(logits.shape)}."
            )
        if target.shape != logits.shape:
            raise ValueError(
                "SCNP target and logits must have identical shapes, "
                f"got {tuple(target.shape)} and {tuple(logits.shape)}."
            )

        # Pseudo masks are treated as hard labels.  Detaching prevents a future
        # soft/teacher target from accidentally receiving gradients here.
        fg = (target.detach() >= 0.5).to(dtype=logits.dtype)
        bg = 1.0 - fg
        k = self.large_value

        # Foreground pixels inherit their weakest same-class neighbor.
        weakest_fg = -F.max_pool2d(
            -(logits * fg + k * bg),
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
        )

        # Background pixels inherit their most dangerous same-class neighbor.
        strongest_bg = F.max_pool2d(
            logits * bg - k * fg,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
        )
        return weakest_fg * fg + strongest_bg * bg


def selected_scnp_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    scnp: SCNP2D,
    pool_ids: Optional[torch.Tensor] = None,
    allowed_pool_ids: Iterable[int] = (0,),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-image SCNP BCE only for selected curriculum pools.

    Returns:
        loss: Scalar loss.  It is differentiable zero if no sample is active.
        active_fraction: Fraction of active images in the current batch.
    """
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), logits.shape[-2:], mode="nearest")
    target = target.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)

    if pool_ids is None:
        active = torch.ones(logits.shape[0], device=logits.device, dtype=torch.bool)
    else:
        pool_ids = pool_ids.to(device=logits.device).reshape(-1)
        if pool_ids.numel() != logits.shape[0]:
            raise ValueError(
                f"pool_ids has {pool_ids.numel()} elements for batch size "
                f"{logits.shape[0]}."
            )
        active = torch.zeros_like(pool_ids, dtype=torch.bool)
        for pool_id in allowed_pool_ids:
            active |= pool_ids == int(pool_id)

    active_fraction = active.float().mean()
    if not bool(active.any()):
        return logits.sum() * 0.0, active_fraction

    active_logits = logits[active]
    active_target = target[active]
    penalized_logits = scnp(active_logits, active_target)
    loss = F.binary_cross_entropy_with_logits(
        penalized_logits,
        active_target,
        reduction="mean",
    )
    return loss, active_fraction

