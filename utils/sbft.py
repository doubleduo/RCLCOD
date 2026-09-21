# -*- coding: utf-8 -*-
"""Minimal SBFT used only on clean-pool RGB images."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


_MASK_CACHE = {}


def _circular_mask(
    height: int,
    width: int,
    cutoff_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return [1,1,H,W] circular low-pass mask after fftshift."""
    key = (height, width, float(cutoff_ratio), str(device), str(dtype))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached

    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = float(cutoff_ratio) * min(height, width) / 2.0

    mask = ((yy - cy).square() + (xx - cx).square() <= radius * radius)
    mask = mask.to(dtype=dtype).view(1, 1, height, width)
    _MASK_CACHE[key] = mask
    return mask


def sbft_lowpass(image: torch.Tensor, cutoff_ratio: float = 0.95) -> torch.Tensor:
    """
    Apply CurriSeg-style circular Fourier low-pass filtering.

    image: [B,C,H,W], expected in [0,1].
    The FFT is computed in float32 for AMP stability.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW image, got {tuple(image.shape)}")
    if not 0.0 < cutoff_ratio <= 1.0:
        raise ValueError(f"cutoff_ratio must be in (0,1], got {cutoff_ratio}")

    input_dtype = image.dtype
    x = image.float()
    height, width = x.shape[-2:]

    spectrum = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))

    mask = _circular_mask(
        height=height,
        width=width,
        cutoff_ratio=cutoff_ratio,
        device=x.device,
        dtype=x.dtype,
    )

    spectrum = spectrum * mask
    spectrum = torch.fft.ifftshift(spectrum, dim=(-2, -1))
    output = torch.fft.ifft2(spectrum, dim=(-2, -1), norm="ortho").real

    return output.clamp(0.0, 1.0).to(dtype=input_dtype)


def apply_clean_pool_sbft(
    data_batch: Dict[str, torch.Tensor],
    epoch: int,
    sbft_cfg: Optional[dict],
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    In the configured second stage, directly replace clean RGB inputs by SBFT inputs.

    - Only samples with data_batch['is_clean'] == 1 are filtered.
    - image_s/image_m/image_l are filtered consistently.
    - mask and all supervision tensors remain unchanged.
    """
    if not sbft_cfg or not bool(sbft_cfg.get("enable", False)):
        return data_batch, 0.0

    start_epoch = int(sbft_cfg.get("start_epoch", 31))
    end_epoch = int(sbft_cfg.get("end_epoch", 50))
    if epoch < start_epoch or epoch > end_epoch:
        return data_batch, 0.0

    if "is_clean" not in data_batch:
        raise KeyError(
            "Missing data_batch['is_clean']; add the clean-pool flag in ImageTrainDataset."
        )

    clean_mask = data_batch["is_clean"].reshape(-1, 1, 1, 1).bool()
    if not bool(clean_mask.any()):
        return data_batch, 0.0

    cutoff_ratio = float(sbft_cfg.get("cutoff_ratio", 0.95))
    output = dict(data_batch)

    for key in ("image_s", "image_m", "image_l"):
        if key not in data_batch:
            continue
        original = data_batch[key]
        low_frequency = sbft_lowpass(original, cutoff_ratio=cutoff_ratio)
        output[key] = torch.where(clean_mask, low_frequency, original)

    applied_ratio = float(clean_mask.float().mean().item())
    output["sbft_applied"] = clean_mask.reshape(-1).float()
    return output, applied_ratio
