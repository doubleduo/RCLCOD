# -*- coding: utf-8 -*-
"""CurriSeg-style circular Fourier low-pass SBFT, used after train augmentation.

The input image must be a HWC uint8 RGB array, before division by 255 and
before the model's PixelNormalizer. The mask/box_mask are never modified.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Real
from collections.abc import Mapping

import numpy as np


_SBFT_DEFAULTS = {
    "enable": False,
    "start_epoch": 60,
    "r": 0.95,
    "p": 0.5,
    "pool": "clean",
}


def parse_sbft_config(config=None):
    """Validate a config; only the clean pool is supported in this ablation."""
    conf = dict(_SBFT_DEFAULTS)
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("train.sbft must be a mapping/dict")
        unknown = set(config) - set(conf)
        if unknown:
            raise ValueError(f"Unknown train.sbft keys: {sorted(unknown)}")
        conf.update(dict(config))

    if not isinstance(conf["enable"], bool):
        raise TypeError("train.sbft.enable must be bool")
    if isinstance(conf["start_epoch"], bool) or not isinstance(conf["start_epoch"], int) or conf["start_epoch"] < 1:
        raise ValueError("train.sbft.start_epoch must be a positive integer")
    for key in ("r", "p"):
        value = conf[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"train.sbft.{key} must be a finite number")
        conf[key] = float(value)
    if not 0.0 < conf["r"] <= 1.0:
        raise ValueError("train.sbft.r must be in (0, 1]")
    if not 0.0 <= conf["p"] <= 1.0:
        raise ValueError("train.sbft.p must be in [0, 1]")
    if conf["pool"] != "clean":
        raise ValueError("This SBFT ablation supports pool='clean' only")
    return conf


@lru_cache(maxsize=16)
def _circular_lowpass(height: int, width: int, radius_ratio: float):
    """Use exactly the circular cutoff definition in CurriSeg Eq. (10)."""
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    cutoff = radius_ratio * min(height, width) / 2.0
    return ((y-cy)**2 + (x-cx)**2 <= cutoff**2)[..., None]


def sbft_lowpass(image: np.ndarray, r: float = 0.95) -> np.ndarray:
    """FFT -> fftshift -> circular low-pass -> inverse FFT, preserving HWC uint8."""
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("SBFT expects uint8 HxWx3 RGB input")
    if not 0 < r <= 1:
        raise ValueError("SBFT r must be in (0,1]")
    h, w = image.shape[:2]
    frequency = np.fft.fftshift(
        np.fft.fft2(image.astype(np.float32), axes=(0, 1)), axes=(0, 1)
    )
    frequency *= _circular_lowpass(h, w, float(r))
    recovered = np.fft.ifft2(
        np.fft.ifftshift(frequency, axes=(0, 1)), axes=(0, 1)
    ).real
    return np.ascontiguousarray(
        np.clip(np.rint(recovered), 0, 255).astype(np.uint8)
    )


def should_apply_sbft(*, config: dict, pool_name: str, epoch: int, augment: bool, random_value: float):
    """Pure gating helper, useful for asserting epoch and pool behavior."""
    return (
        augment and config["enable"] and pool_name == config["pool"]
        and epoch >= config["start_epoch"] and random_value < config["p"]
    )
