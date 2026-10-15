# -*- coding: utf-8 -*-
"""Configurable, independent-p=0.5 augmentations for RCLCOD.

All enabled transforms are applied independently. `mask` and optional
`box_mask` receive spatial transforms but never photometric transforms.

Usage in basemain.ImageTrainDataset:
    self.trains = build_train_augmentations(augment_cfg, self.use_box)
"""
from __future__ import annotations

from collections.abc import Mapping

import albumentations as A
import cv2


_DEFAULTS = {
    "p": 0.5,
    "flip": True,
    "rotate": True,
    "brightness_contrast": True,
    "hsv": True,
    "affine": False,
    "gamma": False,
    "blur": False,
}


def _read_augment_cfg(augment_cfg):
    """Validate options while preserving existing augmentation by default."""
    conf = dict(_DEFAULTS)
    if augment_cfg is not None:
        if not isinstance(augment_cfg, Mapping):
            raise TypeError("train.augment must be a mapping/dict")
        unknown = set(augment_cfg) - set(_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown train.augment keys: {sorted(unknown)}")
        conf.update(dict(augment_cfg))

    p = conf["p"]
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
        raise ValueError("train.augment.p must be a number in [0, 1]")
    conf["p"] = float(p)
    for key in _DEFAULTS:
        if key == "p":
            continue
        if not isinstance(conf[key], bool):
            raise TypeError(f"train.augment.{key} must be True or False")
    return conf


def build_train_augmentations(augment_cfg=None, use_box=False):
    """Build Compose; every enabled augmentation has its OWN probability p.

    A0 / None reproduces the repository's original four transforms and params.
    A1-A7 add independent Affine, Gamma, and/or GaussianBlur at p=0.5.
    """
    conf = _read_augment_cfg(augment_cfg)
    p = conf["p"]
    transforms = []
    if conf["flip"]:
        transforms.append(A.HorizontalFlip(p=p))

    if conf["rotate"]:
        transforms.append(
            A.Rotate(
                limit=90,
                p=p,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REPLICATE,
            )
        )

    if conf["brightness_contrast"]:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=p,
            )
        )

    if conf["hsv"]:
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=p,
            )
        )

    if conf["affine"]:
        transforms.append(
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REPLICATE,
                p=p,
            )
        )

    if conf["gamma"]:
        transforms.append(
            A.RandomGamma(gamma_limit=(85, 115), p=p)
        )

    if conf["blur"]:
        transforms.append(
            A.GaussianBlur(blur_limit=(3, 3), p=p)
        )

    return A.Compose(
        transforms,
        additional_targets={"box_mask": "mask"} if use_box else {},
    )
