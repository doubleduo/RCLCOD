"""Box-PKD geometry; no segmentation GT is used.

Rasterization convention: the band lies INSIDE each box, with horizontal
inset width / 4 and vertical inset height / 4. This is an explicit convention
where the paper does not supply a rasterization script. Union bands are binary:
overlapping boxes do not multiply the extra weight.
"""
import json
import math

import numpy as np


def read_box_prompts(path, height, width, band_ratio=0.25):
    if not 0 < band_ratio <= 0.5:
        raise ValueError("band_ratio must be in (0, 0.5].")
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    # Coordinate metadata must match RGB; silently resizing boxes is unsafe.
    for key, size in (("imageHeight", height), ("imageWidth", width)):
        if payload.get(key) not in (None, size):
            raise ValueError(f"{path}: {key} does not match RGB dimensions.")
    filled = np.zeros((height, width), dtype=np.float32)
    band = np.zeros_like(filled)
    boxes = []
    for shape in payload.get("shapes", []):
        if shape.get("shape_type", "rectangle") not in ("rectangle", "polygon"):
            continue
        points = np.asarray(shape.get("points", []), dtype=np.float64)
        if points.size == 0:
            continue
        points = points.reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError(f"{path}: nonfinite box coordinates.")
        x1, y1 = np.floor(points.min(axis=0)).astype(int)
        x2, y2 = np.ceil(points.max(axis=0)).astype(int)
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
        filled[y1:y2, x1:x2] = 1
        dx = max(1, math.ceil((x2 - x1) * band_ratio))
        dy = max(1, math.ceil((y2 - y1) * band_ratio))
        # Union of per-box bands. Never erase another box's band.
        band[y1:min(y1 + dy, y2), x1:x2] = 1
        band[max(y2 - dy, y1):y2, x1:x2] = 1
        band[y1:y2, x1:min(x1 + dx, x2)] = 1
        band[y1:y2, max(x2 - dx, x1):x2] = 1
    if not boxes:
        raise ValueError(f"No valid rectangle/polygon found in {path}")
    return filled, band, boxes
