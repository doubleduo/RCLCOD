#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save only clearly improved visual cases for step-wise ablation analysis.
Single-file version. Based on the user's SAM2 scribble-assisted refinement pipeline.

What it does:
1) Randomly sample N images.
2) Run a small set of strategy comparisons.
3) Save only images where the strategy clearly improves MAE over its baseline.
4) Draw the applied strategy directly on the RGB image:
   - crop rectangle
   - per-box rectangles
   - added positive / negative points
   - text note for memory logits

Comparisons saved:
- Crop_effect:      G0_Box            -> G1_Crop
- PerBox_effect:    G0_Box            -> G2_PerBox
- Pos_effect:       P0_Repeat         -> P1_Pos
- Neg_effect:       P0_Repeat         -> P2_Neg
- Logits_effect:    P0_Repeat         -> M0_Memory
- PosNegLogits:     P0_Repeat         -> M1_PosNegMemory

Only improved cases are saved. No full 100-image dump.
"""

import argparse
import csv
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


# =========================
# Config
# =========================
@dataclass
class Config:
    sam2_checkpoint: str = "./sam_pseado/weights/sam2_hiera_large.pt"
    sam2_config: str = "sam2_hiera_l.yaml"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    img_dir: str = "./data/Train/Imgs"
    scribble_dir: str = "./data/Train/Scribble"
    gt_dir: str = "./data/Train/GT"
    json_dir: str = "./data/Train/box_label"
    out_dir: str = "./softresult/SAM2_ImprovedOnly"

    size: int = 1024
    seed: int = 2026
    limit: int = 100
    topk_each: int = 20
    min_gain: float = 0.003

    max_initial_per_class: int = 9
    active_min_dist: int = 16
    active_pos: int = 3
    active_neg: int = 3

    small_box_area_thr: float = 0.10
    crop_context: float = 0.35
    crop_min_context: int = 16
    crop_square: bool = True
    box_prompt_margin: int = 5
    per_box_merge_margin: int = 25

    vis_size: int = 320


# =========================
# Basic helpers
# =========================
class DataLoader:
    @staticmethod
    def list_images(directory):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            paths.extend(glob(os.path.join(directory, ext)))
            paths.extend(glob(os.path.join(directory, ext.upper())))
        return sorted(set(paths))

    @staticmethod
    def find_file(directory, name, extensions):
        for ext in extensions:
            for e in (ext.lower(), ext.upper()):
                path = os.path.join(directory, name + e)
                if os.path.isfile(path):
                    return path
        return None

    @staticmethod
    def load_boxes(path, width, height):
        if not path or not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        boxes = []
        for shape in data.get("shapes", []):
            points = np.asarray(shape.get("points", []), np.float32)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
                continue
            box = np.array([points[:, 0].min(), points[:, 1].min(),
                            points[:, 0].max(), points[:, 1].max()], dtype=np.float32)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        return boxes

    @staticmethod
    def parse_scribble(scribble):
        scribble = np.asarray(scribble)
        if np.any(scribble == 1) or np.any(scribble == 2):
            return scribble == 1, scribble == 2
        return scribble > 128, np.zeros_like(scribble, dtype=bool)

    @staticmethod
    def points_from_scribble(scribble, max_per_class=9):
        fg, bg = DataLoader.parse_scribble(scribble)
        points, labels = [], []
        for region, label in ((fg, 1), (bg, 0)):
            ys, xs = np.where(region)
            if len(xs) == 0:
                continue
            dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
            rows = cols = max(1, int(np.ceil(np.sqrt(max_per_class))))
            y_edges = np.linspace(ys.min(), ys.max() + 1, rows + 1)
            x_edges = np.linspace(xs.min(), xs.max() + 1, cols + 1)
            for i in range(rows):
                for j in range(cols):
                    sel = (ys >= y_edges[i]) & (ys < y_edges[i + 1]) & \
                          (xs >= x_edges[j]) & (xs < x_edges[j + 1])
                    if not np.any(sel):
                        continue
                    cx, cy = xs[sel], ys[sel]
                    k = int(np.argmax(dist[cy, cx]))
                    points.append([float(cx[k]), float(cy[k])])
                    labels.append(label)
        if not points:
            return None, None
        return np.asarray(points, np.float32), np.asarray(labels, np.int32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float32), -30, 30)))


def resize_prob(prob, dst_w, dst_h):
    prob = np.asarray(prob, np.float32)
    if prob.shape == (dst_h, dst_w):
        return prob.copy()
    return cv2.resize(prob, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)


def union_box(boxes):
    a = np.asarray(boxes, np.float32)
    return np.asarray([a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()], np.float32)


def clip_box(box, width, height):
    b = np.asarray(box, np.float32).copy()
    b[[0, 2]] = np.clip(b[[0, 2]], 0, width - 1)
    b[[1, 3]] = np.clip(b[[1, 3]], 0, height - 1)
    return b


def expand_box(box, pad, width, height):
    b = np.asarray(box, np.float32).copy()
    b += np.asarray([-pad, -pad, pad, pad], np.float32)
    return clip_box(b, width, height)


def centered_square_crop(box, width, height, context, min_context, square=True):
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = max(float(min_context), max(bw, bh) * float(context))
    crop_w = bw + 2 * pad
    crop_h = bh + 2 * pad
    if square:
        crop_w = crop_h = max(crop_w, crop_h)
    crop_w = min(width, int(np.ceil(crop_w)))
    crop_h = min(height, int(np.ceil(crop_h)))
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    x0 = int(np.clip(np.round(cx - crop_w / 2), 0, width - crop_w))
    y0 = int(np.clip(np.round(cy - crop_h / 2), 0, height - crop_h))
    return x0, y0, x0 + crop_w, y0 + crop_h


def transform_box_to_model(box, crop_rect, model_size, margin):
    x0, y0, x1, y1 = crop_rect
    sx, sy = model_size / (x1 - x0), model_size / (y1 - y0)
    b = np.asarray(box, np.float32)
    b = np.asarray([(b[0] - x0 + .5) * sx - .5 - margin,
                    (b[1] - y0 + .5) * sy - .5 - margin,
                    (b[2] - x0 + .5) * sx - .5 + margin,
                    (b[3] - y0 + .5) * sy - .5 + margin], np.float32)
    return clip_box(b, model_size, model_size)


def binary_iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


# =========================
# Metrics
# =========================
class MetricsEvaluator:
    @staticmethod
    def prepare_pred_gt(pred, gt):
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        pred = pred.astype(np.float32)
        if pred.max() > 1.0:
            pred /= 255.0
        return np.clip(pred, 0.0, 1.0), (gt > 128).astype(np.float32)

    @staticmethod
    def object_score(x):
        if x.size == 0:
            return 0.0
        m, sd = float(x.mean()), float(x.std())
        return (2 * m) / (m * m + 1 + sd + 1e-8)

    @classmethod
    def s_object(cls, p, gt):
        u = float(gt.mean())
        return u * cls.object_score(p[gt == 1]) + (1 - u) * cls.object_score(1 - p[gt == 0])

    @staticmethod
    def ssim_region(p, gt):
        if p.size == 0:
            return 0.0
        x, y = float(p.mean()), float(gt.mean())
        sx, sy = float(((p - x) ** 2).mean()), float(((gt - y) ** 2).mean())
        sxy = float(((p - x) * (gt - y)).mean())
        return 4 * x * y * sxy / ((x * x + y * y) * (sx + sy) + 1e-8)

    @classmethod
    def s_region(cls, p, gt):
        h, w = gt.shape
        if h < 2 or w < 2:
            return 0.0
        ys, xs = np.where(gt > .5)
        x = int(np.clip(np.round(xs.mean()), 1, w - 1))
        y = int(np.clip(np.round(ys.mean()), 1, h - 1))
        return (x * y * cls.ssim_region(p[:y, :x], gt[:y, :x]) +
                (w - x) * y * cls.ssim_region(p[:y, x:], gt[:y, x:]) +
                x * (h - y) * cls.ssim_region(p[y:, :x], gt[y:, :x]) +
                (w - x) * (h - y) * cls.ssim_region(p[y:, x:], gt[y:, x:])) / (h * w)

    @classmethod
    def evaluate(cls, pred_raw, gt_raw):
        p, gt = cls.prepare_pred_gt(pred_raw, gt_raw)
        m = float(gt.mean())
        if m == 0.0:
            s = 1 - float(p.mean())
        elif m == 1.0:
            s = float(p.mean())
        else:
            s = float(np.clip(0.5 * cls.s_object(p, gt) + 0.5 * cls.s_region(p, gt), 0, 1))
        binary = p >= 0.5
        gtb = gt >= 0.5
        tp = float((binary & gtb).sum())
        precision = tp / (float(binary.sum()) + 1e-8)
        recall = tp / (float(gtb.sum()) + 1e-8)
        fb = float(1.3 * precision * recall / (0.3 * precision + recall + 1e-8))
        if gtb.sum() == 0:
            eb = 1 - float(binary.mean())
        elif gtb.sum() == gtb.size:
            eb = float(binary.mean())
        else:
            fm, gtm = binary.astype(np.float32), gt.astype(np.float32)
            fm -= fm.mean(); gtm -= gtm.mean()
            alignment = 2 * fm * gtm / (fm * fm + gtm * gtm + 1e-8)
            eb = float(np.mean((alignment + 1) ** 2 / 4))
        mae = float(np.mean(np.abs(p - gt)))
        return {"Smeasure": s, "maxE": eb, "maxF": fb, "MAE": mae}


# =========================
# SAM2 Engine
# =========================
class SAM2Engine:
    def __init__(self, cfg, predictor=None):
        self.cfg = cfg
        self.device = cfg.device
        if predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(cfg.sam2_config, cfg.sam2_checkpoint, device=cfg.device)
            predictor = SAM2ImagePredictor(model)
        self.predictor = predictor
        self.last_input_shape = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def set_image(self, model_rgb):
        self.last_input_shape = model_rgb.shape[:2]
        with torch.inference_mode():
            self.predictor.set_image(np.ascontiguousarray(model_rgb))

    def predict_once(self, points, labels, box, mask_input):
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else nullcontext())
        with torch.inference_mode(), autocast:
            high_logits, scores, low_logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box[None, :] if box is not None else None,
                mask_input=mask_input,
                multimask_output=False,
                return_logits=True,
            )
        high_logits = np.asarray(high_logits, np.float32)
        low_logits = np.asarray(low_logits, np.float32)
        scores = np.asarray(scores, np.float32)
        if high_logits.ndim == 3:
            high_logits = high_logits[0]
        if low_logits.ndim == 2:
            low_logits = low_logits[None, ...]
        if low_logits.ndim == 4 and low_logits.shape[0] == 1:
            low_logits = low_logits[0]
        if low_logits.ndim != 3 or low_logits.shape[0] != 1:
            raise ValueError(f"Expected 1xHxW low-res mask, got {low_logits.shape}")
        return sigmoid(high_logits), low_logits, float(scores[0] if scores.ndim else scores)


# =========================
# Strategy point samplers
# =========================
def _erase_neighborhood(mask_bool, existing_points, radius):
    available = mask_bool.astype(np.uint8) * 255
    if existing_points is not None and len(existing_points):
        for x, y in np.asarray(existing_points, np.float32):
            cv2.circle(available, (int(round(x)), int(round(y))), int(radius), 0, -1)
    return available > 0


def sample_positive_points(prob, scribble, existing_points, box, max_points=3, min_dist=16):
    fg, bg = DataLoader.parse_scribble(scribble)
    pred_fg = prob >= 0.5

    # 1) Correct confident misses on known FG scribble.
    contradiction = fg & (~pred_fg)
    contradiction = _erase_neighborhood(contradiction, existing_points, min_dist)
    pts, labels = [], []
    p = resize_prob(prob, scribble.shape[1], scribble.shape[0])

    if contradiction.any():
        ys, xs = np.where(contradiction)
        scores = 1.0 - p[ys, xs]
        order = np.argsort(-scores)
        chosen = []
        for idx in order:
            x, y = float(xs[idx]), float(ys[idx])
            if any((x - a) ** 2 + (y - b) ** 2 < min_dist ** 2 for a, b in chosen):
                continue
            chosen.append((x, y))
            pts.append([x, y]); labels.append(1)
            if len(pts) >= max_points:
                return np.asarray(pts, np.float32), np.asarray(labels, np.int32)

    # 2) Reliable interior points inside current foreground and box.
    h, w = scribble.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    roi = np.zeros((h, w), bool)
    if x2 >= x1 and y2 >= y1:
        roi[y1:y2 + 1, x1:x2 + 1] = True

    reliable = pred_fg & (p > 0.7) & roi & (~bg)
    reliable = _erase_neighborhood(reliable, np.vstack([existing_points, np.asarray(pts, np.float32)])
                                   if (existing_points is not None and len(pts)) else (existing_points if existing_points is not None else np.empty((0, 2))),
                                   min_dist)
    if reliable.any():
        dist = cv2.distanceTransform(reliable.astype(np.uint8), cv2.DIST_L2, 5)
        ys, xs = np.where(reliable)
        scores = dist[ys, xs]
        order = np.argsort(-scores)
        chosen = [(p0[0], p0[1]) for p0 in pts]
        for idx in order:
            x, y = float(xs[idx]), float(ys[idx])
            if any((x - a) ** 2 + (y - b) ** 2 < min_dist ** 2 for a, b in chosen):
                continue
            chosen.append((x, y))
            pts.append([x, y]); labels.append(1)
            if len(pts) >= max_points:
                break

    if not pts:
        return None, None
    return np.asarray(pts, np.float32), np.asarray(labels, np.int32)


def sample_box_outside_negatives(model_box, size, max_points=3, margin=18, min_dist=16):
    x1, y1, x2, y2 = map(float, model_box)
    candidates = []
    for frac in (0.15, 0.35, 0.5, 0.65, 0.85):
        xx = x1 + frac * (x2 - x1)
        yy = y1 + frac * (y2 - y1)
        candidates += [(xx, y1 - margin), (xx, y2 + margin), (x1 - margin, yy), (x2 + margin, yy)]
    valid = []
    for x, y in candidates:
        if not (0 <= x < size and 0 <= y < size):
            continue
        if x1 <= x <= x2 and y1 <= y <= y2:
            continue
        if any((x - a) ** 2 + (y - b) ** 2 < min_dist ** 2 for a, b in valid):
            continue
        valid.append((x, y))
    if not valid:
        return None, None
    chosen = valid[:max_points]
    return np.asarray(chosen, np.float32), np.zeros(len(chosen), np.int32)


def sample_negative_points(prob, scribble, existing_points, box, size, max_points=3, min_dist=16):
    # 1) Correct confident false positives on known BG scribble.
    fg, bg = DataLoader.parse_scribble(scribble)
    pred_fg = prob >= 0.5
    contradiction = bg & pred_fg
    contradiction = _erase_neighborhood(contradiction, existing_points, min_dist)
    pts, labels = [], []
    p = resize_prob(prob, scribble.shape[1], scribble.shape[0])
    if contradiction.any():
        ys, xs = np.where(contradiction)
        scores = p[ys, xs]
        order = np.argsort(-scores)
        chosen = []
        for idx in order:
            x, y = float(xs[idx]), float(ys[idx])
            if any((x - a) ** 2 + (y - b) ** 2 < min_dist ** 2 for a, b in chosen):
                continue
            chosen.append((x, y))
            pts.append([x, y]); labels.append(0)
            if len(pts) >= max_points:
                return np.asarray(pts, np.float32), np.asarray(labels, np.int32)

    # 2) Safe ring negatives around box.
    ext_pts, ext_lab = sample_box_outside_negatives(box, size, max_points=max_points - len(pts), margin=18, min_dist=min_dist)
    if ext_pts is not None:
        for q in ext_pts:
            pts.append([float(q[0]), float(q[1])]); labels.append(0)
            if len(pts) >= max_points:
                break
    if not pts:
        return None, None
    return np.asarray(pts, np.float32), np.asarray(labels, np.int32)


# =========================
# Visual helpers
# =========================
def overlay_mask_errors(rgb_bgr, pred, gt):
    pred = pred >= 0.5 if pred.dtype != np.uint8 else pred > 0
    gt = gt > 128 if gt.dtype != np.uint8 else gt > 0
    vis = rgb_bgr.copy()
    fp = pred & (~gt)
    fn = (~pred) & gt
    vis[fp] = (0.55 * vis[fp] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)  # red
    vis[fn] = (0.55 * vis[fn] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)  # blue
    return vis


def draw_boxes_on_rgb(rgb_bgr, boxes, color=(0, 255, 255), thickness=2):
    out = rgb_bgr.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, np.round(box))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(out, f"b{i}", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def draw_crop_rect(rgb_bgr, crop_rect, color=(0, 255, 0), thickness=2):
    out = rgb_bgr.copy()
    x0, y0, x1, y1 = crop_rect
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), color, thickness)
    cv2.putText(out, "crop", (x0, max(0, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def draw_points(rgb_bgr, points, labels, scale_xy=None):
    out = rgb_bgr.copy()
    if points is None or labels is None:
        return out
    points = np.asarray(points, np.float32)
    labels = np.asarray(labels, np.int32)
    for (x, y), lb in zip(points, labels):
        if scale_xy is not None:
            sx, sy = scale_xy
            x, y = x * sx, y * sy
        x, y = int(round(x)), int(round(y))
        if lb == 1:
            cv2.circle(out, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(out, (x, y), 8, (255, 255, 255), 1)
        else:
            cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
            cv2.line(out, (x - 6, y - 6), (x + 6, y + 6), (255, 255, 255), 1)
            cv2.line(out, (x + 6, y - 6), (x - 6, y + 6), (255, 255, 255), 1)
    return out


def put_note(img, text, y=20, color=(0, 255, 255)):
    out = img.copy()
    cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out


def panel_image(cfg, rgb_bgr, gt, base_prob, var_prob, title_a, title_b, rgb_a, rgb_b, mae_a, mae_b, gain, note):
    """Five columns: RGB | GT | annotated RGB | before binary mask | after binary mask."""
    sz = cfg.vis_size
    rgb0 = cv2.resize(rgb_bgr, (sz, sz), interpolation=cv2.INTER_LINEAR)
    gt_bin = (gt > 128).astype(np.uint8) * 255
    gt0 = cv2.cvtColor(cv2.resize(gt_bin, (sz, sz), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    strategy = cv2.resize(rgb_b, (sz, sz), interpolation=cv2.INTER_LINEAR)
    before_bin = (base_prob >= 0.5).astype(np.uint8) * 255
    after_bin = (var_prob >= 0.5).astype(np.uint8) * 255
    before_vis = cv2.cvtColor(cv2.resize(before_bin, (sz, sz), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    after_vis = cv2.cvtColor(cv2.resize(after_bin, (sz, sz), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

    # Put labels in separate bands: never draw on masks or obscure point locations.
    images = [rgb0, gt0, strategy, before_vis, after_vis]
    labels = ["RGB", "GT", "Strategy (boxes/points)",
              f"Before MAE={mae_a:.4f}", f"After MAE={mae_b:.4f}"]
    tiles = []
    for im, label in zip(images, labels):
        band = np.full((30, sz, 3), 255, np.uint8)
        cv2.putText(band, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([band, im]))
    top = np.concatenate(tiles, axis=1)
    banner = np.full((38, top.shape[1], 3), 255, np.uint8)
    cv2.putText(banner, f"Gain(MAE): {gain:.4f} | {note}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 120, 0), 2, cv2.LINE_AA)
    return np.concatenate([banner, top], axis=0)


# =========================
# Runner / Pipeline
# =========================
class SingleFileRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engine = SAM2Engine(cfg)
        os.makedirs(cfg.out_dir, exist_ok=True)
        self._cached_rgb = None
        self._cached_region = None

    def _prepare_single(self, rgb, scribble, box, use_crop):
        h, w = rgb.shape[:2]
        ratio = max(0.0, float((box[2] - box[0]) * (box[3] - box[1]) / (w * h)))
        crop = use_crop and ratio <= self.cfg.small_box_area_thr
        region = centered_square_crop(box, w, h, self.cfg.crop_context,
                                      self.cfg.crop_min_context, self.cfg.crop_square) if crop else (0, 0, w, h)
        if region == (0, 0, w, h):
            crop = False
        x0, y0, x1, y1 = region
        img_crop = rgb[y0:y1, x0:x1]
        sc_crop = scribble[y0:y1, x0:x1]
        model_img = cv2.resize(img_crop, (self.cfg.size, self.cfg.size), interpolation=cv2.INTER_LINEAR)
        model_sc = cv2.resize(sc_crop, (self.cfg.size, self.cfg.size), interpolation=cv2.INTER_NEAREST)
        model_box = transform_box_to_model(box, region, self.cfg.size, self.cfg.box_prompt_margin)

        fg, bg = DataLoader.parse_scribble(model_sc)
        roi = np.zeros((self.cfg.size, self.cfg.size), np.uint8)
        x1m, y1m, x2m, y2m = np.round(model_box).astype(int)
        x1m, x2m = max(0, x1m), min(self.cfg.size - 1, x2m)
        y1m, y2m = max(0, y1m), min(self.cfg.size - 1, y2m)
        if x2m >= x1m and y2m >= y1m:
            roi[y1m:y2m + 1, x1m:x2m + 1] = 1
        model_sc = model_sc.copy()
        outside_fg = fg & (roi == 0)
        model_sc[outside_fg] = 0
        init_points, init_labels = DataLoader.points_from_scribble(model_sc, self.cfg.max_initial_per_class)
        meta = {
            "crop_used": crop,
            "crop_rect": region,
            "model_box": model_box,
            "model_sc": model_sc,
            "init_points": init_points,
            "init_labels": init_labels,
        }
        return model_img, model_box, meta

    def _predict_variant_single(self, rgb, scribble, box, use_crop, second_mode):
        """
        second_mode:
            'none'            -> one call only
            'repeat'          -> second call, no memory, no new points
            'pos'             -> second call, add positive points only, no memory
            'neg'             -> second call, add negative points only, no memory
            'memory'          -> second call, same points + memory logits
            'posnegmemory'    -> second call, add pos/neg + memory logits
        """
        h, w = rgb.shape[:2]
        model_img, model_box, meta = self._prepare_single(rgb, scribble, box, use_crop)
        self.engine.set_image(model_img)

        pts = meta["init_points"]
        labs = meta["init_labels"]
        p0, memory0, _ = self.engine.predict_once(pts, labs, model_box, None)

        actions = {
            "crop_rect": meta["crop_rect"],
            "crop_used": meta["crop_used"],
            "per_boxes": [box],
            "added_points": None,
            "added_labels": None,
            "memory_used": False,
            "note": "",
        }

        final_p = p0
        if second_mode == "none":
            actions["note"] = "box+scribble"
        else:
            new_pts, new_labs = pts, labs
            add_pts, add_labs = None, None

            if second_mode in ("pos", "posnegmemory"):
                add_pts, add_labs = sample_positive_points(p0, meta["model_sc"], pts, model_box,
                                                           max_points=self.cfg.active_pos,
                                                           min_dist=self.cfg.active_min_dist)
            if second_mode in ("neg", "posnegmemory"):
                npts, nlabs = sample_negative_points(p0, meta["model_sc"],
                                                     pts if add_pts is None else np.concatenate([pts, add_pts], axis=0),
                                                     model_box, self.cfg.size,
                                                     max_points=self.cfg.active_neg,
                                                     min_dist=self.cfg.active_min_dist)
                if npts is not None:
                    if add_pts is None:
                        add_pts, add_labs = npts, nlabs
                    else:
                        add_pts = np.concatenate([add_pts, npts], axis=0)
                        add_labs = np.concatenate([add_labs, nlabs], axis=0)

            if add_pts is not None:
                if new_pts is None:
                    new_pts, new_labs = add_pts, add_labs
                else:
                    new_pts = np.concatenate([new_pts, add_pts], axis=0)
                    new_labs = np.concatenate([new_labs, add_labs], axis=0)
                # Store in ORIGINAL image coordinates for accurate RGB visualization.
                x0, y0, x1, y1 = meta["crop_rect"]
                sx, sy = (x1 - x0) / self.cfg.size, (y1 - y0) / self.cfg.size
                mapped = add_pts.copy().astype(np.float32)
                mapped[:, 0] = x0 + (mapped[:, 0] + .5) * sx - .5
                mapped[:, 1] = y0 + (mapped[:, 1] + .5) * sy - .5
                actions["added_points"] = mapped
                actions["added_labels"] = add_labs

            mask_input = None
            if second_mode in ("memory", "posnegmemory"):
                mask_input = memory0
                actions["memory_used"] = True

            final_p, _, _ = self.engine.predict_once(new_pts, new_labs, model_box, mask_input)
            actions["note"] = second_mode

        prob_orig = np.zeros((h, w), np.float32)
        x0, y0, x1, y1 = meta["crop_rect"]
        prob_orig[y0:y1, x0:x1] = resize_prob(final_p, x1 - x0, y1 - y0)
        return prob_orig, actions

    def _predict_variant_multi(self, rgb, scribble, boxes, use_crop, per_box, second_mode):
        if not per_box or len(boxes) == 1:
            box = union_box(boxes)
            prob, actions = self._predict_variant_single(rgb, scribble, box, use_crop, second_mode)
            actions["per_boxes"] = [box] if len(boxes) == 1 else list(boxes)
            return prob, actions

        h, w = rgb.shape[:2]
        merged = np.zeros((h, w), np.float32)
        all_added_pts, all_added_labs = [], []
        crop_rects = []
        for box in boxes:
            prob, act = self._predict_variant_single(rgb, scribble, box, use_crop, second_mode)
            crop_rects.append(act["crop_rect"])
            margin = max(2, round(self.cfg.per_box_merge_margin * max(h, w) / self.cfg.size))
            gated = expand_box(box, margin, w, h)
            x1, y1, x2, y2 = map(int, np.round(gated))
            gate = np.zeros((h, w), np.float32)
            gate[y1:y2 + 1, x1:x2 + 1] = 1.0
            merged = np.maximum(merged, prob * gate)
            if act["added_points"] is not None:
                # Single-box action points already use ORIGINAL image coordinates.
                for pt, lb in zip(act["added_points"], act["added_labels"]):
                    all_added_pts.append([float(pt[0]), float(pt[1])])
                    all_added_labs.append(int(lb))

        actions = {
            "crop_rect": crop_rects[0] if crop_rects else (0, 0, w, h),
            "crop_used": use_crop,
            "per_boxes": list(boxes),
            "added_points": np.asarray(all_added_pts, np.float32) if all_added_pts else None,
            "added_labels": np.asarray(all_added_labs, np.int32) if all_added_labs else None,
            "memory_used": second_mode in ("memory", "posnegmemory"),
            "note": second_mode,
        }
        return merged, actions

    def run_image(self, image_path):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        json_path = DataLoader.find_file(self.cfg.json_dir, basename, (".json",))
        scribble_path = DataLoader.find_file(self.cfg.scribble_dir, basename, (".png", ".jpg"))
        gt_path = DataLoader.find_file(self.cfg.gt_dir, basename, (".png", ".jpg"))
        if not scribble_path or not gt_path:
            return None
        sc = cv2.imread(scribble_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if sc is None or gt is None:
            return None
        if sc.shape != (h, w):
            sc = cv2.resize(sc, (w, h), interpolation=cv2.INTER_NEAREST)
        boxes = DataLoader.load_boxes(json_path, w, h)
        if not boxes:
            return None

        variants = {}
        # geometry
        variants["G0_Box"] = self._predict_variant_multi(rgb, sc, boxes, use_crop=False, per_box=False, second_mode="none")
        variants["G1_Crop"] = self._predict_variant_multi(rgb, sc, boxes, use_crop=True, per_box=False, second_mode="none")
        variants["G2_PerBox"] = self._predict_variant_multi(rgb, sc, boxes, use_crop=False, per_box=True, second_mode="none")
        # prompt / logits on geometry+perbox+crop anchor
        variants["P0_Repeat"] = self._predict_variant_multi(rgb, sc, boxes, use_crop=True, per_box=True, second_mode="repeat")
        variants["P1_Pos"] = self._predict_variant_multi(rgb, sc, boxes, use_crop=True, per_box=True, second_mode="pos")
        variants["P2_Neg"] = self._predict_variant_multi(rgb, sc, boxes, use_crop=True, per_box=True, second_mode="neg")
        variants["M0_Memory"] = self._predict_variant_multi(rgb, sc, boxes, use_crop=True, per_box=True, second_mode="memory")
        variants["M1_PosNegMemory"] = self._predict_variant_multi(rgb, sc, boxes, use_crop=True, per_box=True, second_mode="posnegmemory")

        out = {"rgb_bgr": img, "gt": gt, "boxes": boxes, "results": {}}
        for name, (prob, actions) in variants.items():
            out["results"][name] = {
                "prob": prob,
                "actions": actions,
                "metrics": MetricsEvaluator.evaluate((prob >= 0.5).astype(np.uint8) * 255, gt)
            }
        return basename, out


# =========================
# Save only improved cases
# =========================
def save_improved_case(cfg, save_dir, basename, effect_name, base_name, var_name, data):
    os.makedirs(save_dir, exist_ok=True)
    rgb = data["rgb_bgr"]
    gt = data["gt"]
    base = data["results"][base_name]
    var = data["results"][var_name]

    base_prob = base["prob"]
    var_prob = var["prob"]
    base_mae = base["metrics"]["MAE"]
    var_mae = var["metrics"]["MAE"]
    gain = base_mae - var_mae

    rgb_a = rgb.copy()
    rgb_b = rgb.copy()

    # baseline overlay
    act_a = base["actions"]
    if effect_name.startswith("Crop") and act_a.get("crop_used", False):
        rgb_a = draw_crop_rect(rgb_a, act_a["crop_rect"])
    if effect_name.startswith("PerBox"):
        rgb_a = draw_boxes_on_rgb(rgb_a, act_a["per_boxes"])

    # variant overlay = show the applied strategy explicitly
    act_b = var["actions"]
    rgb_b = draw_boxes_on_rgb(rgb_b, act_b["per_boxes"])
    if act_b.get("crop_used", False):
        rgb_b = draw_crop_rect(rgb_b, act_b["crop_rect"])
    if act_b.get("added_points") is not None:
        rgb_b = draw_points(rgb_b, act_b["added_points"], act_b["added_labels"])
    note = []
    if effect_name.startswith("Crop"):
        note.append("crop")
    if effect_name.startswith("PerBox"):
        note.append("per-box")
    if "Pos" in var_name:
        note.append("positive points")
    if "Neg" in var_name:
        note.append("negative points")
    if act_b.get("memory_used", False):
        note.append("memory logits")
    note_text = ", ".join(note) if note else var_name

    panel = panel_image(cfg, rgb, gt, base_prob, var_prob, base_name, var_name,
                        rgb_a, rgb_b, base_mae, var_mae, gain, note_text)
    cv2.imwrite(os.path.join(save_dir, f"{gain:.4f}_{basename}.jpg"), panel)

    # Export paper-ready assets separately, all at ORIGINAL resolution.
    # No text, no arrows, no color overlays on binary masks.
    asset_dir = os.path.join(save_dir, "separate", basename)
    os.makedirs(asset_dir, exist_ok=True)
    cv2.imwrite(os.path.join(asset_dir, "RGB.png"), rgb)
    cv2.imwrite(os.path.join(asset_dir, "GT.png"), (gt > 128).astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(asset_dir, "Before_Mask.png"),
                (base_prob >= 0.5).astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(asset_dir, "After_Mask.png"),
                (var_prob >= 0.5).astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(asset_dir, "Strategy_RGB.png"), rgb_b)
    # Useful when arranging a manuscript figure: show the corrected region alone.
    before = base_prob >= 0.5
    after = var_prob >= 0.5
    cv2.imwrite(os.path.join(asset_dir, "Added_FG.png"),
                ((~before) & after).astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(asset_dir, "Removed_FG.png"),
                (before & (~after)).astype(np.uint8) * 255)
    with open(os.path.join(asset_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump({"effect": effect_name, "baseline": base_name, "variant": var_name,
                   "MAE_before": base_mae, "MAE_after": var_mae,
                   "MAE_gain": gain, "memory_used": bool(act_b.get("memory_used", False))},
                  f, ensure_ascii=False, indent=2)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sam2-checkpoint", default=Config.sam2_checkpoint)
    parser.add_argument("--sam2-config", default=Config.sam2_config)
    parser.add_argument("--img-dir", default=Config.img_dir)
    parser.add_argument("--scribble-dir", default=Config.scribble_dir)
    parser.add_argument("--gt-dir", default=Config.gt_dir)
    parser.add_argument("--json-dir", default=Config.json_dir)
    parser.add_argument("--output-dir", default=Config.out_dir)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--limit", type=int, default=Config.limit)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--min-gain", type=float, default=Config.min_gain)
    parser.add_argument("--topk-each", type=int, default=Config.topk_each)
    args = parser.parse_args()

    cfg = Config(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        img_dir=args.img_dir,
        scribble_dir=args.scribble_dir,
        gt_dir=args.gt_dir,
        json_dir=args.json_dir,
        out_dir=args.output_dir,
        limit=args.limit,
        seed=args.seed,
        min_gain=args.min_gain,
        topk_each=args.topk_each,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    images = DataLoader.list_images(cfg.img_dir)
    random.Random(cfg.seed).shuffle(images)
    images = images[:cfg.limit]

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "sampled_files.txt"), "w", encoding="utf-8") as f:
        for p in images:
            f.write(p + "\n")

    runner = SingleFileRunner(cfg)

    comparisons = {
        "Crop_effect": ("G0_Box", "G1_Crop"),
        "PerBox_effect": ("G0_Box", "G2_PerBox"),
        "Pos_effect": ("P0_Repeat", "P1_Pos"),
        "Neg_effect": ("P0_Repeat", "P2_Neg"),
        "Logits_effect": ("P0_Repeat", "M0_Memory"),
        "PosNegLogits": ("P0_Repeat", "M1_PosNegMemory"),
    }

    saved = {k: [] for k in comparisons}
    failures = []

    for path in tqdm(images, desc="Saving improved-only cases"):
        try:
            ret = runner.run_image(path)
            if ret is None:
                continue
            basename, data = ret
            for effect_name, (base_name, var_name) in comparisons.items():
                base_mae = data["results"][base_name]["metrics"]["MAE"]
                var_mae = data["results"][var_name]["metrics"]["MAE"]
                gain = base_mae - var_mae
                if gain >= cfg.min_gain:
                    saved[effect_name].append({
                        "basename": basename,
                        "gain": gain,
                        "base": base_name,
                        "variant": var_name,
                        "data": data,
                    })
        except Exception as exc:
            failures.append({"image": path, "error": repr(exc)})
            print(f"[ERROR] {os.path.basename(path)}: {exc}")
            continue

    summary = {"config": vars(cfg), "effects": {}, "failures": failures}
    for effect_name, rows in saved.items():
        rows = sorted(rows, key=lambda d: d["gain"], reverse=True)[:cfg.topk_each]
        save_dir = os.path.join(cfg.out_dir, effect_name)
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "ranking.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "image", "gain_mae", "baseline", "variant"])
            for rank, row in enumerate(rows, 1):
                writer.writerow([rank, row["basename"], f"{row['gain']:.6f}", row["base"], row["variant"]])
                save_improved_case(cfg, save_dir, row["basename"], effect_name, row["base"], row["variant"], row["data"])

        summary["effects"][effect_name] = {
            "num_saved": len(rows),
            "best_gain": float(rows[0]["gain"]) if rows else 0.0,
            "mean_gain_saved": float(np.mean([r["gain"] for r in rows])) if rows else 0.0,
            "folder": save_dir,
        }

    with open(os.path.join(cfg.out_dir, "summary_improved_only.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    for k, v in summary["effects"].items():
        print(f"{k:>16}: saved={v['num_saved']} best_gain={v['best_gain']:.4f} mean_gain_saved={v['mean_gain_saved']:.4f}")
    print("Failures:", len(failures))
    print("Output:", cfg.out_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()
