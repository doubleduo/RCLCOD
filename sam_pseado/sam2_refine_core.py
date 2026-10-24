#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 scribble-assisted active refinement, real small-object crop,
per-box multi-object prediction and consistency-gated horizontal-flip TTA.

Important: because foreground/background scribbles are used as ground-truth
prompts, this is NOT a box-only weak supervision experiment.

Examples:
    python sam2_refine_crop_flipgate.py
    python sam2_refine_crop_flipgate.py --no-flip-gate
    python sam2_refine_crop_flipgate.py --no-flip-gate --no-crop
    python sam2_refine_crop_flipgate.py --flip-mode mean
    python sam2_refine_crop_flipgate.py --refine-iters 0 --no-flip-gate
    python sam2_refine_crop_flipgate.py --self-test
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


@dataclass
class Config:
    sam2_checkpoint: str = "./sam_pseado/weights/sam2_hiera_large.pt"
    sam2_config: str = "sam2_hiera_l.yaml"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir: str = "./data/Train/Imgs"
    scribble_dir: str = "./data/Train/Scribble"
    gt_dir: str = "./data/Train/GT"
    json_dir: str = "./data/Train/box_label"
    out_dir: str = "./softresult/SAM2_Optimized3"

    size: int = 1024
    seed: int = 0
    refine_iters: int = 2
    active_points: bool = True
    active_min_dist: int = 16             # on the model input frame (1024x1024)
    active_conflict_first: bool = True    # favor disagreement with *known* scribble labels
    max_initial_per_class: int = 9

    use_crop: bool = True
    small_box_area_thr: float = 0.10     # fraction of full image occupied by the box
    crop_context: float = 0.35          # fraction of max(box_width, box_height)
    crop_min_context: int = 16          # in original-image pixels
    crop_square: bool = True
    box_prompt_margin: int = 5          # on model frame
    multi_box_merge_margin: int = 25    # on model frame

    flip_gate: bool = True
    flip_mode: str = "gate"              # gate | mean (mean is naive TTA ablation)
    flip_min_iou: float = 0.75
    flip_max_area_ratio: float = 1.5
    flip_scribble_tol: float = 0.015     # allowable rise in scribble contradiction rate
    flip_boundary_radius: int = 7       # in original image px (scaled to image size)
    flip_base_uncertainty: float = 0.22  # |p_orig-0.5| <= 0.22, near original boundary
    flip_other_confidence: float = 0.20 # |p_flip-0.5| >= 0.20
    flip_alpha: float = 0.60

    save_probs: bool = False
    vis_mae_thr: float = 0.030
    vis_gain_thr: float = 0.003

    @property
    def base_dir(self):
        return os.path.join(self.out_dir, "01_SoftLogits_Base")

    @property
    def refine_dir(self):
        return os.path.join(self.out_dir, "03_Refine_NoFlip")

    @property
    def profile_dir(self):
        return os.path.join(self.out_dir, "02_SoftLogits_SO_MO")

    @property
    def vis_dir(self):
        return os.path.join(self.out_dir, "visualizations")


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
        """1=foreground, 2=background, 0=unknown; also supports binary 255=FG.

        IMPORTANT: background zero of a binary scribble is unknown, NOT a negative point.
        """
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


def mirror_box(box, width):
    """Use the inclusive pixel-center coordinate convention on both sides."""
    x1, y1, x2, y2 = np.asarray(box, np.float32)
    return np.asarray([width - 1 - x2, y1, width - 1 - x1, y2], np.float32)


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
    """Returns integer half-open crop [x0,y0,x1,y1] in ORIGINAL image coords."""
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


def parse_known_scribble_error(prob, scribble):
    if scribble is None:
        return float("nan")
    fg, bg = DataLoader.parse_scribble(scribble)
    known = fg | bg
    if not known.any():
        return float("nan")
    binary = prob >= 0.5
    incorrect = (fg & ~binary) | (bg & binary)
    return float(incorrect.sum() / known.sum())


def binary_iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def flip_gate_fusion(original_prob, aligned_flip_prob, scribble, cfg):
    """GT-free flip gate. Returns fused original-or-corrected probabilities and diagnostics.

    Image gate: foreground IoU, area ratio, scribble contradiction rate.
    Pixel gate: prefer original; only consider confident flipped corrections at
    uncertain original pixels near its boundary, never contradict known scribble.
    """
    p = np.clip(np.asarray(original_prob, np.float32), 0., 1.)
    q = np.clip(np.asarray(aligned_flip_prob, np.float32), 0., 1.)
    if p.shape != q.shape:
        raise ValueError(f"flip shape mismatch {p.shape} vs {q.shape}")
    a, b = p >= .5, q >= .5
    area_a, area_b = int(a.sum()), int(b.sum())
    area_ratio = float((area_b + 1) / (area_a + 1))
    iou = binary_iou(a, b)
    err_a = parse_known_scribble_error(p, scribble)
    err_b = parse_known_scribble_error(q, scribble)
    diagnostics = {
        "flip_iou": iou,
        "flip_area_ratio": area_ratio,
        "flip_scribble_err_original": err_a,
        "flip_scribble_err_mirror": err_b,
        "flip_accepted": 0,
        "flip_modified_ratio": 0.0,
        "flip_reason": "disabled",
    }
    if cfg.flip_mode == "mean":
        diagnostics["flip_accepted"] = 1
        diagnostics["flip_reason"] = "naive_mean_ablation"
        diagnostics["flip_modified_ratio"] = float(np.mean(np.abs(p - q) > .01))
        return np.clip(0.5 * (p + q), 0, 1), diagnostics
    if min(area_a, area_b) == 0:
        diagnostics["flip_reason"] = "empty_foreground"
        return p, diagnostics
    if iou < cfg.flip_min_iou:
        diagnostics["flip_reason"] = "low_iou"
        return p, diagnostics
    if not (1.0 / cfg.flip_max_area_ratio <= area_ratio <= cfg.flip_max_area_ratio):
        diagnostics["flip_reason"] = "area_mismatch"
        return p, diagnostics
    if np.isfinite(err_a) and np.isfinite(err_b) and \
            err_b > err_a + cfg.flip_scribble_tol:
        diagnostics["flip_reason"] = "scribble_conflict"
        return p, diagnostics

    # Soft smoothing where both predict the same foreground/background, and
    # small edge corrections where baseline is uncertain but flip is confident.
    h, w = p.shape
    r = max(1, int(round(cfg.flip_boundary_radius * max(h, w) / 1024.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    near_boundary = cv2.dilate(a.astype(np.uint8), kernel).astype(bool) ^ \
                    cv2.erode(a.astype(np.uint8), kernel).astype(bool)
    same = a == b
    edge_confident = (np.abs(p - .5) <= cfg.flip_base_uncertainty) & \
                     (np.abs(q - .5) >= cfg.flip_other_confidence) & near_boundary
    correction = (~same) & edge_confident

    # Protect human-annotated scribbles: flip can never override the known label.
    if scribble is not None:
        fg, bg = DataLoader.parse_scribble(scribble)
        target_is_fg = q >= .5
        correction &= ~((fg & ~target_is_fg) | (bg & target_is_fg))

    # Agreement pixels: soft averaging; changes confidence, not the hard mask.
    mixing = np.zeros_like(p, dtype=np.float32)
    mixing[same & near_boundary] = 0.35
    mixing[correction] = cfg.flip_alpha
    fused = p * (1. - mixing) + q * mixing
    diagnostics["flip_accepted"] = 1
    diagnostics["flip_reason"] = "accepted"
    diagnostics["flip_modified_ratio"] = float(np.mean(mixing > 0))
    diagnostics["flip_correction_ratio"] = float(np.mean(correction))
    return np.clip(fused, 0, 1), diagnostics


class MetricsEvaluator:
    """Keeps the old binary-mask COD evaluation semantics for CSV compatibility.

    NOTE: F_binary_0p5/E_binary_0p5 are NOT maxF/maxE over thresholds.
    Legacy column aliases base_maxF/base_maxE are retained for old consumers.
    """
    @staticmethod
    def prepare_pred_gt(pred, gt):
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        pred = pred.astype(np.float32)
        if pred.max() > 1.:
            pred /= 255.
        return np.clip(pred, 0., 1.), (gt > 128).astype(np.float32)

    @staticmethod
    def object_score(x):
        if x.size == 0:
            return 0.
        m, sd = float(x.mean()), float(x.std())
        return (2 * m) / (m * m + 1 + sd + 1e-8)

    @classmethod
    def s_object(cls, p, gt):
        u = float(gt.mean())
        return u * cls.object_score(p[gt == 1]) + (1-u) * cls.object_score(1-p[gt == 0])

    @staticmethod
    def ssim_region(p, gt):
        if p.size == 0:
            return 0.
        x, y = float(p.mean()), float(gt.mean())
        sx, sy = float(((p-x)**2).mean()), float(((gt-y)**2).mean())
        sxy = float(((p-x)*(gt-y)).mean())
        return 4*x*y*sxy / ((x*x+y*y)*(sx+sy)+1e-8)

    @classmethod
    def s_region(cls, p, gt):
        h, w = gt.shape
        if h < 2 or w < 2:
            return 0.
        ys, xs = np.where(gt > .5)
        x = int(np.clip(np.round(xs.mean()), 1, w-1))
        y = int(np.clip(np.round(ys.mean()), 1, h-1))
        return (x*y*cls.ssim_region(p[:y, :x], gt[:y, :x]) +
                (w-x)*y*cls.ssim_region(p[:y, x:], gt[:y, x:]) +
                x*(h-y)*cls.ssim_region(p[y:, :x], gt[y:, :x]) +
                (w-x)*(h-y)*cls.ssim_region(p[y:, x:], gt[y:, x:])) / (h*w)

    @classmethod
    def evaluate(cls, pred_raw, gt_raw):
        p, gt = cls.prepare_pred_gt(pred_raw, gt_raw)
        m = float(gt.mean())
        if m == 0.:
            s = 1 - float(p.mean())
        elif m == 1.:
            s = float(p.mean())
        else:
            s = float(np.clip(0.5*cls.s_object(p, gt) + 0.5*cls.s_region(p, gt), 0, 1))
        binary = p >= .5
        gtb = gt >= .5
        tp = float((binary & gtb).sum())
        precision = tp / (float(binary.sum())+1e-8)
        recall = tp / (float(gtb.sum())+1e-8)
        fb = float(1.3 * precision * recall / (0.3*precision + recall + 1e-8))
        if gtb.sum() == 0:
            eb = 1 - float(binary.mean())
        elif gtb.sum() == gtb.size:
            eb = float(binary.mean())
        else:
            fm, gtm = binary.astype(np.float32), gt.astype(np.float32)
            fm -= fm.mean()
            gtm -= gtm.mean()
            alignment = 2 * fm * gtm / (fm*fm + gtm*gtm + 1e-8)
            eb = float(np.mean((alignment+1)**2/4))
        return {"Smeasure": s, "maxE": eb, "maxF": fb, "MAE": float(np.mean(np.abs(p-gt)))}


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

    def _predict_once(self, points, labels, box, mask_input):
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.startswith("cuda") and torch.cuda.is_available() and
                    torch.cuda.is_bf16_supported() else nullcontext())
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
        if high_logits.ndim == 3:
            high_logits = high_logits[0]
        if low_logits.ndim == 2:
            low_logits = low_logits[None, ...]
        if low_logits.ndim == 4 and low_logits.shape[0] == 1:
            low_logits = low_logits[0]
        if low_logits.ndim != 3 or low_logits.shape[0] != 1:
            raise ValueError(f"SAM2 expected 1xHxW low-res mask, got {low_logits.shape}")
        return sigmoid(high_logits), low_logits

    def sample_active_point(self, prob, scribble, existing, roi=None):
        """Sample *known* scribble contradiction first; uncertainty second.

        All arrays and points are in SAME SAM2 image input pixel frame.
        No GT or outside-annotation pseudo point is introduced.
        """
        if scribble is None:
            return None, None, "no_scribble"
        h, w = self.last_input_shape
        if scribble.shape[:2] != (h, w):
            raise ValueError(f"Scribble/model frame mismatch {scribble.shape} vs {(h, w)}")
        p = resize_prob(prob, w, h)
        fg, bg = DataLoader.parse_scribble(scribble)
        known = fg | bg
        if roi is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in roi]
            x1, x2 = max(0, x1), min(w-1, x2)
            y1, y2 = max(0, y1), min(h-1, y2)
            allowed = np.zeros((h,w), bool)
            if x2 >= x1 and y2 >= y1:
                allowed[y1:y2+1, x1:x2+1] = True
            known &= allowed
        if not known.any():
            return None, None, "no_known_points"

        if existing is not None and len(existing):
            # Exclude an actual disk, not only the center pixel.
            available = (known.astype(np.uint8) * 255)
            for x, y in np.asarray(existing, np.float32):
                cv2.circle(available, (int(round(x)), int(round(y))),
                           int(self.cfg.active_min_dist), 0, -1)
            known = available > 0
        if not known.any():
            return None, None, "no_new_position"

        predicted_fg = p >= .5
        contradiction = ((fg & ~predicted_fg) | (bg & predicted_fg)) & known
        uncertainty = 1.0 - np.abs(2*p-1)
        if self.cfg.active_conflict_first and contradiction.any():
            # Select most confidently WRONG known point, instead of ambiguous only.
            disagreement_strength = np.where(fg, 1-p, p)
            score = np.where(contradiction, disagreement_strength, -1)
            kind = "scribble_conflict"
        else:
            score = np.where(known, uncertainty, -1)
            kind = "uncertainty"
        y, x = np.unravel_index(int(score.argmax()), score.shape)
        return np.array([[float(x), float(y)]], np.float32), \
               np.asarray([1 if fg[y, x] else 0], np.int32), kind

    def refine_predict(self, points, labels, box, scribble=None, iters_override=None):
        h, w = self.last_input_shape
        if scribble is not None and scribble.shape[:2] != (h, w):
            raise ValueError("Scribble must be resized/cropped to SAM2 input shape")
        n = self.cfg.refine_iters if iters_override is None else int(iters_override)
        current = points.copy() if points is not None else None
        labs = labels.copy() if labels is not None else None
        memory = None
        point_records = []
        prob = None
        # If there is no newly available point, further mask-memory-only iterations
        # can still be useful, so keep running the requested n inference calls.
        for step in range(n+1):
            prob, memory = self._predict_once(current, labs, box, memory)
            if step == n or not self.cfg.active_points or scribble is None:
                continue
            extra, extra_label, kind = self.sample_active_point(prob, scribble, current, roi=box)
            if extra is None:
                continue
            if current is None:
                current, labs = extra, extra_label
            else:
                current = np.concatenate((current, extra), 0)
                labs = np.concatenate((labs, extra_label), 0)
            point_records.append({"step": step + 1, "x": int(extra[0, 0]),
                                  "y": int(extra[0, 1]), "label": int(extra_label[0]),
                                  "reason": kind})
        return prob, point_records


class Pipeline:
    def __init__(self, cfg, predictor=None):
        self.cfg = cfg
        self.engine = SAM2Engine(cfg, predictor=predictor)
        self._cached_rgb = None
        self._cached_region = None
        for d in (cfg.out_dir, cfg.base_dir, cfg.refine_dir, cfg.profile_dir, cfg.vis_dir):
            os.makedirs(d, exist_ok=True)
        if cfg.save_probs:
            for d in ("base_prob", "refine_prob", "flipgate_prob"):
                os.makedirs(os.path.join(cfg.out_dir, d), exist_ok=True)

    def _target_for_box(self, rgb, scribble, box, do_crop):
        """Returns model_rgb, model_scribble, model_box, crop_rect, model_initial_points."""
        h, w = rgb.shape[:2]
        ratio = max(0, float((box[2]-box[0])*(box[3]-box[1]) / (w*h)))
        crop = self.cfg.use_crop and do_crop and ratio <= self.cfg.small_box_area_thr
        region = (centered_square_crop(box, w, h, self.cfg.crop_context,
                                        self.cfg.crop_min_context, self.cfg.crop_square)
                  if crop else (0, 0, w, h))
        if region == (0, 0, w, h):
            crop = False  # no actual zoom; classify this as NORMAL, not SO_CROP
        x0, y0, x1, y1 = region
        img_crop = rgb[y0:y1, x0:x1]
        sc_crop = scribble[y0:y1, x0:x1] if scribble is not None else None
        size = self.cfg.size
        model_img = cv2.resize(img_crop, (size,size), interpolation=cv2.INTER_LINEAR)
        model_sc = (cv2.resize(sc_crop, (size,size), interpolation=cv2.INTER_NEAREST)
                    if sc_crop is not None else None)
        model_box = transform_box_to_model(box, region, size, self.cfg.box_prompt_margin)
        if model_sc is not None:
            # In multi-object images, use only nearby scribbles; do not allow the
            # other object to pollute this box's SAM2 positive points.
            fg, bg = DataLoader.parse_scribble(model_sc)
            roi = np.zeros((size,size), np.uint8)
            x1m,y1m,x2m,y2m = np.round(model_box).astype(int)
            roi[y1m:y2m+1,x1m:x2m+1] = 1
            model_sc = model_sc.copy()
            # Existing background scribbles outside box can still be useful,
            # but remove positive ones outside the local object box.
            outside_fg = fg & (roi == 0)
            model_sc[outside_fg] = 0
            points, labs = DataLoader.points_from_scribble(model_sc, self.cfg.max_initial_per_class)
        else:
            points, labs = None, None
        return model_img, model_sc, model_box, region, points, labs, crop

    def _infer_single_box(self, rgb, scribble, box, refine, do_crop=True):
        """Run SAM2 on whole image or crop, restore probability to original image."""
        h, w = rgb.shape[:2]
        img, sc, model_box, region, pts, labs, cropped = self._target_for_box(
            rgb, scribble, box, do_crop)
        # Reuse image embedding when baseline and refinement refer to exactly
        # the same RGB object and crop; no SAM2 re-encoding is required.
        if self._cached_rgb is not rgb or self._cached_region != region:
            self.engine.set_image(img)
            self._cached_rgb = rgb
            self._cached_region = region
        iters = None if refine else 0
        pred_model, point_records = self.engine.refine_predict(
            pts, labs, model_box, scribble=sc if refine else None,
            iters_override=iters)
        x0, y0, x1, y1 = region
        prob_orig = np.zeros((h,w), np.float32)
        prob_orig[y0:y1,x0:x1] = resize_prob(pred_model, x1-x0, y1-y0)
        return prob_orig, point_records, cropped

    def _infer_multi_box(self, rgb, scribble, boxes):
        """One SAM2 call stack per box; output maxima merge with gated supports."""
        h, w = rgb.shape[:2]
        merged = np.zeros((h,w), np.float32)
        point_total, crop_count = 0, 0
        for box in boxes:
            prob, points, cropped = self._infer_single_box(rgb, scribble, box, refine=True)
            crop_count += int(cropped)
            point_total += len(points)
            # Restrict *each* object's contribution around its own box.
            # Convert model-frame margin to the scale of original image.
            margin = max(2, round(self.cfg.multi_box_merge_margin * max(h,w)/self.cfg.size))
            gated = expand_box(box, margin, w, h)
            x1,y1,x2,y2 = map(int, np.round(gated))
            gate = np.zeros((h,w), np.float32)
            gate[y1:y2+1,x1:x2+1] = 1.0
            merged = np.maximum(merged, prob*gate)
        return merged, point_total, crop_count

    def infer_refined(self, rgb, scribble, boxes):
        if len(boxes) > 1:
            prob, count, crop_count = self._infer_multi_box(rgb, scribble, boxes)
            route = "MO_PER_BOX"
        else:
            box = boxes[0]
            prob, points, cropped = self._infer_single_box(rgb, scribble, box, refine=True)
            count, crop_count = len(points), int(cropped)
            route = "SO_CROP" if cropped else "NORMAL"
        return prob, route, count, crop_count

    def _infer_baseline(self, rgb, scribble, boxes):
        # Faithful initial-prompt baseline: box + initial scribble, 1 call,
        # union box for multiple objects; no crop/active/memory/flip.
        prob, _, _ = self._infer_single_box(rgb, scribble, union_box(boxes),
                                          refine=False, do_crop=False)
        return prob

    def process_image(self, image_path):
        cfg = self.cfg
        basename = os.path.splitext(os.path.basename(image_path))[0]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        json_path = DataLoader.find_file(cfg.json_dir, basename, (".json",))
        scribble_path = DataLoader.find_file(cfg.scribble_dir, basename, (".png", ".jpg"))
        gt_path = DataLoader.find_file(cfg.gt_dir, basename, (".png", ".jpg"))
        if not scribble_path:
            # This script intentionally requires the same scribble supervision as code #1.
            return None
        sc = cv2.imread(scribble_path, cv2.IMREAD_GRAYSCALE)
        if sc is None:
            return None
        if sc.shape != (h,w):
            sc = cv2.resize(sc, (w,h), interpolation=cv2.INTER_NEAREST)
        try:
            boxes = DataLoader.load_boxes(json_path, w, h)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] invalid box JSON for {basename}: {exc}")
            return None
        if not boxes:
            return None

        base = self._infer_baseline(rgb, sc, boxes)
        refined, route, num_points, num_crop = self.infer_refined(rgb, sc, boxes)
        final = refined
        flip_info = {"flip_accepted": 0, "flip_reason": "disabled", "flip_iou": float("nan"),
                     "flip_area_ratio": float("nan"), "flip_scribble_err_original": float("nan"),
                     "flip_scribble_err_mirror": float("nan"), "flip_modified_ratio": 0.0}

        if cfg.flip_gate:
            rgb_flip = np.ascontiguousarray(rgb[:, ::-1, :])
            sc_flip = np.ascontiguousarray(sc[:, ::-1])
            boxes_flip = [mirror_box(b, w) for b in boxes]
            flip_pred, flip_route, _, _ = self.infer_refined(rgb_flip, sc_flip, boxes_flip)
            if flip_route != route:
                print(f"[WARN] crop route changed under flip for {basename}: {route} -> {flip_route}")
            flip_aligned = np.ascontiguousarray(flip_pred[:, ::-1])
            final, flip_info = flip_gate_fusion(refined, flip_aligned, sc, cfg)

        def save_mask(directory, prob):
            cv2.imwrite(os.path.join(directory, basename + ".png"),
                        (prob >= .5).astype(np.uint8) * 255)
        save_mask(cfg.base_dir, base)
        save_mask(cfg.refine_dir, refined)
        save_mask(cfg.profile_dir, final)
        if cfg.save_probs:
            for name, array in (("base_prob", base), ("refine_prob", refined),
                                ("flipgate_prob", final)):
                np.save(os.path.join(cfg.out_dir, name, basename + ".npy"),
                        array.astype(np.float16))

        record = {
            "image_name": os.path.basename(image_path), "base_name": basename,
            "route_type": route, "num_boxes": len(boxes),
            "box_area_ratio": float((union_box(boxes)[2]-union_box(boxes)[0]) *
                                    (union_box(boxes)[3]-union_box(boxes)[1])/(w*h)),
            "has_box": 1, "crop_count": num_crop, "num_active_points": num_points,
            **flip_info,
        }
        if gt_path:
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                results = {}
                for name, prob in (("base",base), ("refine",refined), ("profile",final)):
                    results[name] = MetricsEvaluator.evaluate((prob >= .5).astype(np.uint8)*255, gt)
                    for k,v in results[name].items():
                        record[f"{name}_{k}"] = v
                record["delta_S_new_minus_base"] = results["profile"]["Smeasure"] - results["base"]["Smeasure"]
                record["delta_E_new_minus_base"] = results["profile"]["maxE"] - results["base"]["maxE"]
                record["delta_F_new_minus_base"] = results["profile"]["maxF"] - results["base"]["maxF"]
                record["delta_MAE_base_minus_new"] = results["base"]["MAE"] - results["profile"]["MAE"]
                record["delta_MAE_refine_minus_flipgate"] = results["refine"]["MAE"] - results["profile"]["MAE"]
                if route != "NORMAL" or results["base"]["MAE"] > cfg.vis_mae_thr or \
                        abs(record["delta_MAE_base_minus_new"]) > cfg.vis_gain_thr:
                    self._save_vis(img, gt, base, refined, final, basename, route, results)
        return record

    def _save_vis(self, image_bgr, gt, base, refined, final, name, route, results):
        size = 256
        photo = cv2.resize(image_bgr, (size,size))
        cv2.putText(photo, route, (5,22), cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
        panels = [photo]
        entries = [("GT", gt), ("Base", base), ("Refine", refined), ("FlipGate", final)]
        for label, arr in entries:
            if label != "GT":
                arr = (arr >= .5).astype(np.uint8)*255
            panel = cv2.cvtColor(cv2.resize(arr, (size,size), interpolation=cv2.INTER_NEAREST),
                                 cv2.COLOR_GRAY2BGR)
            key = {"Base": "base", "Refine": "refine", "FlipGate": "profile"}.get(label)
            suffix = (f" {results[key]['MAE']:.4f}" if key is not None else "")
            cv2.putText(panel, label+suffix, (5,22), cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
            panels.append(panel)
        cv2.imwrite(os.path.join(self.cfg.vis_dir, name+"_abl.png"), np.concatenate(panels, 1))


def summarize(records, prefix):
    out = {}
    for k in ("Smeasure", "maxE", "maxF", "MAE"):
        values = np.asarray([r.get(f"{prefix}_{k}", np.nan) for r in records], np.float64)
        valid = values[np.isfinite(values)]
        out[k] = float(valid.mean()) if len(valid) else None
    return out


def _self_test():
    """Pure NumPy/OpenCV tests, no SAM2 installation or weights needed."""
    cfg = Config()
    cfg.flip_gate = True
    # Pixel-center mirroring must be involutive.
    b = np.array([12, 7, 22, 17], dtype=np.float32)
    assert np.allclose(mirror_box(mirror_box(b, 80), 80), b)
    # Crop real input coordinate placement and full-size reconstruction.
    roi = centered_square_crop(b, 160, 100, 0.30, 8)
    assert roi[2]-roi[0] < 160 and roi[3]-roi[1] < 100
    bm = transform_box_to_model(b, roi, 1024, 5)
    assert bm[2] > bm[0] and bm[3] > bm[1]
    # A flip with large disagreement must be rejected.
    p = np.full((64,64), .03, np.float32)
    q = p.copy()
    p[10:35, 10:35] = .92
    q[35:60, 35:60] = .92
    q[10:35, 10:35] = .03
    out, info = flip_gate_fusion(p, q, np.zeros((64,64),np.uint8), cfg)
    assert info['flip_accepted'] == 0 and np.array_equal(out, p)
    # Identical flip is accepted with exactly same output.
    out, info = flip_gate_fusion(p, p.copy(), None, cfg)
    assert info['flip_accepted'] == 1 and np.allclose(out, p)
    # The active point must stay in known scribble and away from existing points.
    class StubPredictor:
        def set_image(self, img):
            pass
    engine = SAM2Engine(cfg, predictor=StubPredictor())
    engine.last_input_shape = (64, 64)
    scribble = np.zeros((64,64), np.uint8)
    scribble[20:40, 20:40] = 1
    pr = np.full((64,64), .05, np.float32)
    newpt, lab, kind = engine.sample_active_point(pr, scribble, np.array([[20,20]], np.float32))
    assert newpt is not None and lab[0] == 1
    assert np.linalg.norm(newpt[0] - np.array([20,20])) > cfg.active_min_dist
    assert kind == 'scribble_conflict'
    # Frame check should reject any mismatch before invoking SAM2.
    try:
        engine.sample_active_point(pr, np.zeros((32,32),np.uint8), None)
        raise AssertionError("expected frame mismatch")
    except ValueError:
        pass
    print("PASS: mirror, crop, flip rejection/acceptance, active sampling, frame consistency")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sam2-checkpoint", default=Config.sam2_checkpoint)
    parser.add_argument("--sam2-config", default=Config.sam2_config)
    parser.add_argument("--img-dir", default=Config.img_dir)
    parser.add_argument("--scribble-dir", default=Config.scribble_dir)
    parser.add_argument("--gt-dir", default=Config.gt_dir)
    parser.add_argument("--json-dir", default=Config.json_dir)
    parser.add_argument("--output-dir", default=Config.out_dir)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--refine-iters", type=int, default=2)
    parser.add_argument("--no-active", action="store_true")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--no-flip-gate", action="store_true")
    parser.add_argument("--flip-mode", choices=["gate","mean"], default="gate")
    parser.add_argument("--flip-min-iou", type=float, default=.75)
    parser.add_argument("--flip-max-area-ratio", type=float, default=1.5)
    parser.add_argument("--small-box-area-thr", type=float, default=.10)
    parser.add_argument("--crop-context", type=float, default=.35)
    parser.add_argument("--save-probs", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    cfg = Config(sam2_checkpoint=args.sam2_checkpoint, sam2_config=args.sam2_config,
                 img_dir=args.img_dir, scribble_dir=args.scribble_dir, gt_dir=args.gt_dir,
                 json_dir=args.json_dir, out_dir=args.output_dir, device=args.device,
                 refine_iters=args.refine_iters, active_points=not args.no_active,
                 use_crop=not args.no_crop, flip_gate=not args.no_flip_gate,
                 flip_mode=args.flip_mode, flip_min_iou=args.flip_min_iou,
                 flip_max_area_ratio=args.flip_max_area_ratio,
                 small_box_area_thr=args.small_box_area_thr,
                 crop_context=args.crop_context, save_probs=args.save_probs)
    if not (0 <= cfg.small_box_area_thr <= 1):
        parser.error("--small-box-area-thr must be in [0,1]")
    if not (0 <= cfg.flip_min_iou <= 1):
        parser.error("--flip-min-iou must be in [0,1]")
    if cfg.refine_iters < 0:
        parser.error("--refine-iters must be >= 0")
    if cfg.flip_max_area_ratio < 1:
        parser.error("--flip-max-area-ratio must be >= 1")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda selected but CUDA unavailable; use --device cpu")

    pipeline = Pipeline(cfg)
    images = DataLoader.list_images(cfg.img_dir)
    if args.limit is not None:
        images = images[:args.limit]
    records = []
    failures = []
    for path in tqdm(images, desc="SAM2 refining"):
        try:
            record = pipeline.process_image(path)
        except Exception as exc:
            failures.append({"image":path,"error":repr(exc)})
            print(f"[ERROR] {os.path.basename(path)}: {exc}")
            continue
        if record is not None:
            records.append(record)
    csv_path = os.path.join(cfg.out_dir,"soft_so_mo_metrics.csv")
    summary_path = os.path.join(cfg.out_dir,"summary_metrics.json")
    if records:
        # A few fields only appear when GT is available; do not discard them.
        fieldnames = list(dict.fromkeys(k for row in records for k in row))
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "Base": summarize(records, "base"),
        "Refine_NoFlip": summarize(records, "refine"),
        "Refine_FlipGate": summarize(records, "profile"),
        "RouteBreakdown": {
            route: {"num_images": len(sub),
                    "profile": summarize(sub, "profile")}
            for route in ("NORMAL", "SO_CROP", "MO_PER_BOX")
            for sub in [[r for r in records if r['route_type']==route]]
        },
        "Diagnostics": {
            "num_images": len(records),
            "num_failures": len(failures),
            "flip_accepted": sum(r['flip_accepted'] for r in records),
            "flip_rejected": sum(cfg.flip_gate and not r['flip_accepted'] for r in records),
            "num_cropped_boxes": sum(r['crop_count'] for r in records),
            "num_added_points": sum(r['num_active_points'] for r in records),
        },
        "MetricNote": "Legacy columns maxE/maxF mean binary E/F at threshold 0.5; not sweep max or adaptive E/F.",
        "Config": vars(cfg),
        "Failures": failures,
    }
    with open(summary_path,"w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("="*72)
    for k in ("Base", "Refine_NoFlip", "Refine_FlipGate"):
        metrics = summary[k]
        print(f"{k:>18}: " + " ".join(f"{m}={metrics[m]:.4f}" for m in ("Smeasure","maxE","maxF","MAE") if metrics[m] is not None))
    print("Diagnostics:", summary['Diagnostics'])
    print("CSV:", csv_path if records else "no valid records")
    print("Summary:", summary_path)
    print("="*72)


if __name__ == '__main__':
    main()
