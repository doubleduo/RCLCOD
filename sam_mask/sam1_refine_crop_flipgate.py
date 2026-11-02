#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM1 (ViT-H/ViT-L/ViT-B) scribble-assisted active refinement.

Faithful SAM1 port of the supplied SAM2 script:
  * Box + human scribble initialization (NOT strict Box-only)
  * 0, 1, or 2+ extra decoder iterations using the previous low-res mask logits
  * Scribble-contradiction/uncertainty guided additional points
  * Small-object crop, multi-object per-box merging and horizontal Flip Gate
  * Same output layout, metrics and visualization format as the SAM2 source
  * Final non-flipped refinement: SAME prompts/memory, extra three-mask diagnostic decode
  * Pairwise candidate IoU, binary-mask SSIM, SIM, scores and stability in CSV/JSON
  * Quality diagnostics DO NOT replace the original single-mask refined prediction

SAM1 changes: SamPredictor, box shape (4,), no SAM2 config, SAM1 checkpoint.
Preprocessing deliberately remains 1024x1024 as in the source for controlled
SAM1-vs-SAM2 architectural comparisons. SamPredictor's internal preprocessing
then sees a 1024x1024 square and prompt coordinates remain in that frame.

Important: because foreground/background scribbles are used as ground-truth
prompts, this is NOT a box-only weak supervision experiment.

Examples (run from RCLCOD root):
    python sam_mask/sam1_refine_crop_flipgate.py --self-test
    python sam_mask/sam1_refine_crop_flipgate.py --limit 100 --refine-iters 1 --no-flip-gate
    python sam_mask/sam1_refine_crop_flipgate.py --limit 100 --refine-iters 2
    python sam_mask/sam1_refine_crop_flipgate.py --no-memory --refine-iters 2 --no-flip-gate
    python sam_mask/sam1_refine_crop_flipgate.py --no-active --refine-iters 2 --no-flip-gate
"""
import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class Config:
    sam_checkpoint: str = "./sam_mask/weights/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir: str = "./data/Train/Imgs"
    scribble_dir: str = "./data/Train/Scribble"
    gt_dir: str = "./data/Train/GT"
    json_dir: str = "./data/Train/box_label"
    out_dir: str = "./softresult/SAM1_Optimized3"

    size: int = 1024
    seed: int = 0
    refine_iters: int = 2
    active_points: bool = True
    use_mask_memory: bool = True
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
    candidate_quality: bool = True     # extra SAM1 3-mask call for post-refinement scoring
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
        """Take at most K spatially distributed known points of EACH label.

        Unlike a raw sqrt(K) grid, K=2 returns at most 2, not 4. This is
        important when reserving known scribble locations for later active
        correction steps. For K=9 it matches the original 3x3 grid budget.
        """
        if max_per_class <= 0:
            return None, None
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
            candidates = []
            for i in range(rows):
                for j in range(cols):
                    selected = (ys >= y_edges[i]) & (ys < y_edges[i + 1]) & \
                               (xs >= x_edges[j]) & (xs < x_edges[j + 1])
                    if not np.any(selected):
                        continue
                    cx, cy = xs[selected], ys[selected]
                    k = int(np.argmax(dist[cy, cx]))
                    candidates.append((float(cx[k]), float(cy[k]), float(dist[cy[k], cx[k]])))
            if not candidates:
                continue
            # Greedy spatial coverage over cell-center candidates. Select the
            # most deeply internal point first; then the most distant point.
            chosen = []
            unused = list(range(len(candidates)))
            while unused and len(chosen) < max_per_class:
                if not chosen:
                    best = max(unused, key=lambda k: (candidates[k][2], -k))
                else:
                    best = max(
                        unused,
                        key=lambda k: (
                            min((candidates[k][0] - candidates[j][0])**2 +
                                (candidates[k][1] - candidates[j][1])**2 for j in chosen),
                            candidates[k][2], -k,
                        ),
                    )
                chosen.append(best)
                unused.remove(best)
            for j in chosen:
                x, y, _ = candidates[j]
                points.append([x, y])
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



def candidate_binary_ssim(a, b):
    """Gaussian-window SSIM for binary masks, matching prior three-candidate SIM.

    The score is computed on the SAM1 model's final image frame (1024x1024).
    """
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('Candidate SSIM requires two HxW arrays of identical shape')
    if np.array_equal(a, b):
        return 1.0
    c1, c2 = 0.01**2, 0.03**2
    side = min(a.shape)
    if side < 3:
        ma, mb = float(a.mean()), float(b.mean())
        va, vb = float(a.var()), float(b.var())
        cov = float(((a-ma) * (b-mb)).mean())
        return float(np.clip(((2*ma*mb+c1) * (2*cov+c2)) /
                             max((ma*ma+mb*mb+c1) * (va+vb+c2), 1e-12), 0., 1.))
    win = min(11, side)
    if win % 2 == 0:
        win -= 1
    mu_a = cv2.GaussianBlur(a, (win, win), 1.5)
    mu_b = cv2.GaussianBlur(b, (win, win), 1.5)
    va = cv2.GaussianBlur(a*a, (win, win), 1.5) - mu_a*mu_a
    vb = cv2.GaussianBlur(b*b, (win, win), 1.5) - mu_b*mu_b
    vab = cv2.GaussianBlur(a*b, (win, win), 1.5) - mu_a*mu_b
    num = (2*mu_a*mu_b+c1) * (2*vab+c2)
    den = (mu_a*mu_a+mu_b*mu_b+c1) * (va+vb+c2)
    return float(np.clip(np.mean(num / np.maximum(den, 1e-12)), 0., 1.))


def compute_candidate_quality(masks, scores, chosen_single_mask=None):
    """Three-candidate pairwise IoU, SSIM and SIM; higher = stronger agreement."""
    masks = np.asarray(masks, bool)
    scores = np.asarray(scores, np.float32).reshape(-1)
    if masks.ndim != 3 or masks.shape[0] != 3 or scores.size != 3:
        raise ValueError(f'Expected 3 masks/scores, got {masks.shape} and {scores.shape}')
    stats = {}
    pair_iou, pair_ssim, pair_sim = [], [], []
    for i, j, label in ((0, 1, '12'), (0, 2, '13'), (1, 2, '23')):
        ci = binary_iou(masks[i], masks[j])
        cs = candidate_binary_ssim(masks[i], masks[j])
        sim = 0.5 * (ci + cs)
        stats[f'candidate_iou_{label}'] = float(ci)
        stats[f'candidate_ssim_{label}'] = float(cs)
        stats[f'candidate_sim_{label}'] = float(sim)
        pair_iou.append(ci)
        pair_ssim.append(cs)
        pair_sim.append(sim)
    stats.update({
        'candidate_iou_mean': float(np.mean(pair_iou)),
        'candidate_ssim_mean': float(np.mean(pair_ssim)),
        'candidate_sim_mean': float(np.mean(pair_sim)),
        'candidate_consensus_s': float(np.mean(pair_sim)),
        'candidate_sim_min': float(np.min(pair_sim)),
        'candidate_sim_max': float(np.max(pair_sim)),
        'candidate_sim_range': float(np.max(pair_sim) - np.min(pair_sim)),
        'candidate_score_1': float(scores[0]),
        'candidate_score_2': float(scores[1]),
        'candidate_score_3': float(scores[2]),
        'selected_candidate_index': int(np.argmax(scores)),
        'sam_score_selected': float(np.max(scores)),
    })
    if chosen_single_mask is not None:
        main = np.asarray(chosen_single_mask, bool)
        if main.shape != masks.shape[1:]:
            raise ValueError('Saved single-mask frame disagrees with candidate masks')
        overlaps = [binary_iou(main, mk) for mk in masks]
        stats['single_vs_candidate_iou_max'] = float(max(overlaps))
        stats['single_vs_candidate_iou_selected'] = float(overlaps[stats['selected_candidate_index']])
    return stats


class SAM1Engine:
    """SAM1 SamPredictor wrapper with scribble points and memory logits.

    IMPORTANT: For the SAM1 API, `box` must be shape (4,), not (1, 4).
    `mask_input` must be shape (1, 256, 256). We preserve the selected
    decoder low-res logits without converting them to probabilities.
    """

    def __init__(self, cfg, predictor=None):
        self.cfg = cfg
        self.device = cfg.device
        if predictor is None:
            if not os.path.isfile(cfg.sam_checkpoint):
                raise FileNotFoundError('SAM1 checkpoint not found: ' + cfg.sam_checkpoint)
            try:
                from segment_anything import sam_model_registry, SamPredictor
            except ImportError as e:
                raise ImportError(
                    'SAM1 not found. From project root run: '
                    'pip install git+https://github.com/facebookresearch/segment-anything.git'
                ) from e
            if cfg.sam_model_type not in sam_model_registry:
                raise ValueError('Unknown SAM1 model type: ' + cfg.sam_model_type)
            model = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
            model.to(device=cfg.device)
            model.eval()
            predictor = SamPredictor(model)
        self.predictor = predictor
        self.last_input_shape = None
        self.last_candidate_quality = None

    def set_image(self, model_rgb):
        self.last_input_shape = model_rgb.shape[:2]
        with torch.inference_mode():
            self.predictor.set_image(np.ascontiguousarray(model_rgb))

    def _predict_once(self, points, labels, box, mask_input):
        pts = None if points is None else np.asarray(points, dtype=np.float32).reshape(-1, 2)
        labs = None if labels is None else np.asarray(labels, dtype=np.int32).reshape(-1)
        if (pts is None) != (labs is None):
            raise ValueError('SAM1 point coordinates and labels must both exist or both be None')
        if pts is not None and len(pts) != len(labs):
            raise ValueError('SAM1 point coordinates and labels count mismatch')
        bbox = None if box is None else np.asarray(box, dtype=np.float32).reshape(4)
        mem = None if mask_input is None else np.asarray(mask_input, dtype=np.float32)
        if mem is not None and (mem.ndim != 3 or mem.shape[0] != 1):
            raise ValueError(f'SAM1 requires mask_input shape (1,H,W), got {mem.shape}')
        with torch.inference_mode():
            high_logits, scores, low_logits = self.predictor.predict(
                point_coords=pts,
                point_labels=labs,
                box=bbox,
                mask_input=mem,
                multimask_output=False,
                return_logits=True,
            )
        high_logits = np.asarray(high_logits, dtype=np.float32)
        low_logits = np.asarray(low_logits, dtype=np.float32)
        if high_logits.ndim != 3 or high_logits.shape[0] != 1:
            raise ValueError(f'SAM1 expected a single high-res mask, got {high_logits.shape}')
        if low_logits.ndim == 2:
            low_logits = low_logits[None, ...]
        if low_logits.ndim != 3 or low_logits.shape[0] != 1:
            raise ValueError(f'SAM1 expected 1xHxW low-res logits, got {low_logits.shape}')
        return sigmoid(high_logits[0]), low_logits


    def _score_final_three(self, points, labels, box, mask_input, single_mask):
        """Extra decoder call with EXACT prompts and mask_input of last single call.

        SAM1 multimask_output=True uses different output mask tokens from False.
        Thus candidate quality is a diagnostic of the final prompt state, not
        the consensus of three masks that were fused into the saved PNG.
        """
        pts = None if points is None else np.asarray(points, np.float32).reshape(-1, 2)
        labs = None if labels is None else np.asarray(labels, np.int32).reshape(-1)
        bbox = None if box is None else np.asarray(box, np.float32).reshape(4)
        mem = None if mask_input is None else np.asarray(mask_input, np.float32)
        with torch.inference_mode():
            logits, scores, low_logits = self.predictor.predict(
                point_coords=pts, point_labels=labs,
                box=bbox, mask_input=mem,
                multimask_output=True, return_logits=True)
        logits = np.asarray(logits, np.float32)
        scores = np.asarray(scores, np.float32).reshape(-1)
        low_logits = np.asarray(low_logits, np.float32)
        if logits.ndim != 3 or logits.shape[0] != 3 or scores.size != 3:
            raise RuntimeError(f'SAM1 final candidate mode needs 3 masks: {logits.shape} {scores.shape}')
        if low_logits.ndim != 3 or low_logits.shape[0] != 3:
            raise RuntimeError(f'Expected 3 low-res logit masks, got {low_logits.shape}')
        if not np.isfinite(logits).all() or not np.isfinite(scores).all():
            raise RuntimeError('Nonfinite values in final SAM1 candidate predictions')
        thr = float(getattr(getattr(self.predictor, 'model', None), 'mask_threshold', 0.0))
        stats = compute_candidate_quality(logits > thr, scores, chosen_single_mask=single_mask)
        chosen = low_logits[stats['selected_candidate_index']]
        lo = int((chosen > thr - 1.0).sum())
        hi = int((chosen > thr + 1.0).sum())
        stats.update({
            'sam_stability_selected': float(hi / lo) if lo else 0.0,
            'quality_source': 'last_refinement_prompts_and_input_memory',
            'candidate_multimask': 1,
            'quality_same_as_saved_mask': 0,
        })
        return stats

    def sample_active_point(self, prob, scribble, existing, roi=None):
        """Sample a newly available known scribble position, not GT."""
        if scribble is None:
            return None, None, 'no_scribble'
        h, w = self.last_input_shape
        if scribble.shape[:2] != (h, w):
            raise ValueError(f'Scribble/model frame mismatch {scribble.shape} vs {(h, w)}')
        p = resize_prob(prob, w, h)
        fg, bg = DataLoader.parse_scribble(scribble)
        known = fg | bg
        if roi is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in roi]
            x1, x2 = max(0, x1), min(w - 1, x2)
            y1, y2 = max(0, y1), min(h - 1, y2)
            allowed = np.zeros((h, w), bool)
            if x2 >= x1 and y2 >= y1:
                allowed[y1:y2 + 1, x1:x2 + 1] = True
            known &= allowed
        if not known.any():
            return None, None, 'no_known_points'
        if existing is not None and len(existing):
            available = known.astype(np.uint8) * 255
            for x, y in np.asarray(existing, np.float32):
                cv2.circle(available, (int(round(x)), int(round(y))),
                           int(self.cfg.active_min_dist), 0, -1)
            known = available > 0
        if not known.any():
            return None, None, 'no_new_position'
        predicted_fg = p >= .5
        contradiction = ((fg & ~predicted_fg) | (bg & predicted_fg)) & known
        uncertainty = 1.0 - np.abs(2 * p - 1)
        if self.cfg.active_conflict_first and contradiction.any():
            disagreement_strength = np.where(fg, 1 - p, p)
            score = np.where(contradiction, disagreement_strength, -1)
            kind = 'scribble_conflict'
        else:
            score = np.where(known, uncertainty, -1)
            kind = 'uncertainty'
        y, x = np.unravel_index(int(score.argmax()), score.shape)
        return (np.asarray([[float(x), float(y)]], np.float32),
                np.asarray([1 if fg[y, x] else 0], np.int32), kind)

    def refine_predict(self, points, labels, box, scribble=None, iters_override=None, collect_quality=False):
        h, w = self.last_input_shape
        if scribble is not None and scribble.shape[:2] != (h, w):
            raise ValueError('Scribble must be resized/cropped to SAM1 input shape')
        n = self.cfg.refine_iters if iters_override is None else int(iters_override)
        current = points.copy() if points is not None else None
        labs = labels.copy() if labels is not None else None
        memory = None
        point_records = []
        self.last_candidate_quality = None
        prob = None
        # n additional SAM1 calls after the first box+scribble call.
        for step in range(n + 1):
            # Record EXACT mask_input fed to the last single-mask decoder call.
            mem_in = memory if self.cfg.use_mask_memory else None
            prob, new_logits = self._predict_once(current, labs, box, mem_in)
            if step == n and collect_quality and self.cfg.candidate_quality:
                self.last_candidate_quality = self._score_final_three(
                    current, labs, box, mem_in, prob >= .5)
            memory = new_logits if self.cfg.use_mask_memory else None
            if step == n or not self.cfg.active_points or scribble is None:
                continue
            extra, extra_label, kind = self.sample_active_point(prob, scribble, current, roi=box)
            if extra is None:
                continue
            if current is None:
                current, labs = extra, extra_label
            else:
                current = np.concatenate((current, extra), axis=0)
                labs = np.concatenate((labs, extra_label), axis=0)
            point_records.append({
                'step': step + 1,
                'x': int(extra[0, 0]), 'y': int(extra[0, 1]),
                'label': int(extra_label[0]), 'reason': kind,
            })
        return prob, point_records


class Pipeline:
    def __init__(self, cfg, predictor=None):
        self.cfg = cfg
        self.engine = SAM1Engine(cfg, predictor=predictor)
        self._cached_rgb = None
        self._cached_region = None
        self.object_quality = []
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
            # other object to pollute this box's SAM1 positive points.
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

    def _infer_single_box(self, rgb, scribble, box, refine, do_crop=True, collect_quality=False):
        """Run SAM1 on whole image or crop, restore probability to original image."""
        h, w = rgb.shape[:2]
        img, sc, model_box, region, pts, labs, cropped = self._target_for_box(
            rgb, scribble, box, do_crop)
        # Reuse image embedding when baseline and refinement refer to exactly
        # the same RGB object and crop; no SAM1 re-encoding is required.
        if self._cached_rgb is not rgb or self._cached_region != region:
            self.engine.set_image(img)
            self._cached_rgb = rgb
            self._cached_region = region
        iters = None if refine else 0
        pred_model, point_records = self.engine.refine_predict(
            pts, labs, model_box, scribble=sc if refine else None,
            iters_override=iters, collect_quality=collect_quality)
        if collect_quality and self.engine.last_candidate_quality is not None:
            self.object_quality.append(dict(self.engine.last_candidate_quality))
        x0, y0, x1, y1 = region
        prob_orig = np.zeros((h,w), np.float32)
        prob_orig[y0:y1,x0:x1] = resize_prob(pred_model, x1-x0, y1-y0)
        return prob_orig, point_records, cropped

    def _infer_multi_box(self, rgb, scribble, boxes, collect_quality=False):
        """One SAM2 call stack per box; output maxima merge with gated supports."""
        h, w = rgb.shape[:2]
        merged = np.zeros((h,w), np.float32)
        point_total, crop_count = 0, 0
        for box in boxes:
            prob, points, cropped = self._infer_single_box(
                rgb, scribble, box, refine=True, collect_quality=collect_quality)
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

    def infer_refined(self, rgb, scribble, boxes, collect_quality=False):
        if collect_quality:
            self.object_quality = []
        if len(boxes) > 1:
            prob, count, crop_count = self._infer_multi_box(
                rgb, scribble, boxes, collect_quality=collect_quality)
            route = "MO_PER_BOX"
        else:
            box = boxes[0]
            prob, points, cropped = self._infer_single_box(
                rgb, scribble, box, refine=True, collect_quality=collect_quality)
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
        refined, route, num_points, num_crop = self.infer_refined(
            rgb, sc, boxes, collect_quality=cfg.candidate_quality)
        quality_per_box = [dict(q) for q in self.object_quality] if cfg.candidate_quality else []
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

        candidate_record = {}
        if cfg.candidate_quality:
            if len(quality_per_box) != len(boxes):
                raise RuntimeError(f'Candidate records {len(quality_per_box)} != boxes {len(boxes)}')
            # When multiple boxes are merged, there is NO single set of 3
            # candidate masks for the merged image. Report worst-object (min)
            # as the conservative image-level score, plus simple object mean.
            prefix = ('candidate_iou_', 'candidate_ssim_', 'candidate_sim_')
            fields = [k for k in quality_per_box[0] if k.startswith(prefix)]
            for k in fields:
                vals = [float(q[k]) for q in quality_per_box if k in q]
                if vals:
                    candidate_record[k] = float(min(vals))
            for k in ('candidate_iou_mean','candidate_ssim_mean','candidate_sim_mean',
                      'candidate_consensus_s','sam_score_selected','sam_stability_selected',
                      'single_vs_candidate_iou_selected','single_vs_candidate_iou_max'):
                candidate_record[k] = float(min(float(q[k]) for q in quality_per_box))
                candidate_record[k + '_object_mean'] = float(np.mean([q[k] for q in quality_per_box]))
            candidate_record['selected_candidate_index'] = (
                int(quality_per_box[0]['selected_candidate_index']) if len(boxes) == 1 else -1)
            for k in ('candidate_score_1','candidate_score_2','candidate_score_3'):
                candidate_record[k] = (float(quality_per_box[0][k]) if len(boxes) == 1 else float('nan'))
            candidate_record['candidate_quality_num_boxes'] = int(len(quality_per_box))
            candidate_record['candidate_quality_source'] = 'last_nonflip_refinement_prompts_memory'
            candidate_record['candidate_quality_vs_saved_refined'] = 'prompt_state_only_not_same_mask_token'
            candidate_record['candidate_quality_vs_saved_final'] = (
                'flip_fused_not_same_output' if cfg.flip_gate else 'prompt_state_only_not_same_mask_token')
            os.makedirs(os.path.join(cfg.out_dir, 'candidate_per_box'), exist_ok=True)
            with open(os.path.join(cfg.out_dir, 'candidate_per_box', basename + '.json'),
                      'w', encoding='utf-8') as f:
                json.dump(quality_per_box, f, ensure_ascii=False, indent=2, allow_nan=False)

        record = {
            "image_name": os.path.basename(image_path), "base_name": basename,
            "route_type": route, "num_boxes": len(boxes),
            "box_area_ratio": float((union_box(boxes)[2]-union_box(boxes)[0]) *
                                    (union_box(boxes)[3]-union_box(boxes)[1])/(w*h)),
            "has_box": 1, "crop_count": num_crop, "num_active_points": num_points,
            **candidate_record,
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
    """Pure NumPy/OpenCV tests, no SAM1 installation or weights needed."""
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
    engine = SAM1Engine(cfg, predictor=StubPredictor())
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
    # Test the SAM1-specific decoder signature and low-resolution-logits memory chain.
    class SAM1PredictorStub:
        def __init__(self):
            self.memory_present = []
        def set_image(self, image):
            self.size = image.shape[:2]
        def predict(self, **kwargs):
            assert kwargs['box'].shape == (4,), 'SAM1 requires a flat 4-vector box'
            assert kwargs['return_logits'] is True
            m = kwargs['mask_input']
            assert m is None or m.shape == (1, 256, 256)
            if kwargs['multimask_output']:
                logits = np.full((3, *self.size), -2., np.float32)
                logits[0, 10:30, 10:30] = 2.0
                logits[1, 10:31, 10:31] = 2.0
                logits[2, 11:30, 11:30] = 2.0
                return (logits, np.array([.9, .8, .7], np.float32),
                        np.full((3, 256, 256), 0.8, np.float32))
            self.memory_present.append(m is not None)
            return (np.full((1, *self.size), -2., np.float32),
                    np.asarray([0.9], np.float32),
                    np.full((1, 256, 256), 0.8, np.float32))
    stub = SAM1PredictorStub()
    engine = SAM1Engine(cfg, predictor=stub)
    engine.set_image(np.zeros((64, 64, 3), np.uint8))
    engine.refine_predict(None, None, np.asarray([4, 4, 60, 60], np.float32),
                          scribble=None, iters_override=2)
    assert stub.memory_present == [False, True, True], stub.memory_present
    stub.memory_present.clear()
    engine.refine_predict(None, None, np.asarray([4, 4, 60, 60], np.float32),
                          scribble=None, iters_override=2, collect_quality=True)
    assert stub.memory_present == [False, True, True], stub.memory_present
    assert abs(engine.last_candidate_quality['candidate_sim_mean'] -
               0.5*(engine.last_candidate_quality['candidate_iou_mean'] +
                    engine.last_candidate_quality['candidate_ssim_mean'])) < 1e-6
    assert engine.last_candidate_quality['quality_same_as_saved_mask'] == 0
    sc_test = np.zeros((64, 64), np.uint8)
    sc_test[10:50, 10:50] = 1
    sc_test[52:62, 10:50] = 2
    init_pts, init_labs = DataLoader.points_from_scribble(sc_test, max_per_class=2)
    assert sum(init_labs == 1) <= 2 and sum(init_labs == 0) <= 2
    print("PASS: SAM1 memory x2, final 3-candidate IoU/SSIM/SIM, point budget, crop, Flip Gate")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sam-checkpoint", default=Config.sam_checkpoint)
    parser.add_argument("--sam-model-type", default=Config.sam_model_type)
    parser.add_argument("--img-dir", default=Config.img_dir)
    parser.add_argument("--scribble-dir", default=Config.scribble_dir)
    parser.add_argument("--gt-dir", default=Config.gt_dir)
    parser.add_argument("--json-dir", default=Config.json_dir)
    parser.add_argument("--output-dir", default=Config.out_dir)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--refine-iters", type=int, default=2)
    parser.add_argument("--no-active", action="store_true")
    parser.add_argument("--max-initial-per-class", type=int, default=Config.max_initial_per_class,
                        help="Number of initial foreground/background scribble points per class (original=9)")
    parser.add_argument("--active-min-dist", type=int, default=Config.active_min_dist,
                        help="Minimum spacing between initial and newly added points in 1024 frame")
    parser.add_argument("--no-memory", action="store_true",
                        help="Ablation: no low-res mask logits passed to subsequent predictions")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--no-flip-gate", action="store_true")
    parser.add_argument("--flip-mode", choices=["gate","mean"], default="gate")
    parser.add_argument("--flip-min-iou", type=float, default=.75)
    parser.add_argument("--flip-max-area-ratio", type=float, default=1.5)
    parser.add_argument("--small-box-area-thr", type=float, default=.10)
    parser.add_argument("--crop-context", type=float, default=.35)
    parser.add_argument("--save-probs", action="store_true")
    parser.add_argument("--no-candidate-quality", action="store_true",
                        help="Skip final three-candidate scoring decoder call")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    cfg = Config(sam_checkpoint=args.sam_checkpoint, sam_model_type=args.sam_model_type,
                 img_dir=args.img_dir, scribble_dir=args.scribble_dir, gt_dir=args.gt_dir,
                 json_dir=args.json_dir, out_dir=args.output_dir, device=args.device,
                 refine_iters=args.refine_iters, active_points=not args.no_active,
                 use_mask_memory=not args.no_memory,
                 max_initial_per_class=args.max_initial_per_class,
                 active_min_dist=args.active_min_dist,
                 use_crop=not args.no_crop, flip_gate=not args.no_flip_gate,
                 flip_mode=args.flip_mode, flip_min_iou=args.flip_min_iou,
                 flip_max_area_ratio=args.flip_max_area_ratio,
                 small_box_area_thr=args.small_box_area_thr,
                 crop_context=args.crop_context, save_probs=args.save_probs,
                 candidate_quality=not args.no_candidate_quality)
    if not (0 <= cfg.small_box_area_thr <= 1):
        parser.error("--small-box-area-thr must be in [0,1]")
    if not (0 <= cfg.flip_min_iou <= 1):
        parser.error("--flip-min-iou must be in [0,1]")
    if cfg.refine_iters < 0:
        parser.error("--refine-iters must be >= 0")
    if cfg.max_initial_per_class < 0:
        parser.error("--max-initial-per-class must be >= 0")
    if cfg.active_min_dist < 0:
        parser.error("--active-min-dist must be >= 0")
    if cfg.sam_model_type not in ('vit_h', 'vit_l', 'vit_b'):
        parser.error('--sam-model-type must be vit_h/vit_l/vit_b')
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
    skipped = []
    for path in tqdm(images, desc="SAM1 refining"):
        try:
            record = pipeline.process_image(path)
        except Exception as exc:
            failures.append({"image":path,"error":repr(exc)})
            print(f"[ERROR] {os.path.basename(path)}: {exc}")
            continue
        if record is not None:
            records.append(record)
        else:
            skipped.append(os.path.basename(path))
    csv_path = os.path.join(cfg.out_dir,"sam1_soft_so_mo_metrics.csv")
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
            "num_attempted": len(images),
            "num_failures": len(failures),
            "num_skipped_missing_inputs": len(skipped),
            "flip_accepted": sum(r['flip_accepted'] for r in records),
            "flip_rejected": sum(cfg.flip_gate and not r['flip_accepted'] for r in records),
            "num_cropped_boxes": sum(r['crop_count'] for r in records),
            "num_added_points": sum(r['num_active_points'] for r in records),
        },
        "MetricNote": "Legacy columns maxE/maxF mean binary E/F at threshold 0.5; not sweep max or adaptive E/F.",
        "Config": vars(cfg),
        "SAMVersion": "SAM1",
        "PromptNote": "Box + human scribble; GT is evaluation only; NOT box-only supervision.",
        "IterationNote": "refine_iters=N gives N extra SAM1 decoder calls after the initial call.",
        "MaskMemory": bool(cfg.use_mask_memory),
        "CandidateQualityNote": (
            "Extra multimask_output=True SAM1 call using EXACT last non-flip "
            "refinement points/box/input-mask-memory. Saved single-mask and "
            "FlipGate outputs are UNCHANGED; 3 diagnostic candidates may "
            "differ from the saved single-mask token. Multi-box image-level "
            "scores aggregate using per-box MIN (conservative)."),
        "CandidateQualitySummary": {
            k: (float(np.mean([float(r[k]) for r in records if k in r]))
                if any(k in r for r in records) else None)
            for k in ('candidate_iou_mean','candidate_ssim_mean',
                      'candidate_sim_mean','sam_score_selected','sam_stability_selected')
        },
        "Failures": failures,
        "Skipped": skipped,
    }
    with open(summary_path,"w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("="*72)
    for k in ("Base", "Refine_NoFlip", "Refine_FlipGate"):
        metrics = summary[k]
        print(f"{k:>18}: " + " ".join(f"{m}={metrics[m]:.4f}" for m in ("Smeasure","maxE","maxF","MAE") if metrics[m] is not None))
    if cfg.candidate_quality:
        print('Final candidate quality:', summary['CandidateQualitySummary'])
        print('Per-box candidate details:', os.path.join(cfg.out_dir, 'candidate_per_box'))
    print("Diagnostics:", summary['Diagnostics'])
    print("CSV:", csv_path if records else "no valid records")
    print("Summary:", summary_path)
    print("="*72)


if __name__ == '__main__':
    main()
