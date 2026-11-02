#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM1 Box+Scribble ablation: Base / GAPR / Scribble-Conflict+MMF.

Core experiments:
  A0_Base              1 SAM1 single-mask decode, union box, <=9 FG/9 BG points.
  A1_GAPR              OCVR zoom and IDP per-box, 1 single-mask decode per box.
  A2_Conflict_MMF      Same first pass as A0, scribble-guided active correction
                       and previous 256x256 SAM1 logits (MMF), 2 extra decodes.
  A3_GAPR_Conflict_MMF Same first pass as A1, active correction + MMF per box.

Optional --extra-controls:
  B1_ConflictOnly, B2_MMFOnly, B3_GAPR_ConflictOnly, B4_GAPR_MMFOnly.

Active scribble points are selected ONLY from human FG/BG annotations:
  1. prioritize contradiction with current prediction and choose the strongest;
  2. if none, select the most uncertain still-available annotated point;
  3. maintain min spacing from existing and newly added points.
One point at most per extra round per inference box, BOTH labels allowed.
For MMF, feed the previous SAM1 low-resolution decoder LOGITS without sigmoid
as mask_input=(1,256,256); *never* use GT or three candidate masks.

GAPR uses rho=max(box_width,box_height)/max(image_height,image_width):
  rho < 1/8 -> lambda=.5 -> zoom 1.5x;
  rho < 1/4 -> lambda=.25 -> zoom 1.25x;
  else      -> lambda=0 -> no crop.
IDP independently decodes per box and merges by max probability.

Examples from the RCLCOD root:
  python sam1_gapr_conflict_mmf_ablation.py --self-test
  python sam1_gapr_conflict_mmf_ablation.py --limit 100 --output-dir softresult/SAM1_CM_100
  python sam1_gapr_conflict_mmf_ablation.py --limit 100 --extra-controls
  python sam1_gapr_conflict_mmf_ablation.py --refine-iters 1

Outputs: method-named PNG folders, metrics_per_image.csv, summary_metrics.json,
active_points.csv, and selected per-case comparison/Strategy_RGB visualizations.
E_binary_0p5 / F_binary_0p5 are fixed-threshold metrics, NOT maxE/maxF.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CORE_METHODS = ("A0_Base", "A1_GAPR", "A2_Conflict_MMF", "A3_GAPR_Conflict_MMF")
CONTROL_METHODS = ("B1_ConflictOnly", "B2_MMFOnly",
                   "B3_GAPR_ConflictOnly", "B4_GAPR_MMFOnly")
METRICS = ("Smeasure", "E_binary_0p5", "F_binary_0p5", "MAE", "IoU", "Dice")


@dataclass
class Config:
    sam_checkpoint: str = "./sam_mask/weights/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    img_dir: str = "./data/Train/Imgs"
    scribble_dir: str = "./data/Train/Scribble"
    gt_dir: str = "./data/Train/GT"
    json_dir: str = "./data/Train/box_label"
    output_dir: str = "./softresult/SAM1_GAPR_Conflict_MMF"
    device: str = "cuda"
    model_size: int = 1024
    seed: int = 0
    initial_per_class: int = 9
    refine_iters: int = 2         # two EXTRA decodes after the initial prediction
    active_min_dist: int = 16    # pixels in 1024x1024 SAM model coordinates
    extra_controls: bool = False
    zoom_small: float = 0.50
    zoom_medium: float = 0.25
    rho_small: float = 0.125
    rho_medium: float = 0.25
    min_context_px: int = 8   # minimal original-image pixel context around box
    context_frac: float = 0.10  # minimal context relative to long edge of box
    merge_margin_px: int = 0  # 0: no extra gate; cropped ROI is natural support
    save_probs: bool = False
    vis_topk: int = 40
    vis_min_gain: float = 0.003
    limit: Optional[int] = None


def image_paths(directory: str) -> List[str]:
    paths: List[str] = []
    for ext in ("jpg", "jpeg", "png", "bmp", "tif", "tiff"):
        paths.extend(glob(os.path.join(directory, "*." + ext)))
        paths.extend(glob(os.path.join(directory, "*." + ext.upper())))
    return sorted(set(paths))


def locate_file(directory: str, stem: str, extensions: Sequence[str]) -> Optional[str]:
    for ext in extensions:
        for e in (ext.lower(), ext.upper()):
            p = os.path.join(directory, stem + e)
            if os.path.isfile(p):
                return p
    return None


def read_boxes(path: str, width: int, height: int) -> List[np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    boxes: List[np.ndarray] = []
    for item in raw.get("shapes", []):
        p = np.asarray(item.get("points", []), dtype=np.float32)
        if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 2:
            continue
        b = np.array((p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()), np.float32)
        b[[0, 2]] = np.clip(b[[0, 2]], 0, width - 1)
        b[[1, 3]] = np.clip(b[[1, 3]], 0, height - 1)
        if b[2] > b[0] and b[3] > b[1]:
            boxes.append(b)
    return boxes


def parse_scribble(scribble: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """1 = FG, 2 = BG, 0 = UNKNOWN. Binary 255=FG, 0=UNKNOWN.

    If annotation uses another label scheme, adapt this function BEFORE running.
    """
    sc = np.asarray(scribble)
    if (sc == 1).any() or (sc == 2).any():
        return sc == 1, sc == 2
    return sc > 128, np.zeros(sc.shape, dtype=bool)


def initial_points(scribble: np.ndarray, k: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Greedy spatial coverage with <=k annotated foreground AND <=k background points."""
    if k == 0:
        return None, None
    fg, bg = parse_scribble(scribble)
    points: List[Tuple[float, float]] = []
    labels: List[int] = []
    for mask, label in ((fg, 1), (bg, 0)):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        divisions = int(math.ceil(math.sqrt(k)))
        ye = np.linspace(int(ys.min()), int(ys.max()) + 1, divisions + 1)
        xe = np.linspace(int(xs.min()), int(xs.max()) + 1, divisions + 1)
        candidates: List[Tuple[float, float, float]] = []
        for yi in range(divisions):
            for xi in range(divisions):
                sel = (ys >= ye[yi]) & (ys < ye[yi + 1]) & (xs >= xe[xi]) & (xs < xe[xi + 1])
                if sel.any():
                    cx, cy = xs[sel], ys[sel]
                    idx = int(np.argmax(dt[cy, cx]))
                    candidates.append((float(cx[idx]), float(cy[idx]), float(dt[cy[idx], cx[idx]])))
        selected: List[int] = []
        remaining = set(range(len(candidates)))
        while remaining and len(selected) < k:
            if not selected:
                j = max(remaining, key=lambda a: (candidates[a][2], -a))
            else:
                j = max(remaining, key=lambda a: (
                    min((candidates[a][0] - candidates[b][0])**2 +
                        (candidates[a][1] - candidates[b][1])**2 for b in selected),
                    candidates[a][2], -a))
            selected.append(j)
            remaining.remove(j)
        for j in selected:
            points.append((candidates[j][0], candidates[j][1]))
            labels.append(label)
    if not points:
        return None, None
    return np.asarray(points, np.float32), np.asarray(labels, np.int32)


def union_box(boxes: Sequence[np.ndarray]) -> np.ndarray:
    a = np.asarray(boxes, dtype=np.float32)
    return np.array([a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()], np.float32)


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    b = np.asarray(box, dtype=np.float32).copy()
    b[[0, 2]] = np.clip(b[[0, 2]], 0, width - 1)
    b[[1, 3]] = np.clip(b[[1, 3]], 0, height - 1)
    return b


def zoom_lambda(box: np.ndarray, width: int, height: int, cfg: Config) -> float:
    rho = max(float(box[2] - box[0]), float(box[3] - box[1])) / float(max(width, height))
    if rho < cfg.rho_small:
        return cfg.zoom_small
    if rho < cfg.rho_medium:
        return cfg.zoom_medium
    return 0.0


def _centered_interval(center: float, length: int, full: int) -> Tuple[int, int]:
    length = min(full, max(1, int(length)))
    start = int(np.clip(round(center - length * 0.5), 0, full - length))
    return start, start + length


def ocvr_region(box: np.ndarray, width: int, height: int, cfg: Config
                ) -> Tuple[Tuple[int, int, int, int], float, float]:
    """Return original-image half-open region, lambda, actual scalar zoom.

    Preserve original aspect ratio when feasible. Enlarge FOV as needed to fully
    contain the box and minimal context; shift (not pad) at image edges.
    """
    lam = zoom_lambda(box, width, height, cfg)
    if lam == 0:
        return (0, 0, width, height), lam, 1.0
    z = 1.0 + lam
    bw, bh = float(box[2] - box[0]), float(box[3] - box[1])
    pad = max(float(cfg.min_context_px), cfg.context_frac * max(bw, bh))
    cw = max(int(math.ceil(width / z)), int(math.ceil(bw + 2.0 * pad + 1)))
    ch = max(int(math.ceil(height / z)), int(math.ceil(bh + 2.0 * pad + 1)))
    # If one axis cannot support planned zoom, reduce zoom uniformly to avoid
    # additional aspect-ratio deformation compared with Base.
    factor = min(float(width) / min(width, cw), float(height) / min(height, ch))
    cw = min(width, max(cw, int(math.ceil(width / factor))))
    ch = min(height, max(ch, int(math.ceil(height / factor))))
    cx, cy = (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)
    x0, x1 = _centered_interval(cx, cw, width)
    y0, y1 = _centered_interval(cy, ch, height)
    # In rare large-context/eccentric-box cases ensure box remains contained.
    assert x0 <= box[0] + 1 and x1 >= box[2] - 1, (box, (x0, x1))
    assert y0 <= box[1] + 1 and y1 >= box[3] - 1, (box, (y0, y1))
    return (x0, y0, x1, y1), lam, min(width / (x1-x0), height / (y1-y0))


def map_box_to_frame(box: np.ndarray, region: Tuple[int, int, int, int], size: int) -> np.ndarray:
    x0, y0, x1, y1 = region
    sx, sy = size / (x1 - x0), size / (y1 - y0)
    b = np.asarray(box, np.float32)
    result = np.array([
        (b[0] - x0 + 0.5) * sx - 0.5,
        (b[1] - y0 + 0.5) * sy - 0.5,
        (b[2] - x0 + 0.5) * sx - 0.5,
        (b[3] - y0 + 0.5) * sy - 0.5,
    ], dtype=np.float32)
    return clip_box(result, size, size)


def point_to_orig(x: float, y: float, region: Tuple[int, int, int, int], size: int
                 ) -> Tuple[float, float]:
    x0, y0, x1, y1 = region
    return (x + 0.5) * (x1-x0)/size + x0 - 0.5, (y + 0.5) * (y1-y0)/size + y0 - 0.5


def make_frame(rgb: np.ndarray, scribble: np.ndarray, region: Tuple[int, int, int, int],
               box: np.ndarray, cfg: Config, per_box: bool
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    x0, y0, x1, y1 = region
    img = cv2.resize(rgb[y0:y1, x0:x1], (cfg.model_size, cfg.model_size), interpolation=cv2.INTER_LINEAR)
    sc = cv2.resize(scribble[y0:y1, x0:x1], (cfg.model_size, cfg.model_size), interpolation=cv2.INTER_NEAREST)
    b = map_box_to_frame(box, region, cfg.model_size)
    if per_box:
        # No other object's foreground scribbles are allowed in an instance's
        # positive-point set. Background scribbles remain available.
        fg, _ = parse_scribble(sc)
        xlo = max(0, int(math.floor(b[0])))
        ylo = max(0, int(math.floor(b[1])))
        xhi = min(cfg.model_size, int(math.ceil(b[2])) + 1)
        yhi = min(cfg.model_size, int(math.ceil(b[3])) + 1)
        allowed = np.zeros(sc.shape, np.uint8)
        allowed[ylo:yhi, xlo:xhi] = 1
        sc = sc.copy()
        sc[fg & (allowed == 0)] = 0
    pts, labs = initial_points(sc, cfg.initial_per_class)
    return img, sc, b, pts, labs


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def restore_prob(prob_model: np.ndarray, region: Tuple[int, int, int, int],
                 height: int, width: int) -> np.ndarray:
    x0, y0, x1, y1 = region
    dst = np.zeros((height, width), np.float32)
    dst[y0:y1, x0:x1] = cv2.resize(np.asarray(prob_model, np.float32),
                                  (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
    return dst


def select_active_scribble_point(
    prob: np.ndarray,
    scribble: np.ndarray,
    existing: Optional[np.ndarray],
    roi_box: np.ndarray,
    min_dist: int,
) -> Tuple[Optional[np.ndarray], Optional[int], str]:
    """One annotated point, prioritizing current scribble contradiction.

    1=FG, 2=BG, 0=unknown in the input label. Never sample unlabeled pixels.
    Distance to all existing points is enforced in MODEL coordinates.
    """
    p = np.asarray(prob, dtype=np.float32)
    fg, bg = parse_scribble(scribble)
    if p.shape != fg.shape:
        raise ValueError("Model prediction and scribble shapes do not match")
    h, w = p.shape
    x1, y1, x2, y2 = np.asarray(roi_box).round().astype(int)
    x1, x2 = max(0, x1), min(w-1, x2)
    y1, y2 = max(0, y1), min(h-1, y2)
    if x2 < x1 or y2 < y1:
        return None, None, "invalid_roi"
    avail = (fg | bg).astype(np.uint8)
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[y1:y2+1, x1:x2+1] = 1
    avail &= roi
    if not np.any(avail):
        return None, None, "no_annotated_point_in_box"
    if existing is not None and len(existing):
        for x, y in np.asarray(existing, dtype=np.float32).reshape(-1, 2):
            cv2.circle(avail, (int(round(x)), int(round(y))), int(min_dist), 0, -1)
    allowed = avail.astype(bool)
    if not allowed.any():
        return None, None, "no_available_annotated_point"
    pred_fg = p >= .5
    contradiction = ((fg & ~pred_fg) | (bg & pred_fg)) & allowed
    if contradiction.any():
        strength = np.where(fg, 1.0 - p, p)
        strength[~contradiction] = -np.inf
        y, x = np.unravel_index(int(np.argmax(strength)), strength.shape)
        kind = "scribble_conflict"
    else:
        uncertainty = 1.0 - np.abs(2.0*p - 1.0)
        uncertainty[~allowed] = -np.inf
        y, x = np.unravel_index(int(np.argmax(uncertainty)), uncertainty.shape)
        kind = "scribble_uncertainty"
    return np.asarray([float(x), float(y)], np.float32), int(fg[y, x]), kind


class SAM1PredictorWrapper:
    def __init__(self, cfg: Config, predictor=None):
        self.cfg = cfg
        if predictor is None:
            if not os.path.isfile(cfg.sam_checkpoint):
                raise FileNotFoundError(f"SAM1 checkpoint missing: {cfg.sam_checkpoint}")
            try:
                from segment_anything import SamPredictor, sam_model_registry
            except ImportError:
                # The supplied RCLCOD repository also includes SAM1 under
                # sam_mask/segment_anything. Support both root and sam_mask runs.
                import sys
                possible_roots = (
                    Path(__file__).resolve().parent / "sam_mask",
                    Path(__file__).resolve().parent,
                    Path.cwd() / "sam_mask",
                )
                for root in possible_roots:
                    if (root / "segment_anything").is_dir():
                        sys.path.insert(0, str(root))
                        break
                try:
                    from segment_anything import SamPredictor, sam_model_registry
                except ImportError as exc:
                    raise ImportError(
                        "Install SAM1 or place this script in the RCLCOD root: "
                        "pip install git+https://github.com/facebookresearch/segment-anything.git"
                    ) from exc
            import torch
            if cfg.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Pass --device cpu or enable CUDA.")
            model = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
            model.to(device=cfg.device)
            model.eval()
            predictor = SamPredictor(model)
        self.predictor = predictor
        self.frame_hw: Optional[Tuple[int, int]] = None

    def set_image(self, img: np.ndarray):
        import torch
        with torch.inference_mode():
            self.predictor.set_image(np.ascontiguousarray(img))
        self.frame_hw = img.shape[:2]

    def single(
        self,
        pts: Optional[np.ndarray],
        labs: Optional[np.ndarray],
        box: np.ndarray,
        mask_input: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """SAM1 single-mask decoder: return probability AND raw low-res logits."""
        import torch
        if (pts is None) != (labs is None):
            raise ValueError("Point coordinates and labels must be present together")
        if pts is not None and len(pts) != len(labs):
            raise ValueError("Point/label count mismatch")
        mem = None if mask_input is None else np.asarray(mask_input, np.float32)
        if mem is not None and mem.shape != (1, 256, 256):
            raise ValueError(f"SAM1 MMF expects mask_input=(1,256,256), got {mem.shape}")
        with torch.inference_mode():
            high_logits, scores, low_logits = self.predictor.predict(
                point_coords=None if pts is None else np.asarray(pts, np.float32).reshape(-1,2),
                point_labels=None if labs is None else np.asarray(labs, np.int32).reshape(-1),
                box=np.asarray(box, np.float32).reshape(4),
                mask_input=mem,
                multimask_output=False,
                return_logits=True,
            )
        high = np.asarray(high_logits, np.float32)
        low = np.asarray(low_logits, np.float32)
        if high.ndim != 3 or high.shape[0] != 1:
            raise ValueError(f"SAM1 must return 1 high-res logits mask, got {high.shape}")
        if low.shape == (256,256):
            low = low[None]
        if low.shape != (1,256,256):
            raise ValueError(f"SAM1 must return (1,256,256) low-res logits, got {low.shape}")
        if not np.isfinite(high).all() or not np.isfinite(low).all():
            raise ValueError("SAM1 returned nonfinite logits")
        return sigmoid(high[0]).astype(np.float32), low.copy()


def _object_score(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    m, sd = float(x.mean()), float(x.std())
    return 2*m/(m*m + 1 + sd + 1e-8)


def _regional_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    x, y = float(pred.mean()), float(gt.mean())
    sx, sy = float(((pred-x)**2).mean()), float(((gt-y)**2).mean())
    cov = float(((pred-x)*(gt-y)).mean())
    return float(4*x*y*cov/((x*x + y*y)*(sx+sy)+1e-8))


def smeasure(pred: np.ndarray, gt: np.ndarray) -> float:
    """COD S-measure on binary predictions, consistent across all four methods."""
    p = np.asarray(pred, np.float32)
    g = np.asarray(gt, np.float32)
    mass = float(g.mean())
    if mass <= 0:
        return 1.0 - float(p.mean())
    if mass >= 1:
        return float(p.mean())
    so = mass*_object_score(p[g > 0.5]) + (1-mass)*_object_score(1-p[g <= 0.5])
    h, w = g.shape
    ys, xs = np.where(g > 0.5)
    x = int(np.clip(np.round(xs.mean()), 1, w-1)) if w > 1 else 0
    y = int(np.clip(np.round(ys.mean()), 1, h-1)) if h > 1 else 0
    sr = 0.0
    for u0, v0, u1, v1 in ((0,0,x,y),(x,0,w,y),(0,y,x,h),(x,y,w,h)):
        region_area = (u1-u0)*(v1-v0)
        if region_area > 0:
            sr += region_area/(w*h)*_regional_ssim(p[v0:v1, u0:u1], g[v0:v1, u0:u1])
    return float(np.clip(0.5*(so + sr), 0., 1.))


def evaluate_binary(prob: np.ndarray, gt_gray: np.ndarray) -> Dict[str, float]:
    """Fixed threshold binary metrics; not sweep-based maxF/maxE."""
    p = np.asarray(prob, np.float32)
    if p.shape != gt_gray.shape:
        p = cv2.resize(p, (gt_gray.shape[1], gt_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    b = (p >= 0.5)
    gt = (gt_gray > 128)
    tp = float((b & gt).sum())
    fp = float((b & ~gt).sum())
    fn = float((~b & gt).sum())
    precision = tp/(tp+fp+1e-8)
    recall = tp/(tp+fn+1e-8)
    fbeta = 1.3*precision*recall/(0.3*precision+recall+1e-8)
    if not gt.any():
        e = 1.0 - float(b.mean())
    elif gt.all():
        e = float(b.mean())
    else:
        fm = b.astype(np.float32)
        gm = gt.astype(np.float32)
        fm -= fm.mean()
        gm -= gm.mean()
        align = 2*fm*gm/(fm*fm+gm*gm+1e-8)
        e = float(np.mean((align+1)**2/4))
    bf, gf = b.astype(np.float32), gt.astype(np.float32)
    return {
        "Smeasure": smeasure(bf, gf),
        "E_binary_0p5": e,
        "F_binary_0p5": float(fbeta),
        "MAE": float(np.mean(b != gt)),
        "IoU": tp/(tp+fp+fn+1e-8),
        "Dice": 2*tp/(2*tp+fp+fn+1e-8),
    }


def _write_csv(path: str, records: List[dict], fields: Optional[List[str]]=None):
    if not records and not fields:
        return
    if fields is None:
        fields = list(dict.fromkeys(k for row in records for k in row))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _save_binary(path: str, prob: np.ndarray):
    ok = cv2.imwrite(path, np.where(prob >= 0.5, 255, 0).astype(np.uint8))
    if not ok:
        raise RuntimeError(f"Could not write {path}")


def _rgb_with_boxes(rgb: np.ndarray, boxes: Sequence[np.ndarray], point_rows: List[dict],
                    regions: List[Tuple[int,int,int,int]], show_method: str
                    ) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for i, box in enumerate(boxes):
        a = tuple(np.round(box[:2]).astype(int))
        b = tuple(np.round(box[2:]).astype(int))
        cv2.rectangle(out, a, b, (30, 200, 255), 2)
        cv2.putText(out, f"B{i}", (a[0], max(10,a[1]-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (30,200,255), 2)
    for region in regions:
        x0,y0,x1,y1 = region
        if x0 != 0 or y0 != 0 or x1 != rgb.shape[1] or y1 != rgb.shape[0]:
            cv2.rectangle(out, (x0,y0), (x1-1,y1-1), (255,150,25), 2)
    for row in point_rows:
        if row["method"] != show_method:
            continue
        xy = (int(round(row["x_original"])), int(round(row["y_original"])))
        color = (30, 200, 35) if int(row["label"]) == 1 else (40, 40, 245)
        cv2.circle(out, xy, 4, color, -1)
        cv2.circle(out, xy, 6, (255, 255, 255), 1)
    return out


def _visualize(path: str, rgb: np.ndarray, gt: np.ndarray, boxes: Sequence[np.ndarray],
               regions: Sequence[Tuple[int,int,int,int]], point_rows: List[dict],
               results: Dict[str,np.ndarray], metrics: Dict[str,Dict[str,float]]):
    tile = 256
    def panel(arr: np.ndarray, title: str, mask: bool=False) -> np.ndarray:
        resized = cv2.resize(arr, (tile,tile),
                             interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(resized, (0,0), (tile,27), (15,15,15), -1)
        cv2.putText(resized, title[:35], (4,18), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (255,255,255), 1, cv2.LINE_AA)
        return resized
    strategy = _rgb_with_boxes(rgb, boxes, point_rows, list(regions),
                               show_method="A3_GAPR_Conflict_MMF")
    masks = [panel((results[m]>=.5).astype(np.uint8)*255,
                   f"{m} MAE={metrics[m]['MAE']:.4f}", mask=True) for m in CORE_METHODS]
    panels = [panel(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "RGB"),
              panel(strategy, "Strategy (green FG/red BG)"),
              panel(gt, "GT", mask=True)]
    blank = np.zeros_like(panels[0])
    comparison = np.vstack([np.hstack(panels + [blank]), np.hstack(masks)])
    cv2.imwrite(path, comparison)
    cv2.imwrite(path.replace("_comparison.png", "_Strategy_RGB.png"), strategy)


class AblationRunner:
    def __init__(self, cfg: Config, predictor=None):
        self.cfg = cfg
        self.sam = SAM1PredictorWrapper(cfg, predictor=predictor)
        self.methods = CORE_METHODS + (CONTROL_METHODS if cfg.extra_controls else ())
        self.point_rows: List[dict] = []
        self.vis_jobs: List[dict] = []
        for m in self.methods:
            os.makedirs(os.path.join(cfg.output_dir, m), exist_ok=True)
            if cfg.save_probs:
                os.makedirs(os.path.join(cfg.output_dir, "probs", m), exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, "visualizations"), exist_ok=True)

    def _refine(
        self,
        first: np.ndarray,
        first_low: np.ndarray,
        pts: Optional[np.ndarray],
        labs: Optional[np.ndarray],
        box: np.ndarray,
        scribble_frame: np.ndarray,
        imgname: str,
        method: str,
        region: Tuple[int,int,int,int],
        object_id: int,
        use_active: bool = True,
        use_mmf: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """Extra SAM1 decoder iterations, with optional active points and MMF.

        Every configuration starts from exactly the SAME first-pass mask/logits,
        not from another ablation's prediction. Human scribble only, never GT.
        """
        current = first.copy()
        low = first_low.copy()
        point_set = None if pts is None else pts.copy()
        label_set = None if labs is None else labs.copy()
        stats = {"num_active": 0, "num_active_fg": 0, "num_active_bg": 0,
                 "num_conflicts": 0, "num_uncertainty": 0,
                 "num_memory_calls": 0, "num_extra_decodes": 0}
        for step in range(1, self.cfg.refine_iters + 1):
            if use_active:
                new, label, reason = select_active_scribble_point(
                    current, scribble_frame, point_set, box, self.cfg.active_min_dist)
                if new is not None:
                    if point_set is None:
                        point_set = new.reshape(1, 2).copy()
                        label_set = np.array([label], dtype=np.int32)
                    else:
                        point_set = np.concatenate((point_set, new.reshape(1,2)), axis=0)
                        label_set = np.concatenate((label_set, np.array([label],np.int32)))
                    stats["num_active"] += 1
                    stats["num_active_fg" if label == 1 else "num_active_bg"] += 1
                    stats["num_conflicts" if reason == "scribble_conflict" else "num_uncertainty"] += 1
                    x, y = float(new[0]), float(new[1])
                    xo, yo = point_to_orig(x, y, region, self.cfg.model_size)
                    self.point_rows.append({
                        "image": imgname, "method": method, "object_id": object_id,
                        "round": step, "x_model": x, "y_model": y,
                        "x_original": float(xo), "y_original": float(yo),
                        "label": label, "reason": reason,
                    })
            mem = low if use_mmf else None
            current, low = self.sam.single(point_set, label_set, box, mask_input=mem)
            stats["num_extra_decodes"] += 1
            stats["num_memory_calls"] += int(use_mmf)
        return current, stats

    def run_image(self, img_path: str) -> Optional[dict]:
        cfg = self.cfg
        stem = Path(img_path).stem
        sc_path = locate_file(cfg.scribble_dir, stem, (".png", ".jpg", ".jpeg"))
        bx_path = locate_file(cfg.json_dir, stem, (".json",))
        if not sc_path or not bx_path:
            return None
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        scribble = cv2.imread(sc_path, cv2.IMREAD_GRAYSCALE)
        if bgr is None or scribble is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if scribble.shape != (h,w):
            scribble = cv2.resize(scribble, (w,h), interpolation=cv2.INTER_NEAREST)
        boxes = read_boxes(bx_path, w, h)
        if not boxes:
            return None
        full = (0,0,w,h)
        union = union_box(boxes)
        out: Dict[str,np.ndarray] = {}
        diag: Dict[str,dict] = {}
        t0 = time.perf_counter()
        first_point_offset = len(self.point_rows)

        # A0 base and A2 conflict+MMF: same union box and first-pass logits.
        full_img, full_sc, b_full, pts_full, labs_full = make_frame(
            rgb, scribble, full, union, cfg, per_box=False)
        self.sam.set_image(full_img)
        base, base_low = self.sam.single(pts_full, labs_full, b_full)
        out["A0_Base"] = restore_prob(base, full, h, w)
        refined, stats = self._refine(base, base_low, pts_full, labs_full, b_full,
                                      full_sc, stem, "A2_Conflict_MMF", full, -1,
                                      use_active=True, use_mmf=True)
        out["A2_Conflict_MMF"] = restore_prob(refined, full, h, w)
        diag["A2_Conflict_MMF"] = stats
        if cfg.extra_controls:
            for name, active, memory in (("B1_ConflictOnly", True, False),
                                          ("B2_MMFOnly", False, True)):
                pred, stats = self._refine(base, base_low, pts_full, labs_full,
                                            b_full, full_sc, stem, name, full, -1,
                                            use_active=active, use_mmf=memory)
                out[name] = restore_prob(pred, full, h, w)
                diag[name] = stats

        # A1 and A3: same per-object first pass. Optional controls also reuse it.
        gapr = np.zeros((h,w), np.float32)
        gapr_refined = np.zeros((h,w), np.float32)
        ctrl_maps = {name: np.zeros((h,w), np.float32)
                     for name in ("B3_GAPR_ConflictOnly", "B4_GAPR_MMFOnly")
                     if cfg.extra_controls}
        obj_stats: Dict[str,List[dict]] = {"A3_GAPR_Conflict_MMF": []}
        for name in ctrl_maps:
            obj_stats[name] = []
        regions: List[Tuple[int,int,int,int]] = []
        zooms: List[float] = []
        lambdas: List[float] = []
        for i, box in enumerate(boxes):
            region, lam, zoom = ocvr_region(box, w, h, cfg)
            regions.append(region)
            zooms.append(zoom)
            lambdas.append(lam)
            img, local_sc, b_obj, pts_obj, labs_obj = make_frame(
                rgb, scribble, region, box, cfg, per_box=True)
            self.sam.set_image(img)
            first, first_low = self.sam.single(pts_obj, labs_obj, b_obj)
            refined, stats = self._refine(first, first_low, pts_obj, labs_obj,
                                          b_obj, local_sc, stem, "A3_GAPR_Conflict_MMF",
                                          region, i, use_active=True, use_mmf=True)
            obj_stats["A3_GAPR_Conflict_MMF"].append(stats)
            probs = {"A1_GAPR": restore_prob(first, region, h, w),
                     "A3_GAPR_Conflict_MMF": restore_prob(refined, region, h, w)}
            if cfg.extra_controls:
                for name, active, memory in (("B3_GAPR_ConflictOnly", True, False),
                                              ("B4_GAPR_MMFOnly", False, True)):
                    pred, stats = self._refine(first, first_low, pts_obj, labs_obj,
                                                b_obj, local_sc, stem, name, region, i,
                                                use_active=active, use_mmf=memory)
                    probs[name] = restore_prob(pred, region, h, w)
                    obj_stats[name].append(stats)
            if cfg.merge_margin_px > 0:
                gate = np.zeros((h,w), np.uint8)
                margin = cfg.merge_margin_px
                x1 = max(0, int(math.floor(box[0]-margin)))
                y1 = max(0, int(math.floor(box[1]-margin)))
                x2 = min(w, int(math.ceil(box[2]+margin))+1)
                y2 = min(h, int(math.ceil(box[3]+margin))+1)
                gate[y1:y2,x1:x2] = 1
                for p in probs.values():
                    p *= gate
            gapr = np.maximum(gapr, probs["A1_GAPR"])
            gapr_refined = np.maximum(gapr_refined, probs["A3_GAPR_Conflict_MMF"])
            for name in ctrl_maps:
                ctrl_maps[name] = np.maximum(ctrl_maps[name], probs[name])
        out["A1_GAPR"] = gapr
        out["A3_GAPR_Conflict_MMF"] = gapr_refined
        out.update(ctrl_maps)
        for method, stats_list in obj_stats.items():
            diag[method] = {k: int(sum(st[k] for st in stats_list))
                            for k in stats_list[0]} if stats_list else {}
        elapsed = time.perf_counter()-t0

        for method, prob in out.items():
            _save_binary(os.path.join(cfg.output_dir, method, stem+".png"), prob)
            if cfg.save_probs:
                np.save(os.path.join(cfg.output_dir, "probs", method, stem+".npy"),
                        prob.astype(np.float16))
        row: Dict[str,object] = {
            "image": os.path.basename(img_path), "num_boxes": len(boxes),
            "num_initial_FG": int((labs_full == 1).sum()) if labs_full is not None else 0,
            "num_initial_BG": int((labs_full == 0).sum()) if labs_full is not None else 0,
            "ocvr_regions_cropped": sum(r != full for r in regions),
            "ocvr_lambda_per_box": ";".join(f"{v:.2f}" for v in lambdas),
            "ocvr_actual_zoom_per_box": ";".join(f"{v:.3f}" for v in zooms),
            "elapsed_sec": elapsed,
        }
        for method, stats in diag.items():
            for key, value in stats.items():
                row[f"{method}_{key}"] = value
        gtpath = locate_file(cfg.gt_dir, stem, (".png", ".jpg", ".jpeg"))
        if gtpath:
            gt = cv2.imread(gtpath, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                results = {method: evaluate_binary(p, gt) for method,p in out.items()}
                for method, mets in results.items():
                    for key, val in mets.items():
                        row[f"{method}_{key}"] = val
                row["gain_MAE_A3_vs_A0"] = (results["A0_Base"]["MAE"] -
                                             results["A3_GAPR_Conflict_MMF"]["MAE"])
                row["gain_MAE_A3_vs_A1"] = (results["A1_GAPR"]["MAE"] -
                                             results["A3_GAPR_Conflict_MMF"]["MAE"])
                if cfg.vis_topk > 0 and (row["gain_MAE_A3_vs_A1"] >= cfg.vis_min_gain):
                    self.vis_jobs.append({"gain": float(row["gain_MAE_A3_vs_A1"]),
                                          "stem": stem, "rgb": rgb, "gt": gt,
                                          "boxes": boxes, "regions": regions,
                                          "point_rows": list(self.point_rows[first_point_offset:]),
                                          "preds": {m: out[m] for m in CORE_METHODS},
                                          "metrics": results})
                    self.vis_jobs.sort(key=lambda j: j["gain"], reverse=True)
                    del self.vis_jobs[cfg.vis_topk:]
        return row

    def save_visualizations(self):
        for rank, job in enumerate(self.vis_jobs, 1):
            path = os.path.join(self.cfg.output_dir, "visualizations",
                                f"{rank:03d}_{job['stem']}_comparison.png")
            _visualize(path, job["rgb"], job["gt"], job["boxes"], job["regions"],
                       job["point_rows"], job["preds"], job["metrics"])


def run(cfg: Config, predictor=None) -> dict:
    if cfg.refine_iters < 1:
        raise ValueError("--refine-iters must be >=1 for Conflict/MMF ablations")
    os.makedirs(cfg.output_dir, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    runner = AblationRunner(cfg, predictor=predictor)
    paths = image_paths(cfg.img_dir)
    if cfg.limit is not None:
        paths = paths[:cfg.limit]
    results: List[dict] = []
    failed: List[dict] = []
    skipped: List[str] = []
    from tqdm import tqdm
    for path in tqdm(paths, desc="SAM1 GAPR + ScribbleConflict/MMF ablation"):
        try:
            rec = runner.run_image(path)
            if rec is None:
                skipped.append(os.path.basename(path))
            else:
                results.append(rec)
        except Exception as exc:
            failed.append({"image":os.path.basename(path), "error":repr(exc)})
            print(f"[ERROR] {os.path.basename(path)}: {exc}")
    runner.save_visualizations()
    _write_csv(os.path.join(cfg.output_dir, "metrics_per_image.csv"), results)
    _write_csv(os.path.join(cfg.output_dir, "active_points.csv"), runner.point_rows,
               ["image", "method", "object_id", "round", "x_model", "y_model",
                "x_original", "y_original", "label", "reason"])
    summary: dict = {
        "Methods": {},
        "Diagnostics": {
            "attempted": len(paths), "processed": len(results),
            "skipped": len(skipped), "failures": len(failed),
            "total_crop_regions": int(sum(r["ocvr_regions_cropped"] for r in results)),
        },
        "Config": vars(cfg),
        "Supervision": "SAM1 Box+Scribble: <=9 FG / <=9 BG initial points; new points only from known scribbles. GT evaluation only.",
        "OCVR": "lambda=0.50/0.25/0.00 means 1.50x/1.25x/1.00x local FOV zoom; IDP max merge",
        "Refinement": "Conflict-first annotated points, uncertainty on annotated scribbles as fallback; previous SAM1 raw 256x256 logits as MMF; no RFC/BCS/Flip Gate.",
        "MetricNote": "Fixed 0.5 threshold; E_binary_0p5/F_binary_0p5 are not standard maxE/maxF.",
        "Failures": failed, "Skipped": skipped,
    }
    for method in runner.methods:
        sub = [r for r in results if f"{method}_MAE" in r]
        summary["Methods"][method] = {
            "num_evaluated": len(sub),
            **{k: (float(np.mean([r[f"{method}_{k}"] for r in sub])) if sub else None)
               for k in METRICS},
        }
        if "Base" not in method and "A1_GAPR" != method:
            for key in ("num_active", "num_active_fg", "num_active_bg",
                        "num_conflicts", "num_uncertainty", "num_memory_calls"):
                summary["Diagnostics"][f"total_{method}_{key}"] = int(sum(
                    r.get(f"{method}_{key}", 0) for r in results))
    for method in ("A2_Conflict_MMF", "A3_GAPR_Conflict_MMF"):
        summary["Diagnostics"][f"images_with_{method}_new_points"] = int(sum(
            r.get(f"{method}_num_active", 0) > 0 for r in results))
    with open(os.path.join(cfg.output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    print("\n"+"="*85)
    for method in runner.methods:
        vals = summary["Methods"][method]
        print(f"{method:26} N={vals['num_evaluated']:4}",
              " ".join(f"{k}={vals[k]:.4f}" for k in
                       ("Smeasure", "E_binary_0p5", "F_binary_0p5", "MAE", "IoU")
                       if vals[k] is not None))
    print("Diagnostics:", summary["Diagnostics"])
    if results and summary["Diagnostics"]["total_A2_Conflict_MMF_num_active"] == 0:
        print("[WARN] No A2 new points were sampled. Check scribble format / initial point coverage / spacing.")
    if results and summary["Diagnostics"]["total_A3_GAPR_Conflict_MMF_num_active"] == 0:
        print("[WARN] No A3 new points were sampled. Check scribble format / initial point coverage / spacing.")
    print("Output:", cfg.output_dir)
    print("="*85)
    return summary


def self_test():
    """Synthetic end-to-end: SAM1 stub checks logits memory and exact prompts."""
    cfg = Config(model_size=64, device="cpu", refine_iters=2,
                 initial_per_class=2, active_min_dist=2, extra_controls=True,
                 vis_topk=1, vis_min_gain=-1.0)
    b = np.array([38,38,58,60], np.float32)
    reg, lam, z = ocvr_region(b, 256,192,cfg)
    assert lam == 0.5 and 1.0 < z <= 1.5 + 1e-4
    assert reg[0]<=b[0] and reg[2]>b[2] and reg[1]<=b[1] and reg[3]>b[3]
    sc = np.zeros((64,64), np.uint8)
    sc[7:24,7:24]=1
    sc[35:48,35:48]=2
    p,l=initial_points(sc,9)
    assert (l==1).sum()<=9 and (l==0).sum()<=9
    pp=np.full((64,64), .9, np.float32)
    pp[7:24,7:24]=.05
    pt, lb, reason=select_active_scribble_point(pp, sc, np.array([[8,8]],np.float32),
                                                    np.array([0,0,63,63],np.float32),5)
    assert pt is not None and lb==1 and reason=="scribble_conflict"
    assert sc[int(pt[1]),int(pt[0])]==1
    assert np.linalg.norm(pt - np.array([8,8])) > 5
    pp[7:24,7:24]=.9
    pp[35:48,35:48]=.05
    _, _, kind=select_active_scribble_point(pp,sc,None,
                                            np.array([0,0,63,63],np.float32),5)
    assert kind=="scribble_uncertainty"
    assert np.isclose(evaluate_binary(np.zeros((64,64),np.float32),
                                      np.zeros((64,64),np.uint8))["MAE"],0)

    class Stub:
        def __init__(self):
            self.all_calls=[]
        def set_image(self,img):
            self.shape=img.shape[:2]
        def predict(self, **kw):
            assert kw["box"].shape == (4,)
            assert not kw["multimask_output"]
            assert kw["return_logits"] is True
            if kw["mask_input"] is not None:
                assert kw["mask_input"].shape == (1,256,256)
                assert np.all(kw["mask_input"]==2.5), "Must feed prior raw logits, not sigmoid/threshold"
            self.all_calls.append({"mem":kw["mask_input"] is not None,
                                   "pts":0 if kw["point_coords"] is None else len(kw["point_coords"])})
            h,w=self.shape
            pred=np.full((1,h,w),-2.5,np.float32)
            box=np.round(kw["box"]).astype(int)
            xa,ya,xb,yb=max(0,box[0]),max(0,box[1]),min(w,box[2]+1),min(h,box[3]+1)
            pred[:,ya:yb,xa:xb]=2.5
            return pred, np.array([.8],np.float32), np.full((1,256,256),2.5,np.float32)

    with tempfile.TemporaryDirectory(prefix="sam1_gapr_cm_test_") as td:
        for sub in ("Imgs","Scribble","GT","box_label"):
            os.makedirs(os.path.join(td,sub))
        rgb=np.zeros((96,128,3),np.uint8)
        rgb[20:50,18:48]=(100,130,180)
        rgb[57:80,70:99]=(160,100,85)
        cv2.imwrite(os.path.join(td,"Imgs","demo.png"),rgb)
        scribble=np.zeros((96,128),np.uint8)
        scribble[25:32,22:29]=1
        scribble[61:67,75:81]=1
        scribble[33:38,22:29]=2   # deliberate wrong predicted FG -> BG contradiction
        cv2.imwrite(os.path.join(td,"Scribble","demo.png"),scribble)
        gt=np.zeros((96,128),np.uint8)
        gt[20:50,18:48]=255
        gt[57:80,70:99]=255
        cv2.imwrite(os.path.join(td,"GT","demo.png"),gt)
        shapes=[{"points":[[18,20],[48,50]]},{"points":[[70,57],[99,80]]}]
        with open(os.path.join(td,"box_label","demo.json"),"w",encoding="utf-8") as f:
            json.dump({"shapes":shapes},f)
        cfg.img_dir=os.path.join(td,"Imgs")
        cfg.scribble_dir=os.path.join(td,"Scribble")
        cfg.gt_dir=os.path.join(td,"GT")
        cfg.json_dir=os.path.join(td,"box_label")
        cfg.output_dir=os.path.join(td,"Output")
        stub=Stub()
        summary=run(cfg,predictor=stub)
        assert summary["Diagnostics"]["processed"]==1,summary["Failures"]
        for method in CORE_METHODS+CONTROL_METHODS:
            assert os.path.isfile(os.path.join(cfg.output_dir,method,"demo.png"))
        assert any(c["mem"] for c in stub.all_calls)
        assert any(not c["mem"] for c in stub.all_calls)
        assert summary["Diagnostics"]["total_A2_Conflict_MMF_num_memory_calls"]==2
        assert summary["Diagnostics"]["total_A3_GAPR_Conflict_MMF_num_memory_calls"]==4
        import csv as _csv
        with open(os.path.join(cfg.output_dir,"active_points.csv"),encoding="utf-8-sig") as f:
            rows=list(_csv.DictReader(f))
        assert any(r["method"]=="A2_Conflict_MMF" for r in rows)
        assert all(r["reason"] in ("scribble_conflict","scribble_uncertainty") for r in rows)
    print("PASS: GAPR zoom + per-box, conflict-first scribble points, MMF raw-logit chain, 8 outputs")


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sam-checkpoint",default=Config.sam_checkpoint)
    parser.add_argument("--sam-model-type",choices=("vit_h","vit_l","vit_b"),default="vit_h")
    parser.add_argument("--img-dir",default=Config.img_dir)
    parser.add_argument("--scribble-dir",default=Config.scribble_dir)
    parser.add_argument("--gt-dir",default=Config.gt_dir)
    parser.add_argument("--json-dir",default=Config.json_dir)
    parser.add_argument("--output-dir",default=Config.output_dir)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--limit",type=int,default=None)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--refine-iters",type=int,default=2,
                        help="EXTRA decodes after initial, one new scribble point each")
    parser.add_argument("--active-min-dist",type=int,default=16,
                        help="Minimum point spacing in the 1024 SAM model frame")
    parser.add_argument("--initial-per-class",type=int,default=9)
    parser.add_argument("--extra-controls",action="store_true",
                        help="Add Conflict-only and MMF-only controls with/without GAPR")
    parser.add_argument("--vis-topk",type=int,default=40)
    parser.add_argument("--vis-min-gain",type=float,default=0.003)
    parser.add_argument("--save-probs",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.refine_iters<1 or args.active_min_dist<0 or args.initial_per_class<0 or args.vis_topk<0:
        parser.error("refine-iters >=1; spacing, initial points and vis-topk >=0")
    if args.limit is not None and args.limit<0:
        parser.error("--limit must be nonnegative")
    return args


def main():
    args=parse_args()
    if args.self_test:
        self_test()
        return
    cfg=Config(sam_checkpoint=args.sam_checkpoint,sam_model_type=args.sam_model_type,
               img_dir=args.img_dir,scribble_dir=args.scribble_dir,gt_dir=args.gt_dir,
               json_dir=args.json_dir,output_dir=args.output_dir,device=args.device,
               limit=args.limit,seed=args.seed,initial_per_class=args.initial_per_class,
               refine_iters=args.refine_iters,active_min_dist=args.active_min_dist,
               extra_controls=args.extra_controls,
               vis_topk=args.vis_topk,vis_min_gain=args.vis_min_gain,
               save_probs=args.save_probs)
    run(cfg)


if __name__=="__main__":
    main()
