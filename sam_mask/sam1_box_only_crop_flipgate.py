#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM1 (ViT-H) box-only pseudo-label generation for COD.

Only annotated boxes are sent as SAM prompts. No scribble, GT, point, or
mask-input prompts are used, including on the crop / flip branches.

Three outputs support fair ablations:
  Base            : one full-image pass with the union box
  CropPerBox      : one pass per box, optional crop for small objects
  CropPerBoxFlip  : optional horizontal flip, consistency-gated fusion

GT files are OPTIONAL and used ONLY for offline evaluation/visualization.

Examples:
    python sam1_box_only_crop_flipgate.py --self-test
    python sam1_box_only_crop_flipgate.py --limit 10 --no-flip-gate
    python sam1_box_only_crop_flipgate.py
    python sam1_box_only_crop_flipgate.py --no-crop --no-flip-gate
"""

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm


SAM_CHECKPOINT = "./sam_mask/weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"


@dataclass
class Config:
    sam_checkpoint: str = SAM_CHECKPOINT
    sam_model_type: str = SAM_MODEL_TYPE
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir: str = "./data/Train/Imgs"
    gt_dir: str = "./data/Train/GT"  # used only for evaluation
    json_dir: str = "./data/Train/box_label"
    out_dir: str = "./softresult/SAM1_BoxOnly"
    size: int = 1024
    seed: int = 0

    use_crop: bool = True
    small_box_area_thr: float = 0.10
    crop_context: float = 0.35
    crop_min_context: int = 16
    crop_square: bool = True
    box_prompt_margin: int = 5  # pixels in SAM input coordinates
    multi_box_merge_margin: int = 25  # pixels in SAM input coordinates

    flip_gate: bool = True
    flip_mode: str = "gate"  # gate | mean
    flip_min_iou: float = 0.75
    flip_max_area_ratio: float = 1.5
    flip_boundary_radius: int = 7
    flip_base_uncertainty: float = 0.22
    flip_other_confidence: float = 0.20
    flip_alpha: float = 0.60

    save_probs: bool = False
    vis_mae_thr: float = 0.030
    vis_gain_thr: float = 0.003

    @property
    def base_dir(self):
        return os.path.join(self.out_dir, "01_BoxBase")

    @property
    def refine_dir(self):
        return os.path.join(self.out_dir, "02_BoxCropPerObject")

    @property
    def profile_dir(self):
        return os.path.join(self.out_dir, "03_BoxCropFlipGate")

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
    def find_file(directory, basename, extensions):
        for ext in extensions:
            for candidate in (ext.lower(), ext.upper()):
                path = os.path.join(directory, basename + candidate)
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
        # Matches the source LabelMe JSON schema: shapes[*].points.
        for shape in data.get("shapes", []):
            points = np.asarray(shape.get("points", []), dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
                continue
            box = np.array([points[:, 0].min(), points[:, 1].min(),
                            points[:, 0].max(), points[:, 1].max()], dtype=np.float32)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
            if np.all(np.isfinite(box)) and box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        return boxes


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float32), -30, 30)))


def resize_prob(prob, width, height):
    prob = np.asarray(prob, np.float32)
    if prob.shape == (height, width):
        return prob.copy()
    return cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)


def mirror_box(box, width):
    x1, y1, x2, y2 = np.asarray(box, np.float32)
    return np.array([width - 1 - x2, y1, width - 1 - x1, y2], np.float32)


def union_box(boxes):
    a = np.asarray(boxes, np.float32)
    return np.array([a[:, 0].min(), a[:, 1].min(),
                     a[:, 2].max(), a[:, 3].max()], np.float32)


def clip_box(box, width, height):
    b = np.asarray(box, np.float32).copy()
    b[[0, 2]] = np.clip(b[[0, 2]], 0, width - 1)
    b[[1, 3]] = np.clip(b[[1, 3]], 0, height - 1)
    return b


def expand_box(box, pad, width, height):
    return clip_box(np.asarray(box, np.float32) +
                    np.array([-pad, -pad, pad, pad], np.float32), width, height)


def centered_square_crop(box, width, height, context, min_context, square=True):
    """Returns half-open [x0,y0,x1,y1] in original image coordinates."""
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = max(float(min_context), max(bw, bh) * float(context))
    crop_w, crop_h = bw + 2 * pad, bh + 2 * pad
    if square:
        crop_w = crop_h = max(crop_w, crop_h)
    crop_w = min(width, int(np.ceil(crop_w)))
    crop_h = min(height, int(np.ceil(crop_h)))
    cx, cy = (x1 + x2) * .5, (y1 + y2) * .5
    x0 = int(np.clip(np.round(cx - crop_w / 2), 0, width - crop_w))
    y0 = int(np.clip(np.round(cy - crop_h / 2), 0, height - crop_h))
    return x0, y0, x0 + crop_w, y0 + crop_h


def transform_box_to_model(box, region, model_size, margin):
    x0, y0, x1, y1 = region
    sx, sy = model_size / (x1 - x0), model_size / (y1 - y0)
    b = np.asarray(box, np.float32)
    model_box = np.array([(b[0] - x0 + .5) * sx - .5 - margin,
                          (b[1] - y0 + .5) * sy - .5 - margin,
                          (b[2] - x0 + .5) * sx - .5 + margin,
                          (b[3] - y0 + .5) * sy - .5 + margin], np.float32)
    return clip_box(model_box, model_size, model_size)


def binary_iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def flip_gate_fusion(original_prob, aligned_flip_prob, cfg):
    """GT-free, scribble-free consistency gate. No prompts are generated."""
    p = np.clip(np.asarray(original_prob, np.float32), 0., 1.)
    q = np.clip(np.asarray(aligned_flip_prob, np.float32), 0., 1.)
    if p.shape != q.shape:
        raise ValueError(f"flip shape mismatch {p.shape} vs {q.shape}")
    a, b = p >= .5, q >= .5
    area_a, area_b = int(a.sum()), int(b.sum())
    area_ratio = float((area_b + 1) / (area_a + 1))
    iou = binary_iou(a, b)
    info = {"flip_iou": iou, "flip_area_ratio": area_ratio,
            "flip_accepted": 0, "flip_modified_ratio": 0., "flip_reason": "disabled"}
    if cfg.flip_mode == "mean":
        info.update(flip_accepted=1, flip_reason="naive_mean_ablation",
                    flip_modified_ratio=float(np.mean(np.abs(p - q) > .01)))
        return np.clip(.5 * (p + q), 0., 1.), info
    if min(area_a, area_b) == 0:
        info["flip_reason"] = "empty_foreground"
        return p, info
    if iou < cfg.flip_min_iou:
        info["flip_reason"] = "low_iou"
        return p, info
    if not (1. / cfg.flip_max_area_ratio <= area_ratio <= cfg.flip_max_area_ratio):
        info["flip_reason"] = "area_mismatch"
        return p, info

    h, w = p.shape
    r = max(1, int(round(cfg.flip_boundary_radius * max(h, w) / 1024.)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near_boundary = (cv2.dilate(a.astype(np.uint8), kernel).astype(bool) ^
                     cv2.erode(a.astype(np.uint8), kernel).astype(bool))
    same = (a == b)
    edge_confident = ((np.abs(p - .5) <= cfg.flip_base_uncertainty) &
                      (np.abs(q - .5) >= cfg.flip_other_confidence) & near_boundary)
    correction = (~same) & edge_confident
    mixing = np.zeros_like(p, dtype=np.float32)
    mixing[same & near_boundary] = .35
    mixing[correction] = cfg.flip_alpha
    fused = p * (1. - mixing) + q * mixing
    info.update(flip_accepted=1, flip_reason="accepted",
                flip_modified_ratio=float(np.mean(mixing > 0)),
                flip_correction_ratio=float(np.mean(correction)))
    return np.clip(fused, 0., 1.), info


class MetricsEvaluator:
    """Preserves source script's binary-mask S/E/F/MAE definitions for comparison.

    'maxF' / 'maxE' are legacy aliases, NOT true max-over-threshold metrics.
    """

    @staticmethod
    def prepare_pred_gt(pred, gt):
        pred, gt = np.asarray(pred), np.asarray(gt)
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
        return u * cls.object_score(p[gt == 1]) + (1 - u) * cls.object_score(1 - p[gt == 0])

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
        if len(xs) == 0:
            return 0.
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
            s = float(np.clip(.5*cls.s_object(p, gt) + .5*cls.s_region(p, gt), 0, 1))
        binary, gtb = p >= .5, gt >= .5
        tp = float((binary & gtb).sum())
        precision = tp / (float(binary.sum()) + 1e-8)
        recall = tp / (float(gtb.sum()) + 1e-8)
        fb = float(1.3 * precision * recall / (.3 * precision + recall + 1e-8))
        if gtb.sum() == 0:
            eb = 1 - float(binary.mean())
        elif gtb.sum() == gtb.size:
            eb = float(binary.mean())
        else:
            fm, gtm = binary.astype(np.float32), gt.astype(np.float32)
            fm -= fm.mean()
            gtm -= gtm.mean()
            alignment = 2*fm*gtm / (fm*fm + gtm*gtm + 1e-8)
            eb = float(np.mean((alignment+1)**2 / 4))
        return {"Smeasure": s, "maxE": eb, "maxF": fb,
                "E_binary_0p5": eb, "F_binary_0p5": fb,
                "MAE": float(np.mean(np.abs(p-gt)))}


class SAM1Engine:
    """The ONLY SAM1 prompt anywhere in this pipeline is ``box``."""

    def __init__(self, cfg, predictor=None):
        self.cfg = cfg
        self.last_input_shape = None
        if predictor is None:
            if not os.path.isfile(cfg.sam_checkpoint):
                raise FileNotFoundError(f"SAM1 checkpoint not found: {cfg.sam_checkpoint}")
            try:
                from segment_anything import sam_model_registry, SamPredictor
            except ImportError as e:
                raise ImportError("Install SAM1: pip install git+https://github.com/facebookresearch/segment-anything.git") from e
            if cfg.sam_model_type not in sam_model_registry:
                raise ValueError(f"Unknown SAM1 model type: {cfg.sam_model_type}")
            sam = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
            sam.to(device=cfg.device)
            sam.eval()
            predictor = SamPredictor(sam)
        self.predictor = predictor

    def set_image(self, rgb):
        self.last_input_shape = rgb.shape[:2]
        with torch.inference_mode():
            self.predictor.set_image(np.ascontiguousarray(rgb))

    def predict_box(self, box):
        if box is None or len(box) != 4:
            raise ValueError("A single box [x1,y1,x2,y2] is required")
        with torch.inference_mode():
            # SAM1's box takes shape (4,), NOT SAM2's (1,4) convention.
            high_logits, iou_scores, low_logits = self.predictor.predict(
                box=np.asarray(box, dtype=np.float32),
                multimask_output=False,
                return_logits=True,
            )
        high_logits = np.asarray(high_logits, np.float32)
        if high_logits.ndim != 3 or high_logits.shape[0] != 1:
            raise ValueError(f"SAM1 expected 1xHxW high-res logits, got {high_logits.shape}")
        return sigmoid(high_logits[0]), float(np.asarray(iou_scores).reshape(-1)[0])


class Pipeline:
    def __init__(self, cfg, predictor=None):
        self.cfg = cfg
        self.engine = SAM1Engine(cfg, predictor=predictor)
        self._cached_rgb = None
        self._cached_region = None
        for d in (cfg.out_dir, cfg.base_dir, cfg.refine_dir, cfg.profile_dir, cfg.vis_dir):
            os.makedirs(d, exist_ok=True)
        if cfg.save_probs:
            for d in ("base_prob", "refine_prob", "flipgate_prob"):
                os.makedirs(os.path.join(cfg.out_dir, d), exist_ok=True)

    def _infer_single_box(self, rgb, box, do_crop):
        h, w = rgb.shape[:2]
        bw, bh = float(box[2]-box[0]), float(box[3]-box[1])
        area_ratio = max(0., bw * bh / (w*h))
        cropped = bool(do_crop and self.cfg.use_crop and area_ratio <= self.cfg.small_box_area_thr)
        region = (centered_square_crop(box, w, h,
                                      self.cfg.crop_context, self.cfg.crop_min_context,
                                      self.cfg.crop_square)
                  if cropped else (0, 0, w, h))
        if region == (0, 0, w, h):
            cropped = False
        x0, y0, x1, y1 = region
        if self._cached_rgb is not rgb or self._cached_region != region:
            model_rgb = cv2.resize(rgb[y0:y1, x0:x1],
                                   (self.cfg.size, self.cfg.size), interpolation=cv2.INTER_LINEAR)
            self.engine.set_image(model_rgb)
            self._cached_rgb, self._cached_region = rgb, region
        model_box = transform_box_to_model(box, region, self.cfg.size,
                                           self.cfg.box_prompt_margin)
        model_prob, score = self.engine.predict_box(model_box)
        prob_orig = np.zeros((h, w), np.float32)
        prob_orig[y0:y1, x0:x1] = resize_prob(model_prob, x1-x0, y1-y0)
        return prob_orig, cropped, score

    def _infer_refined(self, rgb, boxes):
        if len(boxes) == 1:
            prob, cropped, score = self._infer_single_box(rgb, boxes[0], do_crop=True)
            return prob, ("SO_CROP" if cropped else "NORMAL"), int(cropped), score

        h, w = rgb.shape[:2]
        merged = np.zeros((h, w), np.float32)
        crops, scores = 0, []
        for box in boxes:
            prob, cropped, score = self._infer_single_box(rgb, box, do_crop=True)
            crops += int(cropped)
            scores.append(score)
            margin = max(2, round(self.cfg.multi_box_merge_margin * max(h, w)/self.cfg.size))
            gatebox = expand_box(box, margin, w, h)
            x1, y1, x2, y2 = map(int, np.round(gatebox))
            gate = np.zeros((h, w), np.float32)
            gate[y1:y2+1, x1:x2+1] = 1.
            merged = np.maximum(merged, prob * gate)
        return merged, "MO_PER_BOX", crops, float(np.mean(scores))

    def _save_mask(self, directory, basename, prob):
        path = os.path.join(directory, basename + ".png")
        ok = cv2.imwrite(path, (prob >= .5).astype(np.uint8) * 255)
        if not ok:
            raise IOError(f"Cannot write mask: {path}")

    def process_image(self, image_path):
        cfg = self.cfg
        basename = os.path.splitext(os.path.basename(image_path))[0]
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        box_json = DataLoader.find_file(cfg.json_dir, basename, (".json",))
        boxes = DataLoader.load_boxes(box_json, w, h)
        if not boxes:
            return None
        gt_path = DataLoader.find_file(cfg.gt_dir, basename, (".png", ".jpg"))

        # Ablation A: entire image, one UNION BOX; no crop, no flip.
        base, _, base_score = self._infer_single_box(rgb, union_box(boxes), do_crop=False)
        # Ablation B: per box + optional real crop; still BOX-ONLY.
        refined, route, n_crops, refine_score = self._infer_refined(rgb, boxes)
        final = refined
        flip_info = {"flip_iou": None, "flip_area_ratio": None,
                     "flip_accepted": 0, "flip_modified_ratio": 0., "flip_reason": "disabled"}
        if cfg.flip_gate:
            rgb_flip = np.ascontiguousarray(rgb[:, ::-1, :])
            boxes_flip = [mirror_box(b, w) for b in boxes]
            flipped, flip_route, _, _ = self._infer_refined(rgb_flip, boxes_flip)
            if flip_route != route:
                print(f"[WARN] flip changed crop route: {basename} {route} -> {flip_route}")
            aligned = np.ascontiguousarray(flipped[:, ::-1])
            final, flip_info = flip_gate_fusion(refined, aligned, cfg)

        self._save_mask(cfg.base_dir, basename, base)
        self._save_mask(cfg.refine_dir, basename, refined)
        self._save_mask(cfg.profile_dir, basename, final)
        if cfg.save_probs:
            for subdir, arr in (("base_prob", base), ("refine_prob", refined),
                                ("flipgate_prob", final)):
                np.save(os.path.join(cfg.out_dir, subdir, basename + ".npy"),
                        arr.astype(np.float16))

        box_union = union_box(boxes)
        record = {"image_name": os.path.basename(image_path), "base_name": basename,
                  "route_type": route, "num_boxes": len(boxes), "has_box": 1,
                  "box_area_ratio": float((box_union[2]-box_union[0]) *
                                          (box_union[3]-box_union[1]) / (w*h)),
                  "crop_count": n_crops, "base_pred_iou": base_score,
                  "refine_pred_iou": refine_score, **flip_info}
        if gt_path:
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                # Hard masks: same metric protocol as the provided SAM2 file.
                results = {}
                for name, prob in (("base", base), ("refine", refined), ("profile", final)):
                    results[name] = MetricsEvaluator.evaluate((prob >= .5).astype(np.uint8)*255, gt)
                    for k, v in results[name].items():
                        record[f"{name}_{k}"] = v
                record["delta_S_new_minus_base"] = (results["profile"]["Smeasure"] -
                                                    results["base"]["Smeasure"])
                record["delta_E_new_minus_base"] = (results["profile"]["maxE"] -
                                                    results["base"]["maxE"])
                record["delta_F_new_minus_base"] = (results["profile"]["maxF"] -
                                                    results["base"]["maxF"])
                record["delta_MAE_base_minus_new"] = (results["base"]["MAE"] -
                                                      results["profile"]["MAE"])
                record["delta_MAE_refine_minus_flipgate"] = (results["refine"]["MAE"] -
                                                             results["profile"]["MAE"])
                if (route != "NORMAL" or results["base"]["MAE"] > cfg.vis_mae_thr or
                        abs(record["delta_MAE_base_minus_new"]) > cfg.vis_gain_thr):
                    self._save_vis(image_bgr, gt, base, refined, final, basename, route, results)
        return record

    def _save_vis(self, image_bgr, gt, base, refined, final, basename, route, results):
        size = 256
        photo = cv2.resize(image_bgr, (size, size))
        cv2.putText(photo, route, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
        panels = [photo]
        for label, arr, key in (("GT", gt, None), ("BoxBase", base, "base"),
                                ("BoxCrop", refined, "refine"),
                                ("FlipGate", final, "profile")):
            if key is not None:
                arr = (arr >= .5).astype(np.uint8)*255
            panel = cv2.cvtColor(cv2.resize(arr, (size, size),
                                              interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            suffix = f" {results[key]['MAE']:.4f}" if key else ""
            cv2.putText(panel, label+suffix, (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, (0, 0, 255), 1)
            panels.append(panel)
        cv2.imwrite(os.path.join(self.cfg.vis_dir, basename + "_abl.png"),
                    np.concatenate(panels, axis=1))


def summarize(records, prefix):
    out = {}
    for k in ("Smeasure", "maxE", "maxF", "MAE"):
        arr = np.asarray([r.get(f"{prefix}_{k}", np.nan) for r in records], np.float64)
        arr = arr[np.isfinite(arr)]
        out[k] = float(arr.mean()) if len(arr) else None
    return out


def self_test():
    cfg = Config(size=64)
    b = np.array([12, 7, 22, 17], np.float32)
    assert np.allclose(mirror_box(mirror_box(b, 80), 80), b)
    roi = centered_square_crop(b, 160, 100, .30, 8)
    assert roi[2]-roi[0] < 160 and roi[3]-roi[1] < 100
    bm = transform_box_to_model(b, roi, 64, 5)
    assert bm[2] > bm[0] and bm[3] > bm[1]
    p = np.full((64, 64), .03, np.float32)
    q = p.copy()
    p[10:35, 10:35] = .92
    q[35:60, 35:60] = .92
    out, info = flip_gate_fusion(p, q, cfg)
    assert info["flip_accepted"] == 0 and np.array_equal(out, p)
    out, info = flip_gate_fusion(p, p.copy(), cfg)
    assert info["flip_accepted"] == 1 and np.allclose(out, p)

    class StubPredictor:
        def set_image(self, rgb):
            self.image_shape = rgb.shape[:2]
        def predict(self, **kwargs):
            assert set(kwargs) == {"box", "multimask_output", "return_logits"}
            assert kwargs["box"].shape == (4,)
            assert kwargs["multimask_output"] is False
            assert kwargs["return_logits"] is True
            h, w = self.image_shape
            return np.full((1, h, w), -2., np.float32), np.array([.8]), np.zeros((1, 256, 256))

    engine = SAM1Engine(cfg, predictor=StubPredictor())
    engine.set_image(np.zeros((64, 64, 3), np.uint8))
    probability, score = engine.predict_box(bm)
    assert probability.shape == (64, 64) and np.isclose(score, .8)
    # Evaluator works with GT and does not affect inference.
    metrics = MetricsEvaluator.evaluate(np.zeros((64, 64), np.uint8),
                                        np.zeros((64, 64), np.uint8))
    assert np.isfinite(metrics["MAE"])
    print("PASS: SAM1 box-only API, crop, flip gate, GT metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sam-checkpoint", default=SAM_CHECKPOINT)
    parser.add_argument("--sam-model-type", default=SAM_MODEL_TYPE)
    parser.add_argument("--img-dir", default=Config.img_dir)
    parser.add_argument("--gt-dir", default=Config.gt_dir)
    parser.add_argument("--json-dir", default=Config.json_dir)
    parser.add_argument("--output-dir", default=Config.out_dir)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--no-flip-gate", action="store_true")
    parser.add_argument("--flip-mode", choices=("gate", "mean"), default="gate")
    parser.add_argument("--flip-min-iou", type=float, default=.75)
    parser.add_argument("--flip-max-area-ratio", type=float, default=1.5)
    parser.add_argument("--small-box-area-thr", type=float, default=.10)
    parser.add_argument("--crop-context", type=float, default=.35)
    parser.add_argument("--save-probs", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    cfg = Config(sam_checkpoint=args.sam_checkpoint, sam_model_type=args.sam_model_type,
                 img_dir=args.img_dir, gt_dir=args.gt_dir, json_dir=args.json_dir,
                 out_dir=args.output_dir, device=args.device,
                 use_crop=not args.no_crop, flip_gate=not args.no_flip_gate,
                 flip_mode=args.flip_mode, flip_min_iou=args.flip_min_iou,
                 flip_max_area_ratio=args.flip_max_area_ratio,
                 small_box_area_thr=args.small_box_area_thr,
                 crop_context=args.crop_context, save_probs=args.save_probs)
    if not 0 <= cfg.small_box_area_thr <= 1:
        parser.error("--small-box-area-thr must be in [0,1]")
    if not 0 <= cfg.flip_min_iou <= 1:
        parser.error("--flip-min-iou must be in [0,1]")
    if cfg.flip_max_area_ratio < 1:
        parser.error("--flip-max-area-ratio must be >= 1")
    if cfg.crop_context < 0:
        parser.error("--crop-context must be >= 0")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pipeline = Pipeline(cfg)
    images = DataLoader.list_images(cfg.img_dir)
    if args.limit is not None:
        images = images[:args.limit]
    records, failures, skipped = [], [], []
    for path in tqdm(images, desc="SAM1 box-only"):
        try:
            record = pipeline.process_image(path)
            if record is None:
                skipped.append(os.path.basename(path))
            else:
                records.append(record)
        except Exception as exc:
            failures.append({"image": path, "error": repr(exc)})
            print(f"[ERROR] {os.path.basename(path)}: {exc}")

    csv_path = os.path.join(cfg.out_dir, "sam1_box_only_metrics.csv")
    summary_path = os.path.join(cfg.out_dir, "summary_metrics.json")
    if records:
        fieldnames = list(dict.fromkeys(key for row in records for key in row))
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "BoxBase": summarize(records, "base"),
        "BoxCropPerObject": summarize(records, "refine"),
        "BoxCropFlipGate": summarize(records, "profile"),
        "RouteBreakdown": {
            route: {"num_images": len(sub), "profile": summarize(sub, "profile")}
            for route in ("NORMAL", "SO_CROP", "MO_PER_BOX")
            for sub in [[r for r in records if r["route_type"] == route]]
        },
        "Diagnostics": {"num_images": len(records), "num_failures": len(failures),
                        "num_skipped_no_valid_box_or_image": len(skipped),
                        "flip_accepted": sum(r["flip_accepted"] for r in records),
                        "flip_rejected": sum(cfg.flip_gate and not r["flip_accepted"] for r in records),
                        "num_cropped_boxes": sum(r["crop_count"] for r in records)},
        "MetricNote": "Legacy maxE/maxF are binary E/F at threshold 0.5, NOT true maximum E/F; GT evaluation ONLY.",
        "PromptNote": "Only box prompts; no points, scribbles, masks, or GT used as SAM inputs.",
        "Config": vars(cfg),
        "Failures": failures,
        "Skipped": skipped,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("=" * 70)
    for name in ("BoxBase", "BoxCropPerObject", "BoxCropFlipGate"):
        metrics = summary[name]
        result = " ".join(f"{k}={metrics[k]:.4f}" for k in ("Smeasure", "maxE", "maxF", "MAE")
                          if metrics[k] is not None)
        print(f"{name:>20}: {result if result else 'No GT available'}")
    print("Diagnostics:", summary["Diagnostics"])
    print("CSV:", csv_path if records else "no valid records")
    print("Summary:", summary_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
