#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 strict Box-only pseudo-label generation for COD.

Only one human annotation is ever passed to SAM2: one bounding box.
No scribble, positive/negative point, GT, or previous-mask prompt is used.
Crop and horizontal-flip views are optional image transformations, not prompts.

Ablations use the SAME per-instance boxes:
    01_Box_Base         : whole-image per-box SAM2
    02_Box_Crop         : conditional small-object crop, per-box SAM2
    03_Box_Crop_Flip    : crop + consistency-gated horizontal flip

Only --eval-gt reads ground truth, strictly AFTER pseudo labels are saved.
Requires SAM2 (facebookresearch/sam2), torch, opencv-python, numpy, tqdm.
For --eval-gt also install: pip install py_sod_metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class Config:
    sam2_checkpoint: str = "./sam_pseado/weights/sam2_hiera_large.pt"
    sam2_config: str = "sam2_hiera_l.yaml"
    img_dir: str = "./data/Train/Imgs"
    json_dir: str = "./data/Train/box_label"
    gt_dir: str = "./data/Train/GT"
    out_dir: str = "./softresult/SAM2_BoxOnly"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    size: int = 1024
    seed: int = 0
    use_crop: bool = True
    small_box_area_thr: float = 0.10
    crop_context: float = 0.35
    crop_min_context: int = 16
    crop_square: bool = True
    flip_gate: bool = True
    flip_min_iou: float = 0.75
    flip_max_area_ratio: float = 1.50
    flip_boundary_radius: int = 7
    flip_base_uncertainty: float = 0.22
    flip_other_confidence: float = 0.20
    flip_alpha: float = 0.60
    multimask: bool = False
    box_support_margin: float = 0.04  # fraction of box's longer side, output merging only
    save_probs: bool = False
    save_vis: bool = False
    eval_gt: bool = False


def list_images(directory: str) -> list[str]:
    paths = []
    for ext in ("jpg", "jpeg", "png", "bmp", "tif", "tiff"):
        paths.extend(glob(os.path.join(directory, f"*.{ext}")))
        paths.extend(glob(os.path.join(directory, f"*.{ext.upper()}")))
    return sorted(set(paths))


def find_named(directory: str, stem: str, suffixes: tuple[str, ...]) -> str | None:
    for ext in suffixes:
        for e in (ext.lower(), ext.upper()):
            candidate = os.path.join(directory, stem + e)
            if os.path.isfile(candidate):
                return candidate
    return None


def load_boxes(json_path: str | None, width: int, height: int) -> list[np.ndarray]:
    if json_path is None:
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    boxes: list[np.ndarray] = []
    # Compatible with the original LabelMe-style JSON 'shapes/points'.
    for shape in obj.get("shapes", []):
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue
        if not np.isfinite(points).all():
            continue
        b = np.array([points[:, 0].min(), points[:, 1].min(),
                      points[:, 0].max(), points[:, 1].max()], dtype=np.float32)
        b[[0, 2]] = np.clip(b[[0, 2]], 0, width - 1)
        b[[1, 3]] = np.clip(b[[1, 3]], 0, height - 1)
        if b[2] > b[0] and b[3] > b[1]:
            boxes.append(b)
    return boxes


def mirror_box(b: np.ndarray, width: int) -> np.ndarray:
    return np.array([width - 1 - b[2], b[1], width - 1 - b[0], b[3]],
                    dtype=np.float32)


def box_crop(b: np.ndarray, width: int, height: int,
             context: float, min_context: int, square: bool) -> tuple[int, int, int, int]:
    bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
    pad = max(float(min_context), max(bw, bh) * context)
    cw, ch = bw + 2 * pad, bh + 2 * pad
    if square:
        cw = ch = max(cw, ch)
    cw, ch = min(width, max(1, int(np.ceil(cw)))), min(height, max(1, int(np.ceil(ch))))
    cx, cy = float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)
    x0 = int(np.clip(round(cx - cw / 2), 0, width - cw))
    y0 = int(np.clip(round(cy - ch / 2), 0, height - ch))
    return x0, y0, x0 + cw, y0 + ch


def model_box(b: np.ndarray, rect: tuple[int, int, int, int], size: int) -> np.ndarray:
    x0, y0, x1, y1 = rect
    sx, sy = size / (x1 - x0), size / (y1 - y0)
    out = np.array([(b[0] - x0 + .5) * sx - .5,
                    (b[1] - y0 + .5) * sy - .5,
                    (b[2] - x0 + .5) * sx - .5,
                    (b[3] - y0 + .5) * sy - .5], dtype=np.float32)
    out[[0, 2]] = np.clip(out[[0, 2]], 0, size - 1)
    out[[1, 3]] = np.clip(out[[1, 3]], 0, size - 1)
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-np.clip(x.astype(np.float32), -30, 30)))


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def support_rect(b: np.ndarray, width: int, height: int, ratio: float) -> tuple[int, int, int, int]:
    pad = max(2., max(float(b[2] - b[0]), float(b[3] - b[1])) * ratio)
    x0 = max(0, int(np.floor(b[0] - pad)))
    y0 = max(0, int(np.floor(b[1] - pad)))
    x1 = min(width, int(np.ceil(b[2] + pad)) + 1)
    y1 = min(height, int(np.ceil(b[3] + pad)) + 1)
    return x0, y0, x1, y1


def limit_to_support(prob: np.ndarray, box: np.ndarray, margin: float) -> np.ndarray:
    h, w = prob.shape
    x0, y0, x1, y1 = support_rect(box, w, h, margin)
    out = np.zeros_like(prob)
    out[y0:y1, x0:x1] = prob[y0:y1, x0:x1]
    return out


def crop_decision(box: np.ndarray, shape: tuple[int, int], cfg: Config) -> bool:
    h, w = shape
    frac = float((box[2] - box[0]) * (box[3] - box[1]) / (w * h))
    return bool(cfg.use_crop and frac <= cfg.small_box_area_thr)


def flip_gate_fusion(p: np.ndarray, q: np.ndarray, cfg: Config) -> tuple[np.ndarray, dict]:
    """Prediction consistency only; does not use scribble or GT.

    Reject large semantic discrepancies; make only local uncertain-boundary updates.
    The gate measures agreement, not ground-truth accuracy.
    """
    p = np.clip(np.asarray(p, np.float32), 0., 1.)
    q = np.clip(np.asarray(q, np.float32), 0., 1.)
    if p.shape != q.shape:
        raise ValueError(f"flip prediction mismatch: {p.shape} vs {q.shape}")
    a, b = p >= .5, q >= .5
    area_a, area_b = int(a.sum()), int(b.sum())
    iou = binary_iou(a, b)
    area_ratio = (area_b + 1) / (area_a + 1)
    info = {"flip_iou": float(iou), "flip_area_ratio": float(area_ratio),
            "flip_accepted": 0, "flip_modified_ratio": 0., "flip_reason": ""}
    if min(area_a, area_b) == 0:
        info["flip_reason"] = "empty_mask"
        return p, info
    if iou < cfg.flip_min_iou:
        info["flip_reason"] = "inconsistent_iou"
        return p, info
    if not 1. / cfg.flip_max_area_ratio <= area_ratio <= cfg.flip_max_area_ratio:
        info["flip_reason"] = "area_mismatch"
        return p, info

    h, w = p.shape
    r = max(1, int(round(cfg.flip_boundary_radius * max(h, w) / 1024)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    boundary = (cv2.dilate(a.astype(np.uint8), k).astype(bool) ^
                cv2.erode(a.astype(np.uint8), k).astype(bool))
    same = (a == b)
    confident_correction = ((~same) & boundary &
                            (np.abs(p - .5) <= cfg.flip_base_uncertainty) &
                            (np.abs(q - .5) >= cfg.flip_other_confidence))
    alpha = np.zeros_like(p)
    alpha[same & boundary] = .35  # soft probability smoothing only
    alpha[confident_correction] = cfg.flip_alpha
    fused = p * (1 - alpha) + q * alpha
    info.update(flip_accepted=1, flip_modified_ratio=float(np.mean(alpha > 0)),
                flip_reason="accepted", flip_correction_ratio=float(np.mean(confident_correction)))
    return np.clip(fused, 0., 1.), info


class SAM2BoxEngine:
    def __init__(self, cfg: Config, predictor=None):
        self.cfg = cfg
        if predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(cfg.sam2_config, cfg.sam2_checkpoint, device=cfg.device)
            predictor = SAM2ImagePredictor(model)
        self.predictor = predictor
        self.cached_rgb = None
        self.cached_rect = None
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def predict_one_box(self, rgb: np.ndarray, b: np.ndarray,
                        use_crop: bool) -> tuple[np.ndarray, dict]:
        """EVERY predict call is box-only; no other prompt of any kind."""
        h, w = rgb.shape[:2]
        if use_crop:
            rect = box_crop(b, w, h, self.cfg.crop_context,
                            self.cfg.crop_min_context, self.cfg.crop_square)
        else:
            rect = (0, 0, w, h)
        x0, y0, x1, y1 = rect
        cropped = rect != (0, 0, w, h)
        if self.cached_rgb is not rgb or self.cached_rect != rect:
            img = cv2.resize(rgb[y0:y1, x0:x1], (self.cfg.size, self.cfg.size),
                             interpolation=cv2.INTER_LINEAR)
            with torch.inference_mode():
                self.predictor.set_image(np.ascontiguousarray(img))
            self.cached_rgb, self.cached_rect = rgb, rect

        b_in = model_box(b, rect, self.cfg.size)
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if self.cfg.device.startswith("cuda") and torch.cuda.is_available()
                   and torch.cuda.is_bf16_supported() else nullcontext())
        with torch.inference_mode(), context:
            logits, iou_preds, _ = self.predictor.predict(
                box=b_in, point_coords=None, point_labels=None, mask_input=None,
                multimask_output=self.cfg.multimask, return_logits=True)
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[None, :, :]
        scores = np.asarray(iou_preds, dtype=np.float32).reshape(-1)
        if logits.ndim != 3 or len(scores) != logits.shape[0]:
            raise RuntimeError(f"SAM2 inconsistent outputs: {logits.shape}, {scores.shape}")
        if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(scores)):
            raise RuntimeError("SAM2 returned NaN/Inf outputs")

        # Optional multiple candidates: select by SAM2's OWN predicted IoU.
        # This does not require GT, a point, or a mask prompt.
        index = int(np.argmax(scores))
        pred = sigmoid(logits[index])
        # Important: logits are resized back to the original spatial resolution,
        # including the crop's exact integer footprint.
        local = cv2.resize(pred, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        full = np.zeros((h, w), np.float32)
        full[y0:y1, x0:x1] = local
        fg = full >= .5
        inside = np.zeros((h, w), bool)
        sx0, sy0, sx1, sy1 = support_rect(b, w, h, self.cfg.box_support_margin)
        inside[sy0:sy1, sx0:sx1] = True
        raw_area = int(np.count_nonzero(fg))
        outside = int(np.count_nonzero(fg & ~inside))
        full = limit_to_support(full, b, self.cfg.box_support_margin)
        diag = {"sam_score": float(scores[index]), "chosen_candidate": index,
                "num_candidates": len(scores), "used_crop": int(cropped),
                "raw_foreground_pixels": raw_area,
                "raw_outside_support_frac": float(outside / max(1, raw_area)),
                "output_foreground_pixels": int((full >= .5).sum())}
        return full, diag


def merge_box_probabilities(preds: list[np.ndarray]) -> np.ndarray:
    if not preds:
        raise ValueError("cannot merge an empty prediction list")
    return np.maximum.reduce(preds)


class Pipeline:
    def __init__(self, cfg: Config, predictor=None):
        self.cfg = cfg
        self.engine = SAM2BoxEngine(cfg, predictor=predictor)
        self.mask_dirs = {
            "base": Path(cfg.out_dir) / "01_Box_Base",
            "crop": Path(cfg.out_dir) / "02_Box_Crop",
            "final": Path(cfg.out_dir) / "03_Box_Crop_Flip",
        }
        self.box_records: list[dict] = []  # one stable image+box ID for future curriculum use
        for d in self.mask_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        if cfg.save_probs:
            for stage in self.mask_dirs:
                (Path(cfg.out_dir) / "soft_prob" / stage).mkdir(parents=True, exist_ok=True)
        if cfg.save_vis:
            (Path(cfg.out_dir) / "visualizations").mkdir(parents=True, exist_ok=True)

    def process_image(self, path: str) -> tuple[dict | None, dict[str, np.ndarray] | None]:
        stem = Path(path).stem
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"unreadable image: {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        jpath = find_named(self.cfg.json_dir, stem, (".json",))
        boxes = load_boxes(jpath, w, h)
        if not boxes:
            raise ValueError("missing or invalid JSON bounding boxes")

        # A0: per-object, whole-image, box-only baseline.
        base_boxes, base_diags = [], []
        for b in boxes:
            p, info = self.engine.predict_one_box(rgb, b, use_crop=False)
            base_boxes.append(p)
            base_diags.append(info)

        # A1: real input crop for SMALL boxes only; no extra point/mask prompt.
        crop_boxes, crop_diags = [], []
        crop_count = 0
        for i, b in enumerate(boxes):
            if crop_decision(b, (h, w), self.cfg):
                p, info = self.engine.predict_one_box(rgb, b, use_crop=True)
            else:
                p, info = base_boxes[i], base_diags[i]
            crop_boxes.append(p)
            crop_diags.append(info)
            crop_count += int(info["used_crop"])

        # A2: independent flip consistency check FOR EACH BOX, then merge.
        final_boxes: list[np.ndarray] = []
        flip_info = []
        if self.cfg.flip_gate:
            flipped_rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            for b, p in zip(boxes, crop_boxes):
                b_flip = mirror_box(b, w)
                do_crop = crop_decision(b_flip, (h, w), self.cfg)
                pred_flip, _ = self.engine.predict_one_box(flipped_rgb, b_flip, use_crop=do_crop)
                aligned = np.ascontiguousarray(pred_flip[:, ::-1])
                merged, info = flip_gate_fusion(p, aligned, self.cfg)
                merged = limit_to_support(merged, b, self.cfg.box_support_margin)
                final_boxes.append(merged)
                flip_info.append(info)
        else:
            final_boxes = crop_boxes
            flip_info = [{"flip_iou": None, "flip_area_ratio": None,
                          "flip_accepted": 0, "flip_modified_ratio": 0.,
                          "flip_reason": "disabled"} for _ in boxes]

        output = {"base": merge_box_probabilities(base_boxes),
                  "crop": merge_box_probabilities(crop_boxes),
                  "final": merge_box_probabilities(final_boxes)}

        # Save before evaluating GT. Neither stage selection nor candidate selection
        # has any access to ground truth.
        for stage, p in output.items():
            binary = (p >= .5).astype(np.uint8) * 255
            dst = self.mask_dirs[stage] / (stem + ".png")
            if not cv2.imwrite(str(dst), binary):
                raise IOError(f"could not write: {dst}")
            if self.cfg.save_probs:
                np.save(Path(self.cfg.out_dir) / "soft_prob" / stage / (stem + ".npy"),
                        p.astype(np.float16))

        crop_agreement = [binary_iou(bp >= .5, cp >= .5)
                          for bp, cp in zip(base_boxes, crop_boxes)]
        for i, (b, b_info, c_info, flip, cp) in enumerate(zip(
                boxes, base_diags, crop_diags, flip_info, crop_boxes)):
            bx0, by0, bx1, by1 = [float(v) for v in b]
            mask_area = int((cp >= .5).sum())
            self.box_records.append({
                "box_id": f"{stem}__box{i}", "base_name": stem, "box_index": i,
                "x1": bx0, "y1": by0, "x2": bx1, "y2": by1,
                "box_area_ratio": (bx1 - bx0) * (by1 - by0) / (w * h),
                "used_crop": c_info["used_crop"],
                "sam_score_base": b_info["sam_score"],
                "sam_score_crop": c_info["sam_score"],
                "chosen_candidate": c_info["chosen_candidate"],
                "mask_foreground_ratio": mask_area / (w * h),
                "mask_to_box_area": mask_area / max(1., (bx1-bx0) * (by1-by0)),
                "outside_support_frac_before_clip": c_info["raw_outside_support_frac"],
                "crop_vs_base_iou": crop_agreement[i],
                "flip_iou": flip["flip_iou"],
                "flip_area_ratio": flip["flip_area_ratio"],
                "flip_accepted": flip["flip_accepted"],
                "flip_reason": flip["flip_reason"],
                "flip_modified_ratio": flip["flip_modified_ratio"],
            })
        record = {
            "image_name": Path(path).name, "base_name": stem,
            "num_boxes": len(boxes), "crop_count": crop_count,
            "avg_box_area_ratio": float(np.mean([
                (b[2]-b[0])*(b[3]-b[1])/(h*w) for b in boxes])),
            "mean_sam_score_base": float(np.mean([x["sam_score"] for x in base_diags])),
            "mean_sam_score_crop": float(np.mean([x["sam_score"] for x in crop_diags])),
            "mean_crop_vs_base_iou": float(np.mean(crop_agreement)),
            "mean_outside_frac_crop": float(np.mean([
                x["raw_outside_support_frac"] for x in crop_diags])),
            "mean_flip_iou": (float(np.mean([x["flip_iou"] for x in flip_info]))
                              if self.cfg.flip_gate else None),
            "num_flip_accepted": int(sum(x["flip_accepted"] for x in flip_info)),
            "num_flip_rejected": int(sum(not x["flip_accepted"] for x in flip_info))
                                 if self.cfg.flip_gate else 0,
            "num_empty_masks_crop": int(sum(x["output_foreground_pixels"] == 0
                                            for x in crop_diags)),
            "flip_reasons": ";".join(x["flip_reason"] for x in flip_info),
        }
        if self.cfg.save_vis:
            self._save_visualization(stem, img, boxes, output)
        return record, output

    def _save_visualization(self, stem, bgr, boxes, output):
        h, w = bgr.shape[:2]
        display = cv2.resize(bgr, (320, 320))
        for b in boxes:
            p0 = (int(b[0]*320/w), int(b[1]*320/h))
            p1 = (int(b[2]*320/w), int(b[3]*320/h))
            cv2.rectangle(display, p0, p1, (0, 255, 255), 2)
        panels = [display]
        for stage, name in (("base", "A0 Base"), ("crop", "A1 Crop"),
                            ("final", "A2 FlipGate")):
            mask = (cv2.resize(output[stage], (320, 320)) >= .5).astype(np.uint8) * 255
            panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(panel, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (0, 0, 255), 2)
            panels.append(panel)
        cv2.imwrite(str(Path(self.cfg.out_dir) / "visualizations" / (stem + ".jpg")),
                    np.concatenate(panels, axis=1))


class OfflineGTMetrics:
    """Official COD/SOD metrics via PySODMetrics. Only constructed with --eval-gt."""
    def __init__(self):
        try:
            from py_sod_metrics import (Fmeasure, Emeasure, Smeasure,
                                        WeightedFmeasure, MAE)
        except ImportError as e:
            raise RuntimeError("--eval-gt needs: pip install py_sod_metrics") from e
        self.collectors = {
            stage: {"S": Smeasure(), "F": Fmeasure(), "E": Emeasure(),
                    "wF": WeightedFmeasure(), "MAE": MAE()}
            for stage in ("base", "crop", "final")
        }
        self.n = 0

    def add(self, gt: np.ndarray, probs: dict[str, np.ndarray], record: dict):
        if gt.ndim != 2:
            raise ValueError("GT must be single-channel")
        for stage, p in probs.items():
            pp = cv2.resize(p, (gt.shape[1], gt.shape[0]),
                            interpolation=cv2.INTER_LINEAR) if p.shape != gt.shape else p
            pred = np.clip(np.round(pp * 255), 0, 255).astype(np.uint8)
            for m in self.collectors[stage].values():
                m.step(pred=pred, gt=gt)
            pb, gb = pp >= .5, gt > 128
            intersection = int(np.count_nonzero(pb & gb))
            union = int(np.count_nonzero(pb | gb))
            record[f"{stage}_IoU_binary"] = (intersection / union if union else 1.)
            record[f"{stage}_Dice_binary"] = (2 * intersection / (int(pb.sum()) + int(gb.sum()))
                                               if pb.any() or gb.any() else 1.)
            record[f"{stage}_MAE_soft"] = float(np.mean(np.abs(pp - gb.astype(np.float32))))
        record["delta_MAE_base_minus_final"] = record["base_MAE_soft"] - record["final_MAE_soft"]
        self.n += 1

    def summary(self) -> dict:
        if self.n == 0:
            return {"num_gt_evaluated": 0}
        result = {"num_gt_evaluated": self.n}
        for stage, m in self.collectors.items():
            f = m["F"].get_results()["fm"]
            e = m["E"].get_results()["em"]
            result[stage] = {
                "Smeasure": float(m["S"].get_results()["sm"]),
                "maxF": float(np.max(f["curve"])),
                "meanF": float(np.mean(f["curve"])),
                "maxE": float(np.max(e["curve"])),
                "meanE": float(np.mean(e["curve"])),
                "wFmeasure": float(m["wF"].get_results()["wfm"]),
                "MAE": float(m["MAE"].get_results()["mae"]),
            }
        return result


def self_test():
    cfg = Config(device="cpu", size=64, use_crop=True, flip_gate=True)
    class StubPredictor:
        def __init__(self):
            self.calls = []
            self.img_shape = None
        def set_image(self, image):
            self.img_shape = image.shape[:2]
        def predict(self, **kwargs):
            assert kwargs["point_coords"] is None
            assert kwargs["point_labels"] is None
            assert kwargs["mask_input"] is None
            assert kwargs["box"] is not None
            self.calls.append(kwargs)
            h, w = self.img_shape
            x0, y0, x1, y1 = map(int, kwargs["box"].round())
            logits = np.full((1, h, w), -10, dtype=np.float32)
            logits[0, max(0,y0):min(h,y1+1), max(0,x0):min(w,x1+1)] = 10.
            return logits, np.array([.95], np.float32), None
    stub = StubPredictor()
    engine = SAM2BoxEngine(cfg, predictor=stub)
    rgb = np.zeros((100, 160, 3), np.uint8)
    box = np.array([15, 18, 40, 43], np.float32)
    p0, d0 = engine.predict_one_box(rgb, box, use_crop=False)
    p1, d1 = engine.predict_one_box(rgb, box, use_crop=True)
    assert p0.shape == p1.shape == (100, 160)
    assert d1["used_crop"] == 1
    assert np.array_equal(mirror_box(mirror_box(box, 160), 160), box)
    p, info = flip_gate_fusion(p1, p1.copy(), cfg)
    assert info["flip_accepted"] == 1 and np.allclose(p, p1)
    q = np.zeros_like(p1)
    _, info = flip_gate_fusion(p1, q, cfg)
    assert info["flip_accepted"] == 0
    assert len(stub.calls) == 2
    print("PASS: strict box-only calls, crop coordinates, flip gate, no extra prompts")


def parse_args() -> argparse.Namespace:
    c = Config()
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--sam2-checkpoint", default=c.sam2_checkpoint)
    pa.add_argument("--sam2-config", default=c.sam2_config)
    pa.add_argument("--img-dir", default=c.img_dir)
    pa.add_argument("--json-dir", default=c.json_dir)
    pa.add_argument("--gt-dir", default=c.gt_dir)
    pa.add_argument("--output-dir", default=c.out_dir)
    pa.add_argument("--device", default=c.device)
    pa.add_argument("--size", type=int, default=c.size)
    pa.add_argument("--small-box-area-thr", type=float, default=c.small_box_area_thr)
    pa.add_argument("--crop-context", type=float, default=c.crop_context)
    pa.add_argument("--crop-min-context", type=int, default=c.crop_min_context)
    pa.add_argument("--box-support-margin", type=float, default=c.box_support_margin)
    pa.add_argument("--flip-min-iou", type=float, default=c.flip_min_iou)
    pa.add_argument("--flip-max-area-ratio", type=float, default=c.flip_max_area_ratio)
    pa.add_argument("--no-crop", action="store_true")
    pa.add_argument("--no-flip-gate", action="store_true")
    pa.add_argument("--multimask", action="store_true", help="SAM2 3 box candidates; select by predicted IoU")
    pa.add_argument("--save-probs", action="store_true")
    pa.add_argument("--save-vis", action="store_true")
    pa.add_argument("--eval-gt", action="store_true", help="offline ONLY; never used for SAM prompts")
    pa.add_argument("--limit", type=int, default=None)
    pa.add_argument("--self-test", action="store_true")
    return pa.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    cfg = Config(
        sam2_checkpoint=args.sam2_checkpoint, sam2_config=args.sam2_config,
        img_dir=args.img_dir, json_dir=args.json_dir, gt_dir=args.gt_dir,
        out_dir=args.output_dir, device=args.device, size=args.size,
        small_box_area_thr=args.small_box_area_thr, crop_context=args.crop_context,
        crop_min_context=args.crop_min_context, box_support_margin=args.box_support_margin,
        flip_min_iou=args.flip_min_iou, flip_max_area_ratio=args.flip_max_area_ratio,
        use_crop=not args.no_crop, flip_gate=not args.no_flip_gate,
        multimask=args.multimask, save_probs=args.save_probs,
        save_vis=args.save_vis, eval_gt=args.eval_gt)
    if not 0. <= cfg.small_box_area_thr <= 1.:
        raise ValueError("--small-box-area-thr must be in [0,1]")
    if not 0 <= cfg.flip_min_iou <= 1 or cfg.flip_max_area_ratio < 1:
        raise ValueError("invalid flip gate thresholds")
    if cfg.crop_context < 0 or cfg.crop_min_context < 0 or cfg.box_support_margin < 0:
        raise ValueError("context and margin must be nonnegative")
    if cfg.size < 16:
        raise ValueError("--size is too small")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; use --device cpu")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Fail fast if requested metric package is unavailable.
    evaluator = OfflineGTMetrics() if cfg.eval_gt else None
    pipeline = Pipeline(cfg)
    paths = list_images(cfg.img_dir)
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(f"no image found under {cfg.img_dir}")
    rows, failures, missing_gt = [], [], 0
    for path in tqdm(paths, desc="SAM2 strict Box-only"):
        try:
            row, outputs = pipeline.process_image(path)
            if row is None or outputs is None:
                continue
            if evaluator is not None:
                gt_path = find_named(cfg.gt_dir, row["base_name"], (".png", ".jpg", ".bmp"))
                if gt_path is not None:
                    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    if gt is not None:
                        evaluator.add(gt, outputs, row)
                    else:
                        missing_gt += 1
                else:
                    missing_gt += 1
            rows.append(row)
        except Exception as exc:
            failures.append({"image": Path(path).name, "error": repr(exc)})
            print(f"[ERROR] {Path(path).name}: {exc}")

    root = Path(cfg.out_dir)
    if pipeline.box_records:
        with (root / "box_only_per_box_reliability.csv").open(
                "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(pipeline.box_records[0].keys()))
            writer.writeheader()
            writer.writerows(pipeline.box_records)
    if rows:
        fields = list(dict.fromkeys(k for item in rows for k in item))
        with (root / "box_only_image_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "config": asdict(cfg),
        "processed": len(rows), "failed": len(failures),
        "missing_gt": missing_gt,
        "crop_images": int(sum(r["crop_count"] > 0 for r in rows)),
        "num_crop_boxes": int(sum(r["crop_count"] for r in rows)),
        "num_flip_accepted": int(sum(r["num_flip_accepted"] for r in rows)),
        "num_flip_rejected": int(sum(r["num_flip_rejected"] for r in rows)),
        "gt_metrics": evaluator.summary() if evaluator else None,
        "metric_note": "GT never used to generate/select pseudo labels; official maxF/maxE are evaluated from SOFT predictions via PySODMetrics.",
        "failures": failures,
    }
    with (root / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "failures")},
                     indent=2, ensure_ascii=False))
    print("IMAGE CSV:", root / "box_only_image_metrics.csv")
    print("PER-BOX CSV:", root / "box_only_per_box_reliability.csv")
    print("SUMMARY:", root / "summary_metrics.json")


if __name__ == "__main__":
    main()
