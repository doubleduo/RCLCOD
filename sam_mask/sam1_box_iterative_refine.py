#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM1 Box-only -> automatic consensus points -> 1/2 mask-memory refinements.

Place BESIDE sam1_box_only_crop_flipgate.py in sam_mask/.
Uses its Config / DataLoader / SAM1Engine / crop helpers / metric evaluator.

NO scribbles, GT masks, or manual points are used as prompts.  GT is evaluation only.
For fair comparison, Iter0, Iter1, and Iter2 use the SAME crop, box, and init.
SAM1 SamPredictor predicts on each original-aspect-ratio frame by default.

Examples (from RCLCOD root):
  python sam_mask/sam1_box_iterative_refine.py --self-test
  python sam_mask/sam1_box_iterative_refine.py --limit 20 --save-probs \\
       --output-dir ./softresult/SAM1_Iter_Test20
  python sam_mask/sam1_box_iterative_refine.py \\
       --output-dir ./softresult/SAM1_Iter_4040
  python sam_mask/sam1_box_iterative_refine.py --no-auto-points \\
       --output-dir ./softresult/SAM1_MemoryOnly
  python sam_mask/sam1_box_iterative_refine.py --no-memory \\
       --output-dir ./softresult/SAM1_PointOnly
  python sam_mask/sam1_box_iterative_refine.py --init single \\
       --input-mode square --box-margin 5 \\
       --output-dir ./softresult/SAM1_LegacyInput_Iter

The 3 candidate masks are used only to initialize and automatically sample points;
each refinement iteration uses multimask_output=False, with mask_input set to
previous low-resolution logits unless --no-memory is specified.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import sam1_box_only_crop_flipgate as legacy


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float32), -30.0, 30.0)))


def box_binary(shape, box, pad=0):
    """Inclusive-ish XYXY box to image-mask pixels, clipped to image bounds."""
    h, w = shape
    b = np.asarray(box, np.float32)
    x0 = int(np.clip(np.floor(b[0] - pad), 0, w))
    y0 = int(np.clip(np.floor(b[1] - pad), 0, h))
    x1 = int(np.clip(np.ceil(b[2] + pad) + 1, 0, w))
    y1 = int(np.clip(np.ceil(b[3] + pad) + 1, 0, h))
    out = np.zeros((h, w), np.uint8)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 1
    return out.astype(bool)


def choose_point(allowed, existing, min_dist):
    """Deep interior point away from all previously chosen points (in frame coords)."""
    valid = np.asarray(allowed, bool).copy()
    h, w = valid.shape
    if not valid.any():
        return None
    for (x, y) in existing:
        radius = max(2, int(round(min_dist)))
        cv2.circle(valid.view(np.uint8), (int(round(x)), int(round(y))), radius, 0, -1)
    if not valid.any():
        return None
    dist = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    if dist[y, x] <= 0:
        return None
    return (float(x), float(y))


def pair_iou(a, b):
    aa, bb = np.asarray(a, bool), np.asarray(b, bool)
    union = int(np.count_nonzero(aa | bb))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(aa & bb) / union)


def gt_iou(pred, gt):
    pred = np.asarray(pred, bool)
    gt = np.asarray(gt) > 128
    if pred.shape != gt.shape:
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return pair_iou(pred, gt)


def point_candidates(initial_probs, current_prob, box, other_boxes, round_number):
    """Generate FG inside box and BG outside ANY annotated box; no GT.

    In round 1, candidates must be unanimous high/low probabilities.
    In round 2, use at least two-of-three consensus plus current logits;
    this permits another point while still requiring cross-candidate evidence.
    """
    probs = np.asarray(initial_probs, np.float32)
    h, w = probs.shape[1:]
    box = np.asarray(box, np.float32)
    bw = max(float(box[2] - box[0]), 1.0)
    bh = max(float(box[3] - box[1]), 1.0)
    long_side = max(bw, bh)
    # Safe inner FG excludes the uncertain box rim but never requires foreground
    # to be rectangular or to fill the prompt box.
    inside = box_binary((h, w), box, pad=-min(3, 0.02 * min(bw, bh)))
    if round_number == 1:
        fg = np.all(probs >= 0.85, axis=0) & inside
    else:
        fg = (np.sum(probs >= 0.70, axis=0) >= 2) & (current_prob >= 0.80) & inside

    # Outer ring around this box, disallow ALL labeled foreground boxes.
    outer = box_binary((h, w), box, pad=max(8, int(round(0.30 * long_side))))
    guard = np.zeros((h, w), bool)
    for bb in other_boxes:
        guard |= box_binary((h, w), bb, pad=max(3, int(round(0.04 * min(bw, bh)))))
    ring = outer & (~guard)
    if round_number == 1:
        bg = np.all(probs <= 0.10, axis=0) & ring
    else:
        bg = (np.sum(probs <= 0.25, axis=0) >= 2) & (current_prob <= 0.20) & ring
    return fg, bg


class Sam1Iterative:
    def __init__(self, args):
        self.args = args
        cfg = legacy.Config(
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            device=args.device,
            img_dir=args.img_dir,
            gt_dir=args.gt_dir,
            json_dir=args.json_dir,
            out_dir=args.output_dir,
            use_crop=not args.no_crop,
            small_box_area_thr=args.small_box_area_thr,
            crop_context=args.crop_context,
        )
        self.cfg = cfg
        self.engine = legacy.SAM1Engine(cfg)
        self.out = Path(args.output_dir)
        for name in ("00_Box", "01_Iter1", "02_Iter2", "per_image"):
            (self.out / name).mkdir(parents=True, exist_ok=True)
        if args.save_probs:
            for name in ("00_Box", "01_Iter1", "02_Iter2"):
                (self.out / "probs" / name).mkdir(parents=True, exist_ok=True)
        if args.save_vis:
            (self.out / "vis").mkdir(parents=True, exist_ok=True)

    def region_for(self, box, h, w):
        b = np.asarray(box, np.float32)
        ratio = max(0.0, float((b[2] - b[0]) * (b[3] - b[1]) / max(1, h * w)))
        if not self.args.no_crop and ratio <= self.args.small_box_area_thr:
            r = legacy.centered_square_crop(b, w, h, self.args.crop_context, 16, True)
            return r, r != (0, 0, w, h)
        return (0, 0, w, h), False

    def infer_one(self, rgb, box, all_boxes, object_idx):
        h, w = rgb.shape[:2]
        rect, cropped = self.region_for(box, h, w)
        x0, y0, x1, y1 = rect
        frame = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        # Default SAM1 native ResizeLongestSide, preserving aspect ratio.
        if self.args.input_mode == "square":
            frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            prompt_box = legacy.transform_box_to_model(box, rect, 1024, self.args.box_margin)
            all_frame_boxes = [legacy.transform_box_to_model(b, rect, 1024, 0) for b in all_boxes]
        else:
            prompt_box = np.asarray(box, np.float32).copy()
            prompt_box -= np.array([x0, y0, x0, y0], dtype=np.float32)
            prompt_box = legacy.expand_box(prompt_box, self.args.box_margin, frame.shape[1], frame.shape[0])
            all_frame_boxes = []
            for b in all_boxes:
                bb = np.asarray(b, np.float32) - np.array([x0, y0, x0, y0], np.float32)
                all_frame_boxes.append(legacy.clip_box(bb, frame.shape[1], frame.shape[0]))

        self.engine.set_image(frame)
        with torch.inference_mode():
            init_logits, scores, init_low = self.engine.predictor.predict(
                box=np.asarray(prompt_box, np.float32),
                multimask_output=(self.args.init == "three"),
                return_logits=True,
            )
        init_logits = np.asarray(init_logits, np.float32)
        init_low = np.asarray(init_low, np.float32)
        scores = np.asarray(scores, np.float32).reshape(-1)
        if init_logits.ndim != 3 or init_low.ndim != 3:
            raise RuntimeError(f"SAM1 invalid logit shapes: high={init_logits.shape}, low={init_low.shape}")
        if len(scores) != len(init_logits) or len(init_low) != len(init_logits):
            raise RuntimeError("SAM1 candidate count mismatch")
        if not np.isfinite(init_logits).all() or not np.isfinite(init_low).all():
            raise RuntimeError("SAM1 returned nonfinite logits")
        selected = int(np.argmax(scores))
        candidates = sigmoid(init_logits)
        current = candidates[selected]
        memory = init_low[selected:selected+1].copy()
        consensus = float(np.mean([pair_iou(candidates[i] >= .5, candidates[j] >= .5)
                                   for i, j in ((0, 1), (0, 2), (1, 2))])) if len(candidates) == 3 else float("nan")
        predictions = [current.copy()]
        points = []
        point_labels = []
        step_info = []
        bw = max(float(prompt_box[2] - prompt_box[0]), 1.0)
        bh = max(float(prompt_box[3] - prompt_box[1]), 1.0)
        min_dist = max(5.0, self.args.point_separation * max(bw, bh))
        for t in (1, 2):
            new_points = []
            new_labels = []
            if self.args.auto_points and len(candidates) == 3:
                fg, bg = point_candidates(candidates, current, prompt_box, all_frame_boxes, t)
                old_fg = [p for p, label in zip(points, point_labels) if label == 1]
                old_bg = [p for p, label in zip(points, point_labels) if label == 0]
                pfg = choose_point(fg, old_fg, min_dist)
                pbg = choose_point(bg, old_bg, min_dist)
                if pfg is not None:
                    new_points.append(pfg)
                    new_labels.append(1)
                if pbg is not None:
                    new_points.append(pbg)
                    new_labels.append(0)
            points.extend(new_points)
            point_labels.extend(new_labels)
            run = bool(points or self.args.memory)
            if run:
                coords = np.asarray(points, np.float32).reshape(-1, 2) if points else None
                labels = np.asarray(point_labels, np.int32) if points else None
                with torch.inference_mode():
                    next_logits, _, next_low = self.engine.predictor.predict(
                        point_coords=coords,
                        point_labels=labels,
                        box=np.asarray(prompt_box, np.float32),
                        mask_input=memory if self.args.memory else None,
                        multimask_output=False,
                        return_logits=True,
                    )
                logits = np.asarray(next_logits, np.float32)
                lows = np.asarray(next_low, np.float32)
                if logits.shape[0] != 1 or lows.shape[0] != 1:
                    raise RuntimeError(f"SAM1 iteration {t}: unexpected mask shapes")
                current = sigmoid(logits[0])
                memory = lows.copy()
            predictions.append(current.copy())
            to_global = []
            for px, py in new_points:
                if self.args.input_mode == "square":
                    gx = x0 + (px + .5) * ((x1 - x0) / frame.shape[1]) - .5
                    gy = y0 + (py + .5) * ((y1 - y0) / frame.shape[0]) - .5
                else:
                    gx, gy = x0 + px, y0 + py
                to_global.append([float(gx), float(gy)])
            step_info.append({"step": t, "called_sam": int(run),
                              "num_new_pos": int(sum(v == 1 for v in new_labels)),
                              "num_new_neg": int(sum(v == 0 for v in new_labels)),
                              "points_xy_frame": [[float(px), float(py)] for px, py in new_points],
                              "points_xy_global": to_global, "point_labels": new_labels})

        full_probs = []
        for p in predictions:
            # Reverse exactly the geometric resizing performed prior to SAM1.
            local = cv2.resize(p, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((h, w), np.float32)
            canvas[y0:y1, x0:x1] = local
            full_probs.append(canvas)
        return full_probs, {"object_idx": object_idx, "cropped": int(cropped),
                            "num_candidates": len(candidates),
                            "initial_selected_index": selected,
                            "initial_sam_score": float(scores[selected]),
                            "candidate_iou": consensus,
                            "steps": step_info}

    def process_image(self, img_path):
        name = Path(img_path).stem
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read RGB: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        jp = legacy.DataLoader.find_file(self.args.json_dir, name, (".json",))
        boxes = legacy.DataLoader.load_boxes(jp, w, h)
        if not boxes:
            return None
        merged = [np.zeros((h, w), np.float32) for _ in range(3)]
        object_logs = []
        total_crops = 0
        for k, b in enumerate(boxes):
            props, report = self.infer_one(rgb, b, boxes, k)
            total_crops += report["cropped"]
            object_logs.append(report)
            if len(boxes) > 1:
                # Same conservative per-object support gating as original script.
                margin = max(2, round(25 * max(h, w) / 1024))
                gate = box_binary((h, w), b, pad=margin)
            else:
                gate = np.ones((h, w), bool)
            for i in range(3):
                merged[i] = np.maximum(merged[i], props[i] * gate)

        dirs = ("00_Box", "01_Iter1", "02_Iter2")
        for i, d in enumerate(dirs):
            mask = (merged[i] >= 0.5).astype(np.uint8) * 255
            if not cv2.imwrite(str(self.out / d / (name + ".png")), mask):
                raise RuntimeError("Mask write failure: " + name)
            if self.args.save_probs:
                np.save(str(self.out / "probs" / d / (name + ".npy")),
                        merged[i].astype(np.float16), allow_pickle=False)
        (self.out / "per_image" / (name + ".json")).write_text(
            json.dumps({"image": name, "objects": object_logs}, ensure_ascii=False, indent=2),
            encoding="utf-8")

        row = {"image": name, "num_boxes": len(boxes), "cropped_boxes": total_crops,
               "route_type": "MO_PER_BOX" if len(boxes) > 1 else ("SO_CROP" if total_crops else "NORMAL"),
               "candidate_iou": min((o["candidate_iou"] for o in object_logs), default=float("nan")),
               "sam_score": min(o["initial_sam_score"] for o in object_logs)}
        for step in (1, 2):
            row[f"iter{step}_added_positive"] = sum(o["steps"][step - 1]["num_new_pos"] for o in object_logs)
            row[f"iter{step}_added_negative"] = sum(o["steps"][step - 1]["num_new_neg"] for o in object_logs)
            row[f"iter{step}_sam_calls"] = sum(o["steps"][step - 1]["called_sam"] for o in object_logs)
        gt_path = legacy.DataLoader.find_file(self.args.gt_dir, name, (".png", ".jpg"))
        if gt_path:
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                for i, stage in enumerate(("iter0", "iter1", "iter2")):
                    hard = (merged[i] >= .5).astype(np.uint8) * 255
                    scores = legacy.MetricsEvaluator.evaluate(hard, gt)
                    row.update({stage + "_" + k: v for k, v in scores.items()})
                    row[stage + "_IoU"] = gt_iou(hard > 127, gt)
                row["gain_IoU_1"] = row["iter1_IoU"] - row["iter0_IoU"]
                row["gain_IoU_2"] = row["iter2_IoU"] - row["iter0_IoU"]
                row["gain_MAE_1"] = row["iter0_MAE"] - row["iter1_MAE"]
                row["gain_MAE_2"] = row["iter0_MAE"] - row["iter2_MAE"]
                if self.args.save_vis and row["gain_IoU_2"] >= self.args.vis_min_gain:
                    annotated = bgr.copy()
                    for bb in boxes:
                        p0 = tuple(int(round(x)) for x in bb[:2])
                        p1 = tuple(int(round(x)) for x in bb[2:])
                        cv2.rectangle(annotated, p0, p1, (255, 180, 20), 2)
                    for report in object_logs:
                        for step in report["steps"]:
                            for xy, label in zip(step["points_xy_global"], step["point_labels"]):
                                pos = tuple(int(round(c)) for c in xy)
                                cv2.circle(annotated, pos, 5, (0, 220, 0) if label == 1 else (0, 0, 255), -1)
                                cv2.circle(annotated, pos, 6, (255, 255, 255), 1)
                    im = cv2.resize(annotated, (256, 256))
                    cells = [im, cv2.cvtColor(cv2.resize(gt, (256,256), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)]
                    for i in range(3):
                        hard = (merged[i] >= .5).astype(np.uint8) * 255
                        cells.append(cv2.cvtColor(cv2.resize(hard, (256,256), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR))
                    cv2.imwrite(str(self.out / "vis" / (name + ".png")), np.concatenate(cells, axis=1))
        return row


def metrics_summary(rows, prefix):
    result = {}
    for k in ("Smeasure", "maxE", "maxF", "MAE", "IoU"):
        arr = np.array([r.get(prefix + "_" + k, np.nan) for r in rows], np.float64)
        arr = arr[np.isfinite(arr)]
        result[k] = float(arr.mean()) if len(arr) else None
    return result


def self_test():
    b = np.array([15, 15, 45, 45], np.float32)
    allb = [b]
    masks = np.zeros((3, 80, 80), np.float32)
    masks[:, 20:40, 20:40] = 0.95
    current = masks[0].copy()
    fg, bg = point_candidates(masks, current, b, allb, 1)
    assert fg[25, 25] and not fg[2, 2]
    assert bg.any() and not bg[25, 25]
    p1 = choose_point(fg, [], 8)
    p2 = choose_point(fg, [p1], 8)
    assert p1 is not None and p2 is not None and p1 != p2
    assert abs(pair_iou(np.ones((2, 2), bool), np.ones((2, 2), bool)) - 1.0) < 1e-8
    print("PASS: consensus positive/negative, spatial separation, binary IoU")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sam-checkpoint", default=legacy.SAM_CHECKPOINT)
    p.add_argument("--sam-model-type", default=legacy.SAM_MODEL_TYPE)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--img-dir", default="./data/Train/Imgs")
    p.add_argument("--gt-dir", default="./data/Train/GT")
    p.add_argument("--json-dir", default="./data/Train/box_label")
    p.add_argument("--output-dir", default="./softresult/SAM1_BoxIterative")
    p.add_argument("--input-mode", choices=("native", "square"), default="native",
                   help="native preserves aspect; square reproduces old forced 1024x1024 stretch")
    p.add_argument("--box-margin", type=float, default=0.0)
    p.add_argument("--init", choices=("three", "single"), default="three")
    p.add_argument("--no-crop", action="store_true")
    p.add_argument("--small-box-area-thr", type=float, default=0.10)
    p.add_argument("--crop-context", type=float, default=0.35)
    p.add_argument("--no-auto-points", action="store_true")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--point-separation", type=float, default=0.15)
    p.add_argument("--save-probs", action="store_true")
    p.add_argument("--save-vis", action="store_true", help="Save only samples whose Iter2 IoU gains reach --vis-min-gain")
    p.add_argument("--vis-min-gain", type=float, default=0.01)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA not available; use --device cpu")
    if not (0 <= args.small_box_area_thr <= 1): p.error("small_box_area_thr must be [0,1]")
    if args.crop_context < 0 or args.box_margin < 0 or not (0 <= args.point_separation <= 1):
        p.error("invalid crop-context, box-margin or point-separation")
    if args.init == "single" and not args.no_auto_points:
        print("[NOTE] single candidate init => automatic consensus points disabled; use --init three for points")
    args.auto_points = not args.no_auto_points
    args.memory = not args.no_memory
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    pipe = Sam1Iterative(args)
    imgs = legacy.DataLoader.list_images(args.img_dir)
    if args.limit > 0: imgs = imgs[:args.limit]
    rows, errors, skipped = [], [], []
    for path in tqdm(imgs, desc="SAM1 Box / Iterate 1 / Iterate 2"):
        try:
            r = pipe.process_image(path)
            if r is None: skipped.append(str(path))
            else: rows.append(r)
        except Exception as ex:
            errors.append({"image": str(path), "error": repr(ex)})
            if len(errors) <= 15: tqdm.write("[ERROR] " + str(path) + ": " + str(ex))
    root = Path(args.output_dir)
    (root / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    if (errors or skipped) and not args.allow_partial:
        raise RuntimeError(f"Processing incomplete: {len(errors)} errors, {len(skipped)} skipped. Examine errors.json")
    if not rows:
        raise RuntimeError("No images processed")
    records_csv = root / "sam1_iterative_metrics.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(records_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    def gain_counts(a, b, metric="IoU", epsilon=0.01):
        pairs = [(r.get(f"{a}_{metric}"), r.get(f"{b}_{metric}")) for r in rows]
        delta = np.asarray([(float(y) - float(x)) for x, y in pairs
                            if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)],
                           np.float64)
        if delta.size == 0:
            return {"n": 0, "improved": 0, "worsened": 0, "unchanged": 0, "mean_gain": None}
        return {"n": int(delta.size), "improved": int((delta > epsilon).sum()),
                "worsened": int((delta < -epsilon).sum()),
                "unchanged": int((np.abs(delta) <= epsilon).sum()),
                "mean_gain": float(delta.mean())}

    summary = {"Iter0_Box": metrics_summary(rows, "iter0"),
               "Iter1_PointMemory": metrics_summary(rows, "iter1"),
               "Iter2_PointMemory": metrics_summary(rows, "iter2"),
               "Gain_IoU_0_to_1": gain_counts("iter0", "iter1"),
               "Gain_IoU_0_to_2": gain_counts("iter0", "iter2"),
               "Gain_IoU_1_to_2": gain_counts("iter1", "iter2"),
               "Diagnostics": {"num_images": len(rows), "num_failed": len(errors), "num_skipped": len(skipped),
                               "route_counts": dict(Counter(r["route_type"] for r in rows)),
                               "cropped_boxes": int(sum(r["cropped_boxes"] for r in rows)),
                               "iter1_added_positive": int(sum(r["iter1_added_positive"] for r in rows)),
                               "iter1_added_negative": int(sum(r["iter1_added_negative"] for r in rows)),
                               "iter2_added_positive": int(sum(r["iter2_added_positive"] for r in rows)),
                               "iter2_added_negative": int(sum(r["iter2_added_negative"] for r in rows)),
                               "iter1_sam_calls": int(sum(r["iter1_sam_calls"] for r in rows)),
                               "iter2_sam_calls": int(sum(r["iter2_sam_calls"] for r in rows))},
               "Config": vars(args),
               "MetricNote": "Legacy maxE/maxF are E/F measured at binary 0.5, NOT maxima across thresholds.",
               "GTNote": "GT used only for offline evaluation, never prompts, quality selection or point sampling.",
               "BaselineNote": "Iter0 is per-box geometry with native aspect ratio by default; not the old union-box BoxBase."}
    (root / "summary_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("=" * 73)
    for k in ("Iter0_Box", "Iter1_PointMemory", "Iter2_PointMemory"):
        print(k, summary[k])
    print("Diagnostics", summary["Diagnostics"])
    print("CSV:", records_csv, "Summary:", root / "summary_metrics.json")
    print("=" * 73)


if __name__ == "__main__":
    main()
