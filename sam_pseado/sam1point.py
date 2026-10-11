#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PGLR unvalue伪标签重生成（硬标签版）

默认输入：
- 教师二值掩码：./data/nonclean_analysis/epoch055/unvalue/binary
- 原始伪标签：  ./data/Train/S1_GT
- 原图：        ./data/Train/Imgs
- 真实GT：      ./data/Train/GT（仅统计，不参与生成）

生成规则：
1. 教师与原伪标签共同确认的内部区域选正点；
2. 二者共同确认的框内背景区域选负点；
3. 原伪标签提供box和256x256硬mask logits提示；
4. SAM只提供边界候选；
5. 最终硬标签以原伪标签为主体，只在原边界带内接受教师+SAM共同确认的修改；
6. 只保存硬掩码，并统计生成前后S/E/F/MAE与IoU变化。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from py_sod_metrics import Smeasure, Emeasure, Fmeasure, MAE
from segment_anything import sam_model_registry, SamPredictor


def read_binary_mask(path: str, target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if target_hw is not None and mask.shape[:2] != target_hw:
        h, w = target_hw
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask >= 128


def find_image_path(image_dir: str, name: str) -> Optional[str]:
    for suffix in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]:
        path = os.path.join(image_dir, name + suffix)
        if os.path.isfile(path):
            return path
    return None


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else float(inter / union)


def compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def mask_bbox(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def expand_box(box: Sequence[float], h: int, w: int, ratio: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return [
        max(0.0, min(float(w - 1), x1 - bw * ratio)),
        max(0.0, min(float(h - 1), y1 - bh * ratio)),
        max(1.0, min(float(w), x2 + bw * ratio)),
        max(1.0, min(float(h), y2 + bh * ratio)),
    ]


def box_to_mask(box: Sequence[float], h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    x2, y2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
    result = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        result[y1:y2, x1:x2] = True
    return result


def component_distance_points(region: np.ndarray, max_points: int, min_pixels: int) -> List[Tuple[float, float]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
    components = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_pixels:
            components.append((area, idx))
    components.sort(reverse=True)

    points = []
    for _, idx in components[:max_points]:
        component = labels == idx
        distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        if distance[y, x] > 0:
            points.append((float(x), float(y)))
    return points


def build_prompt_points(
    pseudo: np.ndarray,
    teacher: np.ndarray,
    box_mask: np.ndarray,
    pos_erode: int,
    neg_erode: int,
    max_pos: int,
    max_neg: int,
):
    common_fg = pseudo & teacher
    pos_region = erode_mask(common_fg, pos_erode)
    if not pos_region.any():
        pos_region = common_fg
    pos_points = component_distance_points(pos_region, max_pos, 8)

    if not pos_points:
        fallback = erode_mask(pseudo, pos_erode)
        if not fallback.any():
            fallback = pseudo
        pos_points = component_distance_points(fallback, 1, 1)

    common_bg = box_mask & (~pseudo) & (~teacher)
    neg_region = erode_mask(common_bg, neg_erode)
    if not neg_region.any():
        neg_region = common_bg
    neg_points = component_distance_points(neg_region, max_neg, 16)

    coords, labels = [], []
    for point in pos_points:
        coords.append(point)
        labels.append(1)
    for point in neg_points:
        coords.append(point)
        labels.append(0)

    if not coords:
        raise RuntimeError("No valid prompt points")

    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        {"num_positive_points": len(pos_points), "num_negative_points": len(neg_points)},
    )


def build_hard_mask_prompt(pseudo: np.ndarray, magnitude: float) -> np.ndarray:
    low = cv2.resize(pseudo.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
    logits = np.where(low >= 0.5, magnitude, -magnitude).astype(np.float32)
    return logits[None, ...]


def run_single_sam(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    box: Sequence[float],
    mask_prompt: np.ndarray,
    pseudo: np.ndarray,
    teacher: np.ndarray,
) -> Dict:
    predictor.set_image(image_rgb)
    full_logits, sam_scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=np.asarray(box, dtype=np.float32),
        mask_input=mask_prompt,
        multimask_output=True,
        return_logits=True,
    )
    full_logits = np.asarray(full_logits, dtype=np.float32)
    sam_scores = np.asarray(sam_scores, dtype=np.float32).reshape(-1)
    if full_logits.ndim == 2:
        full_logits = full_logits[None]

    threshold = float(predictor.model.mask_threshold)
    masks = full_logits > threshold
    candidates = []
    for idx, mask in enumerate(masks):
        sam_score = float(sam_scores[idx]) if idx < len(sam_scores) else 0.0
        iou_pseudo = compute_iou(mask, pseudo)
        iou_teacher = compute_iou(mask, teacher)
        select_score = 0.45 * sam_score + 0.40 * iou_pseudo + 0.15 * iou_teacher
        candidates.append({
            "mask": mask,
            "sam_score": sam_score,
            "iou_pseudo": iou_pseudo,
            "iou_teacher": iou_teacher,
            "selection_score": select_score,
        })
    return max(candidates, key=lambda x: x["selection_score"])


def conservative_boundary_refinement(
    pseudo: np.ndarray,
    teacher: np.ndarray,
    sam: np.ndarray,
    radius: int,
):
    band = dilate_mask(pseudo, radius) & (~erode_mask(pseudo, radius))
    add = band & (~pseudo) & teacher & sam
    remove = band & pseudo & (~teacher) & (~sam)

    refined = pseudo.copy()
    refined[add] = True
    refined[remove] = False
    return refined, {
        "boundary_band_pixels": int(band.sum()),
        "added_pixels": int(add.sum()),
        "removed_pixels": int(remove.sum()),
        "changed_pixels": int(np.logical_xor(refined, pseudo).sum()),
    }


def sanity_gate(
    refined: np.ndarray,
    pseudo: np.ndarray,
    sam: np.ndarray,
    min_refined_iou: float,
    min_sam_iou: float,
    min_area_ratio: float,
    max_area_ratio: float,
):
    area_ratio = float(refined.sum() / max(1, int(pseudo.sum())))
    refined_iou = compute_iou(refined, pseudo)
    sam_iou = compute_iou(sam, pseudo)
    accepted = bool(
        refined.any()
        and refined_iou >= min_refined_iou
        and sam_iou >= min_sam_iou
        and min_area_ratio <= area_ratio <= max_area_ratio
    )
    return accepted, {
        "refined_pseudo_iou": refined_iou,
        "sam_pseudo_iou": sam_iou,
        "refined_pseudo_area_ratio": area_ratio,
    }


def init_metric_group():
    return {"SM": Smeasure(), "EM": Emeasure(), "FM": Fmeasure(), "MAE": MAE()}


def update_metric_group(group, pred_u8: np.ndarray, gt_u8: np.ndarray):
    group["SM"].step(pred=pred_u8, gt=gt_u8)
    group["EM"].step(pred=pred_u8, gt=gt_u8)
    group["FM"].step(pred=pred_u8, gt=gt_u8)
    group["MAE"].step(pred=pred_u8, gt=gt_u8)


def get_metric_results(group):
    return {
        "S-measure": float(group["SM"].get_results()["sm"]),
        "E-measure(adp)": float(group["EM"].get_results()["em"]["adp"]),
        "F-measure(adp)": float(group["FM"].get_results()["fm"]["adp"]),
        "MAE": float(group["MAE"].get_results()["mae"]),
    }


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", default="./data/nonclean_analysis/epoch055/unvalue/binary")
    parser.add_argument("--pseudo-dir", default="./data/Train/S1_GT")
    parser.add_argument("--image-dir", default="./data/Train/Imgs")
    parser.add_argument("--gt-dir", default="./data/Train/GT")
    parser.add_argument("--sam-checkpoint", default="./weights/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="./SGT/PGLR_Unvalue_Epoch055")
    parser.add_argument("--box-expand-ratio", type=float, default=0.05)
    parser.add_argument("--positive-erode-radius", type=int, default=3)
    parser.add_argument("--negative-erode-radius", type=int, default=3)
    parser.add_argument("--max-positive-points", type=int, default=4)
    parser.add_argument("--max-negative-points", type=int, default=3)
    parser.add_argument("--mask-prompt-logit", type=float, default=6.0)
    parser.add_argument("--boundary-radius", type=int, default=5)
    parser.add_argument("--min-refined-pseudo-iou", type=float, default=0.75)
    parser.add_argument("--min-sam-pseudo-iou", type=float, default=0.50)
    parser.add_argument("--min-area-ratio", type=float, default=0.60)
    parser.add_argument("--max-area-ratio", type=float, default=1.40)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in [args.teacher_dir, args.pseudo_dir, args.image_dir, args.gt_dir, args.sam_checkpoint]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "Pred_Masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_per_image.csv"
    summary_path = output_dir / "metrics_summary.json"

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM {args.sam_model_type} on {device}")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    teacher_paths = sorted(Path(args.teacher_dir).glob("*.png"))
    if not teacher_paths:
        raise RuntimeError(f"No teacher masks found in {args.teacher_dir}")

    before_metrics = init_metric_group()
    after_metrics = init_metric_group()
    rows = []
    skipped = 0
    fallback_count = 0

    for teacher_path in tqdm(teacher_paths, desc="PGLR regenerate", ncols=100):
        name = teacher_path.stem
        pseudo_path = os.path.join(args.pseudo_dir, name + ".png")
        gt_path = os.path.join(args.gt_dir, name + ".png")
        image_path = find_image_path(args.image_dir, name)

        if image_path is None or not os.path.isfile(pseudo_path) or not os.path.isfile(gt_path):
            skipped += 1
            continue

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            skipped += 1
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        pseudo = read_binary_mask(pseudo_path, (h, w))
        teacher = read_binary_mask(str(teacher_path), (h, w))
        gt = read_binary_mask(gt_path, (h, w))

        box = mask_bbox(pseudo)
        box_source = "pseudo"
        if box is None:
            box = mask_bbox(teacher)
            box_source = "teacher_fallback"

        accepted = False
        fallback_reason = ""
        if box is None:
            final = pseudo.copy()
            fallback_reason = "empty_pseudo_and_teacher"
            point_stats = {"num_positive_points": 0, "num_negative_points": 0}
            sam_result = {"sam_score": 0.0, "iou_pseudo": 1.0, "iou_teacher": 1.0, "selection_score": 0.0}
            change_stats = {"boundary_band_pixels": 0, "added_pixels": 0, "removed_pixels": 0, "changed_pixels": 0}
            gate_stats = {"refined_pseudo_iou": 1.0, "sam_pseudo_iou": 1.0, "refined_pseudo_area_ratio": 1.0}
        else:
            box = expand_box(box, h, w, args.box_expand_ratio)
            box_mask = box_to_mask(box, h, w)
            try:
                points, labels, point_stats = build_prompt_points(
                    pseudo, teacher, box_mask,
                    args.positive_erode_radius,
                    args.negative_erode_radius,
                    args.max_positive_points,
                    args.max_negative_points,
                )
                mask_prompt = build_hard_mask_prompt(pseudo, args.mask_prompt_logit)
                with torch.inference_mode():
                    sam_result = run_single_sam(
                        predictor, image_rgb, points, labels, box, mask_prompt, pseudo, teacher
                    )
                refined, change_stats = conservative_boundary_refinement(
                    pseudo, teacher, sam_result["mask"], args.boundary_radius
                )
                accepted, gate_stats = sanity_gate(
                    refined, pseudo, sam_result["mask"],
                    args.min_refined_pseudo_iou,
                    args.min_sam_pseudo_iou,
                    args.min_area_ratio,
                    args.max_area_ratio,
                )
                if accepted:
                    final = refined
                else:
                    final = pseudo.copy()
                    fallback_reason = "sanity_gate_failed"
                    fallback_count += 1
            except Exception as exc:
                final = pseudo.copy()
                fallback_reason = f"sam_error:{type(exc).__name__}"
                fallback_count += 1
                point_stats = {"num_positive_points": 0, "num_negative_points": 0}
                sam_result = {
                    "sam_score": 0.0,
                    "iou_pseudo": 1.0,
                    "iou_teacher": compute_iou(teacher, pseudo),
                    "selection_score": 0.0,
                }
                change_stats = {"boundary_band_pixels": 0, "added_pixels": 0, "removed_pixels": 0, "changed_pixels": 0}
                gate_stats = {"refined_pseudo_iou": 1.0, "sam_pseudo_iou": 1.0, "refined_pseudo_area_ratio": 1.0}

        out_path = mask_dir / f"{name}.png"
        final_u8 = final.astype(np.uint8) * 255
        pseudo_u8 = pseudo.astype(np.uint8) * 255
        gt_u8 = gt.astype(np.uint8) * 255
        if not cv2.imwrite(str(out_path), final_u8):
            raise IOError(f"Cannot save {out_path}")

        update_metric_group(before_metrics, pseudo_u8, gt_u8)
        update_metric_group(after_metrics, final_u8, gt_u8)

        old_iou, new_iou = compute_iou(pseudo, gt), compute_iou(final, gt)
        old_mae, new_mae = compute_mae(pseudo, gt), compute_mae(final, gt)
        rows.append({
            "image": name,
            "accepted_refinement": int(accepted),
            "fallback_reason": fallback_reason,
            "box_source": box_source,
            **point_stats,
            "sam_score": float(sam_result["sam_score"]),
            "sam_iou_with_pseudo": float(sam_result["iou_pseudo"]),
            "sam_iou_with_teacher": float(sam_result["iou_teacher"]),
            "sam_selection_score": float(sam_result["selection_score"]),
            **change_stats,
            **gate_stats,
            "old_iou": old_iou,
            "new_iou": new_iou,
            "delta_iou": new_iou - old_iou,
            "old_mae": old_mae,
            "new_mae": new_mae,
            "delta_mae": old_mae - new_mae,
            "improved_both": int(new_iou > old_iou and new_mae < old_mae),
            "output_mask": str(out_path),
        })

    if not rows:
        raise RuntimeError("No valid samples were processed")

    before = get_metric_results(before_metrics)
    after = get_metric_results(after_metrics)
    old_iou = float(np.mean([r["old_iou"] for r in rows]))
    new_iou = float(np.mean([r["new_iou"] for r in rows]))
    old_mae = float(np.mean([r["old_mae"] for r in rows]))
    new_mae = float(np.mean([r["new_mae"] for r in rows]))

    summary = {
        "input": {
            "teacher_dir": args.teacher_dir,
            "pseudo_dir": args.pseudo_dir,
            "image_dir": args.image_dir,
            "gt_dir": args.gt_dir,
        },
        "count": {
            "processed": len(rows),
            "skipped": skipped,
            "accepted_refinement": int(sum(r["accepted_refinement"] for r in rows)),
            "fallback": fallback_count,
            "iou_improved": int(sum(r["delta_iou"] > 0 for r in rows)),
            "mae_improved": int(sum(r["delta_mae"] > 0 for r in rows)),
            "both_improved": int(sum(r["improved_both"] for r in rows)),
        },
        "before_original_pseudo": {**before, "Mean-IoU": old_iou, "Mean-MAE(binary)": old_mae},
        "after_regenerated": {**after, "Mean-IoU": new_iou, "Mean-MAE(binary)": new_mae},
        "delta_positive_is_better": {
            "S-measure": after["S-measure"] - before["S-measure"],
            "E-measure(adp)": after["E-measure(adp)"] - before["E-measure(adp)"],
            "F-measure(adp)": after["F-measure(adp)"] - before["F-measure(adp)"],
            "MAE": before["MAE"] - after["MAE"],
            "Mean-IoU": new_iou - old_iou,
            "Mean-MAE(binary)": old_mae - new_mae,
        },
    }

    write_csv(str(csv_path), rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== Metric Change ==========")
    print(f"Processed={len(rows)} skipped={skipped} accepted={summary['count']['accepted_refinement']}")
    print(f"Mean IoU: {old_iou:.4f} -> {new_iou:.4f} | delta={new_iou-old_iou:+.4f}")
    print(f"MAE: {before['MAE']:.4f} -> {after['MAE']:.4f} | delta={before['MAE']-after['MAE']:+.4f}")
    print(f"S: {before['S-measure']:.4f} -> {after['S-measure']:.4f} | delta={after['S-measure']-before['S-measure']:+.4f}")
    print(f"E: {before['E-measure(adp)']:.4f} -> {after['E-measure(adp)']:.4f} | delta={after['E-measure(adp)']-before['E-measure(adp)']:+.4f}")
    print(f"F: {before['F-measure(adp)']:.4f} -> {after['F-measure(adp)']:.4f} | delta={after['F-measure(adp)']-before['F-measure(adp)']:+.4f}")
    print(f"IoU improved={summary['count']['iou_improved']} MAE improved={summary['count']['mae_improved']} both={summary['count']['both_improved']}")
    print(f"Masks: {mask_dir}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()