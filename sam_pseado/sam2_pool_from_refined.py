#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply the ORIGINAL SAM2 three-candidate pool assignment to already-refined
NO-FLIP pseudo labels. No SAM inference, no flip, no new pool rule.

Input:
  1) existing 03_Refine_NoFlip/*.png (refined hard masks)
  2) original code-2 image_level_metrics.csv (candidate S, SAM score, stability)
  3) optional refine_prob/*.npy (probability saved by code 1 --save-probs)

If 3) does not exist, --prob-fallback base48 uses the original code-2
48x48 candidate probability map to evaluate agreement with the refined mask.
This is a PROXY, not a refined probability, and can change Q rankings.

GT can only be used for post-hoc evaluation and is NEVER used to assign pools.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
from collections import Counter, defaultdict
import cv2
import numpy as np
from tqdm import tqdm

CANDIDATE_LOW_THR = 0.60
CANDIDATE_HIGH_THR = 0.90

# SAM2 阈值稳定性：对最终选中候选计算
SAM_STABILITY_OFFSET = 1.0


# ============================================================
# 基础质量门控：先筛掉明显不可靠样本，再按 Q 排序
# ============================================================
MIN_FG_CONFIDENCE = 0.70
MIN_BG_CONFIDENCE = 0.80
MIN_SAM_SCORE = 0.80
MIN_SAM_STABILITY = 0.90

# Q 不用于初始高/中/低分级，只用于通过门控后的同级样本排序。
# 区域可靠性与形状可靠性分开计算，避免单一 Q 被背景置信度主导。
REGION_Q_WEIGHTS = {
    "fg_mean": 0.35,
    "fg_p10": 0.15,
    "ring_bg": 0.30,
    "hard_outside_bg": 0.20,
}
SHAPE_Q_WEIGHTS = {
    "sam_score": 0.40,
    "sam_stability": 0.30,
    "candidate_consensus_s": 0.30,
}


# ============================================================
# Q / V / 结构参数
# ============================================================
CORE_ERODE_RADIUS = 3
BACKGROUND_RING_RADIUS = 3
BOUNDARY_BAND_RADIUS = 3
OV_BORDER_DELTA = 4
SO_STRUCTURE_RATIO_THR = 0.10
BA_TOP_RATIO = 0.30

AREA_LOWER_THR = 0.03
AREA_UPPER_THR = 0.95

V_WEIGHTS = {
    "outside": 0.40,
    "box_fit": 0.30,
    "area": 0.20,
    "center": 0.10,
}


# ============================================================
# 典型异常响应阈值
# ============================================================
TINY_MASK_BOX_RATIO_THR = 0.03

# 两级碎片：先忽略极小连通域，再区分普通碎片与严重碎片。
FRAGMENT_EFFECTIVE_MIN_PIXELS = 16
FRAGMENT_EFFECTIVE_MIN_MASK_RATIO = 0.002
FRAGMENT_MIN_COMPONENTS = 3
FRAGMENT_MAX_LARGEST_COMPONENT_RATIO = 0.75
SEVERE_FRAGMENT_MIN_COMPONENTS = 6
SEVERE_FRAGMENT_MAX_LARGEST_COMPONENT_RATIO = 0.55

OUTSIDE_LEAK_RATIO_THR = 0.15
OV_OUTSIDE_LEAK_RATIO_THR = 0.25

BOX_COPY_FILL_RATIO_THR = 0.90
BOX_COPY_BBOX_IOU_THR = 0.95
BOX_COPY_RECTANGULARITY_THR = 0.95

REVERSAL_OUTSIDE_RATIO_THR = 0.50
REVERSAL_INSIDE_RATIO_THR = 0.50
REVERSAL_IMAGE_AREA_RATIO_THR = 0.50
REVERSAL_SMALL_BOX_IMAGE_RATIO_THR = 0.30


# ============================================================
# GT 坏伪标签分析阈值（仅评估，不参与训练分池）
# ============================================================
BAD_IOU_THR = 0.70
BAD_MAE_THR = 0.05
BAD_VIS_MAX_PANEL_SIDE = 800


def extract_all_boxes_from_json(json_path):
    """从 LabelMe 风格 JSON 中提取全部矩形框。"""
    if not os.path.exists(json_path):
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    boxes = []
    for shape in data.get("shapes", []):
        pts = shape.get("points", [])
        if len(pts) < 2:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xmin, ymin = min(xs), min(ys)
        xmax, ymax = max(xs), max(ys)
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])

    return boxes

def clip_box(box, img_h, img_w):
    """将 box 裁剪到图像范围，坐标语义为 [x1, y1, x2, y2)。"""
    x1, y1, x2, y2 = [float(v) for v in box]

    x1 = max(0.0, min(float(img_w - 1), x1))
    y1 = max(0.0, min(float(img_h - 1), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]

def valid_box(box, min_side=1.0):
    return (box[2] - box[0]) >= min_side and (box[3] - box[1]) >= min_side

def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

def build_single_prompt_box(boxes, img_h, img_w):
    """
    每张图只使用一个 SAM box prompt。

    - 一个有效框：直接使用该框。
    - 多个有效框：不逐框推理，使用全部框的最小外接框进行一次标准 SAM 推理。
    """
    valid_boxes = []
    for box in boxes:
        clipped = clip_box(box, img_h, img_w)
        if valid_box(clipped):
            valid_boxes.append(clipped)

    if not valid_boxes:
        return None, []

    prompt_box = [
        min(box[0] for box in valid_boxes),
        min(box[1] for box in valid_boxes),
        max(box[2] for box in valid_boxes),
        max(box[3] for box in valid_boxes),
    ]
    prompt_box = clip_box(prompt_box, img_h, img_w)
    return prompt_box, valid_boxes

def box_to_mask(box, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=bool)
    if box is None:
        return mask

    x1 = max(0, min(img_w, int(math.floor(box[0]))))
    y1 = max(0, min(img_h, int(math.floor(box[1]))))
    x2 = max(0, min(img_w, int(math.ceil(box[2]))))
    y2 = max(0, min(img_h, int(math.ceil(box[3]))))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask

def safe_float(x, default=np.nan):
    try:
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default

def compute_iou(pred_bool, gt_bool):
    pred_bool = np.asarray(pred_bool, dtype=bool)
    gt_bool = np.asarray(gt_bool, dtype=bool)
    inter = int(np.logical_and(pred_bool, gt_bool).sum())
    union = int(np.logical_or(pred_bool, gt_bool).sum())
    if union == 0:
        return 1.0
    return float(inter / union)

def candidate_grade_from_s(candidate_s):
    if candidate_s < CANDIDATE_LOW_THR:
        return "low"
    if candidate_s < CANDIDATE_HIGH_THR:
        return "middle"
    return "high"

def erode_mask(mask_bool, radius):
    mask_u8 = np.asarray(mask_bool, dtype=np.uint8)
    if radius <= 0:
        return mask_u8.astype(bool)
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask_u8, kernel, iterations=1).astype(bool)

def dilate_mask(mask_bool, radius):
    mask_u8 = np.asarray(mask_bool, dtype=np.uint8)
    if radius <= 0:
        return mask_u8.astype(bool)
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)

def mask_bbox(mask_bool):
    ys, xs = np.where(np.asarray(mask_bool, dtype=bool))
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

def bbox_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def mask_centroid(mask_bool):
    ys, xs = np.where(np.asarray(mask_bool, dtype=bool))
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())

def effective_component_stats(mask_bool):
    """
    连通域统计时先忽略极小噪声块。

    有效连通域最小面积：
      max(FRAGMENT_EFFECTIVE_MIN_PIXELS,
          FRAGMENT_EFFECTIVE_MIN_MASK_RATIO * mask_area)
    """
    mask_u8 = np.asarray(mask_bool, dtype=np.uint8)
    total_area = int(mask_u8.sum())

    empty_result = {
        "raw_num_components": 0,
        "effective_num_components": 0,
        "largest_component_ratio": 0.0,
        "largest_effective_component_ratio": 0.0,
        "small_fragment_area_ratio": 0.0,
        "effective_component_min_area": 0,
    }
    if total_area == 0:
        return empty_result

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    if areas.size == 0:
        return empty_result

    min_effective_area = max(
        int(FRAGMENT_EFFECTIVE_MIN_PIXELS),
        int(round(FRAGMENT_EFFECTIVE_MIN_MASK_RATIO * total_area)),
    )
    effective_areas = areas[areas >= min_effective_area]
    small_areas = areas[areas < min_effective_area]

    largest_component_ratio = float(areas.max() / max(1.0, float(total_area)))
    effective_total = float(effective_areas.sum())
    largest_effective_ratio = (
        float(effective_areas.max() / max(1.0, effective_total))
        if effective_areas.size
        else 0.0
    )

    return {
        "raw_num_components": int(areas.size),
        "effective_num_components": int(effective_areas.size),
        "largest_component_ratio": largest_component_ratio,
        "largest_effective_component_ratio": largest_effective_ratio,
        "small_fragment_area_ratio": float(small_areas.sum() / max(1.0, total_area)),
        "effective_component_min_area": int(min_effective_area),
    }

def normalized_area_violation(fill_ratio, lower=0.03, upper=0.95):
    if fill_ratio < lower:
        violation = (lower - fill_ratio) / max(lower, 1e-6)
    elif fill_ratio > upper:
        violation = (fill_ratio - upper) / max(1.0 - upper, 1e-6)
    else:
        violation = 0.0
    return float(np.clip(violation, 0.0, 1.0))

def mean_on_region(values, region_mask, default=np.nan):
    region_mask = np.asarray(region_mask, dtype=bool)
    if region_mask.sum() == 0:
        return default
    return float(np.asarray(values, dtype=np.float32)[region_mask].mean())

def compute_q_evidence(
    prob,
    pred_mask,
    box_mask,
    selected_score,
    selected_stability,
    candidate_consensus_s,
):
    """
    将可靠性拆成两个维度：

    1) q_region：前景内部与困难背景的像素可靠性；
    2) q_shape：SAM2 predicted-IoU、阈值稳定性和候选共识构成的形状可靠性。

    q_dual 使用几何平均作为平衡参考，防止单一维度过高完全补偿另一维度。
    真正排序时同时保存 region / shape / dual 三套排名。
    """
    pred_mask = np.asarray(pred_mask, dtype=bool)
    prob = np.asarray(prob, dtype=np.float32)
    box_mask = np.asarray(box_mask, dtype=bool)

    # ---------- FG：核心区域均值 + 低分位数 ----------
    core = erode_mask(pred_mask, CORE_ERODE_RADIUS)
    if core.sum() == 0:
        core = pred_mask.copy()

    fg_values = prob[core]
    if fg_values.size:
        fg_mean = float(np.clip(fg_values.mean(), 0.0, 1.0))
        fg_p10 = float(np.clip(np.quantile(fg_values, 0.10), 0.0, 1.0))
    else:
        fg_mean = 0.0
        fg_p10 = 0.0

    # ---------- BG：边界环带 + 框外最困难背景 ----------
    dilated = dilate_mask(pred_mask, BACKGROUND_RING_RADIUS)
    ring = dilated & (~pred_mask)
    ring_values = 1.0 - prob[ring]
    ring_bg_conf = (
        float(np.clip(ring_values.mean(), 0.0, 1.0))
        if ring_values.size
        else np.nan
    )

    outside_region = (~box_mask) & (~pred_mask)
    outside_prob = prob[outside_region]
    if outside_prob.size:
        # 框外预测前景概率最高的约 5% 区域决定困难背景置信度。
        hard_outside_prob = float(np.quantile(outside_prob, 0.95))
        hard_outside_bg_conf = float(np.clip(1.0 - hard_outside_prob, 0.0, 1.0))
    else:
        hard_outside_bg_conf = np.nan

    bg_parts = []
    bg_weights = []
    if np.isfinite(ring_bg_conf):
        bg_parts.append(ring_bg_conf)
        bg_weights.append(0.70)
    if np.isfinite(hard_outside_bg_conf):
        bg_parts.append(hard_outside_bg_conf)
        bg_weights.append(0.30)
    bg_conf = float(np.average(bg_parts, weights=bg_weights)) if bg_parts else 0.0

    selected_score = float(np.clip(selected_score, 0.0, 1.0))
    selected_stability = float(np.clip(selected_stability, 0.0, 1.0))
    candidate_consensus_s = float(np.clip(candidate_consensus_s, 0.0, 1.0))

    # 缺失区域用当前组合 BG 回填，保证 q_region 始终位于 [0, 1]。
    ring_for_q = ring_bg_conf if np.isfinite(ring_bg_conf) else bg_conf
    hard_bg_for_q = (
        hard_outside_bg_conf if np.isfinite(hard_outside_bg_conf) else bg_conf
    )

    q_region = (
        REGION_Q_WEIGHTS["fg_mean"] * fg_mean
        + REGION_Q_WEIGHTS["fg_p10"] * fg_p10
        + REGION_Q_WEIGHTS["ring_bg"] * ring_for_q
        + REGION_Q_WEIGHTS["hard_outside_bg"] * hard_bg_for_q
    )
    q_shape = (
        SHAPE_Q_WEIGHTS["sam_score"] * selected_score
        + SHAPE_Q_WEIGHTS["sam_stability"] * selected_stability
        + SHAPE_Q_WEIGHTS["candidate_consensus_s"] * candidate_consensus_s
    )
    q_region = float(np.clip(q_region, 0.0, 1.0))
    q_shape = float(np.clip(q_shape, 0.0, 1.0))
    q_dual = float(math.sqrt(max(0.0, q_region * q_shape)))

    fail_fg = int(fg_mean < MIN_FG_CONFIDENCE)
    fail_bg = int(bg_conf < MIN_BG_CONFIDENCE)
    fail_score = int(selected_score < MIN_SAM_SCORE)
    fail_stability = int(selected_stability < MIN_SAM_STABILITY)
    fails_quality_gate = int(bool(fail_fg or fail_bg or fail_score or fail_stability))

    return {
        # 兼容旧 CSV：fg_confidence 等同于核心均值。
        "fg_confidence": fg_mean,
        "fg_mean_confidence": fg_mean,
        "fg_p10_confidence": fg_p10,
        "fg_core_area": int(core.sum()),
        "fg_core_area_ratio": float(core.mean()),
        "bg_confidence": bg_conf,
        "ring_bg_confidence": ring_bg_conf,
        "hard_outside_bg_confidence": hard_outside_bg_conf,
        "sam_score_selected": selected_score,
        "sam_stability_selected": selected_stability,
        "q_region": q_region,
        "q_shape": q_shape,
        "q_dual": q_dual,
        # 兼容旧代码：q_total 指向双维平衡参考分数。
        "q_total": q_dual,
        "fail_fg_gate": fail_fg,
        "fail_bg_gate": fail_bg,
        "fail_score_gate": fail_score,
        "fail_stability_gate": fail_stability,
        "fails_quality_gate": fails_quality_gate,
    }

def compute_v_violation(pred_mask, box_mask, prompt_box):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    box_mask = np.asarray(box_mask, dtype=bool)

    pred_area = int(pred_mask.sum())
    box_area_pixels = int(box_mask.sum())

    outside_pixel_count = int((pred_mask & (~box_mask)).sum())
    v_outside = float(outside_pixel_count / max(1, pred_area))

    pred_bbox = mask_bbox(pred_mask)
    box_fit_iou = bbox_iou(pred_bbox, prompt_box)
    v_fit = 1.0 - box_fit_iou

    fill_ratio = float(pred_area / max(1, box_area_pixels))
    v_area = normalized_area_violation(
        fill_ratio,
        lower=AREA_LOWER_THR,
        upper=AREA_UPPER_THR,
    )

    pred_center = mask_centroid(pred_mask)
    box_center = mask_centroid(box_mask)
    if pred_center is None or box_center is None:
        center_offset_norm = 1.0
    else:
        dx = pred_center[0] - box_center[0]
        dy = pred_center[1] - box_center[1]
        bw = max(1.0, prompt_box[2] - prompt_box[0])
        bh = max(1.0, prompt_box[3] - prompt_box[1])
        center_offset_norm = float(
            math.sqrt(dx * dx + dy * dy) / math.sqrt(bw * bw + bh * bh)
        )

    v_center = center_offset_norm
    v_total = (
        V_WEIGHTS["outside"] * v_outside
        + V_WEIGHTS["box_fit"] * v_fit
        + V_WEIGHTS["area"] * v_area
        + V_WEIGHTS["center"] * v_center
    )

    pred_bbox_vals = pred_bbox if pred_bbox is not None else [np.nan] * 4
    pred_center_vals = pred_center if pred_center is not None else (np.nan, np.nan)
    box_center_vals = box_center if box_center is not None else (np.nan, np.nan)

    return {
        "outside_pixel_count": outside_pixel_count,
        "v_outside": v_outside,
        "box_fit_iou": box_fit_iou,
        "v_fit": v_fit,
        "mask_box_fill_ratio": fill_ratio,
        "v_area": v_area,
        "center_offset_norm": center_offset_norm,
        "v_center": v_center,
        "v_total": float(v_total),
        "mask_bbox_x1": pred_bbox_vals[0],
        "mask_bbox_y1": pred_bbox_vals[1],
        "mask_bbox_x2": pred_bbox_vals[2],
        "mask_bbox_y2": pred_bbox_vals[3],
        "box_x1": prompt_box[0],
        "box_y1": prompt_box[1],
        "box_x2": prompt_box[2],
        "box_y2": prompt_box[3],
        "mask_centroid_x": pred_center_vals[0],
        "mask_centroid_y": pred_center_vals[1],
        "box_centroid_x": box_center_vals[0],
        "box_centroid_y": box_center_vals[1],
    }

def compute_boundary_uncertainty(prob, pred_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    prob = np.asarray(prob, dtype=np.float32)

    dilated = dilate_mask(pred_mask, BOUNDARY_BAND_RADIUS)
    eroded = erode_mask(pred_mask, BOUNDARY_BAND_RADIUS)
    band = dilated & (~eroded)

    if band.sum() == 0:
        uncertainty = 0.0
    else:
        pixel_uncertainty = 4.0 * prob * (1.0 - prob)
        uncertainty = float(pixel_uncertainty[band].mean())

    return {
        "boundary_uncertainty": uncertainty,
        "boundary_band_area": int(band.sum()),
    }

def compute_structure_info(prompt_box, img_h, img_w, num_boxes_in_json):
    ratio = box_area(prompt_box) / max(1.0, float(img_h * img_w))
    touches_border = (
        prompt_box[0] <= OV_BORDER_DELTA
        or prompt_box[1] <= OV_BORDER_DELTA
        or prompt_box[2] >= img_w - OV_BORDER_DELTA
        or prompt_box[3] >= img_h - OV_BORDER_DELTA
    )
    return {
        "num_boxes_in_json": int(num_boxes_in_json),
        "used_enclosing_union_box": int(num_boxes_in_json > 1),
        "box_area_ratio": float(ratio),
        "is_SO": int(ratio <= SO_STRUCTURE_RATIO_THR),
        "is_OV": int(touches_border),
    }

def compute_bad_response_flags(
    pred_mask,
    box_mask,
    prompt_box,
    img_h,
    img_w,
    is_ov,
):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    box_mask = np.asarray(box_mask, dtype=bool)

    mask_area_pixels = int(pred_mask.sum())
    box_area_pixels = int(box_mask.sum())
    image_area_pixels = int(img_h * img_w)

    mask_image_ratio = float(mask_area_pixels / max(1, image_area_pixels))
    box_image_ratio = float(box_area_pixels / max(1, image_area_pixels))
    mask_box_ratio = float(mask_area_pixels / max(1, box_area_pixels))

    inside_pixels = int((pred_mask & box_mask).sum())
    outside_pixels = int((pred_mask & (~box_mask)).sum())
    inside_ratio = float(inside_pixels / max(1, mask_area_pixels))
    outside_ratio = float(outside_pixels / max(1, mask_area_pixels))

    component_stats = effective_component_stats(pred_mask)
    effective_num_components = component_stats["effective_num_components"]
    largest_effective_ratio = component_stats["largest_effective_component_ratio"]
    pred_bbox = mask_bbox(pred_mask)

    if pred_bbox is None:
        mask_bbox_area = 0.0
        rectangularity = 0.0
        mask_box_bbox_iou = 0.0
    else:
        mask_bbox_area = max(0.0, pred_bbox[2] - pred_bbox[0]) * max(
            0.0, pred_bbox[3] - pred_bbox[1]
        )
        rectangularity = float(mask_area_pixels / max(1.0, mask_bbox_area))
        mask_box_bbox_iou = bbox_iou(pred_bbox, prompt_box)

    is_empty = int(mask_area_pixels == 0)
    is_tiny_collapse = int(
        mask_area_pixels > 0 and mask_box_ratio < TINY_MASK_BOX_RATIO_THR
    )

    # 普通碎片：作为可精修风险，建议降到 Middle。
    is_fragmented = int(
        effective_num_components >= FRAGMENT_MIN_COMPONENTS
        and largest_effective_ratio < FRAGMENT_MAX_LARGEST_COMPONENT_RATIO
    )
    # 严重碎片：才作为确定性异常进入 is_bad_response / Unvalue。
    is_severe_fragmented = int(
        effective_num_components >= SEVERE_FRAGMENT_MIN_COMPONENTS
        and largest_effective_ratio < SEVERE_FRAGMENT_MAX_LARGEST_COMPONENT_RATIO
    )

    outside_thr = OV_OUTSIDE_LEAK_RATIO_THR if is_ov else OUTSIDE_LEAK_RATIO_THR
    is_outside_leak = int(outside_ratio > outside_thr)

    is_box_copy = int(
        mask_box_ratio >= BOX_COPY_FILL_RATIO_THR
        and mask_box_bbox_iou >= BOX_COPY_BBOX_IOU_THR
        and rectangularity >= BOX_COPY_RECTANGULARITY_THR
    )

    reversal_by_outside = (
        outside_ratio > REVERSAL_OUTSIDE_RATIO_THR
        and inside_ratio < REVERSAL_INSIDE_RATIO_THR
    )
    reversal_by_global_background = (
        mask_image_ratio > REVERSAL_IMAGE_AREA_RATIO_THR
        and box_image_ratio < REVERSAL_SMALL_BOX_IMAGE_RATIO_THR
    )
    is_reversal = int(reversal_by_outside or reversal_by_global_background)

    is_bad_response = int(
        bool(
            is_empty
            or is_tiny_collapse
            or is_severe_fragmented
            or is_outside_leak
            or is_box_copy
            or is_reversal
        )
    )
    is_refinable_risk = int(bool(is_fragmented and not is_severe_fragmented))

    return {
        "mask_area_pixels": mask_area_pixels,
        "mask_image_area_ratio": mask_image_ratio,
        "mask_box_area_ratio": mask_box_ratio,
        "inside_mask_ratio": inside_ratio,
        "outside_mask_ratio": outside_ratio,
        **component_stats,
        # 兼容旧字段名。
        "num_components": component_stats["raw_num_components"],
        "mask_rectangularity": rectangularity,
        "mask_bbox_iou_with_box": mask_box_bbox_iou,
        "is_empty_response": is_empty,
        "is_tiny_collapse": is_tiny_collapse,
        "is_fragmented_response": is_fragmented,
        "is_severe_fragmented_response": is_severe_fragmented,
        "is_outside_leak": is_outside_leak,
        "is_box_copy": is_box_copy,
        "is_reversal_response": is_reversal,
        "is_refinable_risk": is_refinable_risk,
        "is_bad_response": is_bad_response,
    }

def assign_q_ranks(image_rows):
    """
    在 high/middle 可用样本中分别保存：
      - 区域可靠性排名 q_region；
      - 形状可靠性排名 q_shape；
      - 双维平衡参考排名 q_dual。

    q_rank_global / q_rank_within_pool 保留为 q_dual 的兼容别名。
    """
    metric_specs = [
        ("q_region", "q_region_rank_global", "q_region_rank_within_pool"),
        ("q_shape", "q_shape_rank_global", "q_shape_rank_within_pool"),
        ("q_dual", "q_dual_rank_global", "q_dual_rank_within_pool"),
    ]

    eligible_indices = [
        idx
        for idx, row in enumerate(image_rows)
        if row["recommended_pool"] in {"high", "middle"}
    ]

    for metric_name, global_field, pool_field in metric_specs:
        sorted_global = sorted(
            eligible_indices,
            key=lambda idx: safe_float(
                image_rows[idx].get(metric_name),
                default=-1.0,
            ),
            reverse=True,
        )
        for rank, idx in enumerate(sorted_global, start=1):
            image_rows[idx][global_field] = rank

        for pool_name in ["high", "middle"]:
            pool_indices = [
                idx
                for idx in eligible_indices
                if image_rows[idx]["recommended_pool"] == pool_name
            ]
            pool_indices.sort(
                key=lambda idx: safe_float(
                    image_rows[idx].get(metric_name),
                    default=-1.0,
                ),
                reverse=True,
            )
            for rank, idx in enumerate(pool_indices, start=1):
                image_rows[idx][pool_field] = rank

    for row in image_rows:
        row["q_rank_global"] = row.get("q_dual_rank_global", -1)
        row["q_rank_within_pool"] = row.get("q_dual_rank_within_pool", -1)


# --------- Pipeline consuming refined masks without another SAM2 call ---------

def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save_csv(path, records, preferred_fields=None):
    records = list(records)
    if not records:
        return
    names = list(preferred_fields or [])
    for record in records:
        for key in record:
            if key not in names:
                names.append(key)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def read_gray_mask(path, expected_hw):
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f'Cannot read refined mask: {path}')
    if arr.shape != expected_hw:
        raise ValueError(f'Refined mask shape {arr.shape} != image shape {expected_hw}: {path}')
    return arr >= 128


def find_base_prob(row, candidate_csv, candidate_prob_dir, image_name):
    candidates = []
    if candidate_prob_dir:
        candidates.append(Path(candidate_prob_dir) / (image_name + '.npy'))
    saved = row.get('prob_48_path', '').strip()
    if saved:
        cand = Path(saved)
        candidates.append(cand)
        if not cand.is_absolute():
            candidates.append(Path(candidate_csv).resolve().parent.parent.parent / saved)
            candidates.append(Path(saved).resolve())
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_prob(path, expected_hw):
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, np.float32)
    if arr.ndim != 2:
        raise ValueError(f'Expected [H,W] probability, got {arr.shape}: {path}')
    if arr.shape != expected_hw:
        arr = cv2.resize(arr, (expected_hw[1], expected_hw[0]), interpolation=cv2.INTER_LINEAR)
    if not np.isfinite(arr).all():
        raise ValueError(f'NaN or Inf in probability: {path}')
    return np.clip(arr, 0., 1.)


def calc_metrics(pred_bool, gt_path, image_hw):
    result = {'has_gt': 0, 'gt_area_ratio': np.nan, 'IoU': np.nan,
              'MAE': np.nan, 'Smeasure': np.nan, 'Emeasure_adp': np.nan,
              'Fmeasure_adp': np.nan, 'gt_bad_by_iou': -1,
              'gt_bad_by_mae': -1, 'gt_is_bad': -1}
    if not gt_path.is_file():
        return result
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return result
    if gt.shape != image_hw:
        gt = cv2.resize(gt, (image_hw[1],image_hw[0]),interpolation=cv2.INTER_NEAREST)
    gt = gt >= 128
    mae = float(np.mean(np.abs(pred_bool.astype(np.float32)-gt.astype(np.float32))))
    iou = compute_iou(pred_bool,gt)
    result.update({'has_gt':1, 'gt_area_ratio':float(gt.mean()), 'IoU':iou, 'MAE':mae,
                   'gt_bad_by_iou':int(iou < BAD_IOU_THR),
                   'gt_bad_by_mae':int(mae > BAD_MAE_THR),
                   'gt_is_bad':int(iou < BAD_IOU_THR or mae > BAD_MAE_THR)})
    # Optional py_sod_metrics to match original code-2 metric names.
    try:
        from py_sod_metrics import Smeasure, Emeasure, Fmeasure
        p = (pred_bool.astype(np.uint8)*255)
        g = (gt.astype(np.uint8)*255)
        sm,em,fm = Smeasure(),Emeasure(),Fmeasure()
        for metric in (sm,em,fm):
            metric.step(pred=p, gt=g)
        result['Smeasure']=float(sm.get_results()['sm'])
        result['Emeasure_adp']=float(em.get_results()['em']['adp'])
        result['Fmeasure_adp']=float(fm.get_results()['fm']['adp'])
    except (ImportError, KeyError, TypeError, ValueError, AttributeError):
        pass
    return result


def recalc_one(image_name, old_row, args):
    mask_path = Path(args.refined_mask_dir) / (image_name + '.png')
    if not mask_path.is_file():
        raise FileNotFoundError(f'No refined mask for {image_name}: {mask_path}')
    # Image dimension from original RGB to avoid silent mismatch.
    image_path = None
    for ext in ('.jpg','.jpeg','.png','.bmp','.JPG','.PNG'):
        candidate = Path(args.img_dir)/(image_name+ext)
        if candidate.is_file():
            image_path=candidate
            break
    if image_path is None:
        raise FileNotFoundError(f'No image found for {image_name} in {args.img_dir}')
    image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Invalid image file: {image_path}')
    h,w=image.shape[:2]
    pred=read_gray_mask(mask_path,(h,w))

    # Source of S and original SAM predicted IoU/stability: CSV from code 2.
    s=safe_float(old_row.get('candidate_consensus_s'),default=np.nan)
    sam_score=safe_float(old_row.get('sam_score_selected'),default=np.nan)
    stability=safe_float(old_row.get('sam_stability_selected'),default=np.nan)
    if not all(np.isfinite(q) for q in (s,sam_score,stability)):
        raise ValueError(f'Missing candidate S / SAM score / stability for {image_name}')

    refined_prob_path=Path(args.refined_prob_dir)/(image_name+'.npy')
    if refined_prob_path.is_file():
        prob=load_prob(refined_prob_path,(h,w))
        source='refined'
        # The hard masks were produced by thresholding exactly these probabilities.
        disagree=float(np.mean((prob >= .5)!=pred))
        if disagree > args.max_prob_mask_disagreement:
            raise ValueError(f'Probability and refined mask disagree on {disagree:.2%} pixels: {image_name}')
    elif args.prob_fallback=='base48':
        saved=find_base_prob(old_row,args.candidate_csv,args.candidate_prob_dir,image_name)
        if saved is None:
            raise FileNotFoundError(f'Refined probability absent AND initial 48x48 probability unavailable: {image_name}. '
                                    'Rerun code 1 with --no-flip-gate --save-probs.')
        prob=load_prob(saved,(h,w))
        source='base48_proxy'
    else:
        raise FileNotFoundError(f'No refined probability for {image_name}: {refined_prob_path}. '
                                'Rerun code 1 with --no-flip-gate --save-probs, '
                                'or choose --prob-fallback base48 (approximate Q).')

    json_path=Path(args.json_dir)/(image_name+'.json')
    boxes=extract_all_boxes_from_json(str(json_path))
    prompt_box, valid_boxes=build_single_prompt_box(boxes,h,w)
    if prompt_box is None:
        raise ValueError(f'Invalid or missing box for {image_name}: {json_path}')
    box_mask=box_to_mask(prompt_box,h,w)
    structure=compute_structure_info(prompt_box,h,w,num_boxes_in_json=len(valid_boxes))

    q=compute_q_evidence(prob,pred,box_mask,sam_score,stability,s)
    v=compute_v_violation(pred,box_mask,prompt_box)
    boundary=compute_boundary_uncertainty(prob,pred)
    anomaly=compute_bad_response_flags(pred,box_mask,prompt_box,h,w,bool(structure['is_OV']))
    grade=candidate_grade_from_s(s)
    force=int(grade=='low' or q['fails_quality_gate']==1 or anomaly['is_bad_response']==1)
    if force:
        pool='unvalue'
    elif grade=='middle' or anomaly['is_refinable_risk']==1:
        pool='middle'
    else:
        pool='high'
    record=dict(old_row)
    record.update({'image':image_name, 'image_width':w, 'image_height':h,
                   'hard_mask_path':str(mask_path), 'prob_48_path':'',
                   'source_mask':'Refine_NoFlip', 'q_prob_source':source,
                   'old_recommended_pool':old_row.get('recommended_pool',''),
                   'candidate_consensus_s':s, 'candidate_s_grade':grade,
                   'sam_score_selected':sam_score,'sam_stability_selected':stability,
                   **structure,
                   'box_x1':prompt_box[0],'box_y1':prompt_box[1],
                   'box_x2':prompt_box[2],'box_y2':prompt_box[3],
                   **q,**v,**boundary,**anomaly,
                   'force_unvalue':force,'recommended_pool':pool,
                   'q_region_rank_global':-1,'q_shape_rank_global':-1,
                   'q_dual_rank_global':-1,'q_region_rank_within_pool':-1,
                   'q_shape_rank_within_pool':-1,'q_dual_rank_within_pool':-1,
                   'q_rank_global':-1,'q_rank_within_pool':-1,
                   'is_BA':0,'ba_threshold':np.nan})
    # Save only truly-refined probabilities as a refined-prob 48x48 output.
    if source=='refined':
        dst=Path(args.output_dir)/'Prob_48x48_FP16'/(image_name+'.npy')
        dst.parent.mkdir(parents=True,exist_ok=True)
        np.save(dst,cv2.resize(prob,(48,48),interpolation=cv2.INTER_AREA).astype(np.float16),allow_pickle=False)
        record['prob_48_path']=str(dst)
    record.update(calc_metrics(pred,Path(args.gt_dir)/(image_name+'.png'),(h,w)) if args.eval_gt
                  else {'has_gt':0,'gt_area_ratio':np.nan,'IoU':np.nan,'MAE':np.nan,
                        'Smeasure':np.nan,'Emeasure_adp':np.nan,'Fmeasure_adp':np.nan,
                        'gt_bad_by_iou':-1,'gt_bad_by_mae':-1,'gt_is_bad':-1})
    # Old base-source bad visualization path must not be confused with a refined one.
    record['bad_vis_path']=''
    return record


def produce(args):
    csv_path=Path(args.candidate_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f'Original code-2 CSV required for candidate S, SAM score/stability: {csv_path}. '
                                'Please run original code 2 once to create it.')
    if not Path(args.refined_mask_dir).is_dir():
        raise FileNotFoundError(f'Cannot find refined masks: {args.refined_mask_dir}')
    old_rows=read_csv_rows(csv_path)
    if not old_rows:
        raise ValueError('Empty original candidate CSV')
    # Explicitly check for duplicate image ids before any processing.
    image_ids=[str(row['image']).strip() for row in old_rows]
    if len(set(image_ids))!=len(image_ids):
        raise ValueError('Duplicate image ids in original CSV')
    out=Path(args.output_dir)
    (out/'csv').mkdir(parents=True,exist_ok=True)
    rows=[]
    errors=[]
    mode_notice=False
    for old_row in tqdm(old_rows[:args.limit] if args.limit else old_rows,desc='Refined original pool'):
        image_name=str(old_row['image']).strip()
        try:
            result=recalc_one(image_name,old_row,args)
            rows.append(result)
            if result['q_prob_source']=='base48_proxy' and not mode_notice:
                tqdm.write('[WARNING] No refine_prob found: using original SAM2 base 48x48 probability as an ANCHOR PROXY; '
                           'Q_region and gate results are approximate. For formal experiments rerun code 1 '
                           'with --no-flip-gate --save-probs.')
                mode_notice=True
        except Exception as exc:
            errors.append({'image':image_name,'error':str(exc)})
            if len(errors)<=15:
                tqdm.write('[ERROR] '+image_name+': '+str(exc))
    rows.sort(key=lambda x:x['image'])
    if not rows:
        raise RuntimeError(f'No valid samples. First error: {errors[:1]}')
    if errors and not args.allow_partial:
        path=out/'errors.json'
        path.write_text(json.dumps(errors,indent=2,ensure_ascii=False),encoding='utf-8')
        raise RuntimeError(f'{len(errors)} image(s) failed. Refusing to save partial pools. Errors: {path}; '
                           'use --allow-partial only if incomplete processing is intentional.')

    boundary_vals=np.array([safe_float(r.get('boundary_uncertainty')) for r in rows],np.float32)
    valid=boundary_vals[np.isfinite(boundary_vals)]
    ba_threshold=float(np.quantile(valid,1.0-BA_TOP_RATIO)) if valid.size else np.nan
    for row in rows:
        row['ba_threshold']=ba_threshold
        b=safe_float(row.get('boundary_uncertainty'))
        row['is_BA']=int(np.isfinite(b) and np.isfinite(ba_threshold) and b>=ba_threshold)
    assign_q_ranks(rows)
    for pool in ('high','middle','unvalue'):
        subset=[r for r in rows if r['recommended_pool']==pool]
        # TXT image IDs, matching usual curriculum file naming conventions.
        (out/f'{pool}.txt').write_text(''.join(r['image']+'\n' for r in subset),encoding='utf-8')
        # An additional filename-based list for Dataset loaders expecting .png.
        (out/f'{pool}_masks.txt').write_text(''.join(str(Path(args.refined_mask_dir)/(r['image']+'.png'))+'\n' for r in subset),encoding='utf-8')
        save_csv(out/'csv'/f'{pool}.csv',subset)
    save_csv(out/'csv'/'image_level_metrics.csv',rows, list(old_rows[0].keys()))
    transitions=Counter((r['old_recommended_pool'],r['recommended_pool']) for r in rows)
    count=Counter(r['recommended_pool'] for r in rows)
    ref_eval=[r for r in rows if r['has_gt']==1]
    summary={'total':len(rows), 'failed':len(errors), 'high':count['high'],
             'middle':count['middle'], 'unvalue':count['unvalue'],
             'probability_sources':dict(Counter(r['q_prob_source'] for r in rows)),
             'candidate_information':'original box-only three-candidate CSV',
             'mask_information':'SAM2 active refined NO-FLIP (may use human scribbles)',
             'pool_rule':'low S OR failed Q gate OR bad mask => unvalue; middle S / refinable fragmentation => middle; else high',
             'transition':{f'{a} -> {b}':n for (a,b),n in sorted(transitions.items())},
             'boundary_ambiguity_threshold':ba_threshold if np.isfinite(ba_threshold) else None,
             'gt_evaluated':len(ref_eval),
             'gt_mean_IoU':float(np.mean([float(r['IoU']) for r in ref_eval])) if ref_eval else None,
             'gt_mean_MAE':float(np.mean([float(r['MAE']) for r in ref_eval])) if ref_eval else None}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n'+'='*72)
    print(f'Total={len(rows)}  High={count["high"]}  Middle={count["middle"]}  Unvalue={count["unvalue"]}')
    print('Quality source:',summary['probability_sources'])
    print('CSV:',out/'csv'/'image_level_metrics.csv')
    print('Pool TXT:',*[str(out/f'{x}.txt') for x in ('high','middle','unvalue')],sep='\n - ')
    print('Summary:',out/'summary.json')
    print('='*72)
    return rows,summary


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--candidate-csv',default='./SGT/SAM2_ThreeCandidate_DualQuality/csv/image_level_metrics.csv')
    parser.add_argument('--candidate-prob-dir',default='./SGT/SAM2_ThreeCandidate_DualQuality/Prob_48x48_FP16')
    parser.add_argument('--refined-mask-dir',default='./softresult/SAM2_Optimized3/03_Refine_NoFlip')
    parser.add_argument('--refined-prob-dir',default='./softresult/SAM2_Optimized3/refine_prob')
    parser.add_argument('--img-dir',default='./data/Train/Imgs')
    parser.add_argument('--json-dir',default='./data/Train/box_label')
    parser.add_argument('--gt-dir',default='./data/Train/GT')
    parser.add_argument('--output-dir',default='./SGT/SAM2_RefineNoFlip_OriginalPools')
    parser.add_argument('--prob-fallback',choices=['base48','error'],default='base48',
                        help='base48 is an approximate proxy if refined probs were not saved. '
                             'error requires genuine refined probabilities for exact Q_region.')
    parser.add_argument('--max-prob-mask-disagreement',type=float,default=0.001)
    parser.add_argument('--eval-gt',action='store_true',help='Use GT for diagnostics ONLY, NEVER pool assignment')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--allow-partial',action='store_true')
    return parser.parse_args(argv)


if __name__=='__main__':
    produce(parse_args())
