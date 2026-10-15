#!/usr/bin/env python3
"""SAM1 -> three-candidate SIM -> TCRQ -> conditional FAVV.
GT is used only after generation for evaluation. Run --help for CLI.
Merged from the two user-supplied scripts; NumPy-only ranking is also importable.
"""
import argparse
import csv
import json
import math
import traceback
from pathlib import Path
from collections import Counter
import os
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
# Loaded only for SAM inference / optional original DPC+RU difficulty.
torch = None
SOD_CLASSES = None
DEVICE = "cpu"


DPC_K = 10

DPC_ALPHA = 0.10

DPC_FEATURE_MODE = "box_roi"

DPC_BATCH_SIZE = 256

ETC_BORDER_MIN_CONTACT_RATIO = 0.02

ETC_BORDER_MIN_PIXELS = 8

ETC_SAFE_BOX_MARGIN_RATIO = 0.01

ETC_SEVERE_OUTSIDE_RATIO = 0.35

ETC_SEVERE_MASK_IMAGE_RATIO = 0.20

MULTI_BOX_POLICY = "enclosing_union_box"

BOUNDARY_BAND_RADIUS = 3

OV_BORDER_DELTA = 4

SO_STRUCTURE_RATIO_THR = 0.10

TINY_MASK_BOX_RATIO_THR = 0.03

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

def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

def compute_iou(pred_bool, gt_bool):
    pred_bool = np.asarray(pred_bool, dtype=bool)
    gt_bool = np.asarray(gt_bool, dtype=bool)
    inter = int(np.logical_and(pred_bool, gt_bool).sum())
    union = int(np.logical_or(pred_bool, gt_bool).sum())
    if union == 0:
        return 1.0
    return float(inter / union)

def compute_binary_ssim(mask_a, mask_b):
    """
    计算两个二值候选之间的标准局部 SSIM，不依赖 scikit-image。
    输入值域为 [0, 1]，采用经典 11×11 Gaussian 窗口；极小图像退化为全局 SSIM。
    """
    a = np.asarray(mask_a, dtype=np.float32)
    b = np.asarray(mask_b, dtype=np.float32)

    if a.shape != b.shape:
        raise ValueError(f"SSIM shape mismatch: {a.shape} vs {b.shape}")

    if np.array_equal(a, b):
        return 1.0

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    min_side = min(a.shape[-2:])

    if min_side < 3:
        mu_a = float(a.mean())
        mu_b = float(b.mean())
        var_a = float(a.var())
        var_b = float(b.var())
        cov_ab = float(((a - mu_a) * (b - mu_b)).mean())
        numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2)
        denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
        value = numerator / max(denominator, 1e-12)
        return float(np.clip(value, 0.0, 1.0))

    window = min(11, min_side)
    if window % 2 == 0:
        window -= 1
    window = max(3, window)

    mu_a = cv2.GaussianBlur(a, (window, window), 1.5)
    mu_b = cv2.GaussianBlur(b, (window, window), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, (window, window), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, (window, window), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (window, window), 1.5) - mu_ab

    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))

def compute_candidate_consensus(candidate_masks):
    """
    三候选两两相似度：

      SIM(Va, Vb) = 0.5 * [IoU(Va, Vb) + SSIM(Va, Vb)]
      S = mean(SIM12, SIM13, SIM23)

    IoU、SSIM 和 SIM 分开返回并写入 CSV，便于后续做分量消融。
    """
    masks = np.asarray(candidate_masks, dtype=bool)
    pairs = [(0, 1, "12"), (0, 2, "13"), (1, 2, "23")]

    result = {
        "candidate_iou_12": np.nan,
        "candidate_iou_13": np.nan,
        "candidate_iou_23": np.nan,
        "candidate_ssim_12": np.nan,
        "candidate_ssim_13": np.nan,
        "candidate_ssim_23": np.nan,
        "candidate_sim_12": np.nan,
        "candidate_sim_13": np.nan,
        "candidate_sim_23": np.nan,
        "candidate_consensus_s": 0.0,
        "candidate_sim_min": np.nan,
        "candidate_sim_max": np.nan,
        "candidate_sim_range": np.nan,
    }

    if masks.ndim != 3 or masks.shape[0] < 2:
        return result

    sims = []
    for a, b, suffix in pairs:
        if a >= masks.shape[0] or b >= masks.shape[0]:
            continue

        iou = float(compute_iou(masks[a], masks[b]))
        ssim = float(compute_binary_ssim(masks[a], masks[b]))
        sim = float(0.5 * (iou + ssim))

        result[f"candidate_iou_{suffix}"] = iou
        result[f"candidate_ssim_{suffix}"] = ssim
        result[f"candidate_sim_{suffix}"] = sim
        sims.append(sim)

    if sims:
        result["candidate_consensus_s"] = float(np.mean(sims))
        result["candidate_sim_min"] = float(np.min(sims))
        result["candidate_sim_max"] = float(np.max(sims))
        result["candidate_sim_range"] = float(np.max(sims) - np.min(sims))

    return result

def calculate_sam_stability_score(
    mask_logit,
    mask_threshold,
    threshold_offset=1.0,
):
    """
    SAM 官方阈值稳定性思想：
    在 threshold + offset 与 threshold - offset 下分别二值化，计算两者 IoU。

    由于高阈值区域必然是低阈值区域的子集，交集为 high_mask 面积，
    并集为 low_mask 面积。空低阈值区域返回 0，用于筛掉空/塌缩预测。
    """
    logit = np.asarray(mask_logit, dtype=np.float32)
    high_mask = logit > (float(mask_threshold) + float(threshold_offset))
    low_mask = logit > (float(mask_threshold) - float(threshold_offset))

    union = int(low_mask.sum())
    if union == 0:
        return 0.0

    intersection = int(high_mask.sum())
    return float(intersection / union)

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

def extract_sam_encoder_feature(predictor, prompt_box):
    """DPC embedding from SAM's ONE image encoding (Box-only).

    SAM1 image encoder returns [1,C,Hf,Wf] on the padded square image.
    ROI coordinates must first pass the SAME ResizeLongestSide transform as SAM.
    No GT, point, scribble, or mask embedding is ever used.
    """
    feat = predictor.get_image_embedding()
    if feat.ndim != 4 or feat.shape[0] != 1:
        raise RuntimeError(f"unexpected SAM embedding shape: {tuple(feat.shape)}")
    if DPC_FEATURE_MODE == "global":
        embedding = feat.mean(dim=(2, 3))[0]
    elif DPC_FEATURE_MODE == "box_roi":
        image_size = float(predictor.model.image_encoder.img_size)
        mapped = predictor.transform.apply_boxes(
            np.asarray([prompt_box], dtype=np.float32), predictor.original_size
        )[0]
        _, _, h_f, w_f = feat.shape
        x0 = max(0, min(w_f - 1, int(np.floor(float(mapped[0]) * w_f / image_size))))
        y0 = max(0, min(h_f - 1, int(np.floor(float(mapped[1]) * h_f / image_size))))
        x1 = max(x0 + 1, min(w_f, int(np.ceil(float(mapped[2]) * w_f / image_size))))
        y1 = max(y0 + 1, min(h_f, int(np.ceil(float(mapped[3]) * h_f / image_size))))
        embedding = feat[0, :, y0:y1, x0:x1].mean(dim=(1, 2))
    else:
        raise ValueError(f"Unknown DPC_FEATURE_MODE: {DPC_FEATURE_MODE}")
    vec = embedding.detach().float().cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(vec)) or not np.any(vec):
        raise RuntimeError("SAM encoder ROI embedding is invalid")
    return vec / max(float(np.linalg.norm(vec)), 1e-12)

def compute_sam_ru(prob, binary_mask):
    """ICCL Eq.(5), but on fixed SAM1 probabilities: SAM uncertainty, not student difficulty."""
    binary_mask = np.asarray(binary_mask, bool)
    if not np.any(binary_mask):
        return 1.0  # empty mask rejected separately
    selected = np.asarray(prob, np.float32)[binary_mask]
    return float(np.mean(1.0 - np.abs(2.0 * selected - 1.0)))

def compute_etc_severe(mask, box):
    """Conservative edge truncation diagnostic, NOT official GME ETC implementation.

    Legacy diagnostic only (does not gate FAVV). Flags when unexpected substantial foreground contact occurs on >=2
    distinct image edges, or on one edge with overwhelming outside-box leakage
    and a large predicted foreground. Expected contact for border-touching boxes
    is not an error. This deliberately protects valid border objects.
    """
    m = np.asarray(mask, bool)
    h, w = m.shape
    fg_pixels = int(m.sum())
    box_mask = box_to_mask(box, h, w)
    leakage = int((m & ~box_mask).sum()) / max(fg_pixels, 1)
    contacts = [int(m[0].sum()), int(m[-1].sum()),
                int(m[:, 0].sum()), int(m[:, -1].sum())]
    limits = [max(ETC_BORDER_MIN_PIXELS, int(np.ceil(w * ETC_BORDER_MIN_CONTACT_RATIO))),
              max(ETC_BORDER_MIN_PIXELS, int(np.ceil(w * ETC_BORDER_MIN_CONTACT_RATIO))),
              max(ETC_BORDER_MIN_PIXELS, int(np.ceil(h * ETC_BORDER_MIN_CONTACT_RATIO))),
              max(ETC_BORDER_MIN_PIXELS, int(np.ceil(h * ETC_BORDER_MIN_CONTACT_RATIO)))]
    margin_x = max(3., w * ETC_SAFE_BOX_MARGIN_RATIO)
    margin_y = max(3., h * ETC_SAFE_BOX_MARGIN_RATIO)
    expected = [float(box[1]) <= margin_y,
                float(box[3]) >= h - 1 - margin_y,
                float(box[0]) <= margin_x,
                float(box[2]) >= w - 1 - margin_x]
    unexpected = [int(c >= t and not e) for c, t, e in zip(contacts, limits, expected)]
    num_unexpected = sum(unexpected)
    severe = int(num_unexpected >= 2 or
                 (num_unexpected >= 1 and leakage >= ETC_SEVERE_OUTSIDE_RATIO
                  and fg_pixels / (h*w) >= ETC_SEVERE_MASK_IMAGE_RATIO))
    return {
        "etc_severe": severe,
        "etc_unexpected_edges": int(num_unexpected),
        "etc_total_contacts": int(sum(c >= t for c,t in zip(contacts, limits))),
        "etc_top_contact": contacts[0],
        "etc_bottom_contact": contacts[1],
        "etc_left_contact": contacts[2],
        "etc_right_contact": contacts[3],
    }

def rank_difficulty_with_dpc(image_rows, image_embeddings):
    """Dataset-global DPC KNN density, then DPC+RU ranking WITHIN quality pools.

    Density computed in l2-normalized SAM encoder embedding space, all images.
    DPC rank is NOT used to move images between quality pools.
    """
    if not image_rows:
        return {"samples": 0}
    keys = [r["image"] for r in image_rows]
    x = np.stack([image_embeddings[k] for k in keys]).astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n = len(keys)
    k = min(DPC_K, n-1)
    if k <= 0:
        distances = np.zeros(n, dtype=np.float32)
    else:
        # Pure torch chunked cdist works without scipy/sklearn and bounds memory.
        feat = torch.from_numpy(x).to(device=DEVICE)
        distances = np.empty(n, np.float32)
        with torch.inference_mode():
            for start in range(0, n, DPC_BATCH_SIZE):
                end = min(n, start + DPC_BATCH_SIZE)
                d = torch.cdist(feat[start:end], feat, p=2)
                local_rows = torch.arange(end-start, device=d.device)
                d[local_rows, torch.arange(start, end, device=d.device)] = float("inf")
                neighbors = torch.topk(d, k=k, largest=False).values
                distances[start:end] = neighbors.mean(dim=1).cpu().numpy()
        del feat
    density = 1.0 / (distances + 1e-6)
    lo, hi = float(density.min()), float(density.max())
    if hi - lo < 1e-10:
        intrinsic = np.zeros(n, dtype=np.float32)
    else:
        intrinsic = (1.0 - (density - lo) / (hi - lo)).astype(np.float32)
    raw_ru = np.asarray([r["sam_region_uncertainty"] for r in image_rows], np.float32)
    valid = np.asarray([r["recommended_pool"] in ("clean", "noisy") for r in image_rows], bool)
    ru_reference = raw_ru[valid] if np.any(valid) else raw_ru
    a, b = float(ru_reference.min()), float(ru_reference.max())
    ru_norm = (np.clip((raw_ru - a) / (b-a), 0., 1.)
               if b - a > 1e-10 else np.zeros_like(raw_ru))
    total = DPC_ALPHA * intrinsic + (1.0 - DPC_ALPHA) * ru_norm
    for idx, r in enumerate(image_rows):
        r["dpc_knn_mean_distance"] = float(distances[idx])
        r["dpc_density"] = float(density[idx])
        r["sam_intrinsic_difficulty"] = float(intrinsic[idx])
        r["sam_ru_normalized"] = float(ru_norm[idx])
        r["difficulty_score"] = float(total[idx])
        r["difficulty_rank_global"] = -1
        r["difficulty_rank_within_pool"] = -1
        r["difficulty_fraction_within_pool"] = float("nan")
    active = [i for i, r in enumerate(image_rows) if r["recommended_pool"] in ("clean", "noisy")]
    for rank, idx in enumerate(sorted(active, key=lambda i:(total[i],keys[i])), 1):
        image_rows[idx]["difficulty_rank_global"] = rank
    for group in ("clean", "noisy"):
        ix = sorted((i for i,r in enumerate(image_rows) if r["recommended_pool"] == group),
                    key=lambda i:(total[i],keys[i]))
        for rank, i in enumerate(ix, 1):
            image_rows[i]["difficulty_rank_within_pool"] = rank
            image_rows[i]["difficulty_fraction_within_pool"] = (rank-0.5)/len(ix)
    return {"samples": n, "effective_knn_k": int(k), "dpc_alpha": DPC_ALPHA,
            "dpc_feature_mode": DPC_FEATURE_MODE,
            "ru_min_valid": a, "ru_max_valid": b,
            "dpc_density_min": lo, "dpc_density_max": hi}

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

    # 普通碎片：仅诊断，不参与新 FAVV。
    is_fragmented = int(
        effective_num_components >= FRAGMENT_MIN_COMPONENTS
        and largest_effective_ratio < FRAGMENT_MAX_LARGEST_COMPONENT_RATIO
    )
    # 严重碎片标记；是否降级由条件化 FAVV 决定。
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

def percentile_midrank(values):
    """(average 1-based rank - .5) / N; exact ties, no rounding/clipping."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError("Ranking needs a finite 1D vector; invalid cues must be excluded first")
    n = len(a)
    result = np.empty(n, np.float64)
    order = np.argsort(a, kind="stable")
    start = 0
    while start < n:
        end = start + 1
        while end < n and a[order[end]] == a[order[start]]:
            end += 1
        result[order[start:end]] = (start + end) / (2.0 * n)
        start = end
    return result


def budget_groups(values, ids, clean_ratio=.25, unvalue_ratio=.10):
    """Descending reliability then ascending ID. Both tail counts use floor(N*r)."""
    if not (0 <= clean_ratio <= 1 and 0 <= unvalue_ratio <= 1 and
            clean_ratio + unvalue_ratio <= 1):
        raise ValueError("Ratios must be in [0,1] and sum <= 1")
    n = len(ids)
    if len(values) != n or len(set(ids)) != n:
        raise ValueError("Reliability and unique image IDs must have the same length")
    order = sorted(range(n), key=lambda i: (-float(values[i]), ids[i]))
    nh, nl = math.floor(n * clean_ratio), math.floor(n * unvalue_ratio)
    group = ["noisy"] * n
    ranks = [0] * n
    for rank, i in enumerate(order, 1):
        ranks[i] = rank
        if rank <= nh:
            group[i] = "clean"
        elif rank > n - nl:
            group[i] = "unvalue"
    return group, ranks


def conditional_favv(row, primary, outside_threshold=.15):
    """No mask editing / no promotion. Multi-object FGC guard is already in row."""
    reasons = []
    if int(row['is_empty_response']):
        reasons.append('EMPTY')
    if float(row['outside_mask_ratio']) > outside_threshold:
        reasons.append('OUTSIDE')
    if reasons:
        return 'unvalue', 'hard_reject', ';'.join(reasons)
    if int(row['fgc_severe']):
        if primary == 'clean':
            return 'noisy', 'downgrade', 'FGC_SEVERE'
        return primary, 'keep', 'FGC_KEEP_' + primary.upper()
    return primary, 'keep', 'NONE'


def qualify_rows(rows, clean_ratio=.25, unvalue_ratio=.10, mode='conditional',
                 method='tcrq', outside_threshold=.15):
    """Whole-dataset pass. No GT fields are read anywhere in this function."""
    if mode not in ('conditional', 'hard', 'off'):
        raise ValueError(mode)
    if method not in ('tcrq', 'min', 'sim'):
        raise ValueError(method)
    # Validate even for an empty dataset.
    budget_groups([], [], clean_ratio, unvalue_ratio)
    if len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image IDs')
    cues = ('sam_score', 'sam_stability', 'candidate_consensus_s')
    valid = [r for r in rows if all(np.isfinite(safe_float(r.get(k))) for k in cues)]
    ids = [r['image'] for r in valid]
    p = np.column_stack([percentile_midrank([r[k] for r in valid]) for k in cues])
    weak = np.sort(p, axis=1)
    scores = {
        'tcrq': np.sqrt(weak[:, 0] * weak[:, 1]),
        'min': weak[:, 0], 'sim': p[:, 2],
        # Leave-one-cue-out: geometric mean of the remaining two percentiles.
        'no_score': np.sqrt(p[:, 1] * p[:, 2]),
        'no_stability': np.sqrt(p[:, 0] * p[:, 2]),
    }
    group_map, rank_map = {}, {}
    for name, score in scores.items():
        group_map[name], rank_map[name] = budget_groups(score, ids, clean_ratio, unvalue_ratio)
    for i, r in enumerate(valid):
        r.update(p_score=float(p[i, 0]), p_stability=float(p[i, 1]), p_sim=float(p[i, 2]),
                 p_weakest=float(weak[i, 0]), p_second_weakest=float(weak[i, 1]),
                 tcrq_score=float(scores['tcrq'][i]), reliability_score=float(scores[method][i]),
                 reliability_rank_global=rank_map[method][i], invalid_cues=0)
        for name in scores:
            r['group_' + name] = group_map[name][i]
        primary = group_map[method][i]
        conditional, action, reason = conditional_favv(r, primary, outside_threshold)
        hard_flag = bool(int(r['is_empty_response']) or
                         float(r['outside_mask_ratio']) > outside_threshold)
        hard_all = 'unvalue' if hard_flag or int(r['fgc_severe']) else primary
        final = {'conditional': conditional, 'hard': hard_all, 'off': primary}[mode]
        r.update(primary_group=primary, conditional_group=conditional,
                 hard_all_group=hard_all, final_group=final, recommended_pool=final,
                 favv_action=action if mode == 'conditional' else mode,
                 favv_reason=reason, favv_mode=mode, hard_invalid=int(hard_flag),
                 force_unvalue=int(final == 'unvalue'),
                 favv_changed=int(final != primary),
                 favv_retained_vs_hard=int(hard_all == 'unvalue' and conditional != 'unvalue'),
                 score_effect=int(group_map['no_score'][i] != group_map['tcrq'][i]),
                 stability_effect=int(group_map['no_stability'][i] != group_map['tcrq'][i]),
                 sim_to_tcrq_changed=int(group_map['sim'][i] != group_map['tcrq'][i]))
    valid_ids = set(ids)
    for r in rows:
        if r['image'] not in valid_ids:
            # Nonfinite evidence is not rankable and never gets artificial high confidence.
            r.update({k: float('nan') for k in ('p_score','p_stability','p_sim',
                      'p_weakest','p_second_weakest','tcrq_score','reliability_score')})
            r.update({f'group_{k}': 'unvalue' for k in scores})
            r.update(invalid_cues=1, reliability_rank_global=-1,
                     primary_group='unvalue', conditional_group='unvalue', hard_all_group='unvalue',
                     final_group='unvalue', recommended_pool='unvalue', favv_action='invalid_cues',
                     favv_reason='NONFINITE_CUE', favv_mode=mode, hard_invalid=1, force_unvalue=1,
                     favv_changed=0, favv_retained_vs_hard=0, score_effect=0,
                     stability_effect=0, sim_to_tcrq_changed=0)
    for group in ('clean', 'noisy', 'unvalue'):
        members = sorted((r for r in rows if r['final_group'] == group),
                         key=lambda r: (-safe_float(r['reliability_score'], -float('inf')), r['image']))
        for rank, r in enumerate(members, 1):
            r['reliability_rank_within_pool'] = rank
    return {'rankable_count': len(valid), 'invalid_cues_count': len(rows)-len(valid),
            'initial_clean_count': math.floor(len(valid)*clean_ratio),
            'initial_unvalue_count': math.floor(len(valid)*unvalue_ratio),
            'method': method, 'favv_mode': mode,
            'final_counts': dict(Counter(r['final_group'] for r in rows))}


def generate_result(image_rgb, box, predictor, stability_offset=1., with_feature=True):
    predictor.set_image(image_rgb)
    logits, scores, _ = predictor.predict(point_coords=None, point_labels=None,
        box=np.asarray(box, np.float32), multimask_output=True, return_logits=True)
    logits = np.asarray(logits, np.float32)
    scores = np.asarray(scores, np.float32).reshape(-1)
    if logits.ndim != 3 or logits.shape[0] != 3 or scores.shape != (3,):
        raise RuntimeError(f'Expected exactly 3 SAM1 candidates, got {logits.shape}, {scores.shape}')
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(scores)):
        raise ValueError('SAM returned nonfinite logits or scores')
    threshold = float(predictor.model.mask_threshold)
    masks = logits > threshold
    selected = int(np.argmax(scores))
    stability = np.array([calculate_sam_stability_score(x, threshold, stability_offset)
                          for x in logits], np.float32)
    return dict(masks=masks, scores=scores, stability=stability, selected=selected,
        prob=sigmoid_np(logits[selected]),
        stability_low=logits[selected] > threshold-stability_offset,
        stability_high=logits[selected] > threshold+stability_offset,
        feature=extract_sam_encoder_feature(predictor, box) if with_feature else None,
        **compute_candidate_consensus(masks))


def save_cache(path, result):
    """Lossless packed masks; no pickle, no logits required for later visualization."""
    np.savez_compressed(path, shape=np.asarray(result['masks'].shape, np.int32),
        masks_bits=np.packbits(result['masks'].reshape(-1)),
        low_bits=np.packbits(result['stability_low'].reshape(-1)),
        high_bits=np.packbits(result['stability_high'].reshape(-1)),
        scores=result['scores'], stability=result['stability'],
        selected=np.asarray(result['selected'], np.int32))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['image']
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    # Portable strict JSON: NaN/Inf become null, never nonstandard JSON literals.
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False),
                          encoding='utf-8')


def process_image(path, predictor, args, out):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Image cannot be read')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    box_path = Path(args.json_dir) / (path.stem+'.json')
    box, boxes = build_single_prompt_box(extract_all_boxes_from_json(str(box_path)), h, w)
    if box is None:
        raise ValueError(f'Missing/invalid LabelMe boxes: {box_path}')
    with torch.inference_mode():
        result = generate_result(rgb, box, predictor, args.stability_offset, args.difficulty == 'dpc_ru')
    idx = result['selected']
    mask = result['masks'][idx]
    bmask = box_to_mask(box, h, w)
    structure = compute_structure_info(box, h, w, len(boxes))
    flags = compute_bad_response_flags(mask, bmask, box, h, w, bool(structure['is_OV']))
    hard_path = out/'Pred_Masks'/(path.stem+'.png')
    prob_path = out/'Prob_48x48_FP16'/(path.stem+'.npy')
    cache_path = out/'Candidates'/(path.stem+'.npz')
    if not cv2.imwrite(str(hard_path), mask.astype(np.uint8)*255):
        raise IOError(f'Cannot save {hard_path}')
    np.save(prob_path, cv2.resize(result['prob'], (48,48), interpolation=cv2.INTER_AREA).astype(np.float16),
            allow_pickle=False)
    save_cache(cache_path, result)
    row = dict(image=path.stem, image_path=str(path.resolve()), image_height=h, image_width=w,
               hard_mask_path=str(hard_path.relative_to(out)),
               prob_48_path=str(prob_path.relative_to(out)),
               candidate_cache_path=str(cache_path.relative_to(out)),
               selected_candidate_index=idx,
               sam_score=float(result['scores'][idx]), sam_stability=float(result['stability'][idx]),
               stability_offset=args.stability_offset,
               **structure, **flags, **compute_etc_severe(mask, box),
               **compute_boundary_uncertainty(result['prob'], mask),
               sam_region_uncertainty=compute_sam_ru(result['prob'], mask))
    row['sam_score_selected']=row['sam_score']
    row['sam_stability_selected']=row['sam_stability']
    row.update({f'box_{k}': float(v) for k,v in zip(('x1','y1','x2','y2'),box)})
    for i in range(3):
        row[f'candidate_score_{i+1}'] = float(result['scores'][i])
        row[f'candidate_stability_{i+1}'] = float(result['stability'][i])
    row.update({k:v for k,v in result.items() if k.startswith('candidate_')})
    row['fgc_severe'] = int(flags['is_severe_fragmented_response'] and len(boxes) <= 1)
    row['fgc_risk_not_rejected_multi_box'] = int(flags['is_severe_fragmented_response'] and len(boxes)>1)
    row.update(has_gt=0, gt_path='', IoU=float('nan'), MAE=float('nan'), gt_grade='unknown')
    if args.gt_dir:
        gt_path = Path(args.gt_dir)/(path.stem+'.png')
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else None
        if gt is not None:
            resized = gt.shape != (h,w)
            if resized:
                gt = cv2.resize(gt,(w,h),interpolation=cv2.INTER_NEAREST)
            target = gt > 128
            iou = compute_iou(mask,target)
            row.update(has_gt=1, gt_path=str(gt_path.resolve()), gt_resized=int(resized), IoU=iou,
                       MAE=float(np.mean(mask != target)),
                       gt_grade='good' if iou>.9 else ('bad' if iou<.7 else 'middle'))
    manifest = dict(image=path.stem, source_boxes=boxes, prompt_box=box,
                    image_height=h, image_width=w, multi_box_policy=MULTI_BOX_POLICY)
    if SOD_CLASSES is not None and row['has_gt']:
        sm,em,fm,mae = [cls() for cls in SOD_CLASSES]
        for metric in (sm,em,fm,mae):
            metric.step(pred=mask.astype(np.uint8)*255, gt=target.astype(np.uint8)*255)
        row.update(Smeasure=float(sm.get_results()['sm']),
                   Emeasure_adp=float(em.get_results()['em']['adp']),
                   Fmeasure_adp=float(fm.get_results()['fm']['adp']))
    return row, manifest, result['feature']


def export_outputs(rows, out, summary, manifests=None):
    pools = out/'pools'
    pools.mkdir(exist_ok=True)
    for group in ('clean','noisy','unvalue'):
        members = [r for r in rows if r['final_group']==group]
        ordered = sorted(members,key=lambda r:(r['reliability_rank_within_pool'],r['image']))
        (pools/(group+'.txt')).write_text(''.join(r['image']+'\n' for r in ordered),encoding='utf-8')
        (pools/(group+'_filenames.txt')).write_text(''.join(r['image']+'.png\n' for r in ordered),encoding='utf-8')
        if any('difficulty_rank_within_pool' in r for r in members):
            ordered = sorted(members,key=lambda r:(r.get('difficulty_rank_within_pool',-1),r['image']))
            (pools/(group+'_easy_to_hard.txt')).write_text(''.join(r['image']+'\n' for r in ordered),encoding='utf-8')
    # Match conditional FAVV's final group sizes using the primary reliability alone.
    active = sorted((r for r in rows if not int(r['invalid_cues'])),
                    key=lambda r:(-float(r['reliability_score']),r['image']))
    nc = sum(r['conditional_group']=='clean' for r in active)
    nu = sum(r['conditional_group']=='unvalue' for r in active)
    for i,r in enumerate(active):
        r['matched_budget_group']='clean' if i<nc else ('unvalue' if i>=len(active)-nu else 'noisy')
    for r in rows:
        if int(r['invalid_cues']): r['matched_budget_group']='unvalue'
    # Export equal-budget TCRQ ablations and FAVV variants without rerunning SAM.
    columns = {'A0_SIM':'group_sim','A1_MIN':'group_min','A2_TCRQ':'group_tcrq',
               'F0_NoFAVV':'primary_group','F1_HardAll':'hard_all_group','F2_Conditional':'conditional_group',
               'F3_MatchedBudget':'matched_budget_group'}
    evaluation=[]
    for name,col in columns.items():
        dest=out/'ablations'/name
        dest.mkdir(parents=True,exist_ok=True)
        for group in ('clean','noisy','unvalue'):
            members=sorted((r for r in rows if r[col]==group),key=lambda r:r['image'])
            (dest/(group+'.txt')).write_text(''.join(r['image']+'\n' for r in members),encoding='utf-8')
            gt=[r for r in members if np.isfinite(safe_float(r.get('IoU')))]
            evaluation.append(dict(experiment=name,group=group,count=len(members),gt_count=len(gt),
                good=sum(float(r['IoU'])>.9 for r in gt),bad=sum(float(r['IoU'])<.7 for r in gt),
                middle=sum(.7<=float(r['IoU'])<=.9 for r in gt)))
    write_csv(out/'csv/image_level_metrics.csv', rows)
    write_csv(out/'csv/ablation_quality.csv',evaluation)
    for label,flag in [('favv_changed','favv_changed'),('favv_retained_vs_hard','favv_retained_vs_hard'),
                       ('score_effect','score_effect'),('stability_effect','stability_effect')]:
        write_csv(out/'csv'/(label+'.csv'),[r for r in rows if int(r[flag])])
    summary['cases']={k:sum(int(r[k]) for r in rows) for k in
                     ('favv_changed','favv_retained_vs_hard','score_effect','stability_effect')}
    write_json(out/'evaluation_results.json',summary)
    if manifests is not None:
        write_json(out/'box_manifest.json',manifests)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--img-dir',default='./data/Train/Imgs')
    p.add_argument('--json-dir',default='./data/Train/box_label')
    p.add_argument('--gt-dir',default=None,help='Optional GT: evaluation only; never used by TCRQ/FAVV')
    p.add_argument('--sod-metrics',action='store_true',help='Optional original S/E/F metrics; requires py_sod_metrics and --gt-dir')
    p.add_argument('--checkpoint',default='./sam_pseado/weights/sam_vit_h_4b8939.pth')
    p.add_argument('--model-type',default='vit_h',choices=['vit_h','vit_l','vit_b'])
    p.add_argument('--device',default='cuda')
    p.add_argument('--out-dir',default='./SGT/SAM1_TCRQ_FAVV')
    p.add_argument('--method',choices=['tcrq','min','sim'],default='tcrq')
    p.add_argument('--favv',choices=['conditional','hard','off'],default='conditional')
    p.add_argument('--clean-ratio',type=float,default=.25)
    p.add_argument('--unvalue-ratio',type=float,default=.10)
    p.add_argument('--outside-threshold',type=float,default=.15)
    p.add_argument('--stability-offset',type=float,default=1.)
    p.add_argument('--difficulty',choices=['dpc_ru','none'],default='dpc_ru',
                   help='Preserve original DPC/RU within-group ordering; does not affect groups')
    p.add_argument('--dpc-k',type=int,default=10)
    p.add_argument('--dpc-alpha',type=float,default=.10)
    p.add_argument('--dpc-feature',choices=['box_roi','global'],default='box_roi')
    p.add_argument('--regrade-csv',help='Reclassify a compatible CSV without SAM; requires all 3 cues and geometry')
    p.add_argument('--limit',type=int,default=0,help='Smoke-test subset only; percentiles then refer to the subset')
    p.add_argument('--skip-errors',action='store_true',help='Continue with explicit errors.csv and use successful subset')
    return p


def main():
    global torch, DEVICE, DPC_K, DPC_ALPHA, DPC_FEATURE_MODE, SOD_CLASSES
    args=parser().parse_args()
    budget_groups([],[],args.clean_ratio,args.unvalue_ratio)
    if not 0<=args.outside_threshold<=1 or args.stability_offset<0 or args.dpc_k<1 or not 0<=args.dpc_alpha<=1:
        raise ValueError('Invalid threshold/DPC options')
    out=Path(args.out_dir).expanduser().resolve()
    # Never mix a new partial run with stale masks, lists, or previews.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new, empty --out-dir: {out}')
    for d in ('Pred_Masks','Prob_48x48_FP16','Candidates','csv'):
        (out/d).mkdir(parents=True,exist_ok=True)
    config=vars(args).copy()
    config.update(schema_version=1,backend='sam1',ranking='average-rank percentile; sqrt(two lowest)',
                  count_rounding='floor',tie_break='image ID ascending',
                  favv_rule=f'empty/outside>{args.outside_threshold} reject; single-box severe FGC clean->noisy',
                  sim='mean over 12/13/23 of 0.5*(IoU+local SSIM)',
                  gt_usage='evaluation only',etc_tiny_boxcopy='diagnostic only',
                  multi_box='enclosing prompt box; FGC downgrade disabled',
                  score='raw selected candidate score, not clipped',
                  geometric_thresholds=dict(min_component_pixels=16,min_component_ratio=.002,
                                            severe_count=6,largest_effective_ratio=.55))
    write_json(out/'generation_config.json',config)
    if args.regrade_csv:
        with open(args.regrade_csv,encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        for r in rows:
            for canonical,legacy in [('sam_score','sam_score_selected'),('sam_stability','sam_stability_selected')]:
                if canonical not in r and legacy in r: r[canonical]=r[legacy]
        required=('image','sam_score','sam_stability','candidate_consensus_s',
                  'is_empty_response','outside_mask_ratio','is_severe_fragmented_response','num_boxes_in_json')
        if not rows or any(k not in rows[0] for k in required):
            raise ValueError('CSV lacks required columns: '+', '.join(k for k in required if not rows or k not in rows[0]))
        source_root=Path(args.regrade_csv).resolve().parent
        if source_root.name=='csv': source_root=source_root.parent
        for r in rows:
            for k in required[1:]: r[k]=float(r[k])
            r['fgc_severe']=int(r['is_severe_fragmented_response'] and r['num_boxes_in_json']<=1)
            for k in ('hard_mask_path','prob_48_path','candidate_cache_path'):
                if r.get(k) and not Path(r[k]).is_absolute(): r[k]=str(source_root/r[k])
            # Old within-pool difficulty ranks become stale when group membership changes.
            for k in list(r):
                if k.startswith('difficulty_'): r.pop(k)
            iou=safe_float(r.get('IoU'))
            r['gt_grade']='unknown' if not np.isfinite(iou) else ('good' if iou>.9 else ('bad' if iou<.7 else 'middle'))
        summary=qualify_rows(rows,args.clean_ratio,args.unvalue_ratio,args.favv,args.method,args.outside_threshold)
        export_outputs(rows,out,summary)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        return
    if cv2 is None: raise ImportError('Install opencv-python-headless')
    import torch as torch_module
    torch=torch_module
    from segment_anything import sam_model_registry, SamPredictor
    if args.sod_metrics:
        if not args.gt_dir: raise ValueError('--sod-metrics requires --gt-dir')
        from py_sod_metrics import Smeasure, Emeasure, Fmeasure, MAE
        SOD_CLASSES=(Smeasure,Emeasure,Fmeasure,MAE)
    DEVICE=args.device
    DPC_K,DPC_ALPHA,DPC_FEATURE_MODE=args.dpc_k,args.dpc_alpha,args.dpc_feature
    if DEVICE.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu or your CUDA PyTorch environment')
    paths=sorted(p.resolve() for p in Path(args.img_dir).iterdir()
                 if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))
    if len({p.stem for p in paths})!=len(paths):
        raise ValueError('Duplicate image stems: would overwrite masks; rename duplicates first')
    if args.limit<0: raise ValueError('--limit must be >= 0')
    if args.limit: paths=paths[:args.limit]
    if not paths: raise ValueError('No images found')
    sam=sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=DEVICE).eval()
    predictor=SamPredictor(sam)
    rows,manifests,embeddings,errors=[],[],{},[]
    for i,path in enumerate(paths,1):
        try:
            r,m,e=process_image(path,predictor,args,out)
            rows.append(r); manifests.append(m)
            if e is not None: embeddings[r['image']]=e
            print(f'[{i}/{len(paths)}] {path.stem}: score={r["sam_score"]:.4f} '
                  f'stability={r["sam_stability"]:.4f} SIM={r["candidate_consensus_s"]:.4f}')
        except Exception as exc:
            errors.append(dict(image=path.stem,error=str(exc),traceback=traceback.format_exc()))
            write_csv(out/'csv/errors.csv',errors)
            if not args.skip_errors: raise
            print(f'ERROR {path.stem}: {exc}')
    if not rows: raise RuntimeError('No successful images; inspect csv/errors.csv')
    rows.sort(key=lambda r:r['image']); manifests.sort(key=lambda r:r['image'])
    summary=qualify_rows(rows,args.clean_ratio,args.unvalue_ratio,args.favv,args.method,args.outside_threshold)
    summary.update(requested_images=len(paths),generated_images=len(rows),failed_images=len(errors))
    # Free model before computing original chunked DPC distances.
    del predictor,sam
    if DEVICE.startswith('cuda'): torch.cuda.empty_cache()
    if args.difficulty=='dpc_ru':
        summary['dpc']=rank_difficulty_with_dpc(rows,embeddings)
        np.save(out/'sam_dpc_embeddings.npy',np.stack([embeddings[r['image']] for r in rows]).astype(np.float16),allow_pickle=False)
        write_csv(out/'csv/sam_dpc_embedding_index.csv',[dict(row_index=i,image=r['image']) for i,r in enumerate(rows)])
    export_outputs(rows,out,summary,manifests)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(f'Completed: {out}')


if __name__=='__main__':
    main()
