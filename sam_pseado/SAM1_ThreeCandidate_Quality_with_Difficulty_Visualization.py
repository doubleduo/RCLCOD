import os
import cv2
import json
import csv
import math
import queue
import threading
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
# 评估指标：GT 仅用于离线评估，不参与伪标签质量分级和排序
from py_sod_metrics import Smeasure, Emeasure, Fmeasure, MAE

# 确保工程中存在 segment_anything，或已安装 segment-anything
from segment_anything import sam_model_registry, SamPredictor


# ============================================================
# 路径与基础配置
# ============================================================
# SAM_CHECKPOINT = "./sam_pseado/weights/sam_vit_h_4b8939.pth"
# SAM_MODEL_TYPE = "vit_h"

SAM_CHECKPOINT = "./sam_pseado/weights/sam_vit_l_0b3195.pth"
SAM_MODEL_TYPE = "vit_l"
DEVICE = "cuda"

IMG_DIR = "./data/Train/Imgs/"
GT_DIR = "./data/Train/GT/"
JSON_DIR = "./data/Train/box_label/"

BASE_OUT_DIR = "./sam_pseado/SGT/SAM1_ThreeCandidate_Quality"
OUT_MASK_DIR = os.path.join(BASE_OUT_DIR, "Pred_Masks")
PROB_48_DIR = os.path.join(BASE_OUT_DIR, "Prob_48x48_FP16")
BAD_VIS_DIR = os.path.join(BASE_OUT_DIR, "Bad_Vis_GT")
CSV_DIR = os.path.join(BASE_OUT_DIR, "csv")
IMAGE_CSV_PATH = os.path.join(CSV_DIR, "image_level_metrics.csv")
JSON_RESULTS_PATH = os.path.join(BASE_OUT_DIR, "evaluation_results.json")
CONFIG_PATH = os.path.join(BASE_OUT_DIR, "generation_config.json")
BOX_MANIFEST_PATH = os.path.join(BASE_OUT_DIR, "box_manifest.json")

# ============================================================
# 困难样本 / 特征 / 分数可视化
# ============================================================
DIAG_ROOT_DIR = os.path.join(BASE_OUT_DIR, "Diagnostics")
DIAG_CACHE_DIR = os.path.join(DIAG_ROOT_DIR, "Cache")
FEATURE_CACHE_DIR = os.path.join(DIAG_CACHE_DIR, "SAM_Feature_Energy_64")
DISAGREE_CACHE_DIR = os.path.join(DIAG_CACHE_DIR, "Candidate_Disagreement_64")
DIFFICULT_VIS_DIR = os.path.join(DIAG_ROOT_DIR, "Difficult_Samples")
TOPK_VIS_DIR = os.path.join(DIAG_ROOT_DIR, "TopK_By_Score")
DIFFICULT_CSV_PATH = os.path.join(CSV_DIR, "difficult_samples.csv")

# 特征缓存很小：每张图仅保存 64x64 左右的 uint8 PNG；不会保存完整 feature tensor。
SAVE_DIAGNOSTIC_CACHE = True
SAVE_DIFFICULT_VIS = True
DIAG_TOPK_PER_METRIC = 24
DIAG_MAX_IMAGES = 180
DIAG_TILE_SIZE = 220
DIAG_PANEL_SIDE = 420

# Top-K 困难排序。asc=True 表示越小越困难；False 表示越大越困难。
DIFFICULT_METRIC_SPECS = [
    ("q_dual", True, "Low_QDual"),
    ("q_region", True, "Low_QRegion"),
    ("q_shape", True, "Low_QShape"),
    ("sam_score_selected", True, "Low_SAMScore"),
    ("sam_stability_selected", True, "Low_Stability"),
    ("candidate_consensus_s", True, "Low_Consensus"),
    ("fg_p10_confidence", True, "Low_FG_P10"),
    ("bg_confidence", True, "Low_BG"),
    ("v_total", False, "High_GeometryViolation"),
    ("boundary_uncertainty", False, "High_BoundaryUncertainty"),
    ("small_fragment_area_ratio", False, "High_FragmentRatio"),
]

# 同一张 GPU 上默认仅放一个 SAM ViT-H。
# 增大到 2 前，建议先实测吞吐量，避免 CUDA 资源竞争。
NUM_WORKERS =6

# GT 只用于离线验证。设为 False 后，无 GT 也可正常生成伪标签与 CSV。
USE_GT_EVAL = True
SAVE_SINGLE_SOD_METRICS = True

# 每张图只进行一次标准 SAM box 推理，不使用：
# 1) box jitter；2) 水平翻转；3) 小目标 crop；4) 逐框多次推理。
# 若 JSON 中意外存在多个有效框，使用所有框的最小外接框作为一次 box prompt。
MULTI_BOX_POLICY = "enclosing_union_box"
MULTIMASK_OUTPUT = True
BEST_MASK_BY = "sam_score"

# 保存 48×48 FP16 概率图，训练时用于低分辨率软区域监督
PROB_SAVE_SIZE = (48, 48)  # cv2 参数顺序为 (w, h)


# ============================================================
# Beyond Weak Supervision：三候选共识 S
# ============================================================
CANDIDATE_LOW_THR = 0.60
CANDIDATE_HIGH_THR = 0.90

# SAM 自带阈值稳定性：对最终选中候选计算
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

metrics_lock = threading.Lock()


# ============================================================
# 通用工具
# ============================================================
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



def candidate_grade_from_s(candidate_s):
    if candidate_s < CANDIDATE_LOW_THR:
        return "low"
    if candidate_s < CANDIDATE_HIGH_THR:
        return "middle"
    return "high"


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


# ============================================================
# SAM encoder feature / candidate disagreement cache
# ============================================================
def robust_normalize_map(value_map, low_pct=2.0, high_pct=98.0):
    arr = np.asarray(value_map, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if hi <= lo + 1e-8:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo + 1e-8:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def extract_sam_feature_energy(predictor):
    """
    将 SAM image encoder feature [1,C,Hf,Wf] 压缩成单通道能量图。
    这里只做诊断可视化，不参与伪标签质量分级或训练。

    energy(x,y) = sqrt(mean_c(feature(c,x,y)^2))
    """
    feature = getattr(predictor, "features", None)
    if feature is None:
        return None
    try:
        feat = feature.detach().float()
        if feat.ndim == 4:
            feat = feat[0]
        if feat.ndim != 3:
            return None
        energy = torch.sqrt(torch.mean(feat * feat, dim=0) + 1e-12)
        energy = energy.cpu().numpy().astype(np.float32)
        return robust_normalize_map(energy)
    except Exception:
        return None


def compute_candidate_disagreement_map(candidate_masks):
    """
    三个 SAM mask 在像素级的不一致程度。
    mean(mask) 为 0/1 时完全一致；接近 1/3 或 2/3 时分歧最大。
    用 4*m*(1-m) 映射到 [0,1] 附近，便于热力图显示。
    """
    masks = np.asarray(candidate_masks, dtype=np.float32)
    if masks.ndim != 3 or masks.shape[0] < 2:
        return None
    mean_mask = np.mean(masks[:3], axis=0)
    disagreement = 4.0 * mean_mask * (1.0 - mean_mask)
    return np.clip(disagreement, 0.0, 1.0).astype(np.float32)


def save_unit_map_png(value_map, save_path, target_size=(64, 64)):
    if value_map is None:
        return ""
    arr = np.asarray(value_map, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    if target_size is not None:
        arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_AREA)
    out = np.round(arr * 255.0).astype(np.uint8)
    ok = cv2.imwrite(save_path, out)
    return save_path if ok else ""


# ============================================================
# 指标
# ============================================================
def init_metric_group():
    return {
        "SM": Smeasure(),
        "EM": Emeasure(),
        "FM": Fmeasure(),
        "MAE": MAE(),
    }


def update_metric_group(metric_group, pred_eval, gt_eval):
    metric_group["SM"].step(pred=pred_eval, gt=gt_eval)
    metric_group["EM"].step(pred=pred_eval, gt=gt_eval)
    metric_group["FM"].step(pred=pred_eval, gt=gt_eval)
    metric_group["MAE"].step(pred=pred_eval, gt=gt_eval)


def get_metric_results(metric_group):
    sm = metric_group["SM"].get_results()["sm"]
    em = metric_group["EM"].get_results()["em"]["adp"]
    fm = metric_group["FM"].get_results()["fm"]["adp"]
    mae = metric_group["MAE"].get_results()["mae"]

    return {
        "S-measure": round(float(sm), 4),
        "E-measure(adp)": round(float(em), 4),
        "F-measure(adp)": round(float(fm), 4),
        "MAE": round(float(mae), 4),
    }


def compute_single_sod_metrics(pred_eval, gt_eval):
    try:
        mg = init_metric_group()
        update_metric_group(mg, pred_eval, gt_eval)
        res = get_metric_results(mg)
        return {
            "Smeasure": res["S-measure"],
            "Emeasure_adp": res["E-measure(adp)"],
            "Fmeasure_adp": res["F-measure(adp)"],
        }
    except Exception:
        return {
            "Smeasure": np.nan,
            "Emeasure_adp": np.nan,
            "Fmeasure_adp": np.nan,
        }


# ============================================================
# 单次标准 SAM 推理：同一图像、同一 box 输出三个候选
# ============================================================
def generate_three_candidate_result(image_rgb, prompt_box, predictor):
    """
    每张图仅执行一次 predictor.set_image() 与一次 box prompt 推理。

    返回：
      - 三候选 full-resolution logits / hard masks / scores
      - 候选共识 S
      - SAM score 最大的最终候选
      - 最终候选的 SAM 阈值稳定性
    """
    predictor.set_image(image_rgb)

    full_logits, scores, low_res_logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.asarray(prompt_box, dtype=np.float32),
        multimask_output=MULTIMASK_OUTPUT,
        return_logits=True,
    )

    full_logits = np.asarray(full_logits, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    low_res_logits = np.asarray(low_res_logits, dtype=np.float32)

    if full_logits.ndim == 2:
        full_logits = full_logits[None, ...]
    if low_res_logits.ndim == 2:
        low_res_logits = low_res_logits[None, ...]

    if full_logits.shape[0] < 3:
        raise RuntimeError(
            f"multimask_output=True but SAM returned {full_logits.shape[0]} candidate(s)"
        )

    mask_threshold = float(predictor.model.mask_threshold)
    candidate_masks = full_logits > mask_threshold
    consensus_stats = compute_candidate_consensus(candidate_masks[:3])

    if BEST_MASK_BY == "sam_score" and scores.size:
        best_idx = int(np.argmax(scores))
    else:
        best_idx = 0

    selected_logit = full_logits[best_idx]
    selected_prob = sigmoid_np(selected_logit)
    selected_mask = selected_logit > mask_threshold
    selected_score = safe_float(scores[best_idx], default=0.0)
    selected_stability = calculate_sam_stability_score(
        mask_logit=selected_logit,
        mask_threshold=mask_threshold,
        threshold_offset=SAM_STABILITY_OFFSET,
    )

    # 诊断缓存：SAM encoder feature 能量 + 三候选像素级分歧。
    feature_energy_map = extract_sam_feature_energy(predictor)
    candidate_disagreement_map = compute_candidate_disagreement_map(candidate_masks[:3])

    candidate_scores = [
        safe_float(scores[idx], default=np.nan) if idx < scores.size else np.nan
        for idx in range(3)
    ]

    return {
        "selected_mask": selected_mask,
        "selected_prob": selected_prob,
        "selected_logit": selected_logit,
        "selected_low_res_logit": low_res_logits[best_idx],
        "selected_idx": best_idx,
        "selected_score": selected_score,
        "selected_stability": selected_stability,
        "candidate_scores": candidate_scores,
        "candidate_masks": candidate_masks[:3],
        "feature_energy_map": feature_energy_map,
        "candidate_disagreement_map": candidate_disagreement_map,
        **consensus_stats,
    }


# ============================================================
# Q / V / 边界与异常响应
# ============================================================

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
    2) q_shape：SAM score、阈值稳定性和候选共识构成的形状可靠性。

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



# ============================================================
# GT 坏伪标签双面板可视化
# ============================================================
def resize_panel(image, max_side):
    h, w = image.shape[:2]
    current = max(h, w)
    if current <= max_side:
        return image
    scale = max_side / float(current)
    return cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def pad_to_same_height(left, right):
    target_h = max(left.shape[0], right.shape[0])

    def pad(image):
        h, _ = image.shape[:2]
        if h == target_h:
            return image
        top = (target_h - h) // 2
        bottom = target_h - h - top
        return cv2.copyMakeBorder(
            image,
            top,
            bottom,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    return pad(left), pad(right)


def save_bad_visualization(image_rgb, prompt_box, pred_bool, gt_bool, mae_val, save_path):
    left = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = [int(round(v)) for v in prompt_box]
    cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)

    pred_bool = np.asarray(pred_bool, dtype=bool)
    gt_bool = np.asarray(gt_bool, dtype=bool)
    tp = pred_bool & gt_bool
    fp = pred_bool & (~gt_bool)
    fn = (~pred_bool) & gt_bool

    right = np.zeros((*gt_bool.shape, 3), dtype=np.uint8)
    right[tp] = (255, 255, 255)
    right[fp] = (0, 0, 255)
    right[fn] = (255, 0, 0)

    left = resize_panel(left, BAD_VIS_MAX_PANEL_SIDE)
    right = resize_panel(right, BAD_VIS_MAX_PANEL_SIDE)
    left, right = pad_to_same_height(left, right)

    gap = np.zeros((left.shape[0], 6, 3), dtype=np.uint8)
    panels = np.concatenate([left, gap, right], axis=1)

    header_h = 24
    header = np.zeros((header_h, panels.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"MAE={mae_val:.4f}",
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(save_path, np.concatenate([header, panels], axis=0))


# ============================================================
# 单图处理
# ============================================================
def autocast_context():
    # SamPredictor.predict() 内部会转换为 NumPy，保持 FP32，避免 BF16 转换问题。
    return nullcontext()


def process_single_image(img_path, predictor_queue, metrics_group):
    predictor = predictor_queue.get()

    try:
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(JSON_DIR, f"{image_name}.json")
        gt_path = os.path.join(GT_DIR, f"{image_name}.png")

        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Skip {image_name}: image read failed")
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]

        boxes_raw = extract_all_boxes_from_json(json_path)
        prompt_box, valid_boxes = build_single_prompt_box(boxes_raw, img_h, img_w)
        if prompt_box is None:
            print(f"Skip {image_name}: no valid box")
            return None

        box_mask = box_to_mask(prompt_box, img_h, img_w)
        structure = compute_structure_info(
            prompt_box,
            img_h,
            img_w,
            num_boxes_in_json=len(valid_boxes),
        )

        with torch.inference_mode(), autocast_context():
            result = generate_three_candidate_result(
                image_rgb=image_rgb,
                prompt_box=prompt_box,
                predictor=predictor,
            )

        pred_bool = result["selected_mask"]
        selected_prob = result["selected_prob"]
        candidate_s = float(result["candidate_consensus_s"])
        s_grade = candidate_grade_from_s(candidate_s)

        # 保存最终 SAM score 最大候选的硬 mask
        hard_mask_path = os.path.join(OUT_MASK_DIR, f"{image_name}.png")
        pred_save = pred_bool.astype(np.uint8) * 255
        if not cv2.imwrite(hard_mask_path, pred_save):
            raise IOError(f"failed to save mask: {hard_mask_path}")

        # 保存最终候选的 48×48 FP16 概率图
        prob_48 = cv2.resize(
            selected_prob,
            PROB_SAVE_SIZE,
            interpolation=cv2.INTER_AREA,
        ).astype(np.float16)
        prob_48_path = os.path.join(PROB_48_DIR, f"{image_name}.npy")
        np.save(prob_48_path, prob_48, allow_pickle=False)

        # 小尺寸诊断缓存：后面完成全局难度排序后再生成大图，无需重新跑 SAM。
        feature_map_path = ""
        candidate_disagreement_path = ""
        if SAVE_DIAGNOSTIC_CACHE:
            feature_map_path = save_unit_map_png(
                result.get("feature_energy_map"),
                os.path.join(FEATURE_CACHE_DIR, f"{image_name}.png"),
                target_size=(64, 64),
            )
            candidate_disagreement_path = save_unit_map_png(
                result.get("candidate_disagreement_map"),
                os.path.join(DISAGREE_CACHE_DIR, f"{image_name}.png"),
                target_size=(64, 64),
            )

        q_stats = compute_q_evidence(
            prob=selected_prob,
            pred_mask=pred_bool,
            box_mask=box_mask,
            selected_score=result["selected_score"],
            selected_stability=result["selected_stability"],
            candidate_consensus_s=candidate_s,
        )
        v_stats = compute_v_violation(
            pred_mask=pred_bool,
            box_mask=box_mask,
            prompt_box=prompt_box,
        )
        boundary_stats = compute_boundary_uncertainty(selected_prob, pred_bool)
        bad_response_stats = compute_bad_response_flags(
            pred_mask=pred_bool,
            box_mask=box_mask,
            prompt_box=prompt_box,
            img_h=img_h,
            img_w=img_w,
            is_ov=bool(structure["is_OV"]),
        )

        # 先按 S 分级；质量门控/确定性异常进入 unvalue；普通碎片降到 middle。
        force_unvalue = int(
            s_grade == "low"
            or q_stats["fails_quality_gate"] == 1
            or bad_response_stats["is_bad_response"] == 1
        )
        if force_unvalue:
            recommended_pool = "unvalue"
        elif s_grade == "middle" or bad_response_stats["is_refinable_risk"] == 1:
            recommended_pool = "middle"
        else:
            recommended_pool = "high"

        # GT 仅用于评估，不参与 candidate grade、quality gate、Q 或 recommended_pool。
        has_gt = bool(USE_GT_EVAL and os.path.exists(gt_path))
        gt_area_ratio = np.nan
        iou_val = np.nan
        mae_val = np.nan
        gt_bad_by_iou = -1
        gt_bad_by_mae = -1
        gt_is_bad = -1
        bad_vis_path = ""
        single_sod = {
            "Smeasure": np.nan,
            "Emeasure_adp": np.nan,
            "Fmeasure_adp": np.nan,
        }

        if has_gt:
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                has_gt = False
            else:
                if gt_mask.shape[:2] != (img_h, img_w):
                    gt_mask = cv2.resize(
                        gt_mask,
                        (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                gt_eval = (gt_mask > 128).astype(np.uint8) * 255
                gt_bool = gt_eval > 128
                gt_area_ratio = float(gt_bool.mean())
                mae_val = float(
                    np.mean(
                        np.abs(
                            pred_bool.astype(np.float32)
                            - gt_bool.astype(np.float32)
                        )
                    )
                )
                iou_val = compute_iou(pred_bool, gt_bool)

                with metrics_lock:
                    update_metric_group(metrics_group, pred_save, gt_eval)

                if SAVE_SINGLE_SOD_METRICS:
                    single_sod = compute_single_sod_metrics(pred_save, gt_eval)

                gt_bad_by_iou = int(iou_val < BAD_IOU_THR)
                gt_bad_by_mae = int(mae_val > BAD_MAE_THR)
                gt_is_bad = int(bool(gt_bad_by_iou or gt_bad_by_mae))

                if gt_is_bad:
                    bad_vis_path = os.path.join(BAD_VIS_DIR, f"{image_name}.jpg")
                    save_bad_visualization(
                        image_rgb=image_rgb,
                        prompt_box=prompt_box,
                        pred_bool=pred_bool,
                        gt_bool=gt_bool,
                        mae_val=mae_val,
                        save_path=bad_vis_path,
                    )

        candidate_scores = result["candidate_scores"]
        image_row = {
            # 身份与文件
            "image": image_name,
            "image_height": img_h,
            "image_width": img_w,
            "hard_mask_path": hard_mask_path,
            "prob_48_path": prob_48_path,
            "feature_map_path": feature_map_path,
            "candidate_disagreement_path": candidate_disagreement_path,
            "diagnostic_vis_path": "",
            "difficulty_reasons": "",
            "is_difficult_visual": 0,
            "bad_vis_path": bad_vis_path,
            "has_gt": int(has_gt),

            # 单次 box prompt
            **structure,
            "box_x1": prompt_box[0],
            "box_y1": prompt_box[1],
            "box_x2": prompt_box[2],
            "box_y2": prompt_box[3],

            # 三候选与最终候选
            "candidate_score_1": candidate_scores[0],
            "candidate_score_2": candidate_scores[1],
            "candidate_score_3": candidate_scores[2],
            "selected_candidate_index": int(result["selected_idx"]),
            "candidate_iou_12": result["candidate_iou_12"],
            "candidate_iou_13": result["candidate_iou_13"],
            "candidate_iou_23": result["candidate_iou_23"],
            "candidate_ssim_12": result["candidate_ssim_12"],
            "candidate_ssim_13": result["candidate_ssim_13"],
            "candidate_ssim_23": result["candidate_ssim_23"],
            "candidate_sim_12": result["candidate_sim_12"],
            "candidate_sim_13": result["candidate_sim_13"],
            "candidate_sim_23": result["candidate_sim_23"],
            "candidate_sim_min": result["candidate_sim_min"],
            "candidate_sim_max": result["candidate_sim_max"],
            "candidate_sim_range": result["candidate_sim_range"],
            "candidate_consensus_s": candidate_s,
            "candidate_s_grade": s_grade,

            # Q、基础门控
            **q_stats,

            # V、边界
            **v_stats,
            **boundary_stats,
            "is_BA": 0,
            "ba_threshold": np.nan,

            # 异常响应
            **bad_response_stats,

            # 分级建议；最终分池由后续独立脚本完成
            "force_unvalue": force_unvalue,
            "recommended_pool": recommended_pool,
            "q_region_rank_global": -1,
            "q_shape_rank_global": -1,
            "q_dual_rank_global": -1,
            "q_region_rank_within_pool": -1,
            "q_shape_rank_within_pool": -1,
            "q_dual_rank_within_pool": -1,
            # 兼容旧字段：默认映射到 q_dual 排名。
            "q_rank_global": -1,
            "q_rank_within_pool": -1,

            # GT 指标，仅用于验证质量判据
            "gt_area_ratio": gt_area_ratio,
            "IoU": iou_val,
            "Smeasure": single_sod["Smeasure"],
            "Emeasure_adp": single_sod["Emeasure_adp"],
            "Fmeasure_adp": single_sod["Fmeasure_adp"],
            "MAE": mae_val,
            "gt_bad_by_iou": gt_bad_by_iou,
            "gt_bad_by_mae": gt_bad_by_mae,
            "gt_is_bad": gt_is_bad,
        }

        manifest_row = {
            "image": image_name,
            "image_height": img_h,
            "image_width": img_w,
            "source_boxes": valid_boxes,
            "prompt_box": prompt_box,
            "multi_box_policy": MULTI_BOX_POLICY,
        }
        return image_row, manifest_row

    except Exception as exc:
        print(f"Error processing {img_path}: {exc}")
        return None

    finally:
        predictor_queue.put(predictor)


# ============================================================
# 保存与汇总
# ============================================================
def write_csv(path, rows, fieldnames):
    if not rows:
        print(f"No rows to write: {path}")
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_generation_config(ba_threshold=None):
    config = {
        "sam_model_type": SAM_MODEL_TYPE,
        "sam_checkpoint": SAM_CHECKPOINT,
        "device": DEVICE,
        "inference_precision": "float32",
        "autocast_enabled": False,
        "num_workers": NUM_WORKERS,
        "use_gt_eval": USE_GT_EVAL,
        "multimask_output": MULTIMASK_OUTPUT,
        "best_mask_by": BEST_MASK_BY,
        "multi_box_policy": MULTI_BOX_POLICY,
        "use_small_crop": False,
        "use_hflip_consistency": False,
        "use_box_jitter_consistency": False,
        "candidate_similarity": "0.5 * (IoU + SSIM)",
        "candidate_consensus": "mean(SIM12, SIM13, SIM23)",
        "candidate_low_threshold": CANDIDATE_LOW_THR,
        "candidate_high_threshold": CANDIDATE_HIGH_THR,
        "sam_stability_offset": SAM_STABILITY_OFFSET,
        "quality_gate": {
            "min_fg_confidence": MIN_FG_CONFIDENCE,
            "min_bg_confidence": MIN_BG_CONFIDENCE,
            "min_sam_score": MIN_SAM_SCORE,
            "min_sam_stability": MIN_SAM_STABILITY,
        },
        "q_usage": "dual_dimension_ranking_after_S_grade_and_filtering",
        "region_q_weights": REGION_Q_WEIGHTS,
        "shape_q_weights": SHAPE_Q_WEIGHTS,
        "dual_q_reference": "sqrt(q_region * q_shape)",
        "core_erode_radius": CORE_ERODE_RADIUS,
        "background_ring_radius": BACKGROUND_RING_RADIUS,
        "boundary_band_radius": BOUNDARY_BAND_RADIUS,
        "v_weights": V_WEIGHTS,
        "area_lower_threshold": AREA_LOWER_THR,
        "area_upper_threshold": AREA_UPPER_THR,
        "anomaly_thresholds": {
            "tiny_mask_box_ratio": TINY_MASK_BOX_RATIO_THR,
            "fragment_effective_min_pixels": FRAGMENT_EFFECTIVE_MIN_PIXELS,
            "fragment_effective_min_mask_ratio": FRAGMENT_EFFECTIVE_MIN_MASK_RATIO,
            "fragment_min_components": FRAGMENT_MIN_COMPONENTS,
            "fragment_max_largest_component_ratio": FRAGMENT_MAX_LARGEST_COMPONENT_RATIO,
            "severe_fragment_min_components": SEVERE_FRAGMENT_MIN_COMPONENTS,
            "severe_fragment_max_largest_component_ratio": SEVERE_FRAGMENT_MAX_LARGEST_COMPONENT_RATIO,
            "outside_leak_ratio": OUTSIDE_LEAK_RATIO_THR,
            "ov_outside_leak_ratio": OV_OUTSIDE_LEAK_RATIO_THR,
            "box_copy_fill_ratio": BOX_COPY_FILL_RATIO_THR,
            "box_copy_bbox_iou": BOX_COPY_BBOX_IOU_THR,
            "box_copy_rectangularity": BOX_COPY_RECTANGULARITY_THR,
            "reversal_outside_ratio": REVERSAL_OUTSIDE_RATIO_THR,
            "reversal_inside_ratio": REVERSAL_INSIDE_RATIO_THR,
            "reversal_image_area_ratio": REVERSAL_IMAGE_AREA_RATIO_THR,
            "reversal_small_box_image_ratio": REVERSAL_SMALL_BOX_IMAGE_RATIO_THR,
        },
        "force_unvalue_rule": (
            "candidate_s_grade == low OR fails_quality_gate OR is_bad_response"
        ),
        "refinable_risk_rule": "ordinary_fragment -> middle; severe_fragment -> unvalue",
        "saved_probability_shape": list(PROB_SAVE_SIZE),
        "saved_probability_dtype": "float16",
        "ba_top_ratio": BA_TOP_RATIO,
        "ba_threshold": ba_threshold,
        "gt_bad_iou_threshold": BAD_IOU_THR,
        "gt_bad_mae_threshold": BAD_MAE_THR,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def collect_image_paths():
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for pattern in patterns:
        paths.extend(glob(os.path.join(IMG_DIR, pattern)))
    return sorted(set(paths))



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



# ============================================================
# 困难样本可视化
# ============================================================
def read_gray_unit_map(path):
    if not path or not os.path.exists(path):
        return None
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    return arr.astype(np.float32) / 255.0


def apply_heatmap_on_rgb(image_rgb, unit_map, alpha=0.48):
    h, w = image_rgb.shape[:2]
    if unit_map is None:
        unit_map = np.zeros((h, w), dtype=np.float32)
    unit_map = cv2.resize(
        np.asarray(unit_map, dtype=np.float32),
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )
    unit_u8 = np.round(np.clip(unit_map, 0.0, 1.0) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(unit_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    mixed = cv2.addWeighted(image_rgb, 1.0 - alpha, heat_rgb, alpha, 0.0)
    return mixed


def draw_box_rgb(image_rgb, row, thickness=2):
    out = image_rgb.copy()
    x1 = int(round(safe_float(row.get("box_x1"), 0)))
    y1 = int(round(safe_float(row.get("box_y1"), 0)))
    x2 = int(round(safe_float(row.get("box_x2"), 0)))
    y2 = int(round(safe_float(row.get("box_y2"), 0)))
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    return out


def overlay_mask_rgb(image_rgb, mask_bool, alpha=0.42):
    out = image_rgb.copy()
    if mask_bool is None:
        return out
    mask = np.asarray(mask_bool, dtype=bool)
    layer = out.copy()
    layer[mask] = np.array([255, 215, 0], dtype=np.uint8)
    return cv2.addWeighted(out, 1.0 - alpha, layer, alpha, 0.0)


def fit_panel_rgb(image_rgb, side=DIAG_PANEL_SIDE):
    h, w = image_rgb.shape[:2]
    scale = min(side / max(1, w), side / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y0 = (side - nh) // 2
    x0 = (side - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def put_panel_title(panel_rgb, title):
    out = panel_rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        out,
        str(title)[:42],
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def probability_difficulty_maps(prob_small):
    prob = np.asarray(prob_small, dtype=np.float32)
    prob = np.clip(prob, 0.0, 1.0)
    uncertainty = 4.0 * prob * (1.0 - prob)
    # 仅作为空间诊断：高值表示模型更倾向前景；真正 BG 指标仍以 CSV 里的 ring/outside 统计为准。
    foreground_response = prob
    return foreground_response, uncertainty


def difficulty_reason_string(row):
    reasons = []
    pool = str(row.get("recommended_pool", ""))
    if pool == "unvalue":
        reasons.append("pool=unvalue")
    elif pool == "middle":
        reasons.append("pool=middle")
    if int(row.get("fails_quality_gate", 0)) == 1:
        reasons.append("quality_gate")
    if int(row.get("is_bad_response", 0)) == 1:
        reasons.append("bad_response")
    if int(row.get("is_refinable_risk", 0)) == 1:
        reasons.append("fragment_risk")
    if int(row.get("is_BA", 0)) == 1:
        reasons.append("boundary_top30%")
    if int(row.get("is_SO", 0)) == 1:
        reasons.append("small_object")
    if int(row.get("is_OV", 0)) == 1:
        reasons.append("border_object")
    if int(row.get("gt_is_bad", -1)) == 1:
        reasons.append("GT_bad_eval")
    return ";".join(reasons)


def safe_metric_text(row, key, digits=3):
    value = safe_float(row.get(key))
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def build_score_card(row, side=DIAG_PANEL_SIDE):
    card = np.full((side, side, 3), 246, dtype=np.uint8)
    title = f"{row.get('image', '')} | {row.get('recommended_pool', '')}"
    cv2.putText(card, title[:42], (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    good_metrics = [
        ("SAM score", "sam_score_selected"),
        ("Stability", "sam_stability_selected"),
        ("Consensus", "candidate_consensus_s"),
        ("FG p10", "fg_p10_confidence"),
        ("BG conf", "bg_confidence"),
        ("Q region", "q_region"),
        ("Q shape", "q_shape"),
        ("Q dual", "q_dual"),
    ]
    bad_metrics = [
        ("V total", "v_total"),
        ("Boundary U", "boundary_uncertainty"),
        ("Frag ratio", "small_fragment_area_ratio"),
        ("Outside", "outside_mask_ratio"),
    ]

    y = 53
    bar_x1, bar_x2 = 164, side - 18
    for label, key in good_metrics:
        value = safe_float(row.get(key), default=np.nan)
        cv2.putText(card, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        if np.isfinite(value):
            v = float(np.clip(value, 0.0, 1.0))
            cv2.rectangle(card, (bar_x1, y - 11), (bar_x2, y + 2), (210, 210, 210), 1)
            cv2.rectangle(card, (bar_x1, y - 10), (bar_x1 + int((bar_x2 - bar_x1) * v), y + 1), (70, 150, 70), -1)
            cv2.putText(card, f"{value:.3f}", (side - 72, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        y += 25

    y += 4
    for label, key in bad_metrics:
        value = safe_float(row.get(key), default=np.nan)
        cv2.putText(card, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        if np.isfinite(value):
            v = float(np.clip(value, 0.0, 1.0))
            cv2.rectangle(card, (bar_x1, y - 11), (bar_x2, y + 2), (210, 210, 210), 1)
            cv2.rectangle(card, (bar_x1, y - 10), (bar_x1 + int((bar_x2 - bar_x1) * v), y + 1), (70, 70, 200), -1)
            cv2.putText(card, f"{value:.3f}", (side - 72, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        y += 25

    gt_line = f"GT: IoU={safe_metric_text(row, 'IoU')}  MAE={safe_metric_text(row, 'MAE')}"
    cv2.putText(card, gt_line, (12, side - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    reason = difficulty_reason_string(row)
    cv2.putText(card, reason[:52], (12, side - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (30, 30, 30), 1, cv2.LINE_AA)
    return card


def load_original_by_name(image_name):
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        path = os.path.join(IMG_DIR, image_name + ext)
        if os.path.exists(path):
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None


def make_diagnostic_panel(row, save_path):
    image_name = str(row.get("image", ""))
    image_rgb = load_original_by_name(image_name)
    if image_rgb is None:
        return ""
    h, w = image_rgb.shape[:2]

    # mask
    mask = cv2.imread(str(row.get("hard_mask_path", "")), cv2.IMREAD_GRAYSCALE)
    mask_bool = None if mask is None else (cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 127)

    # saved small probability
    prob_small = None
    prob_path = str(row.get("prob_48_path", ""))
    if prob_path and os.path.exists(prob_path):
        try:
            prob_small = np.load(prob_path, allow_pickle=False).astype(np.float32)
        except Exception:
            prob_small = None
    if prob_small is None:
        prob_small = np.zeros((48, 48), dtype=np.float32)
    fg_map, uncertainty_map = probability_difficulty_maps(prob_small)

    feature_map = read_gray_unit_map(str(row.get("feature_map_path", "")))
    disagree_map = read_gray_unit_map(str(row.get("candidate_disagreement_path", "")))

    p_original = put_panel_title(fit_panel_rgb(draw_box_rgb(image_rgb, row)), "RGB + box")
    p_mask = put_panel_title(fit_panel_rgb(overlay_mask_rgb(draw_box_rgb(image_rgb, row), mask_bool)), "Selected mask")
    p_feature = put_panel_title(fit_panel_rgb(apply_heatmap_on_rgb(image_rgb, feature_map)), "SAM encoder feature energy")
    p_unc = put_panel_title(fit_panel_rgb(apply_heatmap_on_rgb(image_rgb, uncertainty_map)), "Pixel uncertainty 4p(1-p)")
    p_dis = put_panel_title(fit_panel_rgb(apply_heatmap_on_rgb(image_rgb, disagree_map)), "3-mask disagreement")
    p_fg = put_panel_title(fit_panel_rgb(apply_heatmap_on_rgb(image_rgb, fg_map)), "Foreground response p")
    p_score = build_score_card(row)

    blank = np.zeros_like(p_score)
    top = np.concatenate([p_original, p_mask, p_feature, p_score], axis=1)
    bottom = np.concatenate([p_unc, p_dis, p_fg, blank], axis=1)
    whole = np.concatenate([top, bottom], axis=0)
    bgr = cv2.cvtColor(whole, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(save_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return save_path if ok else ""


def rank_rows_for_metric(image_rows, key, ascending=True):
    valid = []
    for row in image_rows:
        value = safe_float(row.get(key), default=np.nan)
        if np.isfinite(value):
            valid.append((value, row))
    valid.sort(key=lambda x: x[0], reverse=not ascending)
    return [row for _, row in valid]


def make_contact_sheet(rows, metric_key, save_path, ascending=True, max_items=None):
    if max_items is None:
        max_items = DIAG_TOPK_PER_METRIC
    rows = rows[:max_items]
    if not rows:
        return ""
    tile = DIAG_TILE_SIZE
    cols = 4
    rows_n = int(math.ceil(len(rows) / cols))
    sheet = np.zeros((rows_n * tile, cols * tile, 3), dtype=np.uint8)

    for idx, row in enumerate(rows):
        image_rgb = load_original_by_name(str(row.get("image", "")))
        if image_rgb is None:
            continue
        img = draw_box_rgb(image_rgb, row)
        canvas = fit_panel_rgb(img, side=tile)
        value = safe_metric_text(row, metric_key, digits=3)
        cv2.rectangle(canvas, (0, 0), (tile, 34), (0, 0, 0), -1)
        label = f"{idx+1:02d} {row.get('image','')} {value}"
        cv2.putText(canvas, label[:31], (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
        r, c = divmod(idx, cols)
        sheet[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = canvas

    bgr = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(save_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return save_path if ok else ""


def build_difficulty_visualizations(image_rows):
    """
    全局统计完成后再挑困难样本：
      1) 每个分数各取 Top-K 困难；
      2) 并入 middle/unvalue、BA、GT-bad、异常响应的代表；
      3) 最终最多生成 DIAG_MAX_IMAGES 张大诊断图；
      4) difficult_samples.csv 保留全部被判为困难的样本与原因。
    """
    if not image_rows:
        return []

    os.makedirs(DIFFICULT_VIS_DIR, exist_ok=True)
    os.makedirs(TOPK_VIS_DIR, exist_ok=True)

    selected_names = set()
    metric_ranked = {}
    for key, ascending, tag in DIFFICULT_METRIC_SPECS:
        ranked = rank_rows_for_metric(image_rows, key, ascending=ascending)
        metric_ranked[key] = ranked
        for row in ranked[:DIAG_TOPK_PER_METRIC]:
            selected_names.add(str(row.get("image", "")))
        make_contact_sheet(
            ranked,
            metric_key=key,
            save_path=os.path.join(TOPK_VIS_DIR, f"{tag}_{key}.jpg"),
            ascending=ascending,
        )

    # 明确异常优先：即使没有进入某个 Top-K，也应该出现在困难 CSV。
    difficult_rows = []
    for row in image_rows:
        is_diff = int(
            str(row.get("recommended_pool", "")) != "high"
            or int(row.get("is_BA", 0)) == 1
            or int(row.get("is_bad_response", 0)) == 1
            or int(row.get("is_refinable_risk", 0)) == 1
            or int(row.get("gt_is_bad", -1)) == 1
        )
        row["is_difficult_visual"] = is_diff
        row["difficulty_reasons"] = difficulty_reason_string(row)
        if is_diff:
            difficult_rows.append(row)

    # 先可视化多指标极端样本，再补明确异常样本；限制总量避免输出爆炸。
    chosen = []
    seen = set()
    for row in image_rows:
        name = str(row.get("image", ""))
        if name in selected_names and name not in seen:
            chosen.append(row)
            seen.add(name)
    # 让更明显的异常优先补齐。
    fallback = sorted(
        difficult_rows,
        key=lambda r: (
            int(r.get("gt_is_bad", -1)) == 1,
            int(r.get("is_bad_response", 0)) == 1,
            str(r.get("recommended_pool", "")) == "unvalue",
            safe_float(r.get("boundary_uncertainty"), 0.0),
        ),
        reverse=True,
    )
    for row in fallback:
        if len(chosen) >= DIAG_MAX_IMAGES:
            break
        name = str(row.get("image", ""))
        if name not in seen:
            chosen.append(row)
            seen.add(name)

    if SAVE_DIFFICULT_VIS:
        for row in tqdm(chosen[:DIAG_MAX_IMAGES], desc="Diagnostic visualizations"):
            image_name = str(row.get("image", ""))
            path = os.path.join(DIFFICULT_VIS_DIR, f"{image_name}.jpg")
            row["diagnostic_vis_path"] = make_diagnostic_panel(row, path)

    return difficult_rows


def main():
    for directory in [
        BASE_OUT_DIR, OUT_MASK_DIR, PROB_48_DIR, BAD_VIS_DIR, CSV_DIR,
        DIAG_ROOT_DIR, DIAG_CACHE_DIR, FEATURE_CACHE_DIR, DISAGREE_CACHE_DIR,
        DIFFICULT_VIS_DIR, TOPK_VIS_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)

    if torch.cuda.is_available() and str(DEVICE).startswith("cuda"):
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    write_generation_config(ba_threshold=None)

    print(f"Initializing {NUM_WORKERS} SAM1 predictor(s): {SAM_MODEL_TYPE}")
    predictor_queue = queue.Queue()
    for _ in range(NUM_WORKERS):
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device=DEVICE)
        sam.eval()
        predictor_queue.put(SamPredictor(sam))

    img_paths = collect_image_paths()
    metrics_group = init_metric_group()
    image_rows = []
    manifest_rows = []

    print("\nStart standard SAM1 box prompting with three-candidate quality evaluation")
    print(f"Images: {len(img_paths)}")
    print(f"Output: {BASE_OUT_DIR}")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(
                process_single_image,
                img_path,
                predictor_queue,
                metrics_group,
            )
            for img_path in img_paths
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                image_row, manifest_row = result
                image_rows.append(image_row)
                manifest_rows.append(manifest_row)

    image_rows.sort(key=lambda row: row["image"])
    manifest_rows.sort(key=lambda row: row["image"])

    # BA：按全数据 boundary_uncertainty 的最高 30% 标记
    boundary_values = np.asarray(
        [safe_float(row.get("boundary_uncertainty")) for row in image_rows],
        dtype=np.float32,
    )
    valid_boundary = boundary_values[np.isfinite(boundary_values)]
    if valid_boundary.size:
        ba_threshold = float(np.quantile(valid_boundary, 1.0 - BA_TOP_RATIO))
    else:
        ba_threshold = np.nan

    for row in image_rows:
        row["ba_threshold"] = ba_threshold
        value = safe_float(row.get("boundary_uncertainty"))
        row["is_BA"] = int(
            np.isfinite(value)
            and np.isfinite(ba_threshold)
            and value >= ba_threshold
        )

    assign_q_ranks(image_rows)

    # 全局 BA / Q 排名完成后，再构建困难样本与可视化。
    difficult_rows = build_difficulty_visualizations(image_rows)

    # GT 总体指标：只统计实际存在 GT 的图像
    gt_count = int(sum(row["has_gt"] for row in image_rows))
    final_results = {
        "Generated_Count": len(image_rows),
        "GT_Evaluated_Count": gt_count,
    }

    if gt_count > 0:
        try:
            all_res = get_metric_results(metrics_group)
            final_results["All_Pred_Mask_With_GT"] = {
                "Count": gt_count,
                **all_res,
            }
            print(
                f"[GT] Count={gt_count} | "
                f"S={all_res['S-measure']:.4f} | "
                f"E_adp={all_res['E-measure(adp)']:.4f} | "
                f"F_adp={all_res['F-measure(adp)']:.4f} | "
                f"MAE={all_res['MAE']:.4f}"
            )
        except Exception as exc:
            final_results["Metric_Error"] = str(exc)
            print(f"Metric summary failed: {exc}")

    final_results["Quality_Summary"] = {
        "Candidate_Grade_High": int(
            sum(row["candidate_s_grade"] == "high" for row in image_rows)
        ),
        "Candidate_Grade_Middle": int(
            sum(row["candidate_s_grade"] == "middle" for row in image_rows)
        ),
        "Candidate_Grade_Low": int(
            sum(row["candidate_s_grade"] == "low" for row in image_rows)
        ),
        "Recommended_High": int(
            sum(row["recommended_pool"] == "high" for row in image_rows)
        ),
        "Recommended_Middle": int(
            sum(row["recommended_pool"] == "middle" for row in image_rows)
        ),
        "Recommended_Unvalue": int(
            sum(row["recommended_pool"] == "unvalue" for row in image_rows)
        ),
        "Fails_Quality_Gate": int(
            sum(row["fails_quality_gate"] for row in image_rows)
        ),
        "Bad_Response_Count": int(
            sum(row["is_bad_response"] for row in image_rows)
        ),
        "Fragmented_Count": int(
            sum(row["is_fragmented_response"] for row in image_rows)
        ),
        "Severe_Fragmented_Count": int(
            sum(row["is_severe_fragmented_response"] for row in image_rows)
        ),
        "Refinable_Risk_Count": int(
            sum(row["is_refinable_risk"] for row in image_rows)
        ),
        "Reversal_Count": int(
            sum(row["is_reversal_response"] for row in image_rows)
        ),
        "Tiny_Collapse_Count": int(
            sum(row["is_tiny_collapse"] for row in image_rows)
        ),
        "Outside_Leak_Count": int(
            sum(row["is_outside_leak"] for row in image_rows)
        ),
        "Box_Copy_Count": int(
            sum(row["is_box_copy"] for row in image_rows)
        ),
        "Empty_Response_Count": int(
            sum(row["is_empty_response"] for row in image_rows)
        ),
        "BA_Threshold": None if np.isnan(ba_threshold) else ba_threshold,
        "BA_Count": int(sum(row["is_BA"] for row in image_rows)),
        "Candidate_S_Mean": (
            float(np.nanmean([row["candidate_consensus_s"] for row in image_rows]))
            if image_rows
            else None
        ),
        "SAM_Stability_Mean": (
            float(np.nanmean([row["sam_stability_selected"] for row in image_rows]))
            if image_rows
            else None
        ),
        "Q_Region_Mean": (
            float(np.nanmean([row["q_region"] for row in image_rows]))
            if image_rows
            else None
        ),
        "Q_Shape_Mean": (
            float(np.nanmean([row["q_shape"] for row in image_rows]))
            if image_rows
            else None
        ),
        "Q_Dual_Mean": (
            float(np.nanmean([row["q_dual"] for row in image_rows]))
            if image_rows
            else None
        ),
        "Q_Mean": (
            float(np.nanmean([row["q_total"] for row in image_rows]))
            if image_rows
            else None
        ),
    }

    with open(JSON_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    with open(BOX_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2, ensure_ascii=False)

    write_generation_config(
        ba_threshold=None if np.isnan(ba_threshold) else ba_threshold
    )

    image_fieldnames = [
        # 身份与路径
        "image",
        "image_height",
        "image_width",
        "hard_mask_path",
        "prob_48_path",
        "feature_map_path",
        "candidate_disagreement_path",
        "diagnostic_vis_path",
        "difficulty_reasons",
        "is_difficult_visual",
        "bad_vis_path",
        "has_gt",

        # 单次 box prompt
        "num_boxes_in_json",
        "used_enclosing_union_box",
        "box_area_ratio",
        "is_SO",
        "is_OV",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",

        # 三候选共识与最终候选
        "candidate_score_1",
        "candidate_score_2",
        "candidate_score_3",
        "selected_candidate_index",
        "candidate_iou_12",
        "candidate_iou_13",
        "candidate_iou_23",
        "candidate_ssim_12",
        "candidate_ssim_13",
        "candidate_ssim_23",
        "candidate_sim_12",
        "candidate_sim_13",
        "candidate_sim_23",
        "candidate_sim_min",
        "candidate_sim_max",
        "candidate_sim_range",
        "candidate_consensus_s",
        "candidate_s_grade",
        "sam_score_selected",
        "sam_stability_selected",

        # FG / BG / Q / 门控
        "fg_confidence",
        "fg_mean_confidence",
        "fg_p10_confidence",
        "fg_core_area",
        "fg_core_area_ratio",
        "bg_confidence",
        "ring_bg_confidence",
        "hard_outside_bg_confidence",
        "q_region",
        "q_shape",
        "q_dual",
        "q_total",
        "fail_fg_gate",
        "fail_bg_gate",
        "fail_score_gate",
        "fail_stability_gate",
        "fails_quality_gate",

        # 几何违规 V
        "outside_pixel_count",
        "v_outside",
        "box_fit_iou",
        "v_fit",
        "mask_box_fill_ratio",
        "v_area",
        "center_offset_norm",
        "v_center",
        "v_total",
        "mask_bbox_x1",
        "mask_bbox_y1",
        "mask_bbox_x2",
        "mask_bbox_y2",
        "mask_centroid_x",
        "mask_centroid_y",
        "box_centroid_x",
        "box_centroid_y",

        # 边界模糊
        "boundary_uncertainty",
        "boundary_band_area",
        "ba_threshold",
        "is_BA",

        # 异常响应单独打标
        "mask_area_pixels",
        "mask_image_area_ratio",
        "mask_box_area_ratio",
        "inside_mask_ratio",
        "outside_mask_ratio",
        "num_components",
        "raw_num_components",
        "effective_num_components",
        "largest_component_ratio",
        "largest_effective_component_ratio",
        "small_fragment_area_ratio",
        "effective_component_min_area",
        "mask_rectangularity",
        "mask_bbox_iou_with_box",
        "is_empty_response",
        "is_tiny_collapse",
        "is_fragmented_response",
        "is_severe_fragmented_response",
        "is_refinable_risk",
        "is_outside_leak",
        "is_box_copy",
        "is_reversal_response",
        "is_bad_response",

        # 后续分池依据
        "force_unvalue",
        "recommended_pool",
        "q_region_rank_global",
        "q_shape_rank_global",
        "q_dual_rank_global",
        "q_region_rank_within_pool",
        "q_shape_rank_within_pool",
        "q_dual_rank_within_pool",
        "q_rank_global",
        "q_rank_within_pool",

        # GT 指标，仅用于验证
        "gt_area_ratio",
        "IoU",
        "Smeasure",
        "Emeasure_adp",
        "Fmeasure_adp",
        "MAE",
        "gt_bad_by_iou",
        "gt_bad_by_mae",
        "gt_is_bad",
    ]
    write_csv(IMAGE_CSV_PATH, image_rows, image_fieldnames)
    write_csv(DIFFICULT_CSV_PATH, difficult_rows, image_fieldnames)

    print("\nDone")
    print(f"Masks: {OUT_MASK_DIR}")
    print(f"48x48 FP16 probability: {PROB_48_DIR}")
    print(f"Image-level CSV: {IMAGE_CSV_PATH}")
    print(f"Difficult CSV: {DIFFICULT_CSV_PATH}")
    print(f"Difficulty visualizations: {DIFFICULT_VIS_DIR}")
    print(f"Top-K score sheets: {TOPK_VIS_DIR}")
    print(f"Box manifest: {BOX_MANIFEST_PATH}")
    print(f"Summary JSON: {JSON_RESULTS_PATH}")
    print(f"Config JSON: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
