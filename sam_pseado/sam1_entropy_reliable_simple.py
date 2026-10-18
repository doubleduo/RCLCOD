"""Minimal SAM1 box-prompt pseudo-label generation for entropy reliability.

Outputs
-------
Pred_Masks/<name>.png
    Binary pseudo label Y used as the segmentation target.
Prob_48x48_FP16/<name>.npy
    Soft probability P used to calculate image reliability q and pixel
    reliability U during PVT training.
reliability.csv
    One compact row per image.  Only entropy reliability and three inexpensive
    SAM evidence scalars are retained.
summary.json
    Dataset-level generation and optional GT statistics.

This file intentionally removes pool assignment, ranks, hand-crafted Q/V,
boundary percentiles, anomaly flags, per-image S/E/F metrics, visualizations,
and multi-model threading.  It is a SAM1 script (segment_anything), not SAM2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
PROBABILITY_SIZE = (48, 48)  # OpenCV order: (width, height)
ENTROPY_EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SAM1 binary masks and entropy reliability"
    )
    parser.add_argument("--checkpoint", default="./sam_pseado/weights/sam_vit_h_4b8939.pth")
    parser.add_argument("--model-type", default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-dir", default="./data/Train/Imgs")
    parser.add_argument("--box-dir", default="./data/Train/box_label")
    parser.add_argument("--gt-dir", default="./data/Train/GT")
    parser.add_argument(
        "--output-dir",
        default="./SGT/SAM1_EntropyReliable_Box",
        help="Use a new directory to avoid overwriting previous experiments.",
    )
    parser.add_argument(
        "--eval-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="GT is used only for offline IoU/MAE evaluation.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Input and geometry
# -----------------------------------------------------------------------------

def collect_images(image_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_union_box(json_path: Path, height: int, width: int) -> Optional[np.ndarray]:
    """Read LabelMe rectangles and return one clipped enclosing box."""
    if not json_path.is_file():
        return None

    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    boxes: List[Tuple[float, float, float, float]] = []
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2:
            continue

        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1 = float(np.clip(min(xs), 0, max(0, width - 1)))
        y1 = float(np.clip(min(ys), 0, max(0, height - 1)))
        x2 = float(np.clip(max(xs), 0, width))
        y2 = float(np.clip(max(ys), 0, height))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    if not boxes:
        return None

    return np.asarray(
        [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ],
        dtype=np.float32,
    )


# -----------------------------------------------------------------------------
# Reliability and lightweight evidence
# -----------------------------------------------------------------------------

def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(np.asarray(logits, dtype=np.float32), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def entropy_reliability(probability: np.ndarray) -> Tuple[float, float]:
    """Return normalized mean entropy H and image reliability q=1-H."""
    probability = np.asarray(probability, dtype=np.float32)
    p = np.clip(probability, ENTROPY_EPS, 1.0 - ENTROPY_EPS)
    entropy_map = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)
    entropy = float(np.clip(entropy_map.mean(), 0.0, 1.0))
    return entropy, float(1.0 - entropy)


def binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 1.0


def candidate_iou_consensus(candidate_masks: np.ndarray) -> float:
    """Mean pairwise IoU of the three SAM candidates; one scalar only."""
    if candidate_masks.ndim != 3 or candidate_masks.shape[0] < 2:
        return float("nan")
    pairs = [(0, 1), (0, 2), (1, 2)]
    values = [
        binary_iou(candidate_masks[a], candidate_masks[b])
        for a, b in pairs
        if a < candidate_masks.shape[0] and b < candidate_masks.shape[0]
    ]
    return float(np.mean(values)) if values else float("nan")


def sam_stability(
    selected_logit: np.ndarray,
    mask_threshold: float,
    threshold_offset: float = 1.0,
) -> float:
    high = selected_logit > (mask_threshold + threshold_offset)
    low = selected_logit > (mask_threshold - threshold_offset)
    low_area = int(low.sum())
    return float(high.sum() / low_area) if low_area else 0.0


# -----------------------------------------------------------------------------
# SAM inference
# -----------------------------------------------------------------------------

@torch.inference_mode()
def predict_one(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    prompt_box: np.ndarray,
) -> Dict[str, object]:
    predictor.set_image(image_rgb)
    full_logits, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=prompt_box,
        multimask_output=True,
        return_logits=True,
    )

    full_logits = np.asarray(full_logits, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if full_logits.ndim == 2:
        full_logits = full_logits[None]
    if full_logits.shape[0] == 0:
        raise RuntimeError("SAM returned no masks")

    best_index = int(np.argmax(scores)) if scores.size else 0
    threshold = float(predictor.model.mask_threshold)
    candidate_masks = full_logits > threshold
    selected_logit = full_logits[best_index]

    return {
        "mask": selected_logit > threshold,
        "probability": sigmoid(selected_logit),
        "sam_score": float(scores[best_index]) if scores.size else float("nan"),
        "sam_stability": sam_stability(selected_logit, threshold),
        "candidate_iou_consensus": candidate_iou_consensus(candidate_masks[:3]),
    }


# -----------------------------------------------------------------------------
# Optional GT evaluation
# -----------------------------------------------------------------------------

def evaluate_with_gt(
    prediction: np.ndarray,
    gt_path: Path,
) -> Tuple[int, float, float]:
    if not gt_path.is_file():
        return 0, float("nan"), float("nan")

    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return 0, float("nan"), float("nan")
    if gt.shape != prediction.shape:
        gt = cv2.resize(
            gt,
            (prediction.shape[1], prediction.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    gt_bool = gt > 128
    iou = binary_iou(prediction, gt_bool)
    mae = float(np.abs(prediction.astype(np.float32) - gt_bool.astype(np.float32)).mean())
    return 1, iou, mae


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

CSV_FIELDS = [
    "image",
    "q_entropy",
    "entropy",
    "sam_score",
    "sam_stability",
    "candidate_iou_consensus",
    "pred_area_ratio",
    "has_gt",
    "gt_iou",
    "gt_mae",
    "mask_path",
    "probability_path",
]


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_skipped_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    box_dir = Path(args.box_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "Pred_Masks"
    probability_dir = output_dir / "Prob_48x48_FP16"

    mask_dir.mkdir(parents=True, exist_ok=True)
    probability_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    # One ViT-H on one GPU.  Multiple Python threads would duplicate the model
    # and usually waste memory without increasing single-GPU throughput.
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    sam.eval()
    predictor = SamPredictor(sam)

    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, str]] = []

    for image_path in tqdm(images, desc="Generating SAM pseudo labels"):
        name = image_path.stem
        try:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("image read failed")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]

            prompt_box = read_union_box(box_dir / f"{name}.json", height, width)
            if prompt_box is None:
                raise ValueError("valid box not found")

            result = predict_one(predictor, image_rgb, prompt_box)
            prediction = np.asarray(result["mask"], dtype=bool)
            probability = np.asarray(result["probability"], dtype=np.float32)

            mask_path = mask_dir / f"{name}.png"
            probability_path = probability_dir / f"{name}.npy"
            if not cv2.imwrite(str(mask_path), prediction.astype(np.uint8) * 255):
                raise OSError("failed to save binary mask")

            probability_48 = cv2.resize(
                probability,
                PROBABILITY_SIZE,
                interpolation=cv2.INTER_AREA,
            ).astype(np.float16)
            np.save(probability_path, probability_48, allow_pickle=False)

            # Calculate q from exactly the same FP16 map that training reads.
            entropy, q_entropy = entropy_reliability(probability_48)

            if args.eval_gt:
                has_gt, gt_iou, gt_mae = evaluate_with_gt(
                    prediction,
                    gt_dir / f"{name}.png",
                )
            else:
                has_gt, gt_iou, gt_mae = 0, float("nan"), float("nan")

            rows.append(
                {
                    "image": name,
                    "q_entropy": q_entropy,
                    "entropy": entropy,
                    "sam_score": result["sam_score"],
                    "sam_stability": result["sam_stability"],
                    "candidate_iou_consensus": result["candidate_iou_consensus"],
                    "pred_area_ratio": float(prediction.mean()),
                    "has_gt": has_gt,
                    "gt_iou": gt_iou,
                    "gt_mae": gt_mae,
                    "mask_path": str(mask_path),
                    "probability_path": str(probability_path),
                }
            )
        except Exception as error:
            skipped.append({"image": name, "reason": str(error)})

    rows.sort(key=lambda row: str(row["image"]))
    write_csv(output_dir / "reliability.csv", rows)
    write_skipped_csv(output_dir / "skipped.csv", skipped)

    q_values = np.asarray([row["q_entropy"] for row in rows], dtype=np.float64)
    summary = {
        "sam_model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "generated_count": len(rows),
        "skipped_count": len(skipped),
        "q_entropy_mean": float(q_values.mean()) if q_values.size else None,
        "q_entropy_min": float(q_values.min()) if q_values.size else None,
        "q_entropy_max": float(q_values.max()) if q_values.size else None,
        "gt_evaluated_count": int(sum(int(row["has_gt"]) for row in rows)),
        "gt_iou_mean": finite_mean(rows, "gt_iou") if rows else None,
        "gt_mae_mean": finite_mean(rows, "gt_mae") if rows else None,
        "reliability": {
            "image": "q = 1 - mean(normalized Bernoulli entropy(P))",
            "pixel": "U = (2P - 1)^2; calculated during training, not saved twice",
            "probability_shape": [48, 48],
            "probability_dtype": "float16",
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"Generated: {len(rows)} | Skipped: {len(skipped)}")
    print(f"Masks: {mask_dir}")
    print(f"Probabilities: {probability_dir}")
    print(f"Reliability: {output_dir / 'reliability.csv'}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
