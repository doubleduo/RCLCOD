#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate masks with RAM crops and SAM2 box prompts only, then evaluate GT.

Kept pipeline:
    image + LabelMe JSON boxes -> per-instance RAM crop -> 256x256 SAM2
    box prompt -> restore to original image -> binary mask/probability

Removed:
    points, scribbles, mask feedback, iterative refinement, flip augmentation,
    quality routing, and visualization.

Evaluation:
    S-measure and MAE use the continuous probability map. F (beta^2=0.3)
    and E use one fixed threshold of 0.5. GT never enters inference.

Examples:
    python sam2_box_prompt_only.py
    python sam2_box_prompt_only.py --save-probs
    python sam2_box_prompt_only.py --multi-box-mode union
    python sam2_box_prompt_only.py --limit 20
"""

import argparse
import csv
import json
import os
from contextlib import nullcontext
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def list_images(directory):
    paths = []
    for extension in IMAGE_EXTENSIONS:
        paths.extend(glob(os.path.join(directory, extension)))
        paths.extend(glob(os.path.join(directory, extension.upper())))
    return sorted(set(paths))


def find_json(directory, basename):
    for extension in (".json", ".JSON"):
        path = os.path.join(directory, basename + extension)
        if os.path.isfile(path):
            return path
    return None


def find_gt(directory, basename):
    """Find a same-stem GT mask using common image extensions."""
    if not directory:
        return None
    for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        for candidate_extension in (extension, extension.upper()):
            path = os.path.join(directory, basename + candidate_extension)
            if os.path.isfile(path):
                return path
    return None


def load_labelme_boxes(json_path, image_width, image_height):
    """Read all valid LabelMe shapes as XYXY boxes in original-image coordinates."""
    if json_path is None:
        return []

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    boxes = []
    for shape in data.get("shapes", []):
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue

        box = np.asarray(
            [
                points[:, 0].min(),
                points[:, 1].min(),
                points[:, 0].max(),
                points[:, 1].max(),
            ],
            dtype=np.float32,
        )
        box[[0, 2]] = np.clip(box[[0, 2]], 0, image_width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, image_height - 1)
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def union_box(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    return np.asarray(
        [
            boxes[:, 0].min(),
            boxes[:, 1].min(),
            boxes[:, 2].max(),
            boxes[:, 3].max(),
        ],
        dtype=np.float32,
    )


def ram_region_size(box, threshold_m=32.0, k0=2.0, smax=1024.0):
    """Compute the RAM square side length from Eq. (1) in SOPSeg."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    box_width = max(1.0, float(x2 - x1))
    box_height = max(1.0, float(y2 - y1))
    object_size = max(box_width, box_height)

    if threshold_m <= 0 or smax <= threshold_m:
        raise ValueError("RAM requires 0 < threshold_m < smax")
    if k0 < 1.0:
        raise ValueError("RAM requires k0 >= 1 so the crop contains the box")

    k = (smax - k0 * threshold_m) / (smax - threshold_m)
    if object_size < threshold_m:
        region_size = k0 * object_size
    else:
        region_size = k * object_size + (k0 - k) * threshold_m

    # The paper assumes d <= Smax. Keep oversized boxes valid as well.
    region_size = max(object_size, region_size)
    return max(1, int(np.ceil(region_size)))


def ram_crop_rect(box, region_size, ax=0.5, ay=0.5):
    """Return virtual square crop (x0, y0, size) using RAM Eq. (2).

    At inference ax=ay=0.5 gives a centered crop. Coordinates may extend
    outside the image; extract_ram_crop pads those areas without changing S.
    """
    if not 0.0 <= ax <= 1.0 or not 0.0 <= ay <= 1.0:
        raise ValueError("RAM ax and ay must be in [0, 1]")

    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    box_width = max(1.0, float(x2 - x1))
    box_height = max(1.0, float(y2 - y1))
    x0 = int(np.round(x1 - ax * (region_size - box_width)))
    y0 = int(np.round(y1 - ay * (region_size - box_height)))
    return x0, y0, int(region_size)


def extract_ram_crop(rgb_image, crop_rect):
    """Extract an SxS crop; replicate-pad only where it crosses image borders."""
    x0, y0, size = crop_rect
    height, width = rgb_image.shape[:2]
    left = max(0, -x0)
    top = max(0, -y0)
    right = max(0, x0 + size - width)
    bottom = max(0, y0 + size - height)

    padded = cv2.copyMakeBorder(
        rgb_image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REPLICATE,
    )
    px0 = x0 + left
    py0 = y0 + top
    crop = padded[py0 : py0 + size, px0 : px0 + size]
    if crop.shape[:2] != (size, size):
        raise RuntimeError(
            f"RAM crop shape mismatch: expected {(size, size)}, got {crop.shape[:2]}"
        )
    return crop


def transform_box_to_ram_input(box, crop_rect, input_size, margin=0.0):
    """Map an original XYXY box into the resized RAM crop coordinate frame."""
    x0, y0, region_size = crop_rect
    scale = input_size / region_size
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    transformed = np.asarray(
        [
            (x1 - x0 + 0.5) * scale - 0.5 - margin,
            (y1 - y0 + 0.5) * scale - 0.5 - margin,
            (x2 - x0 + 0.5) * scale - 0.5 + margin,
            (y2 - y0 + 0.5) * scale - 0.5 + margin,
        ],
        dtype=np.float32,
    )
    transformed[[0, 2]] = np.clip(transformed[[0, 2]], 0, input_size - 1)
    transformed[[1, 3]] = np.clip(transformed[[1, 3]], 0, input_size - 1)
    return transformed


def paste_ram_probability(canvas, crop_probability, crop_rect):
    """Resize one RAM prediction to SxS and paste its valid part onto canvas."""
    x0, y0, region_size = crop_rect
    image_height, image_width = canvas.shape
    region_probability = cv2.resize(
        crop_probability,
        (region_size, region_size),
        interpolation=cv2.INTER_LINEAR,
    )

    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(image_width, x0 + region_size)
    dst_y1 = min(image_height, y0 + region_size)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return

    src_x0 = dst_x0 - x0
    src_y0 = dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = np.maximum(
        canvas[dst_y0:dst_y1, dst_x0:dst_x1],
        region_probability[src_y0:src_y1, src_x0:src_x1],
    )


def sigmoid(logits):
    logits = np.clip(np.asarray(logits, dtype=np.float32), -30, 30)
    return 1.0 / (1.0 + np.exp(-logits))


class MetricsEvaluator:
    """COD metrics: S-measure, fixed-threshold E/F-measure, and MAE."""

    EPSILON = 1e-8

    @staticmethod
    def prepare_pred_gt(prediction, ground_truth):
        prediction = np.asarray(prediction)
        ground_truth = np.asarray(ground_truth)
        if prediction.shape != ground_truth.shape:
            prediction = cv2.resize(
                prediction,
                (ground_truth.shape[1], ground_truth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        prediction = prediction.astype(np.float32)
        if prediction.size and prediction.max() > 1.0:
            prediction /= 255.0
        ground_truth = ground_truth.astype(np.float32)
        gt_threshold = 0.5 if ground_truth.size and ground_truth.max() <= 1.0 else 128.0
        return np.clip(prediction, 0.0, 1.0), (ground_truth > gt_threshold).astype(
            np.float32
        )

    @classmethod
    def object_score(cls, values):
        if values.size == 0:
            return 0.0
        mean = float(values.mean())
        std = float(values.std())
        return (2.0 * mean) / (mean * mean + 1.0 + std + cls.EPSILON)

    @classmethod
    def object_similarity(cls, prediction, ground_truth):
        foreground_ratio = float(ground_truth.mean())
        foreground = cls.object_score(prediction[ground_truth == 1])
        background = cls.object_score(1.0 - prediction[ground_truth == 0])
        return foreground_ratio * foreground + (1.0 - foreground_ratio) * background

    @classmethod
    def region_ssim(cls, prediction, ground_truth):
        if prediction.size == 0:
            return 0.0
        prediction_mean = float(prediction.mean())
        gt_mean = float(ground_truth.mean())
        prediction_variance = float(((prediction - prediction_mean) ** 2).mean())
        gt_variance = float(((ground_truth - gt_mean) ** 2).mean())
        covariance = float(
            ((prediction - prediction_mean) * (ground_truth - gt_mean)).mean()
        )
        numerator = 4.0 * prediction_mean * gt_mean * covariance
        denominator = (
            (prediction_mean**2 + gt_mean**2)
            * (prediction_variance + gt_variance)
            + cls.EPSILON
        )
        return numerator / denominator

    @classmethod
    def region_similarity(cls, prediction, ground_truth):
        height, width = ground_truth.shape
        if height < 2 or width < 2:
            return 0.0
        ys, xs = np.where(ground_truth > 0.5)
        center_x = int(np.clip(np.round(xs.mean()), 1, width - 1))
        center_y = int(np.clip(np.round(ys.mean()), 1, height - 1))
        weighted_sum = (
            center_x
            * center_y
            * cls.region_ssim(
                prediction[:center_y, :center_x],
                ground_truth[:center_y, :center_x],
            )
            + (width - center_x)
            * center_y
            * cls.region_ssim(
                prediction[:center_y, center_x:],
                ground_truth[:center_y, center_x:],
            )
            + center_x
            * (height - center_y)
            * cls.region_ssim(
                prediction[center_y:, :center_x],
                ground_truth[center_y:, :center_x],
            )
            + (width - center_x)
            * (height - center_y)
            * cls.region_ssim(
                prediction[center_y:, center_x:],
                ground_truth[center_y:, center_x:],
            )
        )
        return weighted_sum / float(height * width)

    @classmethod
    def s_measure(cls, prediction, ground_truth):
        foreground_ratio = float(ground_truth.mean())
        if foreground_ratio == 0.0:
            return 1.0 - float(prediction.mean())
        if foreground_ratio == 1.0:
            return float(prediction.mean())
        score = 0.5 * cls.object_similarity(
            prediction, ground_truth
        ) + 0.5 * cls.region_similarity(prediction, ground_truth)
        return float(np.clip(score, 0.0, 1.0))

    @classmethod
    def f_measure(cls, binary_prediction, binary_ground_truth):
        true_positive = float((binary_prediction & binary_ground_truth).sum())
        precision = true_positive / (float(binary_prediction.sum()) + cls.EPSILON)
        recall = true_positive / (float(binary_ground_truth.sum()) + cls.EPSILON)
        beta_squared = 0.3
        score = (
            (1.0 + beta_squared)
            * precision
            * recall
            / (beta_squared * precision + recall + cls.EPSILON)
        )
        return float(score)

    @classmethod
    def e_measure(cls, binary_prediction, binary_ground_truth):
        foreground_count = int(binary_ground_truth.sum())
        if foreground_count == 0:
            return 1.0 - float(binary_prediction.mean())
        if foreground_count == binary_ground_truth.size:
            return float(binary_prediction.mean())

        prediction = binary_prediction.astype(np.float32)
        ground_truth = binary_ground_truth.astype(np.float32)
        prediction_deviation = prediction - prediction.mean()
        gt_deviation = ground_truth - ground_truth.mean()
        alignment = (
            2.0
            * prediction_deviation
            * gt_deviation
            / (
                prediction_deviation**2
                + gt_deviation**2
                + cls.EPSILON
            )
        )
        enhanced_alignment = ((alignment + 1.0) ** 2) / 4.0
        return float(enhanced_alignment.mean())

    @classmethod
    def evaluate(cls, prediction_raw, ground_truth_raw):
        prediction, ground_truth = cls.prepare_pred_gt(
            prediction_raw, ground_truth_raw
        )
        binary_prediction = prediction >= 0.5
        binary_ground_truth = ground_truth >= 0.5
        return {
            "Smeasure": cls.s_measure(prediction, ground_truth),
            "E": cls.e_measure(binary_prediction, binary_ground_truth),
            "F": cls.f_measure(binary_prediction, binary_ground_truth),
            "MAE": float(np.mean(np.abs(prediction - ground_truth))),
        }


class SAM2BoxPredictor:
    def __init__(self, checkpoint, model_config, device, input_size=256):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device
        self.input_size = input_size
        if input_size % 16 != 0:
            raise ValueError("SAM2 input size must be divisible by 16")

        # This override is essential: otherwise SAM2ImagePredictor silently
        # resizes a supplied 256x256 crop back to the default 1024x1024.
        model = build_sam2(
            model_config,
            checkpoint,
            device=device,
            hydra_overrides_extra=[f"++model.image_size={input_size}"],
        )
        self.predictor = SAM2ImagePredictor(model)

        # Official SAM2ImagePredictor hard-codes the 1024-input feature sizes
        # [(256,256), (128,128), (64,64)]. Make them consistent with 256.
        high_resolution = input_size // 4
        self.predictor._bb_feat_sizes = [
            (high_resolution // (2**level), high_resolution // (2**level))
            for level in range(3)
        ]

        if device.startswith("cuda") and torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def set_image(self, rgb_image):
        if rgb_image.shape[:2] != (self.input_size, self.input_size):
            raise ValueError(
                f"Expected RAM input {(self.input_size, self.input_size)}, "
                f"got {rgb_image.shape[:2]}"
            )
        with torch.inference_mode():
            self.predictor.set_image(np.ascontiguousarray(rgb_image))

    def predict_box(self, model_box):
        autocast = nullcontext()
        if (
            self.device.startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            autocast = torch.autocast("cuda", dtype=torch.bfloat16)

        with torch.inference_mode(), autocast:
            high_res_logits, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=model_box[None, :],
                mask_input=None,
                multimask_output=False,
                return_logits=True,
            )

        logits = np.asarray(high_res_logits, dtype=np.float32)
        if logits.ndim == 3:
            logits = logits[0]
        return sigmoid(logits), float(np.asarray(scores).reshape(-1)[0])


def infer_box_only_ram(
    engine,
    rgb_image,
    boxes,
    mode,
    box_margin,
    ram_threshold_m,
    ram_k0,
    ram_smax,
    ram_ax,
    ram_ay,
):
    """Run per-instance RAM + box-only SAM2 and merge in the original frame."""
    height, width = rgb_image.shape[:2]
    prompt_boxes = [union_box(boxes)] if mode == "union" else boxes
    merged_probability = np.zeros((height, width), np.float32)
    scores = []
    regions = []

    for box in prompt_boxes:
        region_size = ram_region_size(
            box,
            threshold_m=ram_threshold_m,
            k0=ram_k0,
            smax=ram_smax,
        )
        crop_rect = ram_crop_rect(box, region_size, ax=ram_ax, ay=ram_ay)
        crop = extract_ram_crop(rgb_image, crop_rect)
        model_image = cv2.resize(
            crop,
            (engine.input_size, engine.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        model_box = transform_box_to_ram_input(
            box,
            crop_rect=crop_rect,
            input_size=engine.input_size,
            margin=box_margin,
        )
        engine.set_image(model_image)
        probability, score = engine.predict_box(model_box)
        paste_ram_probability(merged_probability, probability, crop_rect)
        scores.append(score)
        regions.append(
            {
                "box": np.asarray(box, dtype=np.float32).tolist(),
                "crop": [int(crop_rect[0]), int(crop_rect[1]), int(crop_rect[2])],
                "score": score,
            }
        )

    return merged_probability, scores, regions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sam2-checkpoint",
        default="./sam_pseado/weights/sam2_hiera_large.pt",
    )
    parser.add_argument("--sam2-config", default="sam2_hiera_l.yaml")
    parser.add_argument("--img-dir", default="./data/Train/Imgs")
    parser.add_argument("--json-dir", default="./data/Train/box_label")
    parser.add_argument(
        "--gt-dir",
        default="./data/Train/GT",
        help="Same-stem binary GT masks. Set to an empty string to disable evaluation.",
    )
    parser.add_argument("--output-dir", default="./softresult/SAM2_BoxOnly_RAM256")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="True SAM2 encoder input size; RAM uses 256 by default.",
    )
    parser.add_argument(
        "--box-margin",
        type=float,
        default=1.25,
        help="Margin in the 256x256 model frame; 1.25 equals 5 px at 1024.",
    )
    parser.add_argument("--ram-threshold-m", type=float, default=32.0)
    parser.add_argument("--ram-k0", type=float, default=2.0)
    parser.add_argument("--ram-smax", type=float, default=1024.0)
    parser.add_argument("--ram-ax", type=float, default=0.5)
    parser.add_argument("--ram-ay", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--multi-box-mode",
        choices=("union", "per_box"),
        default="per_box",
        help="RAM is instance-specific, so per_box is recommended; union is kept for ablation.",
    )
    parser.add_argument("--save-probs", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.input_size <= 0 or args.input_size % 16 != 0:
        raise ValueError("--input-size must be positive and divisible by 16")
    if not 0.0 <= args.ram_ax <= 1.0 or not 0.0 <= args.ram_ay <= 1.0:
        raise ValueError("--ram-ax and --ram-ay must be in [0, 1]")
    if args.ram_k0 < 1.0:
        raise ValueError("--ram-k0 must be >= 1")
    if args.ram_threshold_m <= 0 or args.ram_smax <= args.ram_threshold_m:
        raise ValueError("Require 0 < --ram-threshold-m < --ram-smax")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")

    mask_dir = os.path.join(args.output_dir, "masks")
    probability_dir = os.path.join(args.output_dir, "probs")
    os.makedirs(mask_dir, exist_ok=True)
    if args.save_probs:
        os.makedirs(probability_dir, exist_ok=True)

    engine = SAM2BoxPredictor(
        checkpoint=args.sam2_checkpoint,
        model_config=args.sam2_config,
        device=args.device,
        input_size=args.input_size,
    )

    image_paths = list_images(args.img_dir)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    processed = 0
    skipped = 0
    missing_gt = 0
    errors = []
    metric_rows = []

    for image_path in tqdm(image_paths, desc="SAM2 box-only"):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        try:
            bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if bgr_image is None:
                raise ValueError("OpenCV could not read the image")

            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            height, width = rgb_image.shape[:2]
            json_path = find_json(args.json_dir, basename)
            boxes = load_labelme_boxes(json_path, width, height)
            if not boxes:
                skipped += 1
                continue

            probability, scores, _ = infer_box_only_ram(
                engine,
                rgb_image,
                boxes,
                mode=args.multi_box_mode,
                box_margin=args.box_margin,
                ram_threshold_m=args.ram_threshold_m,
                ram_k0=args.ram_k0,
                ram_smax=args.ram_smax,
                ram_ax=args.ram_ax,
                ram_ay=args.ram_ay,
            )
            binary_mask = (probability >= args.threshold).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(mask_dir, basename + ".png"), binary_mask)

            if args.save_probs:
                np.save(
                    os.path.join(probability_dir, basename + ".npy"),
                    probability.astype(np.float16),
                )

            gt_path = find_gt(args.gt_dir, basename)
            if gt_path is None:
                if args.gt_dir:
                    missing_gt += 1
            else:
                ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if ground_truth is None:
                    raise ValueError(f"OpenCV could not read GT: {gt_path}")
                metrics = MetricsEvaluator.evaluate(probability, ground_truth)
                metric_rows.append(
                    {
                        "image_name": os.path.basename(image_path),
                        "Smeasure": metrics["Smeasure"],
                        "E": metrics["E"],
                        "F": metrics["F"],
                        "MAE": metrics["MAE"],
                        "num_boxes": len(boxes),
                        "mean_sam_score": float(np.mean(scores)) if scores else 0.0,
                    }
                )
            processed += 1

        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
            errors.append({"image": image_path, "error": repr(error)})
            print(f"[ERROR] {os.path.basename(image_path)}: {error}")

    metrics_csv_path = os.path.join(args.output_dir, "metrics.csv")
    metric_fields = [
        "image_name",
        "Smeasure",
        "E",
        "F",
        "MAE",
        "num_boxes",
        "mean_sam_score",
    ]
    with open(metrics_csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    mean_metrics = {}
    if metric_rows:
        for metric_name in ("Smeasure", "E", "F", "MAE"):
            mean_metrics[metric_name] = float(
                np.mean([row[metric_name] for row in metric_rows])
            )

    summary = {
        "processed": processed,
        "skipped_without_valid_box": skipped,
        "failed": len(errors),
        "evaluated_with_gt": len(metric_rows),
        "missing_gt": missing_gt,
        "mean_metrics": mean_metrics,
        "metric_protocol": {
            "Smeasure": "continuous probability map",
            "E": "binary prediction at fixed threshold=0.5",
            "F": "binary prediction at fixed threshold=0.5, beta_squared=0.3",
            "MAE": "continuous probability map",
        },
        "metrics_csv": metrics_csv_path,
        "multi_box_mode": args.multi_box_mode,
        "input_size": args.input_size,
        "ram": {
            "threshold_m": args.ram_threshold_m,
            "k0": args.ram_k0,
            "smax": args.ram_smax,
            "ax": args.ram_ax,
            "ay": args.ram_ay,
        },
        "errors": errors,
    }
    with open(
        os.path.join(args.output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
