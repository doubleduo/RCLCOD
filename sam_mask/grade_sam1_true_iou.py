#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grade SAM1 pseudo-labels using TRUE IoU against GT (offline/oracle only).

Reads the image names and candidate consistency from the existing SAM1 CSV,
loads the saved final profile masks and dataset GT masks, computes actual
IoU(prediction, GT), and exports Clean/Noisy/Reject splits and diagnostics.

This uses training-set GT and MUST NOT be used as a weak-supervision
sample selection mechanism: it is an oracle upper-bound/diagnostic only.

Example (from RCLCOD root):
    python grade_sam1_true_iou.py \
      --csv 'sam1_soft_so_mo_metrics(2).csv' \
      --pred-dir ./softresult/SAM1_Optimized3/02_SoftLogits_SO_MO \
      --gt-dir ./data/Train/GT \
      --output-dir ./softresult/SAM1_TrueIoU_Grade

    python grade_sam1_true_iou.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np

LEVELS = ("Clean", "Noisy", "Reject")


def classify(iou: float, clean: float, noisy: float) -> str:
    return "Clean" if iou >= clean else ("Noisy" if iou >= noisy else "Reject")


def locate_mask(directory: Path, stem: str, extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")) -> Path | None:
    for ext in extensions:
        for variant in (ext, ext.upper()):
            f = directory / (stem + variant)
            if f.is_file():
                return f
    return None


def true_metrics(pred_path: Path, gt_path: Path) -> Dict[str, float | int | bool]:
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if pred is None or gt is None:
        raise ValueError(f"Failed to read mask: {pred_path} or {gt_path}")
    resized = False
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        resized = True
    # Original SAM1 script saves binary 0/255 masks; GT is >128.  For
    # regenerated grayscale masks, thresholding at 128 has the same contract.
    p = pred > 128
    g = gt > 128
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    union = tp + fp + fn
    n = int(g.size)
    iou = float(tp / union) if union else 1.0
    dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
    mae = float((fp + fn) / n)
    return {
        "true_iou": iou, "true_dice": dice, "true_mae": mae,
        "tp": tp, "fp": fp, "fn": fn, "gt_pixels": int(np.count_nonzero(g)),
        "pred_pixels": int(np.count_nonzero(p)),
        "gt_area_ratio": float(np.count_nonzero(g) / n),
        "pred_area_ratio": float(np.count_nonzero(p) / n),
        "size_mismatch_resized": int(resized),
    }


def average(rows: Iterable[dict], name: str):
    vals = []
    for row in rows:
        try:
            v = float(row.get(name, ""))
            if np.isfinite(v):
                vals.append(v)
        except (ValueError, TypeError):
            pass
    return float(np.mean(vals)) if vals else None


def median(rows: Iterable[dict], name: str):
    vals = []
    for row in rows:
        try:
            v = float(row.get(name, ""))
            if np.isfinite(v):
                vals.append(v)
        except (ValueError, TypeError):
            pass
    return float(np.median(vals)) if vals else None


def summarize(rows: List[dict], clean: float, noisy: float) -> dict:
    result = {"total_evaluated": len(rows), "thresholds": {"clean_min": clean, "noisy_min": noisy}, "pools": {}, "cross_tab": {}, "quality_by_candidate_pool": {}, "iou_ranges": {}}
    for pool in LEVELS:
        subset = [r for r in rows if r["true_iou_pool"] == pool]
        result["pools"][pool] = {
            "count": len(subset), "percent": 100 * len(subset) / len(rows) if rows else 0,
            "true_iou_mean": average(subset, "true_iou"),
            "true_iou_median": median(subset, "true_iou"),
            "true_mae_mean": average(subset, "true_mae"),
            "profile_Smeasure_mean": average(subset, "profile_Smeasure"),
            "profile_MAE_mean": average(subset, "profile_MAE"),
            "candidate_iou_mean": average(subset, "candidate_iou_mean"),
        }
    for a in LEVELS:
        sub = [r for r in rows if r.get("candidate_iou_pool") == a]
        result["cross_tab"][a] = {b: sum(r["true_iou_pool"] == b for r in sub) for b in LEVELS}
        result["quality_by_candidate_pool"][a] = {"count":len(sub), "true_iou_mean":average(sub,"true_iou"), "true_mae_mean":average(sub,"true_mae")}
    bands = [("[0,.50)", 0, .5), ("[.50,.70)", .5, .7), ("[.70,.80)", .7, .8), ("[.80,.90)", .8, .9), ("[.90,.95)", .9, .95), ("[.95,1]", .95, 1.0000001)]
    for name, lo, hi in bands:
        result["iou_ranges"][name] = sum(lo <= r["true_iou"] < hi for r in rows)
    result["overall"] = {"true_iou_mean": average(rows,"true_iou"), "true_mae_mean":average(rows,"true_mae"), "candidate_iou_mean":average(rows,"candidate_iou_mean")}
    result["candidate_clean_but_true_reject"] = sum(r.get("candidate_iou_pool") == "Clean" and r["true_iou_pool"] == "Reject" for r in rows)
    return result


def write_csv(path: Path, fields: List[str], rows: Iterable[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not (0 < args.noisy_min < args.clean_min <= 1):
        raise ValueError("Thresholds must satisfy 0 < noisy_min < clean_min <= 1")
    with Path(args.csv).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        input_fields = list(reader.fieldnames or [])
        input_rows = list(reader)
    if "image_name" not in input_fields:
        raise ValueError("Input CSV must have image_name column")
    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    if not pred_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"Mask folders missing: pred={pred_dir} gt={gt_dir}")
    rows, errors = [], []
    for row in input_rows:
        image = row["image_name"].strip()
        stem = Path(image).stem
        pred = locate_mask(pred_dir, stem)
        gt = locate_mask(gt_dir, stem)
        if pred is None or gt is None:
            errors.append({"image_name":image,"error":"missing_pred" if pred is None else "missing_gt"})
            continue
        try:
            metrics = true_metrics(pred, gt)
        except Exception as ex:
            errors.append({"image_name":image,"error":str(ex)})
            continue
        candidate_value = row.get("candidate_iou_mean", "")
        try:
            q = float(candidate_value)
            candidate_pool = classify(q, args.clean_min, args.noisy_min) if np.isfinite(q) else "Missing"
        except (TypeError, ValueError):
            candidate_pool = "Missing"
        row.update(metrics)
        row["true_iou_pool"] = classify(float(metrics["true_iou"]), args.clean_min, args.noisy_min)
        row["candidate_iou_pool"] = candidate_pool
        row["candidate_matches_true_pool"] = int(candidate_pool == row["true_iou_pool"])
        rows.append(row)
    new_fields = ["true_iou", "true_iou_pool", "true_dice", "true_mae", "candidate_iou_pool", "candidate_matches_true_pool", "tp", "fp", "fn", "gt_pixels", "pred_pixels", "gt_area_ratio", "pred_area_ratio", "size_mismatch_resized"]
    fields = input_fields + [k for k in new_fields if k not in input_fields]
    write_csv(out / "sam1_true_iou_graded_full.csv", fields, rows)
    mini_fields = ["image_name", "true_iou", "true_iou_pool", "candidate_iou_mean", "candidate_iou_pool", "candidate_matches_true_pool", "true_mae", "profile_MAE", "profile_Smeasure", "route_type"]
    write_csv(out / "sam1_true_iou_pools.csv", mini_fields, rows)
    for pool in LEVELS:
        names = [r["image_name"] for r in rows if r["true_iou_pool"] == pool]
        (out / f"sam1_true_iou_{pool.lower()}_images.txt").write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
    write_csv(out / "missing_or_failed.csv", ["image_name", "error"], errors)
    s = summarize(rows, args.clean_min, args.noisy_min)
    s["total_input_rows"] = len(input_rows)
    s["missing_or_failed"] = len(errors)
    s["mask_pred_dir"] = str(pred_dir)
    s["mask_gt_dir"] = str(gt_dir)
    s["oracle_warning"] = "True IoU uses training GT; analysis only, never used to generate weak-supervision pools."
    (out / "summary_true_iou.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nTRUE IoU vs GT (offline reference only)")
    print(f"Input={len(input_rows)} Evaluated={len(rows)} Missing/Failed={len(errors)}")
    for pool in LEVELS:
        p = s["pools"][pool]
        print(f'{pool:<7} N={p["count"]:>5} {p["percent"]:6.2f}%  IoU={p["true_iou_mean"] if p["true_iou_mean"] is not None else float("nan"):.4f}  MAE={p["true_mae_mean"] if p["true_mae_mean"] is not None else float("nan"):.4f}')
    print("Candidate IoU pool (rows) vs TRUE IoU pool (columns)")
    print(" " * 19 + " ".join(f"{b:>8}" for b in LEVELS))
    for a in LEVELS:
        print(f"{a:<19}" + " ".join(f"{s['cross_tab'][a][b]:>8}" for b in LEVELS))
    print("Output:", out.resolve())
    if errors and args.strict:
        raise SystemExit(f"Missing/failed {len(errors)} images; check {out / 'missing_or_failed.csv'}")


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pred").mkdir(); (root / "gt").mkdir()
        csv_input = root / "input.csv"
        cases = [
            ("clean", [(1,1),(1,2),(2,1)], [(1,1),(1,2),(2,1)], "Clean"),
            ("noisy", [(1,1),(1,2),(2,1),(2,2)], [(1,1),(1,2),(2,1),(2,2),(3,3)], "Noisy"),
            ("reject", [(1,1),(1,2)], [(3,3),(3,2)], "Reject"),
        ]
        for name, pp, gg, pool in cases:
            p=np.zeros((8,8),np.uint8);g=np.zeros_like(p)
            for y,x in pp:p[y,x]=255
            for y,x in gg:g[y,x]=255
            cv2.imwrite(str(root / "pred" / (name+".png")), p)
            cv2.imwrite(str(root / "gt" / (name+".png")), g)
            actual = true_metrics(root/"pred"/(name+".png"),root/"gt"/(name+".png"))
            assert classify(actual["true_iou"],.95,.8)==pool,(name, actual)
        with csv_input.open("w",encoding="utf-8",newline="") as f:
            wr=csv.DictWriter(f,["image_name","candidate_iou_mean","profile_MAE"])
            wr.writeheader()
            for name,*_ in cases:wr.writerow({"image_name":name+".jpg","candidate_iou_mean":.99,"profile_MAE":.1})
        args=argparse.Namespace(csv=str(csv_input),pred_dir=str(root/"pred"),gt_dir=str(root/"gt"),output_dir=str(root/"out"),clean_min=.95,noisy_min=.8,strict=True)
        main(args)
        result=json.loads((root/"out"/"summary_true_iou.json").read_text())
        assert [result["pools"][q]["count"] for q in LEVELS]==[1,1,1]
        assert result["cross_tab"]["Clean"]=={"Clean":1,"Noisy":1,"Reject":1}
    print("PASS: exact GT IoU, three pools, cross-tab, file output")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv",default="./softresult/SAM1_Optimized3/soft_so_mo_metrics.csv",help="SAM1 metrics CSV containing image_name")
    p.add_argument("--pred-dir",default="./softresult/SAM1_Optimized3/02_SoftLogits_SO_MO",help="final pseudo-mask PNG folder corresponding to this CSV")
    p.add_argument("--gt-dir",default="./data/Train/GT",help="training GT masks folder (offline assessment only)")
    p.add_argument("--output-dir",default="./softresult/SAM1_TrueIoU_Grade")
    p.add_argument("--clean-min",type=float,default=.95)
    p.add_argument("--noisy-min",type=float,default=.80)
    p.add_argument("--strict",action="store_true",help="Exit nonzero if any masks are missing")
    p.add_argument("--self-test",action="store_true")
    parsed=p.parse_args()
    self_test() if parsed.self_test else main(parsed)
