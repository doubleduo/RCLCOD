#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCLCOD: one-time GT visual-difficulty cache + repeatable three-pool reports.

1) First run (RGB and REAL training GT):
   python compute_iccl_gt_difficulty.py \
       --rgb-dir ./data/Train/Imgs --gt-dir ./data/Train/GT \
       --pool-dir ./data/pseudo_pool/sam2refine

2) Every subsequent run, including after editing the three TXT lists:
   python compute_iccl_gt_difficulty.py --summary-only

The second step reads ONLY the cached CSV and high/middle/unvalue TXT lists.
It does NOT read RGB, GT or run Canny / histograms again.

The three metrics reproduce the *definitions* in ICCL Table IV, NOT the
paper's unspecified preprocessing parameters. GT statistics are for OFFLINE
ANALYSIS ONLY; they must not be used to select weakly supervised training data.

Output in --out-dir (default ./iccl_gt_stats):
  per_image_metrics.csv   : ONE-TIME cache (one row for every RGB image)
  metric_settings.json    : parameters/path for the cached computation
  pool_summary.csv        : refreshed every run (ALL + high/middle/unvalue)
  pool_per_image.csv      : refreshed every run (image -> pool -> 3 metrics)
  pool_issues.csv         : refreshed every run (missing/duplicate/overlap)

Requirements: pip install numpy opencv-python
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GT_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
#POOLS = ("high", "middle", "unvalue")
POOLS = ("clean", "noisy", "unvalue")
METRICS = ("S_obj", "D_bound", "Sim_FB")
CACHE_FIELDS = ["image", "gt", "S_obj", "D_bound", "Sim_FB", "status", "error"]


def parse_args():
    p = argparse.ArgumentParser(description="Compute ICCL-style GT difficulty once; analyse RCLCOD three pools repeatedly")
    p.add_argument("--rgb-dir", type=Path, help="RGB training folder, required only when building cache")
    p.add_argument("--gt-dir", type=Path, help="REAL GT folder, NOT SAM2 pseudolabels; only for cache build")
    p.add_argument("--pool-dir", type=Path, default=Path("./data/pseudo_pool/shape"),
                   help="Folder with high.txt, middle.txt, unvalue.txt")
    p.add_argument("--out-dir", type=Path, default=Path("./iccl_gt_stats"))
    p.add_argument("--summary-only", action="store_true", help="Never read RGB/GT, only analyse the existing cache and pools")
    p.add_argument("--force", action="store_true", help="Explicitly rebuild cached metrics from RGB and GT")
    p.add_argument("--resize", type=int, default=384, help="Analysis resolution NxN; 0 = original RGB resolution (default: 384)")
    p.add_argument("--gt-threshold", type=int, default=127)
    p.add_argument("--canny-low", type=int, default=100)
    p.add_argument("--canny-high", type=int, default=200)
    p.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel before Canny (0=off, odd integer>=3)")
    p.add_argument("--hist-bins", type=int, default=8, help="RGB JOINT histogram bins per channel")
    p.add_argument("--expected-total", type=int, default=4040, help="Expected training RGB count; 0=disable warning")
    return p.parse_args()


def sample_key(name: str) -> str:
    """Normalize txt entries such as name.jpg, path/name.png, or 'name.jpg 0'."""
    s = str(name).strip().lstrip("\ufeff")
    if not s or s.startswith("#"):
        return ""
    first = s.split()[0].split(",", 1)[0].strip('"\'')
    base = first.replace("\\", "/").rsplit("/", 1)[-1]
    return Path(base).stem.casefold()


def build_index(folder: Path, allowed: set[str]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    result = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        key = sample_key(path.name)
        if key in result:
            raise ValueError(f"Duplicate filename stems: {result[key]} <-> {path}")
        result[key] = path
    return result


def read_cv(path: Path, flag: int) -> np.ndarray:
    # np.fromfile + imdecode supports Windows paths containing Chinese characters.
    data = np.fromfile(str(path), dtype=np.uint8)
    value = cv2.imdecode(data, flag)
    if value is None:
        raise ValueError(f"OpenCV cannot read: {path}")
    return value


def compute_one(bgr: np.ndarray, gt_gray: np.ndarray, args) -> tuple[float, float, float]:
    if gt_gray.shape != bgr.shape[:2]:
        gt_gray = cv2.resize(gt_gray, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    if args.resize:
        bgr = cv2.resize(bgr, (args.resize, args.resize), interpolation=cv2.INTER_LINEAR)
        gt_gray = cv2.resize(gt_gray, (args.resize, args.resize), interpolation=cv2.INTER_NEAREST)

    if gt_gray.max() <= 1:
        fg = gt_gray > 0
    else:
        fg = gt_gray > args.gt_threshold
    if not fg.any() or fg.all():
        raise ValueError("GT is empty or entirely foreground")

    s_obj = float(fg.mean())
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    gt_mask = (fg.astype(np.uint8) * 255)
    boundary = cv2.morphologyEx(gt_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if not boundary.any():
        raise ValueError("GT boundary is empty")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if args.blur:
        gray = cv2.GaussianBlur(gray, (args.blur, args.blur), 0)
    canny = cv2.Canny(gray, args.canny_low, args.canny_high)
    if canny.any():
        dist = cv2.distanceTransform(255 - canny, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        d_bound = float(dist[boundary].mean())
    else:
        d_bound = math.nan  # No RGB edges: undefined, do not make it a misleading 0.

    channels = [0, 1, 2]
    bins = [args.hist_bins] * 3
    ranges = [0, 256] * 3
    fg_hist = cv2.calcHist([rgb], channels, gt_mask, bins, ranges).reshape(-1)
    bg_hist = cv2.calcHist([rgb], channels, ((~fg).astype(np.uint8) * 255), bins, ranges).reshape(-1)
    fg_hist /= max(float(fg_hist.sum()), 1)
    bg_hist /= max(float(bg_hist.sum()), 1)
    sim_fb = float(np.minimum(fg_hist, bg_hist).sum())
    return s_obj, d_bound, sim_fb


def write_csv(path: Path, fields: list[str], records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: interrupted initial processing never looks like a completed cache.
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    os.replace(temp, path)


def build_cache(args, cache_file: Path, meta_file: Path):
    if not args.rgb_dir or not args.gt_dir:
        raise ValueError("First run (or --force) requires --rgb-dir and --gt-dir with real GT")
    imgs = build_index(args.rgb_dir, IMAGE_EXTS)
    gts = build_index(args.gt_dir, GT_EXTS)
    if not imgs:
        raise ValueError(f"No RGB images in {args.rgb_dir}")
    if args.expected_total and len(imgs) != args.expected_total:
        print(f"[WARN] Found {len(imgs)} RGB images, expected {args.expected_total}. Check RGB path.")
    print(f"[COMPUTE] {len(imgs)} RGB images / {len(gts)} GT masks; build metrics once ...")

    records = []
    for i, (key, image_path) in enumerate(imgs.items(), 1):
        gt_path = gts.get(key)
        row = {"image": image_path.name, "gt": gt_path.name if gt_path else "",
               "S_obj": "", "D_bound": "", "Sim_FB": "", "status": "", "error": ""}
        if gt_path is None:
            row.update(status="missing_gt", error="No GT with same filename stem")
        else:
            try:
                rgb = read_cv(image_path, cv2.IMREAD_COLOR)
                gt = read_cv(gt_path, cv2.IMREAD_GRAYSCALE)
                s, d, c = compute_one(rgb, gt, args)
                row.update(S_obj=f"{s:.9f}", D_bound=f"{d:.9f}" if math.isfinite(d) else "",
                           Sim_FB=f"{c:.9f}", status="ok" if math.isfinite(d) else "no_canny_edges")
            except (ValueError, OSError, cv2.error) as err:
                row.update(status="error", error=str(err))
        records.append(row)
        if i % 500 == 0 or i == len(imgs):
            print(f"  {i}/{len(imgs)} complete")

    write_csv(cache_file, CACHE_FIELDS, records)
    meta = {"rgb_dir": str(args.rgb_dir.resolve()), "gt_dir": str(args.gt_dir.resolve()),
            "image_count": len(imgs), "gt_count": len(gts), "resize": args.resize,
            "gt_threshold": args.gt_threshold, "canny_low": args.canny_low,
            "canny_high": args.canny_high, "blur": args.blur, "hist_bins": args.hist_bins,
            "metrics": list(METRICS), "purpose": "GT-only posthoc analysis; not weak-supervision training"}
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[CACHE SAVED] {cache_file} ({len(records)} rows; never recomputed unless --force)")
    return records


def load_cache(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Cache missing: {path}. First run needs --rgb-dir and --gt-dir")
    records = []
    seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(CACHE_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Invalid/incompatible cache columns: {reader.fieldnames}")
        for r in reader:
            key = sample_key(r["image"])
            if not key or key in seen:
                raise ValueError(f"Duplicate/empty key in cached CSV: {r['image']!r}")
            seen.add(key)
            records.append(r)
    print(f"[CACHE REUSED] {len(records)} images from {path}, no RGB/GT calculations")
    return records


def load_pool_list(path: Path) -> tuple[set[str], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing pool file: {path}")
    names = set()
    problems = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, 1):
            key = sample_key(raw)
            if not key:
                continue
            if key in names:
                problems.append(f"duplicate entry at line {line_no}: {key}")
            names.add(key)
    return names, problems


def valid_number(value) -> float:
    try:
        num = float(value)
        return num if math.isfinite(num) else math.nan
    except (ValueError, TypeError):
        return math.nan


def summarise(group: str, keys: set[str], cache: dict[str, dict]) -> dict:
    available = [cache[k] for k in keys if k in cache]
    row = {"pool": group, "listed": len(keys), "matched_cache": len(available),
           "missing_cache": len(keys) - len(available),
           "complete_three_metrics": sum(all(math.isfinite(valid_number(r[m])) for m in METRICS) for r in available)}
    for metric in METRICS:
        arr = np.asarray([valid_number(r[metric]) for r in available], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        row[f"{metric}_valid"] = int(len(arr))
        row[f"{metric}_mean"] = f"{np.mean(arr):.6f}" if len(arr) else ""
        row[f"{metric}_std"] = f"{np.std(arr, ddof=1):.6f}" if len(arr) > 1 else ("0.000000" if len(arr) else "")
    return row


def make_reports(args, records: list[dict]):
    pool_files = {group: args.pool_dir / f"{group}.txt" for group in POOLS}
    pool_lists = {}
    issues = []
    for group, path in pool_files.items():
        entries, problems = load_pool_list(path)
        pool_lists[group] = entries
        for msg in problems:
            issues.append({"pool": group, "image": "", "reason": msg})

    cache = {sample_key(r["image"]): r for r in records}
    for group, names in pool_lists.items():
        for key in names - cache.keys():
            issues.append({"pool": group, "image": key, "reason": "not found in per_image_metrics.csv"})
    memberships = defaultdict(list)
    for group, names in pool_lists.items():
        for key in names:
            memberships[key].append(group)
    for key, groups in memberships.items():
        if len(groups) > 1:
            issues.append({"pool": "/".join(groups), "image": key, "reason": "appears in multiple pools"})

    for key, row in cache.items():
        if key not in memberships:
            issues.append({"pool": "unassigned", "image": row["image"], "reason": "not listed in any pool"})
        if row["status"] != "ok":
            issues.append({"pool": "/".join(memberships.get(key, ["unassigned"])),
                           "image": row["image"], "reason": f"metric status={row['status']}; {row['error']}"})

    summary = [summarise("ALL", set(cache.keys()), cache)]
    for group in POOLS:
        summary.append(summarise(group, pool_lists[group], cache))
    fields = ["pool", "listed", "matched_cache", "missing_cache", "complete_three_metrics"]
    for metric in METRICS:
        fields += [f"{metric}_valid", f"{metric}_mean", f"{metric}_std"]
    write_csv(args.out_dir / "pool_summary.csv", fields, summary)

    details = []
    for group in POOLS:
        for key in sorted(pool_lists[group]):
            r = cache.get(key)
            details.append({"pool": group, "image": r["image"] if r else key,
                            "S_obj": r["S_obj"] if r else "", "D_bound": r["D_bound"] if r else "",
                            "Sim_FB": r["Sim_FB"] if r else "", "status": r["status"] if r else "missing_cache"})
    write_csv(args.out_dir / "pool_per_image.csv", ["pool", "image", *METRICS, "status"], details)
    write_csv(args.out_dir / "pool_issues.csv", ["pool", "image", "reason"], issues)

    print("\n" + "=" * 93)
    print(f"{'Pool':<12} {'Found/Listed':<17} {'S_obj mean±std':<22} {'D_bound mean±std':<23} {'Sim_FB mean±std'}")
    print("-" * 93)
    for s in summary:
        def m(key):
            a = s[f"{key}_mean"]
            b = s[f"{key}_std"]
            return f"{a}±{b}" if a else "N/A"
        print(f"{s['pool']:<12} {str(s['matched_cache'])+'/'+str(s['listed']):<17} {m('S_obj'):<22} {m('D_bound'):<23} {m('Sim_FB')}")
    print("=" * 93)
    print(f"[REPORT] {args.out_dir / 'pool_summary.csv'}")
    print(f"[DETAIL] {args.out_dir / 'pool_per_image.csv'}")
    print(f"[CHECK] {args.out_dir / 'pool_issues.csv'} ({len(issues)} issues, including invalid metrics)")
    if any(s["missing_cache"] or s["complete_three_metrics"] != s["matched_cache"] for s in summary):
        print("[WARN] Not every sample has 3 valid metrics: check pool_issues.csv")
    unique_pooled = len(memberships)
    print(f"[COUNT] unique pooled names = {unique_pooled}, cache rows = {len(cache)}; "
          f"sum 3 pools = {sum(len(x) for x in pool_lists.values())}")


def main():
    args = parse_args()
    if args.force and args.summary_only:
        raise ValueError("--force and --summary-only cannot be used together")
    if args.resize < 0 or not 1 <= args.hist_bins <= 32 or args.expected_total < 0:
        raise ValueError("Invalid --resize, --hist-bins or --expected-total")
    if args.blur != 0 and (args.blur < 3 or args.blur % 2 != 1):
        raise ValueError("--blur must be 0 or an odd integer >=3")
    if not 0 <= args.gt_threshold <= 255 or not 0 <= args.canny_low < args.canny_high <= 255:
        raise ValueError("Invalid GT or Canny threshold")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cached = args.out_dir / "per_image_metrics.csv"
    meta = args.out_dir / "metric_settings.json"

    if args.force or (not args.summary_only and not cached.exists()):
        rows = build_cache(args, cached, meta)
    else:
        rows = load_cache(cached)
        if meta.exists():
            conf = json.loads(meta.read_text(encoding="utf-8"))
            print("[PARAMETERS] Cached settings:",
                  "resize=", conf.get("resize"), "Canny=", (conf.get("canny_low"), conf.get("canny_high")),
                  "hist-bins=", conf.get("hist_bins"))
        print("To rebuild the GT metrics deliberately, use --force with --rgb-dir and --gt-dir.")
    if args.expected_total and len(rows) != args.expected_total:
        print(f"[WARN] Cache has {len(rows)} rows, not expected {args.expected_total}.")

    make_reports(args, rows)
    print("Note: GT-derived values are for post-hoc analysis, NEVER as weak-supervised training inputs.")


if __name__ == "__main__":
    main()
