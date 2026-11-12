#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM feature-spectrum sample difficulty (SUV, experimental).

Outputs:
    suv_images.csv              Image-level spectrum statistics + SUV
    suv_candidates.csv          Optional mask-conditioned spectrum statistics + SUV
    hard_images.txt             Filenames with top `hard_frac` SUV
    easy_images.txt             Filenames with bottom `easy_frac` SUV
    hard_candidates.csv         Optional top-hard candidate rows
    summary.json

The SUV score adapts the semantic-perception-scale / AUV feature-spectrum
principle of Li et al. (CVPR 2026) to frozen SAM1/SAM2 image embeddings.
It is a *relative feature-difficulty proxy*, NOT calibrated label uncertainty
or a measure of SAM mask accuracy. Here we name it SUV to distinguish this
implementation from the paper's AUV.

Defaults are aligned with your SAM2 Box-only code (run in RCLCOD root):
  checkpoint: ./sam_pseado/weights/sam2_hiera_large.pt
  model config: sam2_hiera_l.yaml
  RGB images:  ./data/Train/Imgs
  pseudo mask: ./softresult/SAM2_BoxOnly/{01_Box_Base,02_Box_Crop,03_Box_Crop_Flip}
  SUV output:  ./softresult/SAM2_BoxOnly/SUV

Run first 100 images (no extra arguments needed):
  python sam_suv_difficulty.py --limit 100
Run all images:
  python sam_suv_difficulty.py
python sam_mask/sam_suv_difficulty.py --no-stage-masks
If stage folders exist, read their masks for mask-conditioned SUV. These are
THREE ABLATION-STAGE outputs, NOT the three multimask candidates from one SAM
predict() call. Use --no-stage-masks for whole-image SUV only.

Optional precomputed SAM2 embeddings (skip inference):
  python sam_suv_difficulty.py --feature-dir ./embeddings

Self-test (does not require SAM or weights):
  python sam_suv_difficulty.py --self-test
"""

import argparse
import csv
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def spectral_entropy(features_chw, spatial_mask=None, max_points=4096,
                     center=False, seed=42, min_points=8):
    """Compute normalized entropy of singular-value squared energies.

    Parameters
    ----------
    features_chw : (C,H,W) float array, one image embedding
    spatial_mask : optional (H,W) boolean array selecting tokens
    max_points : optional subsampling cap for performance (0 = all)
    center : whether to subtract channel means. False matches the literal
             uncentered feature SVD of Eq. (3)-(5) of the paper.
    Returns: dict or None (if no valid tokens/degenerate spectrum).
    """
    feat = np.asarray(features_chw, dtype=np.float32)
    if feat.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got shape={feat.shape}")
    c, h, w = feat.shape
    mat = feat.reshape(c, -1)
    if spatial_mask is not None:
        mask = np.asarray(spatial_mask, dtype=bool)
        if mask.shape != (h, w):
            raise ValueError(f"Mask shape {mask.shape} != feature shape {(h,w)}")
        mat = mat[:, mask.reshape(-1)]
    n_available = mat.shape[1]
    if n_available < min_points:
        return None
    if max_points > 0 and n_available > max_points:
        # Deterministic uniform random subset, avoiding excessive spatial
        # bias from selecting only early rows/columns.
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_available, size=max_points, replace=False)
        mat = mat[:, idx]
    if not np.isfinite(mat).all():
        return None

    mat = np.asarray(mat, dtype=np.float64, order="C")
    if center:
        mat -= mat.mean(axis=1, keepdims=True)

    # Squared singular values = eigenvalues of X @ X.T, avoids full SVD.
    # Prefer small Gram matrix (min(C, N) dimension) for small regions.
    if mat.shape[0] <= mat.shape[1]:
        gram = mat @ mat.T
    else:
        gram = mat.T @ mat
    eig = np.linalg.eigvalsh(gram)
    eig = np.clip(eig, 0, None)
    total = float(eig.sum())
    if total <= 1e-15:
        return None
    p = eig / total
    p = p[p > 1e-12]
    rank_cap = min(mat.shape[0], mat.shape[1])
    entropy = float(-np.sum(p * np.log(p)) / math.log(rank_cap)) if rank_cap > 1 else 0.0
    entropy = float(np.clip(entropy, 0.0, 1.0))
    # Effective rank is useful to spot trivial / collapsed features.
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    energy_top1 = float(eig[-1] / total)
    return {
        "S": entropy,
        "energy_top1": energy_top1,
        "effective_rank": eff_rank,
        "num_tokens": n_available,
        "num_tokens_used": int(mat.shape[1]),
        "channels": int(c),
    }


def valid_image_tokens(orig_hw, feature_hw, sam_version, sam1_input_size):
    """SAM1 pads resized image bottom/right; exclude padding from global stats."""
    h, w = orig_hw
    fh, fw = feature_hw
    if sam_version == "sam2":
        return np.ones((fh, fw), dtype=bool)
    s = sam1_input_size / max(h, w)
    new_h = int(h * s + 0.5)
    new_w = int(w * s + 0.5)
    # Transform a rectangular valid region using bilinear area occupancy.
    # Using center-of-token grid approximates which SAM1 embedding tokens map
    # to original resized pixels; robust for strongly non-square images.
    yy = (np.arange(fh) + 0.5) * sam1_input_size / fh
    xx = (np.arange(fw) + 0.5) * sam1_input_size / fw
    return (yy[:, None] < new_h) & (xx[None, :] < new_w)


def resized_mask_to_tokens(mask_np, feature_hw, sam_version, sam1_input_size):
    """Transform original-resolution binary mask into SAM embedding coords.

    SAM1: resize longest side to encoder size, then bottom/right padding.
    SAM2: resize directly to square image resolution. Both followed by
    nearest resize to feature grid. This is an approximation of the spatial
    support seen by embedding tokens, especially for tiny objects.
    """
    m = np.asarray(mask_np, dtype=bool)
    h, w = m.shape
    if sam_version == "sam1":
        scale = sam1_input_size / max(h, w)
        new_h = int(h * scale + 0.5)
        new_w = int(w * scale + 0.5)
        resized = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(
            (new_w, new_h), resample=Image.Resampling.NEAREST)) > 0
        padded = np.zeros((sam1_input_size, sam1_input_size), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        m_pil = Image.fromarray(padded * 255)
    else:
        m_pil = Image.fromarray((m * 255).astype(np.uint8))
    fh, fw = feature_hw
    return np.asarray(m_pil.resize((fw, fh), resample=Image.Resampling.NEAREST)) > 0


def load_predictor(args):
    import torch
    if args.sam_root:
        sys.path.insert(0, str(Path(args.sam_root).resolve()))
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda specified, but torch.cuda.is_available() is False")
    if args.sam_version == "sam1":
        from segment_anything import sam_model_registry, SamPredictor
        if args.sam1_model_type not in sam_model_registry:
            raise ValueError(f"Unknown SAM1 model type {args.sam1_model_type}")
        model = sam_model_registry[args.sam1_model_type](checkpoint=args.sam_checkpoint)
        model.to(device=device).eval()
        predictor = SamPredictor(model)
        input_size = int(model.image_encoder.img_size)
    else:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        if not args.sam2_config:
            raise ValueError("SAM2 requires --sam2-config (Hydra package config path)")
        model = build_sam2(args.sam2_config, args.sam_checkpoint, device=device)
        model.eval()
        predictor = SAM2ImagePredictor(model)
        input_size = int(model.image_size)
    return predictor, input_size


def get_embedding(predictor, rgb):
    import torch
    with torch.inference_mode():
        predictor.set_image(rgb)
        feat = predictor.get_image_embedding()
        return feat[0].detach().float().cpu().numpy()


def load_precomputed(feature_dir, img_path):
    import torch
    root = Path(feature_dir)
    for ext in (".npy", ".npz", ".pt", ".pth"):
        fp = root / (img_path.stem + ext)
        if not fp.is_file():
            continue
        if ext == ".npy":
            x = np.load(fp, allow_pickle=False)
        elif ext == ".npz":
            with np.load(fp, allow_pickle=False) as z:
                key = "image_embed" if "image_embed" in z else list(z.keys())[0]
                x = z[key]
        else:
            # Only use .pt files produced by you; PyTorch must load tensor data.
            x = torch.load(fp, map_location="cpu", weights_only=True)
            if isinstance(x, dict):
                x = x.get("image_embed", x.get("features", None))
                if x is None:
                    raise ValueError(f"Missing image_embed/features in {fp}")
            if isinstance(x, torch.Tensor):
                x = x.detach().float().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"Expected embedding (C,H,W), got {x.shape} in {fp}")
        return x.astype(np.float32, copy=False)
    raise FileNotFoundError(f"No embedding with basename {img_path.stem} in {root}")


def candidate_file(mask_dir, image_path):
    for ext in MASK_EXTS:
        fp = Path(mask_dir) / (image_path.stem + ext)
        if fp.exists():
            return fp
    return None


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_scores(rows, normalization, score_key="S"):
    """Normalize one population across all samples/candidates, never per image."""
    good = [r for r in rows if r.get(score_key) is not None and r[score_key] >= 0]
    if not good:
        return {"n_valid": 0}
    raw = np.asarray([r[score_key] for r in good], dtype=float)
    log_s = np.log(np.maximum(raw, 1e-12))
    if normalization == "minmax":
        lo, hi = float(log_s.min()), float(log_s.max())
    else:
        lo, hi = (float(x) for x in np.percentile(log_s, [5, 95]))
    if hi - lo < 1e-12:
        for r in good:
            r["SUV"] = 0.5
    else:
        for r, z in zip(good, log_s):
            r["SUV"] = float(np.clip(1.0 - (z - lo) / (hi - lo), 0, 1))
    good_ids = {id(r) for r in good}
    for r in rows:
        if id(r) not in good_ids:
            r["SUV"] = None
    return {"n_valid": len(good), "log_S_lo": lo, "log_S_hi": hi,
            "normalization": normalization,
            "mean_SUV": float(np.mean([r["SUV"] for r in good]))}


def self_test():
    rng = np.random.default_rng(7)
    # An almost rank-1 feature field vs a full-rank feature field.
    v = rng.normal(size=(1, 8, 8))
    rank1 = np.concatenate([v * (i + 1) for i in range(8)], axis=0)
    diverse = rng.normal(size=(8, 8, 8))
    a = spectral_entropy(rank1, max_points=0)
    b = spectral_entropy(diverse, max_points=0)
    print(f"[Self-test] low-rank S={a['S']:.6f} | diverse S={b['S']:.6f}")
    assert a["S"] < b["S"]
    valid = valid_image_tokens((300, 600), (64, 64), "sam1", 1024)
    assert valid.shape == (64, 64) and 0 < valid.sum() < valid.size
    m = np.zeros((300, 600), bool)
    m[100:180, 100:300] = True
    mapped = resized_mask_to_tokens(m, (64, 64), "sam1", 1024)
    assert mapped.shape == (64, 64) and mapped.any()
    out = [{"S": a["S"]}, {"S": b["S"]}]
    normalize_scores(out, "minmax")
    assert out[0]["SUV"] > out[1]["SUV"]
    print("[Self-test] PASS: more concentrated spectrum -> larger SUV")


def parse_args():
    p = argparse.ArgumentParser(description="Experimental SAM feature-spectrum SUV")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--image-dir", type=str, default="./data/Train/Imgs", help="Original RGB images")
    p.add_argument("--output-dir", type=str, default="./softresult/SAM2_BoxOnly/SUV")
    p.add_argument("--sam-version", choices=["sam1", "sam2"], default="sam2")
    p.add_argument("--sam-checkpoint", type=str, default="./sam_pseado/weights/sam2_hiera_large.pt")
    p.add_argument("--sam-root", type=str, default=None,
                   help="Optional local segment-anything or sam2 package root")
    p.add_argument("--sam1-model-type", choices=["vit_b", "vit_l", "vit_h"],
                   default="vit_h")
    p.add_argument("--sam2-config", type=str,
                   default="sam2_hiera_l.yaml")
    p.add_argument("--feature-dir", type=str, default=None,
                   help="Read precomputed image embeddings from .npy/.npz/.pt")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--mask-dirs", nargs="*", default=None,
                   help="Override default mask folders; specify paths for actual SAM candidate masks")
    p.add_argument("--stage-root", type=str, default="./softresult/SAM2_BoxOnly",
                   help="Root of original 01_Box_Base/02_Box_Crop/03_Box_Crop_Flip masks")
    p.add_argument("--no-stage-masks", action="store_true",
                   help="Compute whole-image SUV only; ignore saved Box-only stage masks")
    p.add_argument("--max-points", type=int, default=4096,
                   help="Feature positions sampled per spectrum; 0 uses all")
    p.add_argument("--min-points", type=int, default=8,
                   help="Minimum available feature positions for score")
    p.add_argument("--center-features", action="store_true",
                   help="Optionally center features (ablation, changes metric)")
    p.add_argument("--normalization", choices=["minmax", "p5p95"],
                   default="minmax")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="Process first N images; 0 = all")
    p.add_argument("--hard-frac", type=float, default=0.10)
    p.add_argument("--easy-frac", type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    # By default read the original Box-only ablation masks only when those
    # folders already exist. This does NOT regenerate masks or use GT.
    if args.mask_dirs is None:
        stage_names = ("01_Box_Base", "02_Box_Crop", "03_Box_Crop_Flip")
        args.mask_dirs = [] if args.no_stage_masks else [
            str(Path(args.stage_root) / name) for name in stage_names
            if (Path(args.stage_root) / name).is_dir()
        ]
    if args.mask_dirs:
        print("[SUV] mask stages:", ", ".join(args.mask_dirs))
    else:
        print("[SUV] whole-image mode (no candidate mask folders)")
    if not args.feature_dir and not args.sam_checkpoint:
        raise SystemExit("Need --sam-checkpoint, or --feature-dir to reuse features")
    if args.max_points < 0 or args.min_points < 2:
        raise SystemExit("Require --max-points >= 0 and --min-points >= 2")
    if not 0 <= args.hard_frac <= 1 or not 0 <= args.easy_frac <= 1:
        raise SystemExit("--hard-frac and --easy-frac must be in [0,1]")

    image_dir = Path(args.image_dir)
    images = sorted(fp for fp in image_dir.iterdir()
                    if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS)
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"No input images found in {image_dir}")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    predictor = None
    sam1_input_size = 1024
    if not args.feature_dir:
        predictor, sam1_input_size = load_predictor(args)
    elif args.sam_version == "sam1":
        # SAM1's default image encoder size; change here for custom models.
        sam1_input_size = 1024

    image_rows, candidate_rows, failures = [], [], []
    for i, ipath in enumerate(images, 1):
        try:
            rgb = np.array(Image.open(ipath).convert("RGB"), dtype=np.uint8)
            h, w = rgb.shape[:2]
            f = (load_precomputed(args.feature_dir, ipath) if args.feature_dir
                 else get_embedding(predictor, rgb))
            if f.ndim != 3:
                raise RuntimeError(f"Unexpected SAM embedding {f.shape}")
            valid = valid_image_tokens((h, w), f.shape[-2:],
                                       args.sam_version, sam1_input_size)
            stats = spectral_entropy(f, valid, args.max_points,
                                     args.center_features, args.seed,
                                     args.min_points)
            image_rows.append({"image": ipath.name, "height": h, "width": w,
                               **(stats or {"S": None})})
            for mdir in args.mask_dirs:
                mp = candidate_file(mdir, ipath)
                cname = Path(mdir).name
                row = {"image": ipath.name, "candidate": cname,
                       "mask_path": str(mp) if mp else "", "height": h, "width": w}
                if mp is None:
                    row.update(S=None, status="missing_mask")
                else:
                    mask = np.asarray(Image.open(mp).convert("L")) > 127
                    if mask.shape != (h, w):
                        # True binary mask at original size is expected.
                        mask = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(
                            (w,h), Image.Resampling.NEAREST)) > 0
                    roi = resized_mask_to_tokens(mask, f.shape[-2:],
                                                 args.sam_version, sam1_input_size)
                    # Explicitly ignore image padding, which is uninformative.
                    roi = roi & valid
                    roi_stats = spectral_entropy(f, roi, args.max_points,
                                                 args.center_features, args.seed,
                                                 args.min_points)
                    row.update(roi_stats or {"S": None})
                    row["status"] = "ok" if roi_stats is not None else "insufficient_tokens"
                candidate_rows.append(row)
            if i % 20 == 0 or i == len(images):
                print(f"[{i}/{len(images)}] {ipath.name} | S={None if stats is None else round(stats['S'], 5)}", flush=True)
        except Exception as exc:
            failures.append({"image": ipath.name, "error": str(exc)})
            print(f"[WARNING] {ipath.name}: {exc}", file=sys.stderr)

    s_img = normalize_scores(image_rows, args.normalization)
    s_cand = normalize_scores(candidate_rows, args.normalization) if candidate_rows else {}
    base_cols = ["image", "height", "width", "S", "SUV", "energy_top1",
                 "effective_rank", "channels", "num_tokens", "num_tokens_used"]
    write_csv(outdir / "suv_images.csv", image_rows, base_cols)
    if candidate_rows:
        write_csv(outdir / "suv_candidates.csv", candidate_rows,
                  ["image", "candidate", "mask_path", "status"] + base_cols[1:])

    valid_rows = sorted([r for r in image_rows if r.get("SUV") is not None],
                        key=lambda r: r["SUV"], reverse=True)
    n = len(valid_rows)
    n_hard = int(math.ceil(n * args.hard_frac))
    n_easy = int(math.ceil(n * args.easy_frac))
    hard = valid_rows[:n_hard]
    easy = valid_rows[-n_easy:] if n_easy else []
    (outdir / "hard_images.txt").write_text("\n".join(r["image"] for r in hard) + ("\n" if hard else ""), encoding="utf-8")
    (outdir / "easy_images.txt").write_text("\n".join(r["image"] for r in easy) + ("\n" if easy else ""), encoding="utf-8")

    if candidate_rows:
        valid_cands = sorted([r for r in candidate_rows if r.get("SUV") is not None],
                             key=lambda r: r["SUV"], reverse=True)
        nc = int(math.ceil(len(valid_cands) * args.hard_frac))
        write_csv(outdir / "hard_candidates.csv", valid_cands[:nc],
                  ["image", "candidate", "mask_path", "status"] + base_cols[1:])
    summary = {
        "description": "SUV=1-minmax(log S), S=normalized singular-value energy entropy; high SUV indicates concentrated feature energy and is a relative difficulty proxy, NOT label accuracy.",
        "sam_version": args.sam_version, "number_images": len(images),
        "success_images": len(image_rows), "failure_count": len(failures),
        "image_stats": s_img, "candidate_stats": s_cand,
        "hard_count": len(hard), "easy_count": len(easy),
        "sam1_excludes_padded_tokens": True,
        "spectral_uses_centering": args.center_features,
        "max_points": args.max_points,
        "failures": failures,
        "arguments": vars(args),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[DONE] saved:", outdir.resolve())
    print("  suv_images.csv; hard_images.txt; easy_images.txt; summary.json")
    if candidate_rows:
        print("  suv_candidates.csv; hard_candidates.csv")
    print("  Important: SUV is relative feature-spectrum difficulty, NOT pseudo-label reliability.")


if __name__ == "__main__":
    main()
