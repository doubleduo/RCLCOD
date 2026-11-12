#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-case exporter for SAM1 refinement (mixed ordinary/small/off-center/multi).

Main features:
1) Select a balanced mix of ordinary, small-target, off-center, and multi-object cases.
2) Draw and save annotations separately on RGB:
   - boxes only
   - positive points only (derived from base mask)
   - negative points only (outside all boxes)
   - optional combined views
3) Export first-step(base stage) logits / heatmaps.
4) Keep original refinement visualizations and candidate diagnostics.

GT is used only for ranking / correction visualization, never as a SAM prompt.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def find(folder, stem):
    for ext in EXT:
        for e in (ext, ext.upper()):
            p = Path(folder) / (stem + e)
            if p.exists():
                return p
    return None


def save(path, im):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), im):
        raise OSError(f'Failed to save {path}')


def mask(path, hw):
    a = cv2.imread(str(path), 0)
    if a is None:
        raise RuntimeError(f'Unreadable mask {path}')
    if a.shape != hw:
        a = cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return a > 127


def jet(prob):
    return cv2.applyColorMap(np.uint8(np.clip(prob, 0, 1) * 255), cv2.COLORMAP_JET)


def fit(prob, hw):
    return cv2.resize(np.asarray(prob, np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)


def file_stems(folder):
    if not Path(folder).exists():
        return set()
    return {f.stem for f in Path(folder).glob('*.png')}


def locate_box_annotation(folder, stem):
    folder = Path(folder)
    if not folder.is_dir():
        return None
    for ext in ('.json', '.txt', '.JSON', '.TXT'):
        candidate = folder / (stem + ext)
        if candidate.is_file():
            return candidate
    return next((p for p in folder.iterdir()
                 if p.is_file() and p.stem.lower() == stem.lower()
                 and p.suffix.lower() in ('.json', '.txt')), None)


def load_annotation_boxes(path, w, h, box_format='auto'):
    if path is None:
        return []
    path = Path(path)
    raw_text = path.read_text(encoding='utf-8-sig')
    raw_boxes = []

    def numeric4(v):
        if isinstance(v, (list, tuple)) and len(v) >= 4:
            try:
                return [float(x) for x in v[:4]]
            except (TypeError, ValueError):
                return None
        return None

    def polygon_points(v):
        if not isinstance(v, (list, tuple)) or not v:
            return None
        if len(v) >= 2 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in v):
            try:
                pts = np.array([[float(p[0]), float(p[1])] for p in v])
                return [float(pts[:, 0].min()), float(pts[:, 1].min()),
                        float(pts[:, 0].max()), float(pts[:, 1].max())]
            except (TypeError, ValueError):
                return None
        return None

    def collect(obj, hint='xyxy'):
        if isinstance(obj, dict):
            fmt = str(obj.get('bbox_format', obj.get('box_format', obj.get('format', hint)))).lower()
            for keys, tag in [(['x1', 'y1', 'x2', 'y2'], 'xyxy'),
                              (['xmin', 'ymin', 'xmax', 'ymax'], 'xyxy'),
                              (['left', 'top', 'right', 'bottom'], 'xyxy'),
                              (['x', 'y', 'width', 'height'], 'xywh'),
                              (['cx', 'cy', 'w', 'h'], 'cxcywh')]:
                if all(k in obj for k in keys):
                    raw_boxes.append(([obj[k] for k in keys], tag))
                    return
            for key in ('bbox', 'box', 'bndbox', 'rect', 'rectangle', 'bounding_box'):
                if key in obj:
                    value = obj[key]
                    if isinstance(value, dict):
                        collect(value, fmt)
                    else:
                        coords = numeric4(value)
                        if coords is None:
                            coords = polygon_points(value)
                        if coords is not None:
                            raw_boxes.append((coords, 'xywh' if key == 'bbox' and
                                              (obj.get('bbox_format') == 'xywh' or obj.get('format') == 'coco') else fmt))
                        else:
                            collect(value, fmt)
                    return
            if 'points' in obj:
                coords = polygon_points(obj['points'])
                if coords is not None:
                    raw_boxes.append((coords, 'xyxy'))
                    return
            for _, value in obj.items():
                if isinstance(value, (dict, list, tuple)):
                    collect(value, fmt)
        elif isinstance(obj, (list, tuple)):
            coords = numeric4(obj)
            if coords is not None:
                raw_boxes.append((coords, hint))
            else:
                coords = polygon_points(obj)
                if coords is not None:
                    raw_boxes.append((coords, 'xyxy'))
                else:
                    for item in obj:
                        collect(item, hint)

    if path.suffix.lower() == '.txt':
        for line in raw_text.splitlines():
            parts = line.replace(',', ' ').split()
            if not parts or parts[0].startswith('#'):
                continue
            try:
                nums = [float(x) for x in parts]
            except ValueError:
                continue
            if len(nums) == 5:
                raw_boxes.append((nums[1:5], 'cxcywh'))
            elif len(nums) >= 4:
                raw_boxes.append((nums[:4], 'xyxy'))
    else:
        try:
            collect(json.loads(raw_text))
        except json.JSONDecodeError as e:
            raise ValueError(f'Box label is not valid JSON: {path}: {e}') from e

    result = []
    for coords, fmt in raw_boxes:
        try:
            v = np.array(coords, dtype=float)
        except (ValueError, TypeError):
            continue
        if not np.all(np.isfinite(v)):
            continue
        if box_format != 'auto':
            fmt = box_format
        fmt = str(fmt).lower()
        if fmt in ('center', 'yolo', 'cxcywh'):
            cx, cy, bw, bh = v
            v = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        elif fmt in ('xywh', 'coco'):
            v[2] += v[0]
            v[3] += v[1]
        elif fmt not in ('xyxy', 'pascal_voc'):
            print(f'[WARN] Unrecognized box format {fmt!r} in {path}; assuming xyxy')
        if np.min(v) >= 0 and np.max(v) <= 1:
            v *= np.array([w, h, w, h], dtype=float)
        x1, y1, x2, y2 = np.rint(v).astype(int)
        x1, x2 = sorted((int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))))
        y1, y2 = sorted((int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))))
        if x2 > x1 and y2 > y1:
            b = (x1, y1, x2, y2)
            if b not in result:
                result.append(b)
    return result


def score_existing(out, gt_dir, img_dir, limit, min_gain,
                   max_refined_mae=0.012, min_refined_iou=0.90, min_gain_ratio=0.20):
    b = Path(out) / '01_SoftLogits_Base'
    r = Path(out) / '03_Refine_NoFlip'
    common = sorted(file_stems(b) & file_stems(r))
    scores = []
    for stem in common[:limit] if limit else common:
        ip = find(img_dir, stem)
        gp = find(gt_dir, stem)
        if not ip or not gp:
            continue
        gt_im = cv2.imread(str(gp), 0)
        if gt_im is None:
            continue
        hw = gt_im.shape
        gt = gt_im > 127
        x = mask(b / (stem + '.png'), hw)
        y = mask(r / (stem + '.png'), hw)
        a = float(np.mean(x != gt))
        z = float(np.mean(y != gt))
        gain = a - z
        overlap = int((y & gt).sum())
        union = int((y | gt).sum())
        iou = overlap / union if union else 1.0
        relative = gain / max(a, 1e-8)
        if (gain >= min_gain and relative >= min_gain_ratio and z <= max_refined_mae
                and iou >= min_refined_iou and np.mean(x != y) > 0.001):
            scores.append(dict(stem=stem, base_mae=a, refined_mae=z,
                               refined_iou=iou, relative_gain=relative, gain=gain,
                               change=float(np.mean(x != y))))
    return sorted(scores, key=lambda e: (-(e['gain'] * e['refined_iou']), e['refined_mae']))


def maskoverlay(bgr, m):
    out = bgr.copy()
    paint = np.zeros_like(out)
    paint[:] = (40, 190, 40)
    out[m] = cv2.addWeighted(out, .65, paint, .35, 0)[m]
    return out


def map_from_crop(array, rect, hw):
    h, w = hw
    x0, y0, x1, y1 = rect
    dst = np.zeros(hw, np.float32)
    if array is not None:
        dst[y0:y1, x0:x1] = fit(array, (y1 - y0, x1 - x0))
    return dst


def annotate(bgr, title, detail=''):
    im = bgr.copy()
    h, w = im.shape[:2]
    scale = min(.7, max(.40, w / 900))
    cv2.rectangle(im, (0, 0), (w, min(h, 52)), (245, 245, 245), -1)
    cv2.putText(im, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 2, cv2.LINE_AA)
    if detail:
        cv2.putText(im, detail, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, max(.36, scale * .75), (30, 30, 30), 1, cv2.LINE_AA)
    return im


def tile(images, labels, width=300):
    panels = []
    for im, lab in zip(images, labels):
        im = cv2.resize(im, (width, width), interpolation=cv2.INTER_AREA)
        panels.append(annotate(im, lab))
    return np.concatenate(panels, axis=1)


def build_config(src, args, out):
    cfg = src.Config(sam_checkpoint=args.sam_checkpoint, sam_model_type=args.sam_model_type,
                     img_dir=args.img_dir, scribble_dir=args.scribble_dir,
                     gt_dir=args.gt_dir, json_dir=args.json_dir, out_dir=str(out),
                     refine_iters=args.refine_iters, active_points=not args.no_active,
                     use_mask_memory=not args.no_memory, flip_gate=not args.no_flip_gate,
                     candidate_quality=True, save_probs=True)
    return cfg


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def boxes_union_mask(hw, boxes):
    h, w = hw
    m = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2 + 1, x1:x2 + 1] = 1
    return m.astype(bool)


def choose_spread_points(binary_mask, k):
    ys, xs = np.where(binary_mask)
    if len(xs) == 0 or k <= 0:
        return []
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(pts) <= k:
        return [(int(x), int(y)) for x, y in pts]
    center = pts.mean(axis=0)
    start = int(np.argmin(np.sum((pts - center) ** 2, axis=1)))
    chosen = [start]
    d2 = np.sum((pts - pts[start]) ** 2, axis=1)
    for _ in range(1, min(k, len(pts))):
        nxt = int(np.argmax(d2))
        chosen.append(nxt)
        d2 = np.minimum(d2, np.sum((pts - pts[nxt]) ** 2, axis=1))
    return [(int(pts[i, 0]), int(pts[i, 1])) for i in chosen]


def sample_points_from_base_mask(base_mask, boxes, neg_total=4, base_prob=None,
                                 min_confidence=0.85, interior_margin=0.08,
                                 pos_total=4):
    """Sample a TOTAL of 4 positives and 4 negatives for the visualization.

    Positives must lie inside a box and in Base_Mask foreground, as far as
    possible from box/mask edges. High-confidence pixels are preferred.
    Negatives must lie outside EVERY annotated box and in predicted background.
    These illustrative points are NOT the actual prompt history sent to SAM.
    """
    h, w = base_mask.shape
    if base_prob is not None:
        base_prob = np.clip(np.nan_to_num(fit(base_prob, (h, w)),
                                          nan=0., posinf=1., neginf=0.), 0, 1)
    union = boxes_union_mask((h, w), boxes)
    candidates = base_mask & union
    if not np.any(candidates):
        raise RuntimeError('Base Mask has no foreground inside annotation boxes; cannot sample reliable positives')

    # Confidence+interior scoring across the complete image, per box.
    quality = np.full((h, w), -np.inf, np.float32)
    box_support = []
    for bi, (x1, y1, x2, y2) in enumerate(boxes, 1):
        fg = base_mask[y1:y2+1, x1:x2+1].astype(np.uint8)
        if not fg.any():
            continue
        bh, bw = fg.shape
        pad = cv2.copyMakeBorder(fg, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        distance = cv2.distanceTransform(pad, cv2.DIST_L2, 5)[1:-1, 1:-1]
        inset = np.ones((bh, bw), bool)
        ix = max(1, int(bw * interior_margin)) if bw > 2 else 0
        iy = max(1, int(bh * interior_margin)) if bh > 2 else 0
        if 2*ix < bw and 2*iy < bh:
            inset[:iy] = False; inset[-iy:] = False
            inset[:, :ix] = False; inset[:, -ix:] = False
        valid = (fg > 0) & inset
        if not np.any(valid):
            valid = fg > 0
        confidence = base_prob[y1:y2+1, x1:x2+1] if base_prob is not None else None
        if confidence is not None and np.any(valid & (confidence >= min_confidence)):
            valid &= confidence >= min_confidence
        # Relative interior position dominates, probability resolves ties.
        ds = distance / max(float(distance.max()), 1.0)
        q = ds + (confidence * 0.2 if confidence is not None else 0.0)
        q[~valid] = -np.inf
        quality[y1:y2+1, x1:x2+1] = np.maximum(quality[y1:y2+1, x1:x2+1], q)
        box_support.append((bi, (x1, y1, x2, y2)))

    # First give one point per box (when pos_total permits), then fill using
    # greedy distance-aware sampling. Avoid collapsing onto box corners.
    pos_points = []
    pos_meta = []
    def take_point(allowed):
        ys, xs = np.where(allowed & np.isfinite(quality))
        if len(xs) == 0:
            return None
        val = quality[ys, xs].copy()
        if pos_points:
            d2 = np.full(len(xs), np.inf, np.float32)
            for x0, y0 in pos_points:
                d2 = np.minimum(d2, (xs-x0)**2 + (ys-y0)**2)
            # Keep confidence/interior, prefer spatial separation.
            val += 0.9 * np.minimum(np.sqrt(d2) / max(12., min(h,w)*0.20), 1.)
            val[d2 < max(16., (min(h,w)*0.09)**2)] = -np.inf
        j = int(np.argmax(val))
        if not np.isfinite(val[j]):
            return None
        return (int(xs[j]), int(ys[j]))

    for bi, (x1,y1,x2,y2) in box_support:
        if len(pos_points) >= pos_total:
            break
        restricted = np.zeros((h,w), bool)
        restricted[y1:y2+1,x1:x2+1] = True
        point = take_point(restricted)
        if point is not None:
            pos_points.append(point)
            pos_meta.append({'box_index': bi, 'points_xy': [point]})
    while len(pos_points) < pos_total:
        point = take_point(candidates)
        if point is None:
            break
        pos_points.append(point)
        pos_meta.append({'box_index': None, 'points_xy': [point]})

    outside_background = (~base_mask) & (~union)
    neg_points = choose_spread_points(outside_background, neg_total)
    if len(neg_points) < neg_total:
        # Fallback stays outside boxes, but mark as uncertain when predicted
        # foreground covers the remainder of image.
        fallback = choose_spread_points(~union, neg_total)
        for pt in fallback:
            if pt not in neg_points:
                neg_points.append(pt)
            if len(neg_points) >= neg_total:
                break
    if len(pos_points) != pos_total or len(neg_points) != neg_total:
        print(f'[WARN] Only {len(pos_points)}/{pos_total} positive and '
              f'{len(neg_points)}/{neg_total} outside-box negative points available')
    return pos_points, neg_points[:neg_total], pos_meta


def compute_case_stats(boxes, w, h):
    image_area = float(w * h)
    diag = math.hypot(w, h)
    cx0, cy0 = w / 2.0, h / 2.0
    area_ratios = []
    center_dists = []
    for x1, y1, x2, y2 in boxes:
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_ratios.append((bw * bh) / image_area)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        center_dists.append(math.hypot(cx - cx0, cy - cy0) / diag)
    return {
        'box_count': len(boxes),
        'area_ratios': area_ratios,
        'max_area_ratio': max(area_ratios) if area_ratios else 0.0,
        'mean_area_ratio': float(np.mean(area_ratios)) if area_ratios else 0.0,
        'center_dists': center_dists,
        'mean_center_dist': float(np.mean(center_dists)) if center_dists else 0.0,
        'min_center_dist': min(center_dists) if center_dists else 0.0,
        'max_center_dist': max(center_dists) if center_dists else 0.0,
    }


def draw_boxes(im, boxes, S, line):
    out = im.copy()
    for b in boxes:
        x1, y1, x2, y2 = np.round(np.asarray(b) * S).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), line + 2 * S, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 30, 255), line, cv2.LINE_AA)
    return out


def draw_points(im, points, S, radius, fg=True, added=False):
    out = im.copy()
    color = (20, 245, 35) if fg else (30, 30, 255)
    for x, y in points:
        xS, yS = int(round(x * S)), int(round(y * S))
        cv2.circle(out, (xS, yS), radius + 4 * S, (0, 0, 0), -1, cv2.LINE_AA)
        if added:
            cv2.circle(out, (xS, yS), radius + 8 * S, (0, 230, 255), 3 * S, cv2.LINE_AA)
        cv2.circle(out, (xS, yS), radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, (xS, yS), max(2 * S, radius // 4), (255, 255, 255), -1, cv2.LINE_AA)
    return out


def export_selected(src, args, rows, root):
    cfg = build_config(src, args, root / 'rerun_masks')
    pipeline = src.Pipeline(cfg)
    prediction = pipeline.engine.predictor
    original_predict = prediction.predict
    active = {
        'rect': None,
        'hw': None,
        'stage': None,
        'current_box': -1,
        'calls': [],
        'base_capture': [],
        'refine_capture': []
    }

    def capture_predict(*aa, **kw):
        output = original_predict(*aa, **kw)
        if active['stage'] == 'refine':
            pts = kw.get('point_coords')
            labs = kw.get('point_labels')
            active['calls'].append(dict(
                box_index=active['current_box'],
                points=None if pts is None else np.asarray(pts, np.float32).copy(),
                labels=None if labs is None else np.asarray(labs, np.int32).copy(),
                rect=active['rect'],
                is_multimask=bool(kw.get('multimask_output', False)),
                has_memory=kw.get('mask_input') is not None
            ))
        if kw.get('multimask_output') is True:
            logits, scores, low = output
            rec = dict(rect=active['rect'], hw=active['hw'],
                       logits=np.asarray(logits, np.float32).copy(),
                       scores=np.asarray(scores, np.float32).copy(),
                       low=np.asarray(low, np.float32).copy())
            if active['stage'] == 'base':
                active['base_capture'].append(rec)
            elif active['stage'] == 'refine':
                active['refine_capture'].append(rec)
        return output

    prediction.predict = capture_predict
    original_infer = pipeline._infer_single_box

    def tracked_infer(rgb, scribble, box, refine, do_crop=True, collect_quality=False):
        _, _, _, rect, _, _, _ = pipeline._target_for_box(rgb, scribble, box, do_crop)
        active['rect'] = tuple(rect)
        active['hw'] = rgb.shape[:2]
        active['stage'] = 'refine' if refine else 'base'
        if refine:
            active['current_box'] += 1
        output = original_infer(rgb, scribble, box, refine, do_crop, collect_quality)
        active['stage'] = None
        return output

    pipeline._infer_single_box = tracked_infer
    selected = []

    for rank, row in enumerate(rows, 1):
        stem = row['stem']
        img_path = find(args.img_dir, stem)
        if not img_path:
            continue
        active['calls'] = []
        active['base_capture'] = []
        active['refine_capture'] = []
        active['current_box'] = -1
        active['stage'] = None

        record = pipeline.process_image(str(img_path))
        if record is None:
            print('[WARN] Skipped:', stem)
            continue

        bgr = cv2.imread(str(img_path))
        h, w = bgr.shape[:2]
        output_dir = ensure_dir(root / f'{rank:02d}_{stem}')

        gt_path = find(args.gt_dir, stem)
        gt = cv2.imread(str(gt_path), 0) > 127
        base = mask(Path(cfg.base_dir) / (stem + '.png'), (h, w))
        refined = mask(Path(cfg.refine_dir) / (stem + '.png'), (h, w))
        final = mask(Path(cfg.profile_dir) / (stem + '.png'), (h, w))
        save(output_dir / 'RGB.png', bgr)

        box_label = locate_box_annotation(args.json_dir, stem)
        boxes = load_annotation_boxes(box_label, w, h, args.box_format)
        if not boxes:
            raise RuntimeError(f'No bounding boxes parsed for {stem}; label file={box_label}')

        stats = compute_case_stats(boxes, w, h)
        S = args.draw_scale
        H, W = h * S, w * S
        line = max(5 * S, int(round(min(h, w) * 0.010 * S)))
        radius = max(7 * S, int(round(min(h, w) * 0.014 * S)))
        rgb_up = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_CUBIC)

        rgb_boxes = draw_boxes(rgb_up, boxes, S, line)
        save(output_dir / 'RGB_Boxes.png', rgb_boxes)

        # Use true saved base-stage probability when available (never GT/refined).
        base_prob_path = Path(cfg.out_dir) / 'base_prob' / (stem + '.npy')
        base_prob = np.load(base_prob_path).astype(np.float32) if base_prob_path.exists() else None
        pos_points, neg_points, pos_meta = sample_points_from_base_mask(
            base, boxes, neg_total=args.mask_neg_total, base_prob=base_prob,
            pos_total=args.mask_pos_total,
            min_confidence=args.pos_min_confidence, interior_margin=args.pos_interior_margin)

        rgb_positive = draw_points(rgb_up, pos_points, S, radius, fg=True)
        rgb_negative = draw_points(rgb_up, neg_points, S, radius, fg=False)
        rgb_boxes_positive = draw_points(rgb_boxes, pos_points, S, radius, fg=True)
        rgb_boxes_negative = draw_points(rgb_boxes, neg_points, S, radius, fg=False)
        rgb_boxes_maskpoints = draw_points(rgb_boxes_positive, neg_points, S, radius, fg=False)

        save(output_dir / 'RGB_PositivePoints.png', rgb_positive)
        save(output_dir / 'RGB_NegativePoints.png', rgb_negative)
        save(output_dir / 'RGB_Boxes_PositivePoints.png', rgb_boxes_positive)
        save(output_dir / 'RGB_Boxes_NegativePoints.png', rgb_boxes_negative)
        save(output_dir / 'RGB_Boxes_MaskPoints.png', rgb_boxes_maskpoints)

        legend = np.full((80 * S, 520 * S, 3), 255, np.uint8)
        legend = draw_points(legend, [(35, 36)], 1, max(2, radius // max(S, 1)), fg=True)
        cv2.putText(legend, 'Positive from Base Mask', (60, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65 * S, (10, 10, 10), max(1, S), cv2.LINE_AA)
        legend = draw_points(legend, [(260, 36)], 1, max(2, radius // max(S, 1)), fg=False)
        cv2.putText(legend, 'Negative outside boxes', (285, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65 * S, (10, 10, 10), max(1, S), cv2.LINE_AA)
        save(output_dir / 'MaskPoint_Legend.png', legend)
        print(f'[MASK POINT CHECK] {stem}: boxes={len(boxes)}, mask_pos={len(pos_points)}, mask_neg_outside={len(neg_points)}')

        # Original decoder-prompt visualization retained as reference.
        initial_only = rgb_boxes.copy()
        prompt_img = rgb_boxes.copy()

        def point_original(pt, rect):
            x0, y0, x1, y1 = rect
            xp, yp = pt
            x = int(np.clip(round((x0 + (xp + .5) * (x1 - x0) / cfg.size - .5) * S), 0, W - 1))
            y = int(np.clip(round((y0 + (yp + .5) * (y1 - y0) / cfg.size - .5) * S), 0, H - 1))
            return x, y

        def should_draw_point(xy, label):
            if int(label) == 1:
                return True
            x, y = xy[0] / S, xy[1] / S
            return not any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in boxes)

        def mark_decoder(im, xy, fg, added=False):
            x, y = xy
            color = (20, 245, 35) if fg else (30, 30, 255)
            cv2.circle(im, (x, y), radius + 4 * S, (0, 0, 0), -1, cv2.LINE_AA)
            if added:
                cv2.circle(im, (x, y), radius + 8 * S, (0, 230, 255), 3 * S, cv2.LINE_AA)
            cv2.circle(im, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(im, (x, y), max(2 * S, radius // 4), (255, 255, 255), -1, cv2.LINE_AA)

        count_initial = count_added = 0
        hidden_negative_initial = hidden_negative_added = 0
        calls = active['calls']
        box_indices = sorted(set(c['box_index'] for c in calls))
        no_points = []
        draw_meta = []
        for index in box_indices:
            group = [c for c in calls if c['box_index'] == index and not c['is_multimask']]
            if not group:
                continue
            first, last = group[0], group[-1]
            p0 = first['points'] if first['points'] is not None else np.empty((0, 2))
            l0 = first['labels'] if first['labels'] is not None else np.empty((0,), dtype=int)
            p1 = last['points'] if last['points'] is not None else np.empty((0, 2))
            l1 = last['labels'] if last['labels'] is not None else np.empty((0,), dtype=int)
            if len(p0) == 0:
                no_points.append(index + 1)
            for pt, lab in zip(p0, l0):
                xy = point_original(pt, first['rect'])
                if should_draw_point(xy, lab):
                    mark_decoder(initial_only, xy, int(lab) == 1)
                    mark_decoder(prompt_img, xy, int(lab) == 1)
                    count_initial += 1
                else:
                    hidden_negative_initial += 1
            for pt, lab in zip(p1[len(p0):], l1[len(l0):]):
                xy = point_original(pt, last['rect'])
                if should_draw_point(xy, lab):
                    mark_decoder(prompt_img, xy, int(lab) == 1, added=True)
                    count_added += 1
                else:
                    hidden_negative_added += 1
            draw_meta.append({'box_index': index + 1, 'initial_points': len(p0),
                              'final_points': len(p1), 'decoder_calls': len(group)})
        save(output_dir / 'RGB_Boxes_InitialPoints.png', initial_only)
        save(output_dir / 'RGB_Boxes_Points.png', prompt_img)
        print(f'[POINT CHECK] {stem}: {len(boxes)} boxes, initial={count_initial}, active_added={count_added}, '
              f'negatives_inside_box_hidden={hidden_negative_initial + hidden_negative_added}, actual decoder calls={len(calls)}; scale={S}x')

        for name, arr in [('GT', gt), ('Base_Mask', base), ('Refined_Mask', refined), ('Final_Mask', final)]:
            save(output_dir / (name + '.png'), arr.astype(np.uint8) * 255)
        save(output_dir / 'RGB_BaseOverlay.png', maskoverlay(bgr, base))
        save(output_dir / 'RGB_RefinedOverlay.png', maskoverlay(bgr, refined))

        saved_heat = {}
        for short, title in [('base', 'Base'), ('refine', 'Refined'), ('flipgate', 'Final')]:
            fp = Path(cfg.out_dir) / (short + '_prob') / (stem + '.npy')
            if fp.exists():
                p = np.load(fp).astype(np.float32)
                p = fit(p, (h, w))
                heat = jet(p)
                save(output_dir / (title + '_ProbabilityHeatmap.png'), heat)
                saved_heat[title] = heat

        first_step_dir = ensure_dir(output_dir / 'FirstStep_Logits')
        first_step_meta = []
        merged_first_heat = np.zeros((h, w), np.float32)
        if active['base_capture']:
            for j, obj in enumerate(active['base_capture'], 1):
                rect = obj['rect']
                scores = obj['scores']
                logits = obj['logits']
                obj_dir = ensure_dir(first_step_dir / f'Box{j}')
                best_idx = int(np.argmax(scores))
                for idx in range(min(3, len(logits))):
                    native = logits[idx]
                    np.save(obj_dir / f'Candidate_{idx + 1}_Logits_ModelFrame.npy', native.astype(np.float16))
                    p_native = src.sigmoid(native)
                    p = map_from_crop(p_native, rect, (h, w))
                    hard = map_from_crop((native > 0).astype(np.float32), rect, (h, w)) >= .5
                    save(obj_dir / f'Candidate_{idx + 1}_Heatmap.png', jet(p))
                    save(obj_dir / f'Candidate_{idx + 1}_Mask.png', hard.astype(np.uint8) * 255)
                best_native = logits[best_idx]
                best_prob = map_from_crop(src.sigmoid(best_native), rect, (h, w))
                merged_first_heat = np.maximum(merged_first_heat, best_prob)
                save(obj_dir / 'Best_Logits_Heatmap.png', jet(best_prob))
                np.save(obj_dir / 'Best_Logits_ModelFrame.npy', best_native.astype(np.float16))
                first_step_meta.append({'box_index': j, 'crop_xyxy': list(rect), 'scores': scores.tolist(), 'best_idx': best_idx + 1})
            save(output_dir / 'FirstStep_Logits_Heatmap.png', jet(merged_first_heat))
            np.save(output_dir / 'FirstStep_Logits_ProjectedProb.npy', merged_first_heat.astype(np.float16))
        elif 'Base' in saved_heat:
            save(output_dir / 'FirstStep_Logits_Heatmap.png', saved_heat['Base'])
        else:
            print(f'[WARN] {stem}: no base-stage multimask capture and no saved base_prob heatmap.')

        candidates_meta = []
        for j, obj in enumerate(active['refine_capture'], 1):
            rect = obj['rect']
            scores = obj['scores']
            logits = obj['logits']
            obj_dir = ensure_dir(output_dir / ('Candidates' if len(active['refine_capture']) == 1 else f'Candidates_Box{j}'))
            masks = []
            hm = []
            for idx in range(min(3, len(logits))):
                native = logits[idx]
                p_native = src.sigmoid(native)
                p = map_from_crop(p_native, rect, (h, w))
                hard = map_from_crop((native > 0).astype(np.float32), rect, (h, w)) >= .5
                save(obj_dir / f'Candidate_{idx + 1}_Mask.png', hard.astype(np.uint8) * 255)
                save(obj_dir / f'Candidate_{idx + 1}_Heatmap.png', jet(p))
                np.save(obj_dir / f'Candidate_{idx + 1}_Logits_ModelFrame.npy', native.astype(np.float16))
                masks.append(cv2.cvtColor(hard.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR))
                hm.append(jet(p))
            save(obj_dir / 'Three_Candidate_Masks.png', tile(masks, [f'M{i + 1} score={scores[i]:.3f}' for i in range(len(masks))]))
            save(obj_dir / 'Three_Candidate_Heatmaps.png', tile(hm, [f'L{i + 1}' for i in range(len(hm))]))
            candidates_meta.append({'object': j, 'crop_xyxy': list(rect), 'scores': scores.tolist(),
                                    'note': 'Refine-stage multimask diagnostic.'})

        good = (base != gt) & (refined == gt)
        bad = (base == gt) & (refined != gt)
        compare = bgr.copy()
        compare[good] = (20, 210, 20)
        compare[bad] = (20, 20, 240)
        save(output_dir / 'Improvement_GreenRed_GTReference.png', compare)

        imgs = [rgb_boxes, rgb_positive, rgb_negative,
                cv2.cvtColor(base.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR),
                saved_heat.get('Base', np.zeros_like(bgr)),
                cv2.cvtColor(refined.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR),
                compare]
        labels = ['RGB+Boxes', 'Positive points', 'Negative points', 'Base Mask', 'Base Prob', 'Refined Mask', 'Correction (GT only)']
        save(output_dir / 'Overview.png', tile(imgs, labels, width=400))

        meta = {
            **row,
            'rank': rank,
            'box_annotation_file': str(box_label),
            'box_coordinates': boxes,
            'case_stats': stats,
            'boxes': len(boxes),
            'route_type': record['route_type'],
            'mask_point_strategy': {
                'positive_from': 'One confidence-guided, interior Base_Mask foreground point per box',
                'negative_from': 'background outside all annotation boxes',
                'positive_total_requested': args.mask_pos_total,
                'pos_min_confidence': args.pos_min_confidence,
                'pos_interior_margin': args.pos_interior_margin,
                'mask_neg_total': args.mask_neg_total,
                'mask_positive_points': pos_points,
                'mask_negative_points': neg_points,
                'per_box_positive_stats': pos_meta
            },
            'first_step_candidate_sets': first_step_meta,
            'candidate_sets': candidates_meta,
            'initial_points_drawn': count_initial,
            'active_points_drawn': count_added,
            'negative_points_inside_boxes_hidden': hidden_negative_initial + hidden_negative_added,
            'point_decoder_audit': draw_meta,
            'no_points_boxes': no_points,
            'notes': [
                'RGB_Boxes.png, RGB_PositivePoints.png, and RGB_NegativePoints.png are saved separately.',
                'Cases are selected as a mixed set: ordinary plus optional small, off-center, and multi-object examples.',
                'RGB_Boxes_MaskPoints.png uses one high-confidence interior Base_Mask positive per box and negatives only outside all annotation boxes. These are illustrative points, not necessarily SAM inputs.',
                'FirstStep_Logits contains base-stage multimask diagnostic logits when available.',
                'GT used only for post-hoc case ranking and correction visualization.'
            ]
        }
        (output_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'[{rank}/{len(rows)}] {stem} MAE improvement {row["gain"]:.4f}, first_step_captures={len(active["base_capture"])}, refine_captures={len(active["refine_capture"])}')
        selected.append(meta)

    (root / 'selected_cases.json').write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-script', default='./sam_mask/sam1_refine_crop_flipgate.py')
    p.add_argument('--sam-output', default='./softresult/SAM1_Optimized3')
    p.add_argument('--sam-model-type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
    p.add_argument('--sam-checkpoint', default='./sam_mask/weights/sam_vit_h_4b8939.pth')
    p.add_argument('--img-dir', default='./data/Train/Imgs')
    p.add_argument('--scribble-dir', default='./data/Train/Scribble')
    p.add_argument('--gt-dir', default='./data/Train/GT')
    p.add_argument('--json-dir', default='./data/Train/box_label')
    p.add_argument('--out-dir', default='./gesp_four_cases')
    p.add_argument('--num', type=int, default=4)
    p.add_argument('--limit', type=int, default=100, help='0 means all')
    p.add_argument('--min-gain', type=float, default=.006)
    p.add_argument('--max-refined-mae', type=float, default=.012)
    p.add_argument('--min-refined-iou', type=float, default=.90)
    p.add_argument('--min-gain-ratio', type=float, default=.20)
    p.add_argument('--refine-iters', type=int, default=1)
    p.add_argument('--no-memory', action='store_true')
    p.add_argument('--no-active', action='store_true')
    p.add_argument('--draw-scale', type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument('--no-flip-gate', action='store_true')
    p.add_argument('--box-format', choices=['auto', 'xyxy', 'xywh', 'cxcywh'], default='auto')
    p.add_argument('--pos-min-confidence', type=float, default=0.85, help='Base probability threshold for reliable positive candidates')
    p.add_argument('--pos-interior-margin', type=float, default=0.08, help='Inset fraction of each box to keep positives away from the box boundary')
    p.add_argument('--mask-neg-total', type=int, default=4)
    p.add_argument('--mask-pos-total', type=int, default=4, help='Total number of high-confidence interior positive points')
    p.add_argument('--case-name', default='COD10K-CAM-1-Aquatic-11-Octopus-431', help='Exact image stem of a candidate-Noisy case; empty string selects from quality-ranked cases')

    # A mixed case portfolio: normal examples are NOT filtered out.
    p.add_argument('--min-boxes', type=int, default=1,
                   help='Global minimum valid boxes; default 1 allows ordinary cases')
    p.add_argument('--small-max-area', type=float, default=0.10,
                   help='Small if ANY annotation box area/image area <= threshold')
    p.add_argument('--offcenter-min-dist', type=float, default=0.12,
                   help='Off-center if ANY box center normalized distance >= threshold')
    p.add_argument('--small-num', type=int, default=1, help='Reserve up to N small-target examples')
    p.add_argument('--offcenter-num', type=int, default=1, help='Reserve up to N off-center examples')
    p.add_argument('--multi-num', type=int, default=1, help='Reserve up to N multi-object examples')
    p.add_argument('--normal-num', type=int, default=1,
                   help='Reserve up to N ordinary examples; best remaining if no strict ordinary cases')
    args = p.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    import importlib.util
    source = Path(args.source_script)
    if not source.exists():
        p.error(f'SAM1 script not found: {source}')
    local_pkg = str(source.resolve().parent)
    if local_pkg not in sys.path:
        sys.path.insert(0, local_pkg)
    spec = importlib.util.spec_from_file_location('user_sam1_pipeline', source)
    src = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = src
    spec.loader.exec_module(src)

    if args.case_name:
        stem = Path(args.case_name).stem
        if find(args.img_dir, stem) is None:
            p.error(f'Chosen Noisy image not found in --img-dir: {stem}')
        if find(args.gt_dir, stem) is None:
            p.error(f'GT file not found for chosen Noisy case: {stem}')
        # This specific sample was checked in sam1_three_pool_hit_labels_4040.csv:
        # candidate_iou=0.883121, true_iou=0.842235, both Noisy.
        rows = [dict(stem=stem, base_mae=float('nan'), refined_mae=float('nan'),
                     refined_iou=0.8422351566078564, relative_gain=float('nan'),
                     gain=0., change=float('nan'), candidate_iou_mean=0.8831211735087456,
                     candidate_iou_pool='Noisy', true_iou_pool='Noisy')]
        print(f'[NOISY CASE] {stem}: candidate IoU 0.883121 -> Noisy (0.80-0.95), true IoU 0.842235 -> Noisy')
    else:
        rows = score_existing(args.sam_output, args.gt_dir, args.img_dir, args.limit,
                              args.min_gain, args.max_refined_mae,
                              args.min_refined_iou, args.min_gain_ratio)
    fallback_used = False
    if not rows and not args.case_name:
        fallback_used = True
        print('[INFO] No qualifying existing masks. Running SAM1 on a subset to rank improvements...')
        probe_cfg = build_config(src, args, root / 'ranking_run')
        probe_cfg.save_probs = False
        probe_cfg.candidate_quality = False
        probe_cfg.flip_gate = False
        probe = src.Pipeline(probe_cfg)
        paths = src.DataLoader.list_images(args.img_dir)[:args.limit] if args.limit else src.DataLoader.list_images(args.img_dir)
        for i, path in enumerate(paths, 1):
            try:
                probe.process_image(path)
            except Exception as e:
                print(f'[WARN] ranking {Path(path).name}: {e}')
            if i % 20 == 0:
                print(f'[INFO] Ranking {i}/{len(paths)}')
        rows = score_existing(probe_cfg.out_dir, args.gt_dir, args.img_dir, None,
                              args.min_gain, args.max_refined_mae,
                              args.min_refined_iou, args.min_gain_ratio)

    if rows:
        print(f'[INFO] {len(rows)} clean improvement candidates; best:')
        for item in rows[:min(12, len(rows))]:
            print('  {stem}: MAE {base_mae:.4f} -> {refined_mae:.4f} (gain {gain:.4f}, IoU {refined_iou:.3f})'.format(**item))

    eligible = []
    skipped = {'missing_image': 0, 'missing_boxes': 0}
    for row in rows:
        stem = row['stem']
        img_file = find(args.img_dir, stem)
        if img_file is None:
            skipped['missing_image'] += 1
            continue
        img = cv2.imread(str(img_file))
        if img is None:
            skipped['missing_image'] += 1
            continue
        h, w = img.shape[:2]
        label = locate_box_annotation(args.json_dir, stem)
        boxes = load_annotation_boxes(label, w, h, args.box_format)
        if len(boxes) < max(1, args.min_boxes):
            skipped['missing_boxes'] += 1
            continue
        cs = compute_case_stats(boxes, w, h)
        # Each category is independent, NOT all three conditions at once.
        small = any(a <= args.small_max_area for a in cs['area_ratios'])
        offcenter = any(d >= args.offcenter_min_dist for d in cs['center_dists'])
        multi = len(boxes) >= 2
        # Strictly typical: single, non-small, centered. If none available,
        # a best-ranked otherwise-unreserved example can fill normal slot.
        normal = len(boxes) == 1 and not small and not offcenter
        eligible.append({**row,
                         'annotation_box_count': len(boxes),
                         'case_stats': cs,
                         'is_small': small,
                         'is_offcenter': offcenter,
                         'is_multi': multi,
                         'is_normal': normal,
                         'box_annotation_file': str(label) if label else None})

    budget = max(0, args.num)
    selected = []
    used = set()
    def take(category, count, condition):
        for r in eligible:
            if len(selected) >= budget or count <= 0:
                break
            if r['stem'] not in used and condition(r):
                selected.append({**r, 'selection_category': category})
                used.add(r['stem'])
                count -= 1
        return count

    # Prioritize an ordinary example, then cover the three requested variations.
    remaining_normal = take('normal', args.normal_num, lambda r: r['is_normal'])
    if remaining_normal:
        # Do not lose ordinary representation when all strict 'normal' cases are excluded.
        take('normal_fallback', remaining_normal, lambda r: not r['is_multi'])
    requested = [
        ('small', args.small_num, lambda r: r['is_small']),
        ('offcenter', args.offcenter_num, lambda r: r['is_offcenter']),
        ('multi', args.multi_num, lambda r: r['is_multi']),
    ]
    for category, count, condition in requested:
        missing = take(category, count, condition)
        if missing:
            print(f'[CATEGORY WARN] {category}: {missing} requested examples not available as distinct cases')

    # Fill any remaining slots by original MAE ranking, normal or special.
    take('best_remaining', budget - len(selected), lambda r: True)
    rows = selected
    print(f'[CASE FILTER] eligible={len(eligible)}, skipped={skipped}, selected={len(rows)}')
    for r in rows:
        cs = r['case_stats']
        types = [name for name, key in [('small', 'is_small'), ('offcenter', 'is_offcenter'),
                                      ('multi', 'is_multi'), ('normal', 'is_normal')] if r[key]]
        print(f"[SELECT] {r['selection_category']}: {r['stem']} | actual_tags={types} | "
              f"boxes={r['annotation_box_count']} | min_area={min(cs['area_ratios']):.4f} | "
              f"max_center_dist={cs['max_center_dist']:.4f} | MAE_gain={r['gain']:.4f}")
    if not rows:
        print('[INFO] No eligible cases with readable boxes and valid improvement; '
              'try lowering --min-gain or increasing --limit.')
        return

    export_selected(src, args, rows, root)
    if fallback_used:
        import shutil
        shutil.rmtree(root / 'ranking_run', ignore_errors=True)
    print('[DONE] Figure-ready cases in:', root)


if __name__ == '__main__':
    main()
