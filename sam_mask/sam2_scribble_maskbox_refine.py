#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM2: first pass Scribble-only, then mask-derived box + soft mask feedback.
No annotated boxes are read. No Crop/MO/Flip by design.
Examples:
 python sam2_scribble_maskbox_refine.py
 python sam2_scribble_maskbox_refine.py --limit 100
 python sam2_scribble_maskbox_refine.py --no-active
 python sam2_scribble_maskbox_refine.py --refine-iters 0
 python sam2_scribble_maskbox_refine.py --self-test
"""
import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from glob import glob
from contextlib import nullcontext
import cv2
import numpy as np
import torch
from tqdm import tqdm

@dataclass
class Config:
    sam2_checkpoint: str = "./sam_pseado/weights/sam2_hiera_large.pt"
    sam2_config: str = "sam2_hiera_l.yaml"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir: str = "./data/Train/Imgs"
    scribble_dir: str = "./data/Train/Scribble"
    gt_dir: str = "./data/Train/GT"
    out_dir: str = "./softresult/SAM2_Scribble_MaskBox"
    size: int = 1024
    seed: int = 0
    refine_iters: int = 2
    active_points: bool = True
    active_min_dist: int = 16
    active_conflict_first: bool = True
    max_initial_per_class: int = 9
    box_threshold: float = 0.5
    box_min_component_area: int = 16
    box_margin_ratio: float = 0.05
    box_margin_min: int = 5
    save_probs: bool = False

class DataLoader:
    @staticmethod
    def list_images(directory):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            paths.extend(glob(os.path.join(directory, ext)))
            paths.extend(glob(os.path.join(directory, ext.upper())))
        return sorted(set(paths))

    @staticmethod
    def find_file(directory, name, extensions):
        for ext in extensions:
            for e in (ext.lower(), ext.upper()):
                path = os.path.join(directory, name + e)
                if os.path.isfile(path):
                    return path
        return None

    @staticmethod
    def load_boxes(path, width, height):
        if not path or not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        boxes = []
        for shape in data.get("shapes", []):
            points = np.asarray(shape.get("points", []), np.float32)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
                continue
            box = np.array([points[:, 0].min(), points[:, 1].min(),
                            points[:, 0].max(), points[:, 1].max()], dtype=np.float32)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        return boxes

    @staticmethod
    def parse_scribble(scribble):
        """1=foreground, 2=background, 0=unknown; also supports binary 255=FG.

        IMPORTANT: background zero of a binary scribble is unknown, NOT a negative point.
        """
        scribble = np.asarray(scribble)
        if np.any(scribble == 1) or np.any(scribble == 2):
            return scribble == 1, scribble == 2
        return scribble > 128, np.zeros_like(scribble, dtype=bool)

    @staticmethod
    def points_from_scribble(scribble, max_per_class=9):
        fg, bg = DataLoader.parse_scribble(scribble)
        points, labels = [], []
        for region, label in ((fg, 1), (bg, 0)):
            ys, xs = np.where(region)
            if len(xs) == 0:
                continue
            dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
            rows = cols = max(1, int(np.ceil(np.sqrt(max_per_class))))
            y_edges = np.linspace(ys.min(), ys.max() + 1, rows + 1)
            x_edges = np.linspace(xs.min(), xs.max() + 1, cols + 1)
            for i in range(rows):
                for j in range(cols):
                    sel = (ys >= y_edges[i]) & (ys < y_edges[i + 1]) & \
                          (xs >= x_edges[j]) & (xs < x_edges[j + 1])
                    if not np.any(sel):
                        continue
                    cx, cy = xs[sel], ys[sel]
                    k = int(np.argmax(dist[cy, cx]))
                    points.append([float(cx[k]), float(cy[k])])
                    labels.append(label)
        if not points:
            return None, None
        return np.asarray(points, np.float32), np.asarray(labels, np.int32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float32), -30, 30)))


def resize_prob(prob, dst_w, dst_h):
    prob = np.asarray(prob, np.float32)
    if prob.shape == (dst_h, dst_w):
        return prob.copy()
    return cv2.resize(prob, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)


def mask_to_box(prob, cfg, scribble=None):
    """Find bounding box from initial predicted foreground, in 1024 input coordinates.
    Filter tiny components, union remaining. If empty, use foreground scribble bounds;
    if no foreground scribble, return None (point-only refinement).
    """
    binary = (prob >= cfg.box_threshold).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    selected = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.box_min_component_area:
            selected[components == i] = 1
    fallback = "mask"
    ys, xs = np.where(selected > 0)
    if len(xs) == 0:
        fallback = "scribble_fg"
        if scribble is not None:
            fg, _ = DataLoader.parse_scribble(scribble)
            ys, xs = np.where(fg)
    if len(xs) == 0:
        return None, "none"
    h, w = prob.shape
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    margin = max(cfg.box_margin_min, int(round(max(x2-x1+1,y2-y1+1) * cfg.box_margin_ratio)))
    box = np.array([max(0,x1-margin),max(0,y1-margin),min(w-1,x2+margin),min(h-1,y2+margin)], np.float32)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, "none"
    return box, fallback

class MetricsEvaluator:
    """Keeps the old binary-mask COD evaluation semantics for CSV compatibility.

    NOTE: F_binary_0p5/E_binary_0p5 are NOT maxF/maxE over thresholds.
    Legacy column aliases base_maxF/base_maxE are retained for old consumers.
    """
    @staticmethod
    def prepare_pred_gt(pred, gt):
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        pred = pred.astype(np.float32)
        if pred.max() > 1.:
            pred /= 255.
        return np.clip(pred, 0., 1.), (gt > 128).astype(np.float32)

    @staticmethod
    def object_score(x):
        if x.size == 0:
            return 0.
        m, sd = float(x.mean()), float(x.std())
        return (2 * m) / (m * m + 1 + sd + 1e-8)

    @classmethod
    def s_object(cls, p, gt):
        u = float(gt.mean())
        return u * cls.object_score(p[gt == 1]) + (1-u) * cls.object_score(1-p[gt == 0])

    @staticmethod
    def ssim_region(p, gt):
        if p.size == 0:
            return 0.
        x, y = float(p.mean()), float(gt.mean())
        sx, sy = float(((p-x)**2).mean()), float(((gt-y)**2).mean())
        sxy = float(((p-x)*(gt-y)).mean())
        return 4*x*y*sxy / ((x*x+y*y)*(sx+sy)+1e-8)

    @classmethod
    def s_region(cls, p, gt):
        h, w = gt.shape
        if h < 2 or w < 2:
            return 0.
        ys, xs = np.where(gt > .5)
        x = int(np.clip(np.round(xs.mean()), 1, w-1))
        y = int(np.clip(np.round(ys.mean()), 1, h-1))
        return (x*y*cls.ssim_region(p[:y, :x], gt[:y, :x]) +
                (w-x)*y*cls.ssim_region(p[:y, x:], gt[:y, x:]) +
                x*(h-y)*cls.ssim_region(p[y:, :x], gt[y:, :x]) +
                (w-x)*(h-y)*cls.ssim_region(p[y:, x:], gt[y:, x:])) / (h*w)

    @classmethod
    def evaluate(cls, pred_raw, gt_raw):
        p, gt = cls.prepare_pred_gt(pred_raw, gt_raw)
        m = float(gt.mean())
        if m == 0.:
            s = 1 - float(p.mean())
        elif m == 1.:
            s = float(p.mean())
        else:
            s = float(np.clip(0.5*cls.s_object(p, gt) + 0.5*cls.s_region(p, gt), 0, 1))
        binary = p >= .5
        gtb = gt >= .5
        tp = float((binary & gtb).sum())
        precision = tp / (float(binary.sum())+1e-8)
        recall = tp / (float(gtb.sum())+1e-8)
        fb = float(1.3 * precision * recall / (0.3*precision + recall + 1e-8))
        if gtb.sum() == 0:
            eb = 1 - float(binary.mean())
        elif gtb.sum() == gtb.size:
            eb = float(binary.mean())
        else:
            fm, gtm = binary.astype(np.float32), gt.astype(np.float32)
            fm -= fm.mean()
            gtm -= gtm.mean()
            alignment = 2 * fm * gtm / (fm*fm + gtm*gtm + 1e-8)
            eb = float(np.mean((alignment+1)**2/4))
        return {"Smeasure": s, "maxE": eb, "maxF": fb, "MAE": float(np.mean(np.abs(p-gt)))}


class SAM2Engine:
    def __init__(self, cfg, predictor=None):
        self.cfg = cfg
        self.device = cfg.device
        if predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(cfg.sam2_config, cfg.sam2_checkpoint, device=cfg.device)
            predictor = SAM2ImagePredictor(model)
        self.predictor = predictor
        self.last_input_shape = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def set_image(self, model_rgb):
        self.last_input_shape = model_rgb.shape[:2]
        with torch.inference_mode():
            self.predictor.set_image(np.ascontiguousarray(model_rgb))

    def _predict_once(self, points, labels, box, mask_input):
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.startswith("cuda") and torch.cuda.is_available() and
                    torch.cuda.is_bf16_supported() else nullcontext())
        with torch.inference_mode(), autocast:
            high_logits, scores, low_logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box[None, :] if box is not None else None,
                mask_input=mask_input,
                multimask_output=False,
                return_logits=True,
            )
        high_logits = np.asarray(high_logits, np.float32)
        low_logits = np.asarray(low_logits, np.float32)
        if high_logits.ndim == 3:
            high_logits = high_logits[0]
        if low_logits.ndim == 2:
            low_logits = low_logits[None, ...]
        if low_logits.ndim == 4 and low_logits.shape[0] == 1:
            low_logits = low_logits[0]
        if low_logits.ndim != 3 or low_logits.shape[0] != 1:
            raise ValueError(f"SAM2 expected 1xHxW low-res mask, got {low_logits.shape}")
        return sigmoid(high_logits), low_logits

    def sample_active_point(self, prob, scribble, existing, roi=None):
        """Sample *known* scribble contradiction first; uncertainty second.

        All arrays and points are in SAME SAM2 image input pixel frame.
        No GT or outside-annotation pseudo point is introduced.
        """
        if scribble is None:
            return None, None, "no_scribble"
        h, w = self.last_input_shape
        if scribble.shape[:2] != (h, w):
            raise ValueError(f"Scribble/model frame mismatch {scribble.shape} vs {(h, w)}")
        p = resize_prob(prob, w, h)
        fg, bg = DataLoader.parse_scribble(scribble)
        known = fg | bg
        if roi is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in roi]
            x1, x2 = max(0, x1), min(w-1, x2)
            y1, y2 = max(0, y1), min(h-1, y2)
            allowed = np.zeros((h,w), bool)
            if x2 >= x1 and y2 >= y1:
                allowed[y1:y2+1, x1:x2+1] = True
            known &= allowed
        if not known.any():
            return None, None, "no_known_points"

        if existing is not None and len(existing):
            # Exclude an actual disk, not only the center pixel.
            available = (known.astype(np.uint8) * 255)
            for x, y in np.asarray(existing, np.float32):
                cv2.circle(available, (int(round(x)), int(round(y))),
                           int(self.cfg.active_min_dist), 0, -1)
            known = available > 0
        if not known.any():
            return None, None, "no_new_position"

        predicted_fg = p >= .5
        contradiction = ((fg & ~predicted_fg) | (bg & predicted_fg)) & known
        uncertainty = 1.0 - np.abs(2*p-1)
        if self.cfg.active_conflict_first and contradiction.any():
            # Select most confidently WRONG known point, instead of ambiguous only.
            disagreement_strength = np.where(fg, 1-p, p)
            score = np.where(contradiction, disagreement_strength, -1)
            kind = "scribble_conflict"
        else:
            score = np.where(known, uncertainty, -1)
            kind = "uncertainty"
        y, x = np.unravel_index(int(score.argmax()), score.shape)
        return np.array([[float(x), float(y)]], np.float32), \
               np.asarray([1 if fg[y, x] else 0], np.int32), kind

    def refine_from_maskbox(self, points, labels, scribble):
        """1st decoding: no box. Refinement: mask-derived box from first prediction."""
        base, memory = self._predict_once(points, labels, None, None)
        box, source = mask_to_box(base, self.cfg, scribble)
        current = points.copy() if points is not None else None
        labs = labels.copy() if labels is not None else None
        point_records = []
        refined = base
        for step in range(self.cfg.refine_iters):
            # active points are sampled from original known scribbles, not limited by pseudo box
            if self.cfg.active_points:
                extra, extra_lab, reason = self.sample_active_point(refined, scribble, current, roi=None)
                if extra is not None:
                    if current is None:
                        current, labs = extra, extra_lab
                    else:
                        current = np.concatenate((current, extra), axis=0)
                        labs = np.concatenate((labs, extra_lab), axis=0)
                    point_records.append({"step":step+1,"x":int(extra[0,0]),"y":int(extra[0,1]),"label":int(extra_lab[0]),"reason":reason})
            refined, memory = self._predict_once(current, labs, box, memory)
        return base, refined, box, source, point_records

class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.engine = SAM2Engine(cfg)
        for suffix in ("base_mask", "refine_mask", "visualizations"):
            os.makedirs(os.path.join(cfg.out_dir,suffix),exist_ok=True)
        if cfg.save_probs:
            for suffix in ("base_prob", "refine_prob"):
                os.makedirs(os.path.join(cfg.out_dir,suffix),exist_ok=True)

    def process_image(self, image_path):
        cfg = self.cfg
        basename = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return None
        h, w = image.shape[:2]
        scribble_path = DataLoader.find_file(cfg.scribble_dir, basename, (".png", ".jpg"))
        if not scribble_path:
            return None
        sc = cv2.imread(scribble_path, cv2.IMREAD_GRAYSCALE)
        if sc is None:
            return None
        if sc.shape != (h,w):
            sc = cv2.resize(sc,(w,h),interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        model_rgb = cv2.resize(rgb,(cfg.size,cfg.size),interpolation=cv2.INTER_LINEAR)
        model_sc = cv2.resize(sc,(cfg.size,cfg.size),interpolation=cv2.INTER_NEAREST)
        pts, labs = DataLoader.points_from_scribble(model_sc,cfg.max_initial_per_class)
        # First pass: no box at all; second pass: pseudo box from initial predicted mask.
        self.engine.set_image(model_rgb)
        base, refined, box, box_source, records = self.engine.refine_from_maskbox(pts,labs,model_sc)
        base_orig = resize_prob(base,w,h)
        refined_orig = resize_prob(refined,w,h)
        for dirname, prob in (("base_mask",base_orig),("refine_mask",refined_orig)):
            cv2.imwrite(os.path.join(cfg.out_dir,dirname,basename+".png"),((prob >= .5)*255).astype(np.uint8))
        if cfg.save_probs:
            for dirname,prob in (("base_prob",base_orig),("refine_prob",refined_orig)):
                np.save(os.path.join(cfg.out_dir,dirname,basename+".npy"),prob.astype(np.float16))
        record = {"image_name":os.path.basename(image_path),"base_name":basename,
                  "box_source":box_source,"pseudo_box_x1":float(box[0]) if box is not None else "",
                  "pseudo_box_y1":float(box[1]) if box is not None else "",
                  "pseudo_box_x2":float(box[2]) if box is not None else "",
                  "pseudo_box_y2":float(box[3]) if box is not None else "",
                  "num_active_points":len(records),
                  "num_conflict_points":sum(p["reason"] == "scribble_conflict" for p in records),
                  "num_uncertain_points":sum(p["reason"] == "uncertainty" for p in records)}
        gt_path = DataLoader.find_file(cfg.gt_dir,basename,(".png",".jpg"))
        if gt_path:
            gt = cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                for prefix,prob in (("base",base_orig),("refine",refined_orig)):
                    # binary at 0.5, matching semantics of original file.
                    metric = MetricsEvaluator.evaluate((prob>=.5).astype(np.uint8)*255,gt)
                    for name,value in metric.items():
                        record[prefix+"_"+name] = value
        return record


def summarize(rows,prefix):
    result={}
    for k in ("Smeasure","maxE","maxF","MAE"):
        arr=[r[prefix+"_"+k] for r in rows if prefix+"_"+k in r]
        result[k] = float(np.mean(arr)) if arr else None
    return result


def self_test():
    cfg=Config()
    fg=np.zeros((80,100),dtype=np.float32)
    fg[20:40,35:55]=1.0
    box,kind=mask_to_box(fg,cfg)
    assert kind == "mask" and box is not None and box[0]<35 and box[2]>54
    blank=np.zeros_like(fg)
    sc=np.zeros_like(fg,dtype=np.uint8)
    sc[30,40]=1
    box,kind=mask_to_box(blank,cfg,sc)
    assert kind == "scribble_fg" and box is not None
    box,kind=mask_to_box(blank,cfg,np.zeros_like(sc))
    assert box is None and kind == "none"
    print("PASS: pseudo-mask box, scribble fallback, no-box fallback")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sam2-checkpoint", default=Config.sam2_checkpoint)
    ap.add_argument("--sam2-config",default=Config.sam2_config)
    ap.add_argument("--img-dir",default=Config.img_dir)
    ap.add_argument("--scribble-dir",default=Config.scribble_dir)
    ap.add_argument("--gt-dir",default=Config.gt_dir)
    ap.add_argument("--output-dir",default=Config.out_dir)
    ap.add_argument("--refine-iters",type=int,default=2)
    ap.add_argument("--no-active",action="store_true")
    ap.add_argument("--box-threshold",type=float,default=.5)
    ap.add_argument("--box-margin-ratio",type=float,default=.05)
    ap.add_argument("--box-min-component-area",type=int,default=16)
    ap.add_argument("--save-probs",action="store_true")
    ap.add_argument("--limit",type=int,default=None)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:
        self_test()
        return
    cfg=Config(sam2_checkpoint=args.sam2_checkpoint,sam2_config=args.sam2_config,
               img_dir=args.img_dir,scribble_dir=args.scribble_dir,gt_dir=args.gt_dir,
               out_dir=args.output_dir,refine_iters=args.refine_iters,
               active_points=not args.no_active,box_threshold=args.box_threshold,
               box_margin_ratio=args.box_margin_ratio,
               box_min_component_area=args.box_min_component_area,save_probs=args.save_probs)
    if cfg.refine_iters < 0 or not 0 <= cfg.box_threshold <= 1 or cfg.box_margin_ratio < 0 or cfg.box_min_component_area < 1:
        ap.error("invalid refinement/box parameters")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pipeline=Pipeline(cfg)
    paths=DataLoader.list_images(cfg.img_dir)
    if args.limit is not None:
        paths=paths[:args.limit]
    rows=[]; failures=[]
    for path in tqdm(paths,desc="SAM2 scribble -> maskbox refinement"):
        try:
            r=pipeline.process_image(path)
            if r is not None:
                rows.append(r)
        except Exception as exc:
            failures.append({"image":os.path.basename(path),"error":repr(exc)})
            print("[ERROR]",os.path.basename(path),repr(exc))
    if rows:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(cfg.out_dir,"metrics.csv"),"w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)
    summary={"Base":summarize(rows,"base"),"Refine_MaskBox":summarize(rows,"refine"),
             "Diagnostics":{"num_images":len(rows),"num_failures":len(failures),
                            "mask_boxes":sum(r["box_source"]=="mask" for r in rows),
                            "scribble_fallback_boxes":sum(r["box_source"]=="scribble_fg" for r in rows),
                            "no_boxes":sum(r["box_source"]=="none" for r in rows),
                            "num_added_points":sum(r["num_active_points"] for r in rows)},
             "MetricNote":"maxE/maxF are legacy names for binary E/F at threshold 0.5, not true maxima.",
             "Config":vars(cfg),"Failures":failures}
    with open(os.path.join(cfg.out_dir,"summary_metrics.json"),"w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({"Base":summary["Base"],"Refine_MaskBox":summary["Refine_MaskBox"],"Diagnostics":summary["Diagnostics"]},indent=2))

if __name__ == "__main__":
    main()
