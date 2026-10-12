#!/usr/bin/env python3
"""Box + human scribble SAM2 refinement. GT is evaluation-only. No flip TTA."""
import argparse

import csv

import json

import os

import random

from contextlib import nullcontext

from dataclasses import dataclass

from glob import glob

from typing import Dict, List, Optional, Tuple



import cv2

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from tqdm import tqdm





@dataclass
class Config:
    sam2_checkpoint: str = "./sam_mask/weights/sam2_hiera_base_plus.pt"
    sam2_config: str = "sam2_hiera_b+.yaml"
    device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    img_dir: str = "./data/Train/Imgs"
    scribble_dir: str = "./data/Train/Scribble"
    gt_dir: str = "./data/Train/GT"
    json_dir: str = "./data/Train/box_label"
    out_dir: str = "./softresult/SAM2_ReliableRefine"
    size: int = 1024
    seed: int = 0
    refine_iters: int = 3
    active_min_dist: int = 16
    max_initial_per_class: int = 9
    use_crop: bool = True
    small_box_area_thr: float = .10
    crop_context: float = .35
    crop_min_context: int = 16
    crop_square: bool = True
    box_prompt_margin: int = 5
    auto_points: bool = False
    use_features: bool = True
    stable_iou: float = .995
    feature_margin: float = .10
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





def union_box(boxes):

    a = np.asarray(boxes, np.float32)

    return np.asarray([a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()], np.float32)





def clip_box(box, width, height):

    b = np.asarray(box, np.float32).copy()

    b[[0, 2]] = np.clip(b[[0, 2]], 0, width - 1)

    b[[1, 3]] = np.clip(b[[1, 3]], 0, height - 1)

    return b





def expand_box(box, pad, width, height):

    b = np.asarray(box, np.float32).copy()

    b += np.asarray([-pad, -pad, pad, pad], np.float32)

    return clip_box(b, width, height)





def centered_square_crop(box, width, height, context, min_context, square=True):

    """Returns integer half-open crop [x0,y0,x1,y1] in ORIGINAL image coords."""

    x1, y1, x2, y2 = map(float, box)

    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

    pad = max(float(min_context), max(bw, bh) * float(context))

    crop_w = bw + 2 * pad

    crop_h = bh + 2 * pad

    if square:

        crop_w = crop_h = max(crop_w, crop_h)

    crop_w = min(width, int(np.ceil(crop_w)))

    crop_h = min(height, int(np.ceil(crop_h)))

    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

    x0 = int(np.clip(np.round(cx - crop_w / 2), 0, width - crop_w))

    y0 = int(np.clip(np.round(cy - crop_h / 2), 0, height - crop_h))

    return x0, y0, x0 + crop_w, y0 + crop_h





def transform_box_to_model(box, crop_rect, model_size, margin):

    x0, y0, x1, y1 = crop_rect

    sx, sy = model_size / (x1 - x0), model_size / (y1 - y0)

    b = np.asarray(box, np.float32)

    b = np.asarray([(b[0] - x0 + .5) * sx - .5 - margin,

                    (b[1] - y0 + .5) * sy - .5 - margin,

                    (b[2] - x0 + .5) * sx - .5 + margin,

                    (b[3] - y0 + .5) * sy - .5 + margin], np.float32)

    return clip_box(b, model_size, model_size)





def parse_known_scribble_error(prob, scribble):

    if scribble is None:

        return float("nan")

    fg, bg = DataLoader.parse_scribble(scribble)

    known = fg | bg

    if not known.any():

        return float("nan")

    binary = prob >= 0.5

    incorrect = (fg & ~binary) | (bg & binary)

    return float(incorrect.sum() / known.sum())





def binary_iou(a, b):

    a, b = np.asarray(a, bool), np.asarray(b, bool)

    union = int((a | b).sum())

    return float((a & b).sum() / union) if union else 1.0





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






def conflicts(prob, sc):
    fg, bg = DataLoader.parse_scribble(sc)
    hard = prob >= .5
    return {'fg_miss': int((fg & ~hard).sum()),
            'bg_false': int((bg & hard).sum()),
            'fg_total': int(fg.sum()), 'bg_total': int(bg.sum())}


def no_worse(a, b):
    return a['fg_miss'] <= b['fg_miss'] and a['bg_false'] <= b['bg_false']


class Engine:
    def __init__(self, cfg, predictor=None):
        self.cfg = cfg
        if predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            predictor = SAM2ImagePredictor(build_sam2(cfg.sam2_config, cfg.sam2_checkpoint, device=cfg.device))
        self.predictor = predictor

    def set_image(self, image):
        with (torch.inference_mode() if torch is not None else nullcontext()):
            self.predictor.set_image(np.ascontiguousarray(image))

    def predict(self, points, labels, box, memory=None):
        ctx = torch.autocast('cuda', dtype=torch.bfloat16) if torch is not None and self.cfg.device.startswith('cuda') and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else nullcontext()
        with (torch.inference_mode() if torch is not None else nullcontext()), ctx:
            high, scores, low = self.predictor.predict(point_coords=points, point_labels=labels,
                box=box[None], mask_input=memory, multimask_output=True, return_logits=True)
        high, low = np.asarray(high, np.float32), np.asarray(low, np.float32)
        scores = np.asarray(scores, np.float32).reshape(-1)
        if high.ndim == 4 and high.shape[0] == 1: high = high[0]
        if low.ndim == 4 and low.shape[0] == 1: low = low[0]
        if high.ndim != 3 or low.ndim != 3 or len(high) != len(low):
            raise ValueError(f'Unexpected SAM2 candidate shapes {high.shape}, {low.shape}')
        return sigmoid(high), scores, low

    def features(self, size):
        """Public embedding getter; no RGB replacement masquerading as SAM features."""
        if not hasattr(self.predictor, 'get_image_embedding'): return None
        with torch.inference_mode():
            feat = self.predictor.get_image_embedding().detach().float()
            # Keep native embedding resolution to avoid a gigabyte-sized dense feature map.
            feat = torch.nn.functional.normalize(feat, dim=1)
        return feat[0].cpu().numpy()


class Pipeline:
    def __init__(self, cfg, predictor=None):
        self.cfg, self.engine = cfg, Engine(cfg, predictor)
        self.root = cfg.out_dir
        for name in ('Base', 'PerBox_Initial', 'Final', 'history', 'visualizations'):
            os.makedirs(os.path.join(self.root, name), exist_ok=True)

    def frame(self, rgb, sc, box, region):
        x0,y0,x1,y1 = region
        n=self.cfg.size
        image=cv2.resize(rgb[y0:y1,x0:x1], (n,n))
        fg,bg=DataLoader.parse_scribble(sc)
        # Restrict positive scribbles to this object; retain real background context.
        roi=np.zeros(sc.shape,bool)
        a,b,c,d=np.round(box).astype(int); roi[b:d+1,a:c+1]=True
        local=np.zeros(sc.shape,np.uint8); local[fg & roi]=1; local[bg]=2
        local=cv2.resize(local[y0:y1,x0:x1],(n,n),interpolation=cv2.INTER_NEAREST)
        mb=transform_box_to_model(box,region,n,self.cfg.box_prompt_margin)
        self.engine.set_image(image)
        feat=self.engine.features(n) if self.cfg.auto_points and self.cfg.use_features else None
        return local, mb, feat

    @staticmethod
    def restore(p, region, shape):
        out=np.zeros(shape,np.float32)
        x0,y0,x1,y1=region
        out[y0:y1,x0:x1]=resize_prob(p,x1-x0,y1-y0)
        return out

    def choose(self, probs, scores, lows, sc, previous=None):
        valid=[]
        for i,p in enumerate(probs):
            if not np.isfinite(p).all() or not np.isfinite(lows[i]).all(): continue
            c=conflicts(p,sc)
            if c['fg_total'] and not (p>=.5).any(): continue
            if previous is not None and not no_worse(c, previous['conflicts']): continue
            # Balanced human-error rank; SAM score only breaks ties.
            e=c['fg_miss']/max(1,c['fg_total'])+c['bg_false']/max(1,c['bg_total'])
            valid.append((e, -float(scores[i]) if np.isfinite(scores[i]) else 0., i,c))
        if not valid: return None
        _,_,i,c=min(valid)
        return {'prob':probs[i].copy(),'memory':lows[i:i+1].copy(),'conflicts':c,
                'candidate':i,'sam_score':float(scores[i]) if np.isfinite(scores[i]) else None}

    def next_point(self, state, sc, pts, rejected, probs, reference, feat, box):
        p=state['prob']; fg,bg=DataLoader.parse_scribble(sc)
        available=np.ones(p.shape,np.uint8)
        for x,y in list(pts if pts is not None else [])+rejected:
            cv2.circle(available,(round(float(x)),round(float(y))),self.cfg.active_min_dist,0,-1)
        wrong=((fg & (p<.5)) | (bg & (p>=.5))) & (available>0)
        if wrong.any():
            strength=np.where(fg,1-p,p)
            y,x=np.unravel_index(np.where(wrong,strength,-1).argmax(),p.shape)
            return np.array([[x,y]],np.float32),np.array([int(fg[y,x])],np.int32),'human_conflict'
        if not self.cfg.auto_points or reference is None or len(probs)<3: return None
        pos=(probs>.9).all(0) & (reference>.9)
        neg=(probs<.1).all(0) & (reference<.1)
        # Auto-points require both human feature prototypes by default.
        if self.cfg.use_features:
            if feat is None or not fg.any() or not bg.any(): return None
            fh,fw=feat.shape[1:]
            # Thin scribbles need coordinate pooling rather than disappearing on resize.
            fy,fx=np.where(fg); by,bx=np.where(bg)
            pf=feat[:,np.minimum(fh-1,fy*fh//fg.shape[0]),np.minimum(fw-1,fx*fw//fg.shape[1])].mean(1)
            pb=feat[:,np.minimum(fh-1,by*fh//bg.shape[0]),np.minimum(fw-1,bx*fw//bg.shape[1])].mean(1)
            pf/=max(np.linalg.norm(pf),1e-8); pb/=max(np.linalg.norm(pb),1e-8)
            if float(pf@pb)>.95: return None
            delta=cv2.resize(np.einsum('chw,c->hw',feat,pf-pb),(p.shape[1],p.shape[0]),interpolation=cv2.INTER_LINEAR)
            pos &= delta>self.cfg.feature_margin
            neg &= delta<-self.cfg.feature_margin
        roi=np.zeros(p.shape,np.uint8)
        a,b,c,d=np.round(box).astype(int); roi[b:d+1,a:c+1]=1
        # Negative points only in nearby context; never on another object's FG.
        context=cv2.dilate(roi,np.ones((65,65),np.uint8))>0
        pos &= (roi>0) & ~bg & ~fg
        neg &= context & ~fg & ~bg
        best=None
        for reg,label in ((pos,1),(neg,0)):
            interior=cv2.distanceTransform(reg.astype(np.uint8),cv2.DIST_L2,5)
            usable=(interior>=3)&(available>0)
            # A point should carry corrective information, not restate current mask.
            usable &= (p<.5) if label else (p>=.5)
            if usable.any():
                y,x=np.unravel_index(np.where(usable,interior,-1).argmax(),p.shape)
                cand=(float(interior[y,x]),x,y,label)
                if best is None or cand[0]>best[0]: best=cand
        if best is None:return None
        _,x,y,label=best
        return np.array([[x,y]],np.float32),np.array([label],np.int32),'auto_consensus'

    def run_view(self,rgb,sc,box,region,name,history_dir,reference_orig=None):
        local,mb,feat=self.frame(rgb,sc,box,region)
        pts,labs=DataLoader.points_from_scribble(local,self.cfg.max_initial_per_class)
        reference=None
        if reference_orig is not None:
            x0,y0,x1,y1=region
            reference=resize_prob(reference_orig[y0:y1,x0:x1],self.cfg.size,self.cfg.size)
        probs,scores,lows=self.engine.predict(pts,labs,mb)
        state=self.choose(probs,scores,lows,local)
        if state is None: raise RuntimeError('No valid initial SAM2 candidate')
        initial=self.restore(state['prob'],region,sc.shape)
        records=[]; accepted=[]; rejected=[]; fingerprints=[]
        def save(st,step,status,source,reason):
            out=self.restore(st['prob'],region,sc.shape)
            cv2.imwrite(os.path.join(history_dir,f'{name}_{step:02d}_{status}.png'),(out>=.5).astype(np.uint8)*255)
            prev=accepted[-1]['prob'] if accepted else st['prob']
            c=st['conflicts']
            rec={'view':name,'step':step,'status':status,'source':source,'reason':reason,
                 'area':int((out>=.5).sum()),'iou_previous':binary_iou(st['prob']>=.5,prev>=.5),
                 'fg_conflict_rate':c['fg_miss']/c['fg_total'] if c['fg_total'] else None,
                 'bg_conflict_rate':c['bg_false']/c['bg_total'] if c['bg_total'] else None,
                 **c,'candidate':st['candidate'],'sam_score':st['sam_score'],'region':list(region)}
            records.append(rec)
            if self.cfg.save_probs: np.save(os.path.join(history_dir,f'{name}_{step:02d}_{status}.npy'),out.astype(np.float16))
        save(state,0,'accepted','initial','initial_candidate')
        accepted.append({**state,'step':0}); fingerprints.append(state['prob']>=.5)
        stop='max_iterations'
        for step in range(1,self.cfg.refine_iters+1):
            new=self.next_point(state,local,pts,rejected,probs,reference,feat,mb)
            if new is None:
                stop='no_new_corrective_information'; break
            extra,label,source=new
            trial_pts=extra if pts is None else np.concatenate((pts,extra))
            trial_labs=label if labs is None else np.concatenate((labs,label))
            trial_probs,trial_scores,trial_lows=self.engine.predict(trial_pts,trial_labs,mb,state['memory'])
            trial=self.choose(trial_probs,trial_scores,trial_lows,local,state)
            if trial is None and source=='human_conflict':
                # Human label survives; clear potentially wrong mask prior.
                trial_probs,trial_scores,trial_lows=self.engine.predict(trial_pts,trial_labs,mb,None)
                trial=self.choose(trial_probs,trial_scores,trial_lows,local,state)
            if trial is None:
                raw=self.choose(trial_probs,trial_scores,trial_lows,local)
                if raw is not None:save(raw,step,'rejected',source,'human_conflict_increased')
                else: records.append({'view':name,'step':step,'status':'rejected','source':source,'reason':'hard_invalid'})
                rejected.append(extra[0].tolist())
                if source=='human_conflict':pts,labs=trial_pts,trial_labs
                # state/logits untouched. Human retry uses no memory next round.
                if source=='human_conflict':state={**state,'memory':None}
                continue
            iou=binary_iou(trial['prob']>=.5,state['prob']>=.5)
            improved=trial['conflicts']['fg_miss']<state['conflicts']['fg_miss'] or trial['conflicts']['bg_false']<state['conflicts']['bg_false']
            if source=='auto_consensus' and not improved:
                # Require independent fixed-view support on changed pixels.
                changed=(trial['prob']>=.5)!=(state['prob']>=.5)
                if not changed.any() or not np.all((trial['prob'][changed]>=.5)==(reference[changed]>=.5)):
                    save(trial,step,'rejected',source,'no_fixed_view_support');rejected.append(extra[0].tolist());continue
            save(trial,step,'accepted',source,'human_improvement' if improved else 'fixed_view_supported')
            records[-1]['point']={'x':float(extra[0,0]),'y':float(extra[0,1]),'label':int(label[0]),'source':source}
            if any(binary_iou(trial['prob']>=.5,old)>=self.cfg.stable_iou for old in fingerprints[:-1]):
                stop='oscillation'; break
            state=trial; pts,labs=trial_pts,trial_labs;probs=trial_probs
            accepted.append({**state,'step':step});fingerprints.append(state['prob']>=.5)
            if iou>=self.cfg.stable_iou and not improved:
                stop='stable_without_human_improvement';break
        # Earlier wins equal human evidence; auto changes only win fixed support.
        def rank(st):
            c=st['conflicts']; error=c['fg_miss']/max(1,c['fg_total'])+c['bg_false']/max(1,c['bg_total'])
            support=0.
            if reference is not None:
                confident=(reference>.9)|(reference<.1)
                support=float(((st['prob']>=.5)==(reference>=.5))[confident].mean()) if confident.any() else 0.
            return error,-support,st['step']
        selected=min(accepted,key=rank)
        return self.restore(selected['prob'],region,sc.shape),initial,records,{'view':name,'selected_step':selected['step'],'stop_reason':stop,'conflicts':selected['conflicts'],'region':list(region)}

    def object(self,rgb,sc,box,index,hdir):
        h,w=sc.shape; full=(0,0,w,h)
        gp,base,logs,sel=self.run_view(rgb,sc,box,full,f'box{index}_global',hdir)
        candidates=[(gp,sel)]; selections=[sel]
        area=(box[2]-box[0])*(box[3]-box[1])/(w*h)
        if self.cfg.use_crop and area<=self.cfg.small_box_area_thr:
            region=centered_square_crop(box,w,h,self.cfg.crop_context,self.cfg.crop_min_context,True)
            if region!=full:
                cp,_,cl,cs=self.run_view(rgb,sc,box,region,f'box{index}_crop',hdir,gp)
                logs+=cl; candidates.append((cp,cs));selections.append(cs)
                x0,y0,x1,y1=region; m=cp[y0:y1,x0:x1]>=.5
                edge=m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any()
                if edge or binary_iou(cp>=.5,gp>=.5)<.75:
                    expanded=centered_square_crop(box,w,h,self.cfg.crop_context*2,self.cfg.crop_min_context*2,True)
                    if expanded!=region:
                        ep,_,el,es=self.run_view(rgb,sc,box,expanded,f'box{index}_crop_expand',hdir,gp)
                        logs+=el;candidates.append((ep,es));selections.append(es)
        fg,bg=DataLoader.parse_scribble(sc);roi=np.zeros(sc.shape,bool)
        a,b,c,d=np.round(box).astype(int);roi[b:d+1,a:c+1]=True
        relevant=np.zeros(sc.shape,np.uint8); relevant[fg&roi]=1;relevant[bg]=2
        chosen=candidates[0]
        best=conflicts(chosen[0],relevant)
        for candidate in candidates[1:]:
            cc=conflicts(candidate[0],relevant)
            # Crop must improve human evidence; otherwise retain global.
            if no_worse(cc,best) and (cc['fg_miss']<best['fg_miss'] or cc['bg_false']<best['bg_false']):
                chosen=candidate;best=cc
        return chosen[0],base,logs,{'box':index,'selected_view':chosen[1]['view'],'views':selections,'selection_reason':'human_improvement_else_global'}

    def process(self,path):
        cfg=self.cfg;name=os.path.splitext(os.path.basename(path))[0]
        image=cv2.imread(path);assert image is not None,'Unreadable image'
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);h,w=image.shape[:2]
        sp=DataLoader.find_file(cfg.scribble_dir,name,('.png','.jpg'))
        jp=DataLoader.find_file(cfg.json_dir,name,('.json',))
        if sp is None:raise ValueError('Missing human scribble')
        sc=cv2.imread(sp,0)
        if sc is None:raise ValueError('Unreadable scribble')
        sc=cv2.resize(sc,(w,h),interpolation=cv2.INTER_NEAREST)
        boxes=DataLoader.load_boxes(jp,w,h)
        if not boxes:raise ValueError('Missing valid box')
        hdir=os.path.join(self.root,'history',name);os.makedirs(hdir,exist_ok=True)
        # Legacy union-box initial-prompt baseline, no crop/refinement.
        local,mb,_=self.frame(rgb,sc,union_box(boxes),(0,0,w,h))
        pts,labs=DataLoader.points_from_scribble(local,cfg.max_initial_per_class)
        pp,ss,ll=self.engine.predict(pts,labs,mb); st=self.choose(pp,ss,ll,local)
        if st is None:raise RuntimeError('No valid union baseline')
        base=resize_prob(st['prob'],w,h)
        initial=np.zeros((h,w),np.float32);final=initial.copy();logs=[];decisions=[]
        for i,box in enumerate(boxes):
            p,b,l,d=self.object(rgb,sc,box,i,hdir)
            # Gate per-object contributions, without converting unknown pixels to prompts.
            gate=np.zeros((h,w),np.float32)
            margin=max(2,round(25*max(h,w)/cfg.size))
            x1,y1,x2,y2=np.round(expand_box(box,margin,w,h)).astype(int)
            gate[y1:y2+1,x1:x2+1]=1
            final=np.maximum(final,p*gate);initial=np.maximum(initial,b*gate)
            logs+=l;decisions.append(d)
        # Validate post-merge too: merge gate can remove a valid foreground scribble.
        if not no_worse(conflicts(final,sc),conflicts(initial,sc)):
            final=initial.copy();decisions.append({'merge_rollback':'human_conflict_increased'})
        for folder,p in (('Base',base),('PerBox_Initial',initial),('Final',final)):
            cv2.imwrite(os.path.join(self.root,folder,name+'.png'),(p>=.5).astype(np.uint8)*255)
        with open(os.path.join(hdir,'iterations.json'),'w',encoding='utf-8') as f:
            json.dump({'iterations':logs,'decisions':decisions},f,ensure_ascii=False,indent=2,allow_nan=False)
        record={'image':name,'num_boxes':len(boxes),**conflicts(final,sc)}
        gtpath=DataLoader.find_file(cfg.gt_dir,name,('.png','.jpg')) if cfg.gt_dir else None
        if gtpath:
            gt=cv2.imread(gtpath,0)
            if gt is not None:
                for prefix,p in (('base',base),('perbox',initial),('final',final)):
                    for key,val in MetricsEvaluator.evaluate((p>=.5).astype(np.uint8)*255,gt).items():
                        record[f'{prefix}_{key}']=val
                panels=[cv2.resize(image,(256,256))]
                for label,p in (('GT',gt),('Base',base),('PerBox',initial),('Final',final)):
                    mask=p if label=='GT' else (p>=.5).astype(np.uint8)*255
                    panel=cv2.cvtColor(cv2.resize(mask,(256,256),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
                    cv2.putText(panel,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),1);panels.append(panel)
                cv2.imwrite(os.path.join(self.root,'visualizations',name+'.png'),np.concatenate(panels,1))
        return record


def self_test():
    import tempfile
    cfg=Config(size=64,device='cpu',refine_iters=3,use_crop=False)
    class Stub:
        def set_image(self,img):pass
        def predict(self,**kw):
            p=np.full((3,64,64),-4.,np.float32);p[:,16:48,16:48]=4.
            # A newly introduced point causes a catastrophic empty foreground.
            if kw['point_coords'] is not None and len(kw['point_coords'])>2:p[:]=-4.
            return p,np.array([.9,.8,.7]),np.zeros((3,16,16),np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        cfg.out_dir=tmp; pipe=Pipeline(cfg,Stub())
        rgb=np.zeros((64,64,3),np.uint8);sc=np.zeros((64,64),np.uint8)
        sc[20,20]=1;sc[52,52]=1;sc[5,5]=2
        cfg.max_initial_per_class=1
        out,initial,logs,selection=pipe.run_view(rgb,sc,np.array([10,10,60,60]),(0,0,64,64),'test',tmp)
        assert np.array_equal(out,initial),'Rejected prediction must not replace valid history'
        assert any(x['status']=='rejected' for x in logs),logs
        assert selection['selected_step']==0
        p=np.zeros((16,16));a=np.zeros((16,16),np.uint8);a[4,4]=1;a[2,2]=2
        c=conflicts(p,a);q=p.copy();q[4,4]=1
        assert no_worse(conflicts(q,a),c)
        q[2,2]=1;assert not no_worse(conflicts(q,a),c)
        assert DataLoader.parse_scribble(np.zeros((2,2),np.uint8))[1].sum()==0
        # No informative points stops without memory-only repeats.
        sc=np.zeros((64,64),np.uint8);sc[20,20]=1
        _,_,logs,selection=pipe.run_view(rgb,sc,np.array([10,10,60,60]),(0,0,64,64),'stable',tmp)
        assert len(logs)==1 and selection['stop_reason']=='no_new_corrective_information'
    print('PASS: rollback, human priority, unknown background, early stop')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('sam2_checkpoint','sam2_config','img_dir','scribble_dir','gt_dir','json_dir','device'):
        parser.add_argument('--'+key.replace('_','-'),default=getattr(Config,key))
    parser.add_argument('--output-dir',default=Config.out_dir)
    parser.add_argument('--refine-iters',type=int,default=3)
    parser.add_argument('--no-crop',action='store_true')
    parser.add_argument('--auto-points',action='store_true')
    parser.add_argument('--no-feature-gate',action='store_true',help='Ablation only: allow consensus without feature prototypes')
    parser.add_argument('--save-probs',action='store_true')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    if args.refine_iters<0:parser.error('--refine-iters must be nonnegative')
    cfg=Config(**{k:getattr(args,k) for k in ('sam2_checkpoint','sam2_config','img_dir','scribble_dir','gt_dir','json_dir','device')},out_dir=args.output_dir,
        refine_iters=args.refine_iters,use_crop=not args.no_crop,auto_points=args.auto_points,
        use_features=not args.no_feature_gate,save_probs=args.save_probs)
    if torch is None:raise RuntimeError('Real SAM2 inference requires torch')
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    pipeline=Pipeline(cfg);paths=DataLoader.list_images(cfg.img_dir)
    if args.limit is not None:paths=paths[:args.limit]
    records=[];failures=[]
    for path in tqdm(paths,desc='Reliable SAM2 refinement'):
        try:records.append(pipeline.process(path))
        except Exception as exc:
            failures.append({'image':os.path.basename(path),'error':str(exc)});print(f'[ERROR] {path}: {exc}')
    if records:
        with open(os.path.join(cfg.out_dir,'image_metrics.csv'),'w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)))
            writer.writeheader();writer.writerows(records)
    summary={'config':vars(cfg),'num_images':len(records),'failures':failures,
        'metric_note':'Inherited maxF/maxE are binary F/E at 0.5, not threshold-sweep maxima.'}
    for prefix in ('base','perbox','final'):
        summary[prefix]={}
        for key in ('Smeasure','maxE','maxF','MAE'):
            vals=[r[f'{prefix}_{key}'] for r in records if f'{prefix}_{key}' in r]
            summary[prefix][key]=float(np.mean(vals)) if vals else None
    with open(os.path.join(cfg.out_dir,'summary_metrics.json'),'w',encoding='utf-8') as f:
        json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
