#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCLCOD SAM2 reliability-controlled global/local refinement (NO FLIP).

Drop-in replacement filename retained for existing imports.  No flip inference,
flip fusion, or flip-related CLI flags remain.

Supervision modes:
  --prompt-mode scribble: Box + true human scribbles (NOT box-only WSCOD)
  --prompt-mode box:     Box + conservatively mined pseudo-points (box-only)

No ground truth is read or used in prompt generation/selection.  If available,
GT evaluation belongs in sam2_refine_quality_pool.py (--eval-gt).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def json_compatible(value):
    """Convert NumPy values (including non-finite values) to strict JSON types.

    SAM2 and quality metrics can return np.float32/np.float64/np.int64,
    which the standard json encoder cannot always serialize.
    """
    if isinstance(value, np.generic):
        return json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    """Write complete strict JSON atomically, avoiding truncated JSON on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_compatible(value), ensure_ascii=False,
                         indent=2, allow_nan=False)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(payload, encoding='utf-8')
    tmp.replace(path)


@dataclass
class Config:
    sam2_checkpoint: str = './sam_pseado/weights/sam2_hiera_large.pt'
    sam2_config: str = 'sam2_hiera_l.yaml'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_dir: str = './data/Train/Imgs'
    scribble_dir: str = './data/Train/Scribble'
    json_dir: str = './data/Train/box_label'
    gt_dir: str = './data/Train/GT'  # compatibility only; never used for selection
    out_dir: str = './softresult/SAM2_Optimized3'
    prompt_mode: str = 'scribble'       # 'box' requires no scribble file
    size: int = 1024
    refine_iters: int = 2
    active_points: bool = True
    auto_points: bool = True
    max_initial_per_class: int = 9
    active_min_dist: int = 16          # pixels at 1024 model resolution
    max_auto_positive: int = 2
    max_auto_negative: int = 2
    auto_feature_margin: float = 0.06 # SAM2 embedding prototype cosine margin
    use_crop: bool = True
    small_box_area_thr: float = 0.10
    crop_context: float = 0.35
    crop_min_context: int = 16
    box_prompt_margin: int = 5
    min_mask_box_ratio: float = .005
    max_mask_box_ratio: float = 1.50
    max_outside_ratio: float = .25
    early_iou: float = .995
    early_area_change: float = .005
    min_accept_gain: float = .01
    cross_view_min_iou: float = .75
    seed: int = 0
    save_probs: bool = True
    save_iterations: bool = False

    @property
    def base_dir(self): return str(Path(self.out_dir) / '01_SoftLogits_Base')

    @property
    def refine_dir(self): return str(Path(self.out_dir) / '03_Refine_NoFlip')


class DataLoader:
    @staticmethod
    def list_images(folder):
        p = Path(folder)
        return sorted([str(x) for x in p.iterdir() if x.suffix.lower() in
                       {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}]) if p.is_dir() else []

    @staticmethod
    def find_file(folder, name, extensions):
        folder = Path(folder)
        for ext in extensions:
            for e in (ext.lower(), ext.upper()):
                path = folder / f'{name}{e}'
                if path.is_file(): return str(path)
        return None

    @staticmethod
    def load_boxes(path, width, height):
        if not path or not os.path.isfile(path): return []
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        boxes = []
        for item in data.get('shapes', []):
            points = np.asarray(item.get('points', []), dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2: continue
            if not np.isfinite(points).all(): continue
            box = np.asarray([points[:, 0].min(), points[:, 1].min(),
                              points[:, 0].max(), points[:, 1].max()], np.float32)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width-1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height-1)
            if box[2] > box[0] and box[3] > box[1]: boxes.append(box)
        return boxes

    @staticmethod
    def parse_scribble(scribble):
        if scribble is None: return None, None
        s = np.asarray(scribble)
        # Important: for binary scribble zero is UNKNOWN, never background.
        if np.any(s == 1) or np.any(s == 2):
            return s == 1, s == 2
        return s > 128, np.zeros(s.shape, bool)

    @staticmethod
    def points_from_scribble(scribble, max_per_class=9):
        fg, bg = DataLoader.parse_scribble(scribble)
        if fg is None: return None, None
        coords, labels = [], []
        for region, label in ((fg, 1), (bg, 0)):
            ys, xs = np.where(region)
            if not len(xs): continue
            dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
            side = max(1, int(np.ceil(np.sqrt(max_per_class))))
            for xset in np.array_split(np.arange(xs.min(), xs.max()+1), side):
                if not len(xset): continue
                for yset in np.array_split(np.arange(ys.min(), ys.max()+1), side):
                    if not len(yset): continue
                    sel = np.isin(xs, xset) & np.isin(ys, yset)
                    if not sel.any(): continue
                    xp, yp = xs[sel], ys[sel]
                    ix = int(np.argmax(dist[yp, xp]))
                    coords.append([float(xp[ix]), float(yp[ix])]); labels.append(label)
                    if sum(v == label for v in labels) >= max_per_class: break
                if sum(v == label for v in labels) >= max_per_class: break
        return ((np.asarray(coords,np.float32), np.asarray(labels,np.int32))
                if coords else (None,None))


def sigmoid(x):
    x = np.clip(np.asarray(x, np.float32), -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def binary_iou(a, b):
    a, b = np.asarray(a,bool), np.asarray(b,bool)
    u = int(np.logical_or(a,b).sum())
    return float(np.logical_and(a,b).sum()/u) if u else 1.0


def union_box(boxes):
    b = np.asarray(boxes, np.float32)
    return np.array([b[:,0].min(), b[:,1].min(), b[:,2].max(), b[:,3].max()],np.float32)


def box_mask(box, h, w):
    m = np.zeros((h,w),bool)
    x1,y1,x2,y2 = box
    x1,x2 = np.clip([int(np.floor(x1)),int(np.ceil(x2))+1],0,w)
    y1,y2 = np.clip([int(np.floor(y1)),int(np.ceil(y2))+1],0,h)
    m[y1:y2,x1:x2] = True
    return m


def crop_region(box, w, h, context, min_context):
    x1,y1,x2,y2=map(float,box)
    side=max(x2-x1,y2-y1)+2*max(min_context,max(x2-x1,y2-y1)*context)
    cw=min(w,max(2,int(np.ceil(side))))
    ch=min(h,max(2,int(np.ceil(side))))
    cx,cy=(x1+x2)/2,(y1+y2)/2
    ox=int(np.clip(round(cx-cw/2),0,w-cw));oy=int(np.clip(round(cy-ch/2),0,h-ch))
    return (ox,oy,ox+cw,oy+ch)


def to_view_points(points, region, size):
    if points is None or len(points)==0: return None
    x0,y0,x1,y1=region
    a=np.asarray(points,np.float32).copy()
    a[:,0]=(a[:,0]-x0+.5)*(size/(x1-x0))-.5
    a[:,1]=(a[:,1]-y0+.5)*(size/(y1-y0))-.5
    return a


def from_view_mask(arr, region, shape, is_binary=False):
    h,w=shape; x0,y0,x1,y1=region
    out=np.zeros((h,w),np.uint8 if is_binary else np.float32)
    out[y0:y1,x0:x1]=cv2.resize(np.asarray(arr,np.uint8 if is_binary else np.float32),
                               (x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST if is_binary else cv2.INTER_LINEAR)
    return out


def binary_ssim(a,b):
    a,b=np.asarray(a,np.float32),np.asarray(b,np.float32)
    if np.array_equal(a,b): return 1.0
    c1,c2=.01**2,.03**2
    k=min(11,min(a.shape));k-=int(k%2==0)
    if k<3: return float(binary_iou(a>.5,b>.5))
    ma=cv2.GaussianBlur(a,(k,k),1.5);mb=cv2.GaussianBlur(b,(k,k),1.5)
    va=cv2.GaussianBlur(a*a,(k,k),1.5)-ma*ma
    vb=cv2.GaussianBlur(b*b,(k,k),1.5)-mb*mb
    vab=cv2.GaussianBlur(a*b,(k,k),1.5)-ma*mb
    x=((2*ma*mb+c1)*(2*vab+c2))/np.maximum((ma*ma+mb*mb+c1)*(va+vb+c2),1e-12)
    return float(np.clip(x.mean(),0,1))


def consensus(masks):
    sims=[]; d={}
    for i,j,s in ((0,1,'12'),(0,2,'13'),(1,2,'23')):
        iou=binary_iou(masks[i],masks[j])
        ssim=binary_ssim(masks[i],masks[j])
        d[f'candidate_iou_{s}']=iou;d[f'candidate_ssim_{s}']=ssim
        d[f'candidate_sim_{s}']=.5*(iou+ssim)
        sims.append(d[f'candidate_sim_{s}'])
    d.update(candidate_consensus_s=float(np.mean(sims)),
             candidate_sim_min=float(min(sims)),candidate_sim_max=float(max(sims)),
             candidate_sim_range=float(max(sims)-min(sims)))
    return d


def sample_separated_points(region, existing, max_count, min_dist):
    m=np.asarray(region,bool).astype(np.uint8)
    old=np.asarray(existing,np.float32).reshape(-1,2) if existing is not None else np.empty((0,2),np.float32)
    points=[]
    for _ in range(max_count):
        if not m.any(): break
        dist=cv2.distanceTransform(m,cv2.DIST_L2,5)
        y,x=np.unravel_index(int(np.argmax(dist)),dist.shape)
        if dist[y,x]<2: break
        xy=np.array([x,y],np.float32)
        if not len(old) or np.min(np.linalg.norm(old-xy,axis=1))>=min_dist:
            points.append(xy)
        cv2.circle(m,(x,y),int(min_dist),0,-1)
    return points


class SAM2Engine:
    def __init__(self,cfg,predictor=None):
        self.cfg=cfg
        if predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model=build_sam2(cfg.sam2_config,cfg.sam2_checkpoint,device=cfg.device)
            model.eval()
            predictor=SAM2ImagePredictor(model)
        self.predictor=predictor

    def set_image(self,img):
        self.predictor.set_image(np.ascontiguousarray(img))

    def predict_three(self,points,labels,box,memory=None,scribble=None):
        use_amp=self.cfg.device.startswith('cuda') and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        with torch.inference_mode():
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=use_amp) if self.cfg.device.startswith('cuda') and torch.cuda.is_available() else torch.no_grad():
                logit,score,low=self.predictor.predict(
                    point_coords=points,point_labels=labels,
                    box=np.asarray(box,np.float32)[None,:],mask_input=memory,
                    multimask_output=True,return_logits=True)
        logit=np.asarray(logit,np.float32)
        low=np.asarray(low,np.float32)
        score=np.asarray(score,np.float32).reshape(-1)
        if logit.ndim==2:logit=logit[None]
        if low.ndim==2:low=low[None]
        if logit.shape[0]!=3 or low.shape[0]!=3 or score.size!=3:
            raise RuntimeError(f'Expected 3 candidates: {logit.shape}, {low.shape}, {score.shape}')
        if not (np.isfinite(logit).all() and np.isfinite(low).all() and np.isfinite(score).all()):
            raise ValueError('NaN/Inf in SAM2 candidate results')
        thr=float(getattr(self.predictor,'mask_threshold',getattr(getattr(self.predictor,'model',None),'mask_threshold',0.0)))
        hard=logit>thr
        quality=consensus(hard)
        # Candidate selection is NOT solely highest predicted-IoU: known
        # human scribble contradictions and hard geometry failures take priority.
        fg,bg=DataLoader.parse_scribble(scribble)
        candidate_costs=[]
        for ci in range(3):
            cm=hard[ci]; ratio=float(cm.sum()/max(1,box_mask(box,*cm.shape).sum()))
            outside=float((cm & ~box_mask(box,*cm.shape)).sum()/max(1,cm.sum()))
            fail=(cm.sum()==0 or ratio<self.cfg.min_mask_box_ratio or
                  ratio>self.cfg.max_mask_box_ratio or outside>self.cfg.max_outside_ratio)
            errors=[]
            if fg is not None:
                if fg.any():errors.append(float(np.mean(~cm[fg])))
                if bg.any():errors.append(float(np.mean(cm[bg])))
            serr=float(np.mean(errors)) if errors else 0.0
            candidate_costs.append((int(fail),4.0*serr+1-float(np.clip(score[ci],0,1))))
        idx=int(min(range(3),key=lambda ci:candidate_costs[ci]))
        chosen=logit[idx]
        nlow=int((chosen>thr-1.0).sum())
        stab=float((chosen>thr+1.0).sum()/nlow) if nlow else 0.0
        quality.update(selected_candidate_index=idx,
                       sam_score_selected=float(np.clip(score[idx],0,1)),
                       sam_stability_selected=stab,
                       candidate_score_1=float(score[0]),candidate_score_2=float(score[1]),candidate_score_3=float(score[2]),
                       quality_frame='selected_refinement_decoder',quality_mask_threshold=thr)
        return dict(prob=sigmoid(chosen-thr),masks=hard,quality=quality,
                    memory=low[idx:idx+1].copy())

    def feature_margin(self,pack,model_box):
        """SAM2 embedding foreground-vs-background prototype margin (not RGB proxy).
        Returns None if the seed or negative prototypes are unreliable/unavailable.
        """
        fn=getattr(self.predictor,'get_image_embedding',None)
        if not callable(fn):return None
        try:
            e=fn().detach().float()
            if e.ndim==3:e=e.unsqueeze(0)
            if e.ndim!=4 or e.shape[0]!=1:return None
            _,c,hf,wf=e.shape
            common=np.logical_and.reduce(pack['masks'])
            fg=cv2.resize(common.astype(np.uint8),(wf,hf),interpolation=cv2.INTER_NEAREST).astype(bool)
            # Inner core only; do not use uncertain silhouettes as foreground prototypes.
            fg=cv2.erode(fg.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            bmask=box_mask(model_box,self.cfg.size,self.cfg.size)
            bg=~cv2.resize(bmask.astype(np.uint8),(wf,hf),interpolation=cv2.INTER_NEAREST).astype(bool)
            if fg.sum()<4 or bg.sum()<4:return None
            f=F.normalize(e,dim=1).squeeze(0).reshape(c,-1)
            fg_idx=torch.from_numpy(fg.reshape(-1).copy()).to(f.device)
            bg_idx=torch.from_numpy(bg.reshape(-1).copy()).to(f.device)
            fp=F.normalize(f[:,fg_idx].mean(1),dim=0)
            bp=F.normalize(f[:,bg_idx].mean(1),dim=0)
            if float((fp*bp).sum())>.995:return None
            margin=((fp-bp)@f).reshape(1,1,hf,wf)
            margin=F.interpolate(margin,size=(self.cfg.size,self.cfg.size),mode='bilinear',align_corners=False)
            return margin[0,0].cpu().numpy().astype(np.float32)
        except (RuntimeError,ValueError,TypeError,AttributeError):
            return None


class Pipeline:
    def __init__(self,cfg,predictor=None):
        self.cfg=cfg
        self.engine=SAM2Engine(cfg,predictor=predictor)
        self.last_objects=[]
        for d in (cfg.base_dir,cfg.refine_dir,str(Path(cfg.out_dir)/'refine_prob'),
                  str(Path(cfg.out_dir)/'base_prob')):
            Path(d).mkdir(parents=True,exist_ok=True)

    def _view(self,rgb,sc,box,cropped):
        h,w=rgb.shape[:2]
        region=crop_region(box,w,h,self.cfg.crop_context,self.cfg.crop_min_context) if cropped else (0,0,w,h)
        x0,y0,x1,y1=region
        img=cv2.resize(rgb[y0:y1,x0:x1],(self.cfg.size,self.cfg.size),interpolation=cv2.INTER_LINEAR)
        s=cv2.resize(sc[y0:y1,x0:x1],(self.cfg.size,self.cfg.size),interpolation=cv2.INTER_NEAREST) if sc is not None else None
        b=to_view_points(np.asarray(box,np.float32).reshape(2,2),region,self.cfg.size).reshape(4)
        b+=np.array([-self.cfg.box_prompt_margin,-self.cfg.box_prompt_margin,self.cfg.box_prompt_margin,self.cfg.box_prompt_margin],np.float32)
        b[[0,2]]=np.clip(b[[0,2]],0,self.cfg.size-1)
        b[[1,3]]=np.clip(b[[1,3]],0,self.cfg.size-1)
        if s is not None:
            fg,bg=DataLoader.parse_scribble(s)
            if fg is not None:
                s=s.copy(); s[fg & ~box_mask(b,self.cfg.size,self.cfg.size)]=0
        pts,labs=DataLoader.points_from_scribble(s,self.cfg.max_initial_per_class)
        return dict(img=img,sc=s,box=b,region=region,pts=pts,labs=labs,cropped=cropped)

    def _hard_invalid(self,prob,model_box):
        m=prob>=.5
        b=box_mask(model_box,*m.shape)
        area=int(m.sum()); ba=max(1,int(b.sum()))
        outside=float((m & ~b).sum()/max(area,1))
        ratio=area/ba
        reason=[]
        if area==0:reason.append('empty')
        if ratio<self.cfg.min_mask_box_ratio:reason.append('collapse')
        if ratio>self.cfg.max_mask_box_ratio:reason.append('overgrown')
        if outside>self.cfg.max_outside_ratio:reason.append('outside')
        if area>0:
            n,_,stats,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),8)
            parts=stats[1:,cv2.CC_STAT_AREA].astype(np.float32)
            effective=parts[parts>=max(16,round(.002*area))]
            if effective.size>=6 and effective.max()/max(1.0,effective.sum())<.55:
                reason.append('severe_fragmentation')
            ys,xs=np.where(m)
            bbox_area=float((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))
            rectangularity=area/max(1.0,bbox_area)
            mask_box=box_mask(model_box,*m.shape)
            coords=np.where(mask_box)
            if coords[0].size:
                bx0,bx1=coords[1].min(),coords[1].max()
                by0,by1=coords[0].min(),coords[0].max()
                ix=max(0,min(xs.max(),bx1)-max(xs.min(),bx0)+1)*max(0,min(ys.max(),by1)-max(ys.min(),by0)+1)
                bbox_iou=ix/max(1.0,bbox_area+mask_box.sum()-ix)
                if ratio>=.90 and bbox_iou>=.95 and rectangularity>=.95:
                    reason.append('box_copy')
            if area/m.size>.50 and mask_box.sum()/m.size<.30:
                reason.append('global_reversal')
        return reason,ratio,outside

    @staticmethod
    def _scribble_error(prob,s):
        fg,bg=DataLoader.parse_scribble(s)
        if fg is None:return 0.0,False
        m=prob>=.5
        # Class balanced error: prevent plentiful BG scribble overwhelming FG.
        errors=[]
        if fg.any():errors.append(float(np.mean(~m[fg])))
        if bg.any():errors.append(float(np.mean(m[bg])))
        return (float(np.mean(errors)),bool(errors)) if errors else (0.0,False)

    def _analyze(self,pack,view,prev_prob,step):
        prob=pack['prob']; q=pack['quality']
        invalid,ratio,outside=self._hard_invalid(prob,view['box'])
        err,has_s=self._scribble_error(prob,view['sc'])
        iou=binary_iou(prob>=.5,prev_prob>=.5) if prev_prob is not None else 1.0
        change=abs(float((prob>=.5).sum())-float((prev_prob>=.5).sum()))/max(1,float((prev_prob>=.5).sum())) if prev_prob is not None else 0.0
        # Lower is better; scribble contradiction given higher weight.
        cost=4.0*err+(1.0-q['sam_score_selected'])+0.3*(1-q['sam_stability_selected'])+0.2*(1-q['candidate_consensus_s'])+outside
        return dict(step=step,valid=not bool(invalid),invalid_reason=invalid,mask_box_ratio=ratio,
                    outside_ratio=outside,scribble_error=err,has_scribble=has_s,
                    iou_prev=iou,area_change_prev=change,mask_area=int((prob>=.5).sum()),cost=float(cost),
                    sam_score=q['sam_score_selected'],consensus=q['candidate_consensus_s'])

    def _human_next(self,pack,view,points):
        fg,bg=DataLoader.parse_scribble(view['sc'])
        if fg is None:return None
        m=pack['prob']>=.5; p=pack['prob']
        conflict=(fg & ~m)|(bg & m)
        if not conflict.any():return None
        strength=np.where(fg,1-p,p)
        ys,xs=np.where(conflict)
        order=np.argsort(-strength[ys,xs])
        for ix in order:
            point=np.array([float(xs[ix]),float(ys[ix])],np.float32)
            if points is None or not len(points) or np.linalg.norm(points-point,axis=1).min()>=self.cfg.active_min_dist:
                return point,int(fg[ys[ix],xs[ix]]),'scribble_conflict'
        return None

    def _make_auto_points(self,global_pack,global_view,local_pack,local_view,hw,box):
        if not self.cfg.auto_points:return []
        # Require all three candidates agree in both views (when crop exists).
        region_g=global_view['region']
        g_fg=from_view_mask(np.logical_and.reduce(global_pack['masks']),region_g,hw,True)>0
        g_bg=from_view_mask(~np.logical_or.reduce(global_pack['masks']),region_g,hw,True)>0
        gp=from_view_mask(global_pack['prob'],region_g,hw)
        if local_pack is not None:
            region_l=local_view['region']
            l_fg=from_view_mask(np.logical_and.reduce(local_pack['masks']),region_l,hw,True)>0
            l_bg=from_view_mask(~np.logical_or.reduce(local_pack['masks']),region_l,hw,True)>0
            lp=from_view_mask(local_pack['prob'],region_l,hw)
            fg=g_fg & l_fg & (gp>.8) & (lp>.8)
            bg=g_bg & l_bg
        else:
            fg=g_fg & (gp>.85)
            bg=g_bg
        h,w=hw
        bm=box_mask(box,h,w)
        fg &= bm
        if np.any(fg):
            k=max(3,int(round(max(h,w)/512)))|1
            fg=cv2.erode(fg.astype(np.uint8),np.ones((k,k),np.uint8)).astype(bool)
        # SAM2 feature prototypes from global image embedding provide an independent check.
        self.engine.set_image(global_view['img'])
        gm=self.engine.feature_margin(global_pack,global_view['box'])
        if gm is not None:
            margin=from_view_mask(gm,region_g,hw)
            fg &= margin>self.cfg.auto_feature_margin
            bg &= margin < -self.cfg.auto_feature_margin
        # Pseudo-background outside a relatively tight annotated box is safer.
        outside=~cv2.dilate(bm.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        bg &= outside
        if local_pack is not None:
            # local view cannot assert regions outside its support; global-only outside is acceptable.
            bg = g_bg & outside
        scale=max(h,w)/float(self.cfg.size)
        dist=max(4,int(round(self.cfg.active_min_dist*scale)))
        pos=sample_separated_points(fg,None,self.cfg.max_auto_positive,dist)
        neg=sample_separated_points(bg,np.asarray(pos,np.float32).reshape(-1,2) if pos else None,
                                    self.cfg.max_auto_negative,dist)
        return [(p,1,'auto_fg') for p in pos]+[(p,0,'auto_bg') for p in neg]

    def _refine_view(self,view,auto_points,hw,trace_prefix=None):
        self.engine.set_image(view['img'])
        current=None if view['pts'] is None else view['pts'].copy()
        labs=None if view['labs'] is None else view['labs'].copy()
        memory=None;prev=None;best=None;fallback=None;history=[]
        auto_queue=[]
        for orig_xy,label,src in auto_points:
            p=to_view_points(np.asarray(orig_xy,np.float32).reshape(1,2),view['region'],self.cfg.size)[0]
            if np.all((p>=0)&(p<self.cfg.size)):
                auto_queue.append((p,int(label),src))
        for step in range(self.cfg.refine_iters+1):
            pack=self.engine.predict_three(current,labs,view['box'],memory,scribble=view['sc'])
            info=self._analyze(pack,view,prev,step)
            if self.cfg.save_iterations and trace_prefix:
                trace_dir=Path(self.cfg.out_dir)/'iteration_masks'
                trace_dir.mkdir(parents=True,exist_ok=True)
                step_path=trace_dir/f'{trace_prefix}_it{step:02d}.png'
                step_mask=from_view_mask((pack['prob']>=.5).astype(np.uint8),view['region'],hw,True)
                if not cv2.imwrite(str(step_path),step_mask*255):
                    raise IOError(f'Cannot write iteration mask {step_path}')
                info['mask_path']=str(step_path)
            history.append(info)
            # Best snapshot is chosen only from valid iterations. Never keep a new hard-invalid state.
            if info['valid'] and (best is None or info['cost']+self.cfg.min_accept_gain<best['info']['cost']):
                best={'pack':pack,'info':info,'step':step}
            if not info['valid']:
                # A hard-invalid candidate must never silently become High/Middle.
                # Retain a fallback only so the image can be saved to Unvalue.
                rank=(len(info['invalid_reason']), info['cost'])
                if fallback is None or rank<fallback['rank']:
                    fallback={'pack':pack,'info':info,'step':step,'rank':rank}
            if not info['valid'] and step>0:
                info['stop_reason']='hard_invalid_rollback';break
            point=self._human_next(pack,view,current) if self.cfg.active_points else None
            if point is None and self.cfg.active_points:
                while auto_queue:
                    proposal=auto_queue.pop(0)
                    if current is None or not len(current) or np.linalg.norm(current-proposal[0],axis=1).min()>=self.cfg.active_min_dist:
                        point=proposal;break
            if not info['valid'] and step==0 and point is None:
                info['stop_reason']='initial_invalid_no_rescue_point';break
            stable=prev is not None and info['iou_prev']>=self.cfg.early_iou and info['area_change_prev']<=self.cfg.early_area_change
            if stable and point is None:
                info['stop_reason']='stable_no_new_point';break
            if step==self.cfg.refine_iters:
                info['stop_reason']='iteration_limit';break
            if point is not None:
                xy,label,kind=point
                info['added_point_type']=kind
                if current is None:
                    current=np.asarray(xy,np.float32).reshape(1,2)
                    labs=np.array([label],np.int32)
                else:
                    current=np.concatenate([current,np.asarray(xy,np.float32).reshape(1,2)])
                    labs=np.concatenate([labs,np.array([label],np.int32)])
            else:
                info['added_point_type']='mask_feedback_only'
            prev=pack['prob']
            memory=pack['memory']
        invalid_fallback=(best is None)
        if invalid_fallback:
            if fallback is None:
                return None, history  # only genuinely missing/invalid decoder output
            best=fallback
            best['info']['stop_reason']=best['info'].get('stop_reason','all_states_invalid_fallback')
        chosen=best['pack']
        prob_orig=from_view_mask(chosen['prob'],view['region'],hw)
        valid=np.zeros(hw,bool)
        x0,y0,x1,y1=view['region'];valid[y0:y1,x0:x1]=True
        return dict(prob=prob_orig,valid_region=valid,quality=chosen['quality'],
                    selected_step=best['step'],selected_info=best['info'],history=history,
                    invalid_fallback=int(invalid_fallback),
                    invalid_reasons=list(best['info']['invalid_reason']) if invalid_fallback else [],
                    route='CROP' if view['cropped'] else 'GLOBAL'),history

    def _choose_views(self,global_result,local_result):
        if local_result is None:
            return global_result, ('GLOBAL_INVALID_FALLBACK' if global_result is not None and global_result['invalid_fallback'] else 'GLOBAL_ONLY')
        if global_result is None:
            return local_result, ('CROP_INVALID_FALLBACK' if local_result['invalid_fallback'] else 'GLOBAL_FAILED_LOCAL')
        # Never prefer an invalid crop over a valid global result (or vice versa).
        gi=bool(global_result['invalid_fallback'])
        li=bool(local_result['invalid_fallback'])
        if gi != li:
            return (local_result,'CROP_VALID_RESCUE') if gi else (global_result,'GLOBAL_VALID_RESCUE')
        if gi and li:
            def severity(r):
                return (len(r['invalid_reasons']),r['selected_info']['cost'])
            return ((global_result,'GLOBAL_INVALID_FALLBACK') if severity(global_result)<=severity(local_result)
                    else (local_result,'CROP_INVALID_FALLBACK'))
        sim=binary_iou(global_result['prob']>=.5,local_result['prob']>=.5)
        gc=global_result['selected_info'];lc=local_result['selected_info']
        if sim<self.cfg.cross_view_min_iou:
            # Large disagreements are not evidence that one view is correct.
            if lc['has_scribble'] and (gc['scribble_error']-lc['scribble_error'])>.05:
                return local_result,'CROP_SCRIBBLE_RESCUE'
            return global_result,'GLOBAL_DISAGREEMENT_FALLBACK'
        if lc['cost']+self.cfg.min_accept_gain<gc['cost']:
            return local_result,'CROP_SUPPORTED'
        return global_result,'GLOBAL_PREFERRED'

    def process_image(self,image_path):
        name=Path(image_path).stem
        img=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if img is None:raise ValueError(f'cannot read RGB image {image_path}')
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w=rgb.shape[:2]
        js=DataLoader.find_file(self.cfg.json_dir,name,['.json'])
        boxes=DataLoader.load_boxes(js,w,h)
        if not boxes:raise ValueError(f'no valid box: {name}')
        sc=None
        if self.cfg.prompt_mode=='scribble':
            scpath=DataLoader.find_file(self.cfg.scribble_dir,name,['.png','.jpg'])
            if not scpath:raise FileNotFoundError(f'scribble required: {name}')
            sc=cv2.imread(scpath,cv2.IMREAD_GRAYSCALE)
            if sc is None:raise ValueError(f'cannot read scribble: {name}')
            if sc.shape!=(h,w):sc=cv2.resize(sc,(w,h),interpolation=cv2.INTER_NEAREST)
        # Original-like baseline: single whole-image union box; first three-candidate pass.
        global_baseline=self._view(rgb,sc,union_box(boxes),False)
        self.engine.set_image(global_baseline['img'])
        base_pack=self.engine.predict_three(global_baseline['pts'],global_baseline['labs'],global_baseline['box'],scribble=global_baseline['sc'])
        base=from_view_mask(base_pack['prob'],global_baseline['region'],(h,w))
        per_obj=[];self.last_objects=[]
        final=np.zeros((h,w),np.float32)
        for k,box in enumerate(boxes):
            global_view=self._view(rgb,sc,box,False)
            self.engine.set_image(global_view['img'])
            g0=self.engine.predict_three(global_view['pts'],global_view['labs'],global_view['box'],scribble=global_view['sc'])
            small=float((box[2]-box[0])*(box[3]-box[1])/max(1,h*w))<=self.cfg.small_box_area_thr
            local_view=None;l0=None
            if self.cfg.use_crop and small:
                maybe=self._view(rgb,sc,box,True)
                if maybe['region']!=(0,0,w,h):
                    local_view=maybe
                    self.engine.set_image(local_view['img'])
                    l0=self.engine.predict_three(local_view['pts'],local_view['labs'],local_view['box'],scribble=local_view['sc'])
            inferred=self._make_auto_points(g0,global_view,l0,local_view,(h,w),box)
            gr,gh=self._refine_view(global_view,inferred,(h,w),f'{name}_box{k:02d}_GLOBAL')
            lr,lh=(None,[])
            if local_view is not None:
                lr,lh=self._refine_view(local_view,inferred,(h,w),f'{name}_box{k:02d}_CROP')
            chosen,decision=self._choose_views(gr,lr)
            if chosen is None:raise RuntimeError(f'all candidate states invalid for {name} / box {k}')
            # Conservative spatial support of this object for multi-box union.
            pad=max(2,int(round(25*max(h,w)/self.cfg.size)))
            gate=box_mask([box[0]-pad,box[1]-pad,box[2]+pad,box[3]+pad],h,w)
            obj_prob=chosen['prob']*gate.astype(np.float32)
            final=np.maximum(final,obj_prob)
            obj=dict(box_id=k,box=[float(x) for x in box],route=chosen['route'],decision=decision,
                     selected_step=chosen['selected_step'],
                     invalid_fallback=int(chosen['invalid_fallback']),
                     invalid_fallback_reasons=list(chosen['invalid_reasons']),
                     **chosen['quality'],
                     selected_info=chosen['selected_info'],global_history=gh,local_history=lh,
                     num_auto_positive=sum(p[1]==1 for p in inferred),
                     num_auto_negative=sum(p[1]==0 for p in inferred),
                     global_local_iou=(binary_iou(gr['prob']>=.5,lr['prob']>=.5) if gr is not None and lr is not None else None),
                     merge_gate_changed_ratio=float(np.mean((chosen['prob']>=.5)&~gate)),
                     valid_region_ratio=float(np.mean(chosen['valid_region'])))
            per_obj.append(obj)
            # Global prediction supplies real evidence outside the local crop.
            global_prob = gr['prob'] if gr is not None else chosen['prob']
            qprob = np.where(chosen['valid_region'], chosen['prob'], global_prob).astype(np.float32)
            qvalid=np.ones((h,w),bool) if gr is not None else chosen['valid_region']
            self.last_objects.append({'record':obj,'prob':obj_prob,'valid_region':chosen['valid_region'],
                                      'quality_prob':qprob, 'quality_valid_region':qvalid,
                                      'box':box})
        base_png=(base>=.5).astype(np.uint8)*255
        final_png=(final>=.5).astype(np.uint8)*255
        if not cv2.imwrite(str(Path(self.cfg.base_dir)/(name+'.png')),base_png):raise IOError('base PNG failed')
        if not cv2.imwrite(str(Path(self.cfg.refine_dir)/(name+'.png')),final_png):raise IOError('refined PNG failed')
        if self.cfg.save_probs:
            np.save(str(Path(self.cfg.out_dir)/'base_prob'/(name+'.npy')),base.astype(np.float32))
            np.save(str(Path(self.cfg.out_dir)/'refine_prob'/(name+'.npy')),final.astype(np.float32))
        self.last_per_box=per_obj
        hist_dir=Path(self.cfg.out_dir)/'per_box_history'
        hist_dir.mkdir(parents=True,exist_ok=True)
        hist_path=hist_dir/f'{name}.json'
        write_json(hist_path, per_obj)
        return dict(image_name=Path(image_path).name,base_name=name,image_height=h,image_width=w,
                    route_type='MULTI_BOX' if len(boxes)>1 else per_obj[0]['route'],
                    num_boxes=len(boxes),crop_count=sum(int(x['route']=='CROP') for x in per_obj),
                    accepted_steps=sum(x['selected_step'] for x in per_obj),
                    auto_pos=sum(x['num_auto_positive'] for x in per_obj),
                    auto_neg=sum(x['num_auto_negative'] for x in per_obj),
                    prompt_mode=self.cfg.prompt_mode,refine_iters=self.cfg.refine_iters,
                    per_box_history_path=str(hist_path),
                    box_area_ratio=float((union_box(boxes)[2]-union_box(boxes)[0])*(union_box(boxes)[3]-union_box(boxes)[1])/(w*h)),
                    hard_mask_path=str(Path(self.cfg.refine_dir)/(name+'.png')),
                    full_prob_path=str(Path(self.cfg.out_dir)/'refine_prob'/(name+'.npy')))


def _self_test():
    h,w=80,128
    b=np.array([20,15,45,38],np.float32)
    roi=crop_region(b,w,h,.35,8)
    m=np.zeros((1024,1024),np.float32);m[180:600,230:670]=.8
    out=from_view_mask(m,roi,(h,w))
    assert out.shape==(h,w) and np.all(out[:roi[1]]==0)
    points=to_view_points(np.array([[21,16],[41,33]],np.float32),roi,1024)
    assert np.all((points>=0)&(points<1024))
    sc=np.zeros((h,w),np.uint8);sc[30:45,30:45]=1
    fg,bg=DataLoader.parse_scribble(sc)
    assert fg.sum()>0 and not bg.any()
    pts,labels=DataLoader.points_from_scribble(sc,3)
    assert len(pts) and np.all(labels==1)
    assert binary_iou(out>=.5,out>=.5)==1
    # Mock decoder: check multi-candidate same-call bookkeeping and backtracking.
    class DummyPredictor:
        mask_threshold=0
        def set_image(self,im): pass
        def predict(self,**kwargs):
            logits=np.full((3,1024,1024),-4,np.float32)
            logits[:,160:700,170:640]=4
            logits[1,165:700,170:640]=4
            low=np.zeros((3,256,256),np.float32)
            return logits,np.array([.94,.92,.90],np.float32),low
    cfg=Config(size=1024,device='cpu',refine_iters=2,prompt_mode='box',auto_points=False)
    engine=SAM2Engine(cfg,predictor=DummyPredictor())
    pack=engine.predict_three(None,None,np.array([100,100,730,730]))
    assert pack['quality']['selected_candidate_index']==0 and pack['memory'].shape==(1,256,256)
    print('SELF TEST PASS: crop coordinates, scribble semantics, candidate/memory shapes')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sam2-checkpoint',default=Config.sam2_checkpoint)
    ap.add_argument('--sam2-config',default=Config.sam2_config)
    ap.add_argument('--img-dir',default=Config.img_dir)
    ap.add_argument('--scribble-dir',default=Config.scribble_dir)
    ap.add_argument('--json-dir',default=Config.json_dir)
    ap.add_argument('--output-dir',default=Config.out_dir)
    ap.add_argument('--device',default=Config.device)
    ap.add_argument('--prompt-mode',choices=['box','scribble'],default='scribble')
    ap.add_argument('--refine-iters',type=int,default=2)
    ap.add_argument('--no-active',action='store_true')
    ap.add_argument('--no-auto-points',action='store_true')
    ap.add_argument('--no-crop',action='store_true')
    ap.add_argument('--small-box-area-thr',type=float,default=.10)
    ap.add_argument('--crop-context',type=float,default=.35)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--save-iterations',action='store_true',help='Save every selected-iteration mask and history')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args(argv)
    if args.self_test:_self_test();return
    if args.refine_iters<0:ap.error('--refine-iters must be >= 0')
    if not 0<=args.small_box_area_thr<=1:ap.error('--small-box-area-thr must be in [0,1]')
    if args.crop_context<0:ap.error('--crop-context must be >=0')
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    cfg=Config(sam2_checkpoint=args.sam2_checkpoint,sam2_config=args.sam2_config,
               img_dir=args.img_dir,scribble_dir=args.scribble_dir,json_dir=args.json_dir,
               out_dir=args.output_dir,device=args.device,prompt_mode=args.prompt_mode,
               refine_iters=args.refine_iters,active_points=not args.no_active,
               auto_points=not args.no_auto_points,use_crop=not args.no_crop,
               small_box_area_thr=args.small_box_area_thr,crop_context=args.crop_context,
               save_iterations=args.save_iterations)
    images=DataLoader.list_images(cfg.img_dir)
    if args.limit:images=images[:args.limit]
    pipe=Pipeline(cfg)
    rows=[];errors=[]
    for image in tqdm(images,desc='SAM2 reliability-controlled NoFlip'):
        try:rows.append(pipe.process_image(image))
        except Exception as exc:errors.append({'image':image,'error':str(exc)})
    out=Path(cfg.out_dir)
    if rows:
        with open(out/'refine_records.csv','w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'errors.json', errors)
    print(f'Refined {len(rows)}; errors {len(errors)}. Output {out}')


if __name__=='__main__':main()
