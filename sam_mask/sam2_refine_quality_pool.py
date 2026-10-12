#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM2 final-step three-candidate active refinement + original quality pools.

DEPENDENCIES: place alongside these two files:
  sam2_refine_crop_flipgate.py  (previous script, contains crop / active points)
  sam2_pool_from_refined.py     (previous script, contains original Q/V/flags/ranks)

Critical change: final SAM2 decoder call uses multimask_output=True, not a
post-hoc re-scoring of the previously saved binary PNG. The selected mask and
its SAM score/stability come from the SAME final decoder output. Three-candidate
S is calculated from the SAME final prompt/memory. This produces NEW refined
masks, which can differ from those previously saved by multimask_output=False.
No flips. The original pool decision rule and thresholds remain unchanged.

Run:
 python sam2_refine_quality_pool.py --eval-gt
 python sam2_refine_quality_pool.py --limit 12 --output-dir ./SGT/test_final3
"""
import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import sam2_refine_crop_flipgate as ref
import sam2_pool_from_refined as quality


def binary_ssim(a, b):
    # Exactly the method of original second script (11x11 Gaussian SSIM).
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if np.array_equal(a, b):
        return 1.0
    if a.shape != b.shape:
        raise ValueError('SSIM candidate size mismatch')
    c1, c2 = .01 ** 2, .03 ** 2
    side = min(a.shape)
    if side < 3:
        ma, mb = float(a.mean()), float(b.mean())
        va, vb = float(a.var()), float(b.var())
        cov = float(((a-ma)*(b-mb)).mean())
        return float(np.clip(((2*ma*mb+c1)*(2*cov+c2)) /
                             max((ma*ma+mb*mb+c1)*(va+vb+c2),1e-12),0,1))
    win = min(11,side)
    win -= (win % 2 == 0)
    mu_a=cv2.GaussianBlur(a,(win,win),1.5)
    mu_b=cv2.GaussianBlur(b,(win,win),1.5)
    va=cv2.GaussianBlur(a*a,(win,win),1.5)-mu_a*mu_a
    vb=cv2.GaussianBlur(b*b,(win,win),1.5)-mu_b*mu_b
    vab=cv2.GaussianBlur(a*b,(win,win),1.5)-mu_a*mu_b
    num=(2*mu_a*mu_b+c1)*(2*vab+c2)
    den=(mu_a*mu_a+mu_b*mu_b+c1)*(va+vb+c2)
    return float(np.clip((num/np.maximum(den,1e-12)).mean(),0,1))


def compute_consensus(masks):
    masks = np.asarray(masks, bool)
    if masks.ndim != 3 or masks.shape[0] != 3:
        raise RuntimeError('Three SAM2 candidate masks are required, got '+str(masks.shape))
    stats = {}
    sims = []
    for i,j,s in ((0,1,'12'),(0,2,'13'),(1,2,'23')):
        iou = quality.compute_iou(masks[i],masks[j])
        ssim = binary_ssim(masks[i],masks[j])
        sim = .5 * (iou+ssim)
        stats.update({f'candidate_iou_{s}':iou,f'candidate_ssim_{s}':ssim,
                      f'candidate_sim_{s}':sim})
        sims.append(sim)
    stats.update({'candidate_consensus_s':float(np.mean(sims)),
                  'candidate_sim_min':float(np.min(sims)),
                  'candidate_sim_max':float(np.max(sims)),
                  'candidate_sim_range':float(np.max(sims)-np.min(sims))})
    return stats


def logit_stability(logit, threshold=0.0, offset=1.0):
    x=np.asarray(logit,np.float32)
    low=int((x>threshold-offset).sum())
    return float((x>threshold+offset).sum()/low) if low else 0.0


class FinalQualityEngine(ref.SAM2Engine):
    """Identical active-point refinement until the LAST call.

    Last call yields three candidate logits and SAM2 scores, and the candidate
    with max predicted IoU becomes the actual saved final pseudo-mask.
    """
    def refine_predict(self, points, labels, box, scribble=None, iters_override=None):
        h,w = self.last_input_shape
        if scribble is not None and scribble.shape[:2] != (h,w):
            raise ValueError('Scribble/model frame mismatch')
        n = self.cfg.refine_iters if iters_override is None else int(iters_override)
        current = points.copy() if points is not None else None
        labs = labels.copy() if labels is not None else None
        memory = None
        records = []
        # Precisely n single-mask passes; final pass below is the (n+1)th pass.
        for step in range(n):
            prob, memory = self._predict_once(current,labs,box,memory)
            if not self.cfg.active_points or scribble is None:
                continue
            extra,extra_lab,why=self.sample_active_point(prob,scribble,current,roi=box)
            if extra is None:
                continue
            if current is None:
                current,labs=extra,extra_lab
            else:
                current=np.concatenate([current,extra],axis=0)
                labs=np.concatenate([labs,extra_lab],axis=0)
            records.append({'step':step+1,'x':int(extra[0,0]),'y':int(extra[0,1]),
                            'label':int(extra_lab[0]),'reason':why})

        autocast = (torch.autocast('cuda',dtype=torch.bfloat16)
                    if self.device.startswith('cuda') and torch.cuda.is_available()
                    and torch.cuda.is_bf16_supported() else nullcontext())
        with torch.inference_mode(), autocast:
            full_logits,scores,low_logits=self.predictor.predict(
                point_coords=current,point_labels=labs,
                box=box[None,:] if box is not None else None,
                mask_input=memory,multimask_output=True,return_logits=True)
        logits=np.asarray(full_logits,np.float32)
        scores=np.asarray(scores,np.float32).reshape(-1)
        if logits.ndim == 2: logits=logits[None]
        if logits.shape[0] != 3 or scores.size != 3:
            raise RuntimeError(f'SAM2 final 3-candidate mode returned {logits.shape}, {scores.shape}')
        if not np.isfinite(logits).all() or not np.isfinite(scores).all():
            raise RuntimeError('Invalid final SAM2 logit/score')
        thr=float(getattr(self.predictor,'mask_threshold',
                          getattr(getattr(self.predictor,'model',None),'mask_threshold',0.0)))
        masks=logits>thr
        stats=compute_consensus(masks)
        idx=int(np.argmax(scores))
        chosen=logits[idx]
        self.last_quality={**stats,'sam_score_selected':float(np.clip(scores[idx],0,1)),
                           'sam_stability_selected':logit_stability(chosen,thr),
                           'selected_candidate_index':idx,
                           'candidate_score_1':float(scores[0]),
                           'candidate_score_2':float(scores[1]),
                           'candidate_score_3':float(scores[2]),
                           'quality_frame':'last_refinement_decoder',
                           'quality_mask_threshold':thr}
        # ref.Pipeline saves (prob >= 0.5); for SAM2 usual mask_threshold=0.0
        # this is consistent with masks[idx]. For nonzero mask_threshold, shift.
        return ref.sigmoid(chosen-thr), records


class FinalQualityPipeline(ref.Pipeline):
    def __init__(self,cfg,predictor=None):
        cfg.flip_gate=False
        cfg.save_probs=True
        super().__init__(cfg,predictor=predictor)
        self.engine=FinalQualityEngine(cfg,predictor=self.engine.predictor)
        self.object_quality=[]

    def _infer_baseline(self,rgb,scribble,boxes):
        # Preserve the original baseline's single-mask decoder, rather than
        # accidentally changing it to three candidates via this subclass.
        h,w=rgb.shape[:2]
        img,sc,model_box,region,pts,labs,_=self._target_for_box(
            rgb,scribble,ref.union_box(boxes),do_crop=False)
        if self._cached_rgb is not rgb or self._cached_region != region:
            self.engine.set_image(img)
            self._cached_rgb=rgb
            self._cached_region=region
        prob,_=ref.SAM2Engine.refine_predict(
            self.engine,pts,labs,model_box,scribble=None,iters_override=0)
        x0,y0,x1,y1=region
        mapped=np.zeros((h,w),np.float32)
        mapped[y0:y1,x0:x1]=ref.resize_prob(prob,x1-x0,y1-y0)
        return mapped

    def _infer_single_box(self,rgb,scribble,box,refine,do_crop=True):
        prob,pts,cropped=super()._infer_single_box(rgb,scribble,box,refine,do_crop)
        if refine:
            item=dict(self.engine.last_quality)
            item['source_box']=[float(x) for x in box]
            item['cropped']=int(cropped)
            self.object_quality.append(item)
        return prob,pts,cropped

    def process_image(self,image_path):
        self.object_quality=[]
        record=super().process_image(image_path)
        if record is None:
            return None
        if not self.object_quality:
            raise RuntimeError('No final quality record was captured')
        return record


def evaluate_final_pool(pipeline, ref_record, cfg, out, use_gt):
    name=str(ref_record['base_name'])
    imgpath=ref.DataLoader.find_file(cfg.img_dir,name,('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))
    if not imgpath:
        raise FileNotFoundError('Original RGB image missing for '+name)
    rgb=cv2.imread(imgpath,cv2.IMREAD_COLOR)
    h,w=rgb.shape[:2]
    final_prob=quality.load_prob(Path(cfg.out_dir)/'refine_prob'/(name+'.npy'),(h,w))
    hard_path=Path(cfg.refine_dir)/(name+'.png')
    final_mask=quality.read_gray_mask(hard_path,(h,w))
    if not np.array_equal(final_mask,final_prob>=.5):
        # FP16 may introduce rare equal-to-0.5 threshold differences.
        dif=float(np.mean(final_mask != (final_prob>=.5)))
        if dif > .002:
            raise ValueError(f'Probability/hard mask mismatch ({dif:.2%}) on {name}')
    boxes=ref.DataLoader.load_boxes(str(Path(cfg.json_dir)/(name+'.json')),w,h)
    if not boxes: raise ValueError('No valid box: '+name)
    # For multiple objects, no ONE SAM2 predicted-IoU score exists for their
    # merged mask; use conservative MIN of scores/stabilities/consensus.
    obj=pipeline.object_quality
    if len(obj)!=len(boxes):
        raise ValueError(f'Per-box final quality count mismatch {len(obj)} vs {len(boxes)}')
    metrics=['candidate_consensus_s','sam_score_selected','sam_stability_selected']
    agg={k:float(min(float(o[k]) for o in obj)) for k in metrics}
    # Keep all candidate-component fields, conservative min for multi-box.
    for k in ('candidate_iou_12','candidate_iou_13','candidate_iou_23',
              'candidate_ssim_12','candidate_ssim_13','candidate_ssim_23',
              'candidate_sim_12','candidate_sim_13','candidate_sim_23',
              'candidate_sim_min','candidate_sim_max'):
        agg[k]=float(min(o[k] for o in obj))
    agg['candidate_sim_range']=float(max(o['candidate_sim_range'] for o in obj))
    agg['selected_candidate_index']=(int(obj[0]['selected_candidate_index']) if len(obj)==1 else -1)
    for k in ('candidate_score_1','candidate_score_2','candidate_score_3'):
        agg[k]=(float(obj[0][k]) if len(obj)==1 else float('nan'))
    score,stab,s=agg['sam_score_selected'],agg['sam_stability_selected'],agg['candidate_consensus_s']
    box=ref.union_box(boxes)
    boxmask=quality.box_to_mask(box,h,w)
    structure=quality.compute_structure_info(box,h,w,len(boxes))
    q=quality.compute_q_evidence(final_prob,final_mask,boxmask,score,stab,s)
    v=quality.compute_v_violation(final_mask,boxmask,box)
    boundary=quality.compute_boundary_uncertainty(final_prob,final_mask)
    flags=quality.compute_bad_response_flags(final_mask,boxmask,box,h,w,bool(structure['is_OV']))
    grade=quality.candidate_grade_from_s(s)
    force=int(grade=='low' or q['fails_quality_gate']==1 or flags['is_bad_response']==1)
    if force: pool='unvalue'
    elif grade=='middle' or flags['is_refinable_risk']==1: pool='middle'
    else: pool='high'
    output={**ref_record,**structure,**agg,**q,**v,**boundary,**flags,
            'image':name,'image_height':h,'image_width':w,
            'hard_mask_path':str(hard_path),
            'full_prob_path':str(Path(cfg.out_dir)/'refine_prob'/(name+'.npy')),
            'candidate_s_grade':grade,'force_unvalue':force,
            'recommended_pool':pool,'source_mask':'FinalRefinedThreeCandidate_NoFlip',
            'quality_source':'final_refine_three_candidate',
            'multi_box_quality_aggregation':'single' if len(boxes)==1 else 'min_per_object',
            'used_enclosing_union_box':0,
            'quality_uses_per_box':1,
            'num_boxes_in_json':len(boxes),'quality_num_objects':len(obj),
            'q_region_rank_global':-1,'q_shape_rank_global':-1,'q_dual_rank_global':-1,
            'q_region_rank_within_pool':-1,'q_shape_rank_within_pool':-1,
            'q_dual_rank_within_pool':-1,'q_rank_global':-1,'q_rank_within_pool':-1,
            'is_BA':0,'ba_threshold':float('nan'),'bad_vis_path':''}
    (out/'Prob_48x48_FP16').mkdir(parents=True,exist_ok=True)
    (out/'per_box_quality').mkdir(parents=True,exist_ok=True)
    prob48=out/'Prob_48x48_FP16'/(name+'.npy')
    np.save(prob48,cv2.resize(final_prob,(48,48),interpolation=cv2.INTER_AREA).astype(np.float16),allow_pickle=False)
    output['prob_48_path']=str(prob48)
    quality_per_box=out/'per_box_quality'/f'{name}.json'
    quality_per_box.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    output['per_box_quality_path']=str(quality_per_box)
    if use_gt:
        output.update(quality.calc_metrics(final_mask,Path(cfg.gt_dir)/(name+'.png'),(h,w)))
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--sam2-checkpoint',default=ref.Config.sam2_checkpoint)
    p.add_argument('--sam2-config',default=ref.Config.sam2_config)
    p.add_argument('--device',default=ref.Config.device)
    p.add_argument('--img-dir',default=ref.Config.img_dir)
    p.add_argument('--scribble-dir',default=ref.Config.scribble_dir)
    p.add_argument('--json-dir',default=ref.Config.json_dir)
    p.add_argument('--gt-dir',default=ref.Config.gt_dir)
    p.add_argument('--output-dir',default='./SGT/SAM2_FinalRefine_OriginalPools')
    p.add_argument('--refine-iters',type=int,default=2)
    p.add_argument('--small-box-area-thr',type=float,default=.10)
    p.add_argument('--crop-context',type=float,default=.35)
    p.add_argument('--no-crop',action='store_true')
    p.add_argument('--no-active',action='store_true')
    p.add_argument('--eval-gt',action='store_true')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args(argv)
    if args.refine_iters<0: p.error('refine-iters must be >=0')
    if not (0<=args.small_box_area_thr<=1): p.error('small-box-area-thr must be in [0,1]')
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    out=Path(args.output_dir)
    cfg=ref.Config(sam2_checkpoint=args.sam2_checkpoint,sam2_config=args.sam2_config,
                   img_dir=args.img_dir,scribble_dir=args.scribble_dir,
                   json_dir=args.json_dir,gt_dir=args.gt_dir,
                   out_dir=str(out),device=args.device,refine_iters=args.refine_iters,
                   small_box_area_thr=args.small_box_area_thr,crop_context=args.crop_context,
                   use_crop=not args.no_crop,active_points=not args.no_active,
                   save_probs=True,flip_gate=False)
    for d in (out/'csv',out/'per_box_quality',out/'Prob_48x48_FP16'): d.mkdir(parents=True,exist_ok=True)
    pipe=FinalQualityPipeline(cfg)
    images=ref.DataLoader.list_images(cfg.img_dir)
    if args.limit: images=images[:args.limit]
    records=[]; errors=[]
    for path in tqdm(images,desc='SAM2 refined final-3 / original pools'):
        try:
            info=pipe.process_image(path)
            if info is None:
                raise RuntimeError('Missing or invalid image/scribble/box')
            row=evaluate_final_pool(pipe,info,cfg,out,args.eval_gt)
            records.append(row)
        except Exception as e:
            errors.append({'image':path,'error':repr(e)})
            if len(errors)<=12: tqdm.write(f'[ERROR] {os.path.basename(path)}: {e}')
    if errors:
        (out/'errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    if errors and not args.allow_partial:
        raise RuntimeError(f'{len(errors)} samples failed; no pool CSV/TXT written. First: {errors[0]}')
    if not records: raise RuntimeError('No valid samples processed')
    records.sort(key=lambda x:x['image'])
    val=np.asarray([quality.safe_float(r.get('boundary_uncertainty')) for r in records],np.float32)
    val=val[np.isfinite(val)]
    thr=float(np.quantile(val,1-quality.BA_TOP_RATIO)) if val.size else float('nan')
    for r in records:
        b=quality.safe_float(r.get('boundary_uncertainty'))
        r['ba_threshold']=thr
        r['is_BA']=int(np.isfinite(thr) and np.isfinite(b) and b>=thr)
    quality.assign_q_ranks(records)
    for group in ('high','middle','unvalue'):
        sub=[r for r in records if r['recommended_pool']==group]
        (out/f'{group}.txt').write_text(''.join(r['image']+'\n' for r in sub),encoding='utf-8')
        (out/f'{group}_masks.txt').write_text(''.join(r['hard_mask_path']+'\n' for r in sub),encoding='utf-8')
        quality.save_csv(out/'csv'/f'{group}.csv',sub)
    quality.save_csv(out/'csv'/'image_level_metrics.csv',records)
    n=Counter(r['recommended_pool'] for r in records)
    summary={'images':len(records),'failures':len(errors),'high':n['high'],
             'middle':n['middle'],'unvalue':n['unvalue'],
             'flip':False,'quality':'final refined last-call three candidate',
             'multi_box':'per-box three candidate; conservative min across boxes',
             'scribble_assisted':True,
             'pool_rule':'original: S low OR quality gate failed OR bad response => unvalue; '
                         'S middle OR ordinary fragmented => middle; else high',
             'caveat':'masks may differ from old multimask_output=False results'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n'+'='*70)
    print('Final refined masks: '+str(out/'03_Refine_NoFlip'))
    print('S / score / stability: measured on same FINAL SAM2 3-candidate call')
    print(f"Processed {len(records)}  High {n['high']}  Middle {n['middle']}  Unvalue {n['unvalue']}")
    print('CSV:',out/'csv'/'image_level_metrics.csv')
    print('='*70)


if __name__=='__main__':
    main()
