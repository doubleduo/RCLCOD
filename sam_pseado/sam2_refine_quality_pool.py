#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM2 NoFlip quality pools aligned to reliability-controlled refinement.

Requires the adjacent modified sam2_refine_crop_flipgate.py and existing
sam2_pool_from_refined.py from RCLCOD (commit 9e05c95).  Per-instance SIM,
SAM scores, and masks correspond to each instance's SELECTED accepted decoder
iteration. GT never enters prompt generation, acceptance, or pool assignment.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import sam2_refine_crop as ref
import sam2_pool_from_refined as quality


def etc_info(mask,box,border_tolerance=4):
    """Count actual image-edge activations, not edges of artificial crop window."""
    m=np.asarray(mask,bool)
    h,w=m.shape
    hit=[bool(m[0,:].any()),bool(m[-1,:].any()),bool(m[:,0].any()),bool(m[:,-1].any())]
    box_near=[box[1]<=border_tolerance,box[3]>=h-1-border_tolerance,
              box[0]<=border_tolerance,box[2]>=w-1-border_tolerance]
    return {'etc':int(sum(hit)),
            'etc_suspicious':int(sum(hh and not bn for hh,bn in zip(hit,box_near))),
            'etc_top':int(hit[0]),'etc_bottom':int(hit[1]),
            'etc_left':int(hit[2]),'etc_right':int(hit[3])}


def compute_region_quality(prob, hard_mask, bmask, all_boxes_mask, valid_region, score,stability,similarity):
    """Original RCLCOD weights with outside-of-box confidence restricted to measured pixels.

    For Crop, caller supplies real global prediction as probability fallback
    outside crop. Background for one object excludes ALL annotated boxes.
    This avoids treating unobserved zero fill / other annotated objects as easy BG.
    """
    p=np.asarray(prob,np.float32)
    m=np.asarray(hard_mask,bool)
    valid=np.asarray(valid_region,bool)
    core=quality.erode_mask(m, quality.CORE_ERODE_RADIUS)
    if not core.any():core=m.copy()
    fg=p[core & valid]
    fg_mean=float(np.clip(fg.mean(),0,1)) if fg.size else 0.0
    fg_p10=float(np.clip(np.quantile(fg,.10),0,1)) if fg.size else 0.0
    ring=quality.dilate_mask(m,quality.BACKGROUND_RING_RADIUS) & ~m & valid & ~all_boxes_mask
    ring_val=1.0-p[ring]
    ring_bg=float(np.clip(ring_val.mean(),0,1)) if ring_val.size else float('nan')
    outside=(~all_boxes_mask)&(~m)&valid
    hardval=p[outside]
    hard_bg=float(np.clip(1.0-np.quantile(hardval,.95),0,1)) if hardval.size else float('nan')
    parts=[];weights=[]
    if np.isfinite(ring_bg):parts.append(ring_bg);weights.append(.7)
    if np.isfinite(hard_bg):parts.append(hard_bg);weights.append(.3)
    bg=float(np.average(parts,weights=weights)) if parts else 0.0
    ring_q=ring_bg if np.isfinite(ring_bg) else bg
    hard_q=hard_bg if np.isfinite(hard_bg) else bg
    rw=quality.REGION_Q_WEIGHTS;sw=quality.SHAPE_Q_WEIGHTS
    qr=float(np.clip(rw['fg_mean']*fg_mean+rw['fg_p10']*fg_p10+
                    rw['ring_bg']*ring_q+rw['hard_outside_bg']*hard_q,0,1))
    qs=float(np.clip(sw['sam_score']*score+sw['sam_stability']*stability+
                    sw['candidate_consensus_s']*similarity,0,1))
    fail_fg=int(fg_mean<quality.MIN_FG_CONFIDENCE)
    fail_bg=int(bg<quality.MIN_BG_CONFIDENCE)
    fail_score=int(score<quality.MIN_SAM_SCORE)
    fail_stability=int(stability<quality.MIN_SAM_STABILITY)
    return dict(q_region=qr,q_shape=qs,q_dual=float(math.sqrt(qr*qs)),q_total=float(math.sqrt(qr*qs)),
                fg_mean_confidence=fg_mean,fg_p10_confidence=fg_p10,fg_confidence=fg_mean,
                bg_confidence=bg,ring_bg_confidence=ring_bg,
                hard_outside_bg_confidence=hard_bg,
                sam_score_selected=float(score),sam_stability_selected=float(stability),
                fail_fg_gate=fail_fg,fail_bg_gate=fail_bg,
                fail_score_gate=fail_score,fail_stability_gate=fail_stability,
                fails_quality_gate=int(bool(fail_fg or fail_bg or fail_score or fail_stability)),
                quality_valid_background_ratio=float(outside.mean()))


def evaluate_final_pool(pipeline,base_record,cfg,out,use_gt):
    name=base_record['base_name'];h=int(base_record['image_height']);w=int(base_record['image_width'])
    final_prob=np.load(str(Path(cfg.out_dir)/'refine_prob'/f'{name}.npy'),allow_pickle=False)
    if final_prob.shape!=(h,w) or final_prob.dtype!=np.float32:
        raise ValueError(f'Expected full-resolution FP32 probability: {name}: {final_prob.shape}/{final_prob.dtype}')
    final_mask=quality.read_gray_mask(Path(cfg.refine_dir)/f'{name}.png',(h,w))
    disagreement=float(np.mean((final_prob>=.5)!=final_mask))
    if disagreement:
        raise ValueError(f'Hard mask != saved FP32 prob at {disagreement:.3%} pixels: {name}')
    boxes=[np.asarray(o['box'],np.float32) for o in pipeline.last_objects]
    if len(boxes)!=base_record['num_boxes']:
        raise ValueError(f'Object mismatch {name}')
    all_boxmask=np.zeros((h,w),bool)
    for box in boxes:all_boxmask|=ref.box_mask(box,h,w)
    scored=[]
    for item in pipeline.last_objects:
        record=dict(item['record']);box=item['box']
        prob=np.asarray(item['prob'],np.float32)
        mask=prob>=.5
        # Original full-image BG from a selected global refinement is preferable
        # to unobserved zeros outside crop. The generator saves this as quality_prob.
        qprob=np.asarray(item.get('quality_prob',prob),np.float32)
        qvalid=np.asarray(item.get('quality_valid_region',item['valid_region']),bool)
        score=record['sam_score_selected']
        stability=record['sam_stability_selected']
        s=record['candidate_consensus_s']
        q=compute_region_quality(qprob,mask,ref.box_mask(box,h,w),all_boxmask,qvalid,score,stability,s)
        geom=quality.compute_v_violation(mask,ref.box_mask(box,h,w),box)
        anomalies=quality.compute_bad_response_flags(mask,ref.box_mask(box,h,w),box,h,w,
                                                     bool(box[0]<=quality.OV_BORDER_DELTA or
                                                          box[1]<=quality.OV_BORDER_DELTA or
                                                          box[2]>=w-quality.OV_BORDER_DELTA or
                                                          box[3]>=h-quality.OV_BORDER_DELTA))
        edge=etc_info(mask,box)
        score_grade=quality.candidate_grade_from_s(s)
        hard=int(bool(record.get('invalid_fallback',0)) or score_grade=='low' or q['fails_quality_gate'] or anomalies['is_bad_response'])
        if hard:pool='unvalue'
        elif score_grade=='middle' or anomalies['is_refinable_risk'] or edge['etc_suspicious']:
            pool='middle'
        else:pool='high'
        scored.append({**record,**q,**anomalies,**geom,**edge,
                       'candidate_s_grade':score_grade,'force_unvalue':hard,'recommended_pool':pool})
    (out/'per_box_quality').mkdir(parents=True,exist_ok=True)
    per_box_path=out/'per_box_quality'/f'{name}.json'
    ref.write_json(per_box_path, scored)
    return aggregate_from_saved_quality(scored,base_record,cfg,out,final_prob,final_mask,use_gt,per_box_path)

def aggregate_from_saved_quality(scored,base_record,cfg,out,final_prob,final_mask,use_gt,per_box_path):
    """Identical image-level pool logic for live inference and offline recovery.

    All per-instance quality measurements were generated by the same selected
    decoder state as the saved mask. No SAM2 inference is needed here.
    """
    name=base_record['base_name'];h=int(base_record['image_height']);w=int(base_record['image_width'])
    if not scored: raise ValueError('Empty per-box quality: '+name)
    # Image-level conservative aggregate: ANY invalid object => image unvalue.
    grades=[p['recommended_pool'] for p in scored]
    final_grade='unvalue' if 'unvalue' in grades else 'middle' if 'middle' in grades else 'high'
    boxes=[np.asarray(o['box'],np.float32) for o in scored]
    all_boxmask=np.zeros((h,w),bool)
    for box in boxes: all_boxmask |= ref.box_mask(box,h,w)
    sim=min(p['candidate_consensus_s'] for p in scored)
    score=min(p['sam_score_selected'] for p in scored)
    stab=min(p['sam_stability_selected'] for p in scored)
    qr=min(p['q_region'] for p in scored)
    qs=min(p['q_shape'] for p in scored)
    qdual=float(math.sqrt(qr*qs))
    # Image-level boundary ambiguity from *actual* final probability.
    boundary=quality.compute_boundary_uncertainty(final_prob,final_mask)
    union=ref.union_box(boxes)
    struct=quality.compute_structure_info(union,h,w,len(boxes))
    # Explicitly avoid treating multiple known objects as fragmentation noise.
    merged_geom=quality.compute_v_violation(final_mask,all_boxmask,union)
    merged_etc=etc_info(final_mask,union)
    aggregate=dict(image=name,hard_mask_path=str(Path(cfg.refine_dir)/f'{name}.png'),
                   full_prob_path=str(Path(cfg.out_dir)/'refine_prob'/f'{name}.npy'),
                   per_box_quality_path=str(per_box_path),
                   candidate_consensus_s=sim,sam_score_selected=score,
                   sam_stability_selected=stab,
                   candidate_s_grade=quality.candidate_grade_from_s(sim),
                   q_region=qr,q_shape=qs,q_dual=qdual,q_total=qdual,
                   recommended_pool=final_grade,
                   force_unvalue=int(final_grade=='unvalue'),
                   fails_quality_gate=int(any(p['fails_quality_gate'] for p in scored)),
                   is_bad_response=int(any(p['is_bad_response'] for p in scored)),
                   is_refinable_risk=int(any(p['is_refinable_risk'] for p in scored)),
                   is_fragmented_response=int(any(p['is_fragmented_response'] for p in scored)),
                   is_severe_fragmented_response=int(any(p['is_severe_fragmented_response'] for p in scored)),
                   is_reversal_response=int(any(p['is_reversal_response'] for p in scored)),
                   outside_pixel_count=merged_geom['outside_pixel_count'],
                   v_outside=merged_geom['v_outside'],v_total=merged_geom['v_total'],
                   etc=merged_etc['etc'],etc_suspicious=merged_etc['etc_suspicious'],
                   num_high_objects=sum(p=='high' for p in grades),
                   num_middle_objects=sum(p=='middle' for p in grades),
                   num_unvalue_objects=sum(p=='unvalue' for p in grades),
                   num_invalid_fallback_objects=sum(int(p.get('invalid_fallback',0)) for p in scored),
                   has_invalid_fallback=int(any(p.get('invalid_fallback',0) for p in scored)),
                   quality_source='selected_decoder_with_explicit_invalid_fallback',
                   multi_box_quality_aggregation='worst_object_conservative',
                   source_mask='ReliabilityControlled_NoFlip',
                   q_region_rank_global=-1,q_shape_rank_global=-1,
                   q_dual_rank_global=-1,q_region_rank_within_pool=-1,
                   q_shape_rank_within_pool=-1,q_dual_rank_within_pool=-1,
                   q_rank_global=-1,q_rank_within_pool=-1,
                   is_BA=0,ba_threshold=float('nan'))
    aggregate={**base_record,**struct,**boundary,**aggregate}
    prob48=out/'Prob_48x48_FP16';prob48.mkdir(parents=True,exist_ok=True)
    path=prob48/f'{name}.npy'
    np.save(path,cv2.resize(final_prob,(48,48),interpolation=cv2.INTER_AREA).astype(np.float16),allow_pickle=False)
    aggregate['prob_48_path']=str(path)
    if use_gt:
        aggregate.update(quality.calc_metrics(final_mask,Path(cfg.gt_dir)/f'{name}.png',(h,w)))
    else:
        aggregate.update(has_gt=0,IoU=float('nan'),MAE=float('nan'))
    return aggregate


def cached_row(image_path,cfg,out,use_gt):
    """Recover a complete previously successful image without loading SAM2.

    Reconstructs image-level quality and rank inputs from persisted FP32
    probability, exact PNG, and per-box quality JSON of v2.1/v2.2.
    """
    name=Path(image_path).stem
    pq=out/'per_box_quality'/f'{name}.json'
    pro=out/'refine_prob'/f'{name}.npy'
    png=out/'03_Refine_NoFlip'/f'{name}.png'
    history=out/'per_box_history'/f'{name}.json'
    if not all(x.is_file() for x in (pq,pro,png,history)):
        return None
    try:
        scored=json.loads(pq.read_text(encoding='utf-8'))
        if not isinstance(scored,list) or not scored:raise ValueError('per-box JSON empty')
        prob=np.load(str(pro),allow_pickle=False)
        if prob.ndim!=2 or prob.dtype!=np.float32: raise ValueError('invalid probability cache')
        h,w=prob.shape
        hard=quality.read_gray_mask(png,(h,w))
        if np.any((prob>=.5)!=hard):raise ValueError('probability/PNG mismatch')
        boxes=[np.asarray(r['box'],np.float32) for r in scored]
        if not all(b.shape==(4,) and np.isfinite(b).all() for b in boxes):raise ValueError('corrupt cached boxes')
        if any('q_region' not in r or 'q_shape' not in r or 'recommended_pool' not in r for r in scored):
            raise ValueError('incomplete per-box scores')
        ub=ref.union_box(boxes)
        base=dict(image_name=Path(image_path).name,base_name=name,
                  image_height=h,image_width=w,num_boxes=len(scored),
                  route_type='MULTI_BOX' if len(scored)>1 else scored[0]['route'],
                  crop_count=sum(int(r.get('route')=='CROP') for r in scored),
                  accepted_steps=sum(int(r['selected_step']) for r in scored),
                  auto_pos=sum(int(r.get('num_auto_positive',0)) for r in scored),
                  auto_neg=sum(int(r.get('num_auto_negative',0)) for r in scored),
                  prompt_mode=cfg.prompt_mode,refine_iters=cfg.refine_iters,
                  per_box_history_path=str(history),
                  box_area_ratio=float((ub[2]-ub[0])*(ub[3]-ub[1])/(h*w)),
                  hard_mask_path=str(png),full_prob_path=str(pro))
        return aggregate_from_saved_quality(scored,base,cfg,out,prob,hard,use_gt,pq)
    except (ValueError,KeyError,IndexError,TypeError,RuntimeError,UnicodeError,OSError) as exc:
        tqdm.write(f'[CACHE INVALID] {name}: {exc}. Will recompute if permitted.')
        return None


def _self_test():
    m=np.zeros((24,32),bool)
    m[:3,4:8]=True
    assert etc_info(m,np.array([3,0,12,11]))['etc']==1
    assert etc_info(m,np.array([3,10,12,20]))['etc_suspicious']==1
    prob=np.full((24,32),.05,np.float32)
    prob[10:17,8:20]=.92
    hard=prob>=.5
    b=ref.box_mask([7,9,21,18],24,32)
    q=compute_region_quality(prob,hard,b,b,np.ones_like(b),.95,.95,.95)
    assert 0<=q['q_region']<=1 and 0<=q['q_dual']<=1
    print('SELF TEST PASS: ETC, full-prob region quality, gate fields')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sam2-checkpoint',default=ref.Config.sam2_checkpoint)
    ap.add_argument('--sam2-config',default=ref.Config.sam2_config)
    ap.add_argument('--img-dir',default=ref.Config.img_dir)
    ap.add_argument('--scribble-dir',default=ref.Config.scribble_dir)
    ap.add_argument('--json-dir',default=ref.Config.json_dir)
    ap.add_argument('--gt-dir',default=ref.Config.gt_dir)
    ap.add_argument('--output-dir',default='./SGT/SAM2_FinalRefine_OriginalPools')
    ap.add_argument('--device',default=ref.Config.device)
    ap.add_argument('--prompt-mode',choices=['box','scribble'],default='scribble')
    ap.add_argument('--refine-iters',type=int,default=2)
    ap.add_argument('--no-active',action='store_true')
    ap.add_argument('--no-auto-points',action='store_true')
    ap.add_argument('--no-crop',action='store_true')
    ap.add_argument('--small-box-area-thr',type=float,default=.10)
    ap.add_argument('--crop-context',type=float,default=.35)
    ap.add_argument('--eval-gt',action='store_true')
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--allow-partial',action='store_true')
    ap.add_argument('--save-iterations',action='store_true')
    ap.add_argument('--resume',action='store_true',help='Reuse all completed images, run SAM2 only for missing images')
    ap.add_argument('--recover-only',action='store_true',help='Recover cached pool CSV/TXT without loading SAM2; incomplete coverage is reported')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args(argv)
    if args.self_test:_self_test();return
    if args.refine_iters<0:ap.error('refine-iters must be >=0')
    if args.resume and args.recover_only:ap.error('--resume and --recover-only are mutually exclusive')
    if not 0<=args.small_box_area_thr<=1:ap.error('small-box-area-thr must be [0,1]')
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    out=Path(args.output_dir)
    cfg=ref.Config(sam2_checkpoint=args.sam2_checkpoint,sam2_config=args.sam2_config,
                   device=args.device,img_dir=args.img_dir,scribble_dir=args.scribble_dir,
                   json_dir=args.json_dir,gt_dir=args.gt_dir,out_dir=str(out),
                   prompt_mode=args.prompt_mode,refine_iters=args.refine_iters,
                   active_points=not args.no_active,auto_points=not args.no_auto_points,
                   use_crop=not args.no_crop,small_box_area_thr=args.small_box_area_thr,
                   crop_context=args.crop_context,save_probs=True,
                   save_iterations=args.save_iterations)
    images=ref.DataLoader.list_images(cfg.img_dir)
    if args.limit:images=images[:args.limit]
    rows=[];errors=[]
    pipe=None
    restored=0;computed=0
    for img in tqdm(images,desc='SAM2 resume/recovery NoFlip + pools'):
        name=Path(img).stem
        try:
            cached=cached_row(img,cfg,out,args.eval_gt) if (args.resume or args.recover_only) else None
            if cached is not None:
                rows.append(cached);restored+=1
                continue
            if args.recover_only:
                errors.append({'image':str(img),'error':'No complete cached result; SAM2 not run in recover-only mode'})
                continue
            if pipe is None:
                pipe=ref.Pipeline(cfg)  # lazy: --recover-only never builds SAM2
            basic=pipe.process_image(img)
            rows.append(evaluate_final_pool(pipe,basic,cfg,out,args.eval_gt))
            computed+=1
        except Exception as exc:
            errors.append({'image':str(img),'error':repr(exc)})
            if len(errors)<=15:tqdm.write(f'[ERROR] {Path(img).name}: {exc}')
    out.mkdir(parents=True,exist_ok=True)
    # Preserve error diagnostics from the expensive previous run before
    # --resume / --recover-only writes the current missing-image list.
    import shutil
    old_errors=out/'errors.json'
    previous_errors=out/'errors_previous_run.json'
    if old_errors.is_file() and not previous_errors.exists():
        shutil.copy2(old_errors,previous_errors)
    ref.write_json(out/'errors.json', errors)
    if not rows:
        raise RuntimeError('No complete cached images found and no SAM2 images succeeded')
    if errors:
        tqdm.write(f'[INCOMPLETE] {len(errors)} / {len(images)} images missing. '
                   f'CSV/TXT WILL be written with complete=false; see {out/"errors.json"}.')
    rows.sort(key=lambda r:r['image'])
    bvals=np.array([quality.safe_float(r['boundary_uncertainty']) for r in rows],np.float32)
    bvals=bvals[np.isfinite(bvals)]
    bthr=float(np.quantile(bvals,1-quality.BA_TOP_RATIO)) if bvals.size else float('nan')
    for row in rows:
        val=quality.safe_float(row['boundary_uncertainty'])
        row['ba_threshold']=bthr
        row['is_BA']=int(np.isfinite(bthr) and np.isfinite(val) and val>=bthr)
    quality.assign_q_ranks(rows)
    (out/'csv').mkdir(exist_ok=True,parents=True)
    for pool in ('high','middle','unvalue'):
        sub=[r for r in rows if r['recommended_pool']==pool]
        ranked=sorted(sub,key=lambda r:(-r['q_dual'],r['image']))
        # Preserve old alphabetically sorted lists for loaders expecting their format.
        (out/f'{pool}.txt').write_text(''.join(r['image']+'\n' for r in sub),encoding='utf-8')
        (out/f'{pool}_masks.txt').write_text(''.join(r['hard_mask_path']+'\n' for r in sub),encoding='utf-8')
        (out/f'{pool}_ranked.txt').write_text(''.join(r['image']+'\n' for r in ranked),encoding='utf-8')
        quality.save_csv(out/'csv'/f'{pool}.csv',sub)
    quality.save_csv(out/'csv'/'image_level_metrics.csv',rows)
    groups=Counter(r['recommended_pool'] for r in rows)
    summary=dict(images=len(rows),expected_images=len(images),failed=len(errors),
                 invalid_fallback_images=sum(bool(r.get('has_invalid_fallback',0)) for r in rows),
                 complete=(len(errors)==0 and len(rows)==len(images)),
                 restored_from_cache=restored,inferred_this_run=computed,
                 high=groups['high'],middle=groups['middle'],unvalue=groups['unvalue'],
                 prompt_mode=cfg.prompt_mode,flip=False,
                 source='reliability-controlled selected decoder + per-box quality',
                 high_middle_unvalue='worst object conservatively',
                 gt_used_for_selection=False)
    ref.write_json(out/'summary.json', summary)
    print('Pool result:',summary)
    print('CSV:',out/'csv'/'image_level_metrics.csv')

if __name__=='__main__':main()
