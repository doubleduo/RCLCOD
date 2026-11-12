#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select at most K improved images, re-run actual SAM1 and export GESP figure assets.
Requires the user's sam1_refine_crop_flipgate.py in sam_mask/. GT is used ONLY to
rank visualization examples, never used as a SAM prompt.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np

EXT = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')

def find(folder, stem):
    for ext in EXT:
        for e in (ext,ext.upper()):
            p=Path(folder)/(stem+e)
            if p.exists(): return p
    return None

def save(path, im):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),im): raise OSError(f'Failed to save {path}')

def mask(path, hw):
    a=cv2.imread(str(path),0)
    if a is None: raise RuntimeError(f'Unreadable mask {path}')
    if a.shape!=hw: a=cv2.resize(a,(hw[1],hw[0]),interpolation=cv2.INTER_NEAREST)
    return a>127

def jet(prob):
    return cv2.applyColorMap(np.uint8(np.clip(prob,0,1)*255),cv2.COLORMAP_JET)

def fit(prob, hw):
    return cv2.resize(np.asarray(prob,np.float32),(hw[1],hw[0]),interpolation=cv2.INTER_LINEAR)

def file_stems(folder):
    if not Path(folder).exists(): return set()
    return {f.stem for f in Path(folder).glob('*.png')}

def score_existing(out,gt_dir,img_dir,limit,min_gain):
    b=Path(out)/'01_SoftLogits_Base'; r=Path(out)/'03_Refine_NoFlip'
    common=sorted(file_stems(b)&file_stems(r))
    scores=[]
    for stem in common[:limit] if limit else common:
        ip=find(img_dir,stem); gp=find(gt_dir,stem)
        if not ip or not gp: continue
        gt_im=cv2.imread(str(gp),0)
        if gt_im is None: continue
        hw=gt_im.shape; gt=gt_im>127
        x=mask(b/(stem+'.png'),hw); y=mask(r/(stem+'.png'),hw)
        a=float(np.mean(x!=gt)); z=float(np.mean(y!=gt)); gain=a-z
        if gain>=min_gain and np.mean(x!=y)>0.001:
            scores.append(dict(stem=stem,base_mae=a,refined_mae=z,gain=gain,change=float(np.mean(x!=y))))
    return sorted(scores,key=lambda e:(-e['gain'],e['refined_mae']))

def maskoverlay(bgr, m):
    out=bgr.copy(); paint=np.zeros_like(out);paint[:]=(40,190,40)
    out[m]=cv2.addWeighted(out,.65,paint,.35,0)[m]
    return out

def map_from_crop(array, rect, hw):
    h,w=hw; x0,y0,x1,y1=rect
    dst=np.zeros(hw,np.float32)
    if array is not None: dst[y0:y1,x0:x1]=fit(array,(y1-y0,x1-x0))
    return dst

def annotate(bgr, title, detail=''):
    im=bgr.copy(); h,w=im.shape[:2]
    scale=min(.7,max(.40,w/900))
    cv2.rectangle(im,(0,0),(w,min(h,52)),(245,245,245),-1)
    cv2.putText(im,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,scale,(20,20,20),2,cv2.LINE_AA)
    if detail: cv2.putText(im,detail,(8,42),cv2.FONT_HERSHEY_SIMPLEX,max(.36,scale*.75),(30,30,30),1,cv2.LINE_AA)
    return im

def tile(images, labels, width=300):
    panels=[]
    for im,lab in zip(images,labels):
        im=cv2.resize(im,(width,width),interpolation=cv2.INTER_AREA)
        panels.append(annotate(im,lab))
    return np.concatenate(panels,axis=1)

def build_config(src,args,out):
    cfg=src.Config(sam_checkpoint=args.sam_checkpoint,sam_model_type=args.sam_model_type,
                   img_dir=args.img_dir,scribble_dir=args.scribble_dir,
                   gt_dir=args.gt_dir,json_dir=args.json_dir,out_dir=str(out),
                   refine_iters=args.refine_iters,active_points=not args.no_active,
                   use_mask_memory=not args.no_memory,flip_gate=not args.no_flip_gate,
                   candidate_quality=True,save_probs=True)
    return cfg

def export_selected(src,args,rows,root):
    cfg=build_config(src,args,root/'rerun_masks')
    pipeline=src.Pipeline(cfg)
    # Capture actual logits/scores from the FINAL multimask decoder call. The
    # main saved refined mask uses the single-mask token and is kept separate.
    prediction=pipeline.engine.predictor
    original_predict=prediction.predict
    active={"rect":None,"hw":None,"capture":[]}
    def capture_predict(*aa,**kw):
        output=original_predict(*aa,**kw)
        if kw.get('multimask_output') is True:
            logits,scores,low=output
            active['capture'].append(dict(rect=active['rect'],hw=active['hw'],
                  logits=np.asarray(logits,np.float32).copy(),
                  scores=np.asarray(scores,np.float32).copy(),
                  low=np.asarray(low,np.float32).copy()))
        return output
    prediction.predict=capture_predict
    original_infer=pipeline._infer_single_box
    def tracked_infer(rgb,scribble,box,refine,do_crop=True,collect_quality=False):
        # Match exactly the crop/transform from the original pipeline.
        _,_,_,rect,_,_,_=pipeline._target_for_box(rgb,scribble,box,do_crop)
        active['rect']=tuple(rect); active['hw']=rgb.shape[:2]
        return original_infer(rgb,scribble,box,refine,do_crop,collect_quality)
    pipeline._infer_single_box=tracked_infer
    selected=[]
    for rank,row in enumerate(rows,1):
        stem=row['stem']; img_path=find(args.img_dir,stem)
        if not img_path: continue
        active['capture']=[]
        record=pipeline.process_image(str(img_path))
        if record is None: print('[WARN] Skipped:',stem); continue
        bgr=cv2.imread(str(img_path)); h,w=bgr.shape[:2]
        output_dir=root/f'{rank:02d}_{stem}'; output_dir.mkdir(parents=True,exist_ok=True)
        gt_path=find(args.gt_dir,stem)
        gt=cv2.imread(str(gt_path),0)>127
        base=mask(Path(cfg.base_dir)/(stem+'.png'),(h,w))
        refined=mask(Path(cfg.refine_dir)/(stem+'.png'),(h,w))
        final=mask(Path(cfg.profile_dir)/(stem+'.png'),(h,w))
        save(output_dir/'RGB.png',bgr)
        json_file=find(args.json_dir,stem)
        boxes=src.DataLoader.load_boxes(str(json_file),w,h) if json_file else []
        scribble_file=find(args.scribble_dir,stem)
        scr=cv2.imread(str(scribble_file),0) if scribble_file else None
        if scr is not None and scr.shape!=(h,w):
            scr=cv2.resize(scr,(w,h),interpolation=cv2.INTER_NEAREST)
        box_img=bgr.copy()
        for k,b in enumerate(boxes):
            x1,y1,x2,y2=np.round(b).astype(int)
            cv2.rectangle(box_img,(x1,y1),(x2,y2),(0,0,255),max(2,round(w/350)))
            cv2.putText(box_img,f'Box {k+1}',(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
        save(output_dir/'RGB_Boxes.png',box_img)
        prompt_img=box_img.copy()
        if scr is not None:
            # Show actual initial scribble-selected points, inversely mapping
            # model-frame coordinates into the original image/crop.
            for b in boxes:
                _,_,_,rect,pts,labs,_=pipeline._target_for_box(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),scr,b,True)
                if pts is None: continue
                x0,y0,x1,y1=rect
                for (xp,yp),lab in zip(pts,labs):
                    x=int(np.clip(round(x0+(xp+.5)*(x1-x0)/cfg.size-.5),0,w-1))
                    y=int(np.clip(round(y0+(yp+.5)*(y1-y0)/cfg.size-.5),0,h-1))
                    cv2.circle(prompt_img,(x,y),max(4,round(w/150)),(0,255,0) if lab else (0,0,255),-1)
                    cv2.circle(prompt_img,(x,y),max(4,round(w/150)),(255,255,255),1)
        save(output_dir/'RGB_Boxes_Points.png',prompt_img)
        for name,arr in [('GT',gt),('Base_Mask',base),('Refined_Mask',refined),('Final_Mask',final)]:
            save(output_dir/(name+'.png'),arr.astype(np.uint8)*255)
        save(output_dir/'RGB_BaseOverlay.png',maskoverlay(bgr,base))
        save(output_dir/'RGB_RefinedOverlay.png',maskoverlay(bgr,refined))
        saved_heat={}
        for short,title in [('base','Base'),('refine','Refined'),('flipgate','Final')]:
            fp=Path(cfg.out_dir)/(short+'_prob')/(stem+'.npy')
            if fp.exists():
                p=np.load(fp).astype(np.float32)
                p=fit(p,(h,w)); heat=jet(p)
                save(output_dir/(title+'_ProbabilityHeatmap.png'),heat)
                saved_heat[title]=heat
        # Actual three candidate masks/scores and logits; for multi-box one set per box.
        candidates_meta=[]
        for j,obj in enumerate(active['capture'],1):
            rect=obj['rect']; scores=obj['scores']; logits=obj['logits']
            obj_dir=output_dir/('Candidates' if len(active['capture'])==1 else f'Candidates_Box{j}')
            obj_dir.mkdir(exist_ok=True)
            masks=[]; hm=[]
            for idx in range(3):
                native=logits[idx]; p_native=src.sigmoid(native)
                p=map_from_crop(p_native,rect,(h,w))
                # hard mask threshold is exactly 0 logits, not thresholded interpolated logits
                hard=map_from_crop((native>0).astype(np.float32),rect,(h,w))>=.5
                save(obj_dir/f'Candidate_{idx+1}_Mask.png',hard.astype(np.uint8)*255)
                save(obj_dir/f'Candidate_{idx+1}_Heatmap.png',jet(p))
                # native logits saved unaltered for scientific reproducibility
                np.save(obj_dir/f'Candidate_{idx+1}_Logits_ModelFrame.npy',native.astype(np.float16))
                masks.append(cv2.cvtColor(hard.astype(np.uint8)*255,cv2.COLOR_GRAY2BGR)); hm.append(jet(p))
            save(obj_dir/'Three_Candidate_Masks.png',tile(masks,[f'M{i+1} score={scores[i]:.3f}' for i in range(3)]))
            save(obj_dir/'Three_Candidate_Heatmaps.png',tile(hm,[f'L{i+1}' for i in range(3)]))
            candidates_meta.append({'object':j,'crop_xyxy':list(rect),'scores':scores.tolist(),
                       'note':'Final-state multimask diagnostic. Not identical to saved Refined_Mask single-token output.'})
        # Good-correction pixels are GT-based illustration ONLY (not part of inference).
        good=(base!=gt)&(refined==gt); bad=(base==gt)&(refined!=gt)
        compare=bgr.copy(); compare[good]=(20,210,20); compare[bad]=(20,20,240)
        save(output_dir/'Improvement_GreenRed_GTReference.png',compare)
        imgs=[box_img,prompt_img,cv2.cvtColor(base.astype(np.uint8)*255,cv2.COLOR_GRAY2BGR),
              saved_heat.get('Base',np.zeros_like(bgr)),cv2.cvtColor(refined.astype(np.uint8)*255,cv2.COLOR_GRAY2BGR),
              saved_heat.get('Refined',np.zeros_like(bgr)),compare]
        save(output_dir/'Overview.png',tile(imgs,['RGB + Boxes','RGB + Box + Points','Base Mask','Base Prob','Refined Mask','Refined Prob','Correction (GT only)'],width=250))
        meta={**row,'rank':rank,'boxes':len(boxes),'route_type':record['route_type'],
              'candidate_sets':candidates_meta,'notes':[
               'GT used only for post-hoc case ranking and correction visualization.',
               'Model logits are from last multimask=True diagnostic decode, same final prompts and input mask memory.',
               'Refined single-mask output and multimask candidates need not coincide.',
               'ProbabilityHeatmap is sigmoid/logit probability, not raw logit coloring.',
               'FlipGate may make Final_Mask differ from Refined_Mask.']}
        (output_dir/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f'[{rank}/{len(rows)}] {stem} MAE improvement {row["gain"]:.4f}, candidates={len(active["capture"])}')
        selected.append(meta)
    (root/'selected_cases.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-script',default='./sam_mask/sam1_refine_crop_flipgate.py')
    p.add_argument('--sam-output',default='./softresult/SAM1_Optimized3',help='Existing output to choose cases from; optional')
    p.add_argument('--sam-model-type',default='vit_h',choices=['vit_h','vit_l','vit_b'])
    p.add_argument('--sam-checkpoint',default='./sam_mask/weights/sam_vit_h_4b8939.pth')
    p.add_argument('--img-dir',default='./data/Train/Imgs');p.add_argument('--scribble-dir',default='./data/Train/Scribble')
    p.add_argument('--gt-dir',default='./data/Train/GT');p.add_argument('--json-dir',default='./data/Train/box_label')
    p.add_argument('--out-dir',default='./gesp_four_cases');p.add_argument('--num',type=int,default=4)
    p.add_argument('--limit',type=int,default=100,help='Max images for fallback selection inference')
    p.add_argument('--min-gain',type=float,default=.003)
    p.add_argument('--refine-iters',type=int,default=1)
    p.add_argument('--no-memory',action='store_true');p.add_argument('--no-active',action='store_true')
    p.add_argument('--no-flip-gate',action='store_true',help='Disable FlipGate; recommended for controlled comparison')
    args=p.parse_args();root=Path(args.out_dir);root.mkdir(parents=True,exist_ok=True)
    import importlib.util
    source=Path(args.source_script)
    if not source.exists():p.error(f'SAM1 script not found: {source}')
    spec=importlib.util.spec_from_file_location('user_sam1_pipeline',source)
    src=importlib.util.module_from_spec(spec);sys.modules[spec.name]=src;spec.loader.exec_module(src)
    rows=score_existing(args.sam_output,args.gt_dir,args.img_dir,args.limit,args.min_gain)
    fallback_used=False
    if not rows:
        fallback_used=True
        print('[INFO] No qualifying existing masks. Running SAM1 on a subset to rank improvements...')
        probe_cfg=build_config(src,args,root/'ranking_run')
        probe_cfg.save_probs=False;probe_cfg.candidate_quality=False
        probe_cfg.flip_gate=False
        probe=src.Pipeline(probe_cfg)
        paths=src.DataLoader.list_images(args.img_dir)[:args.limit] if args.limit else src.DataLoader.list_images(args.img_dir)
        for i,path in enumerate(paths,1):
            try: probe.process_image(path)
            except Exception as e: print(f'[WARN] ranking {Path(path).name}: {e}')
            if i%20==0:print(f'[INFO] Ranking {i}/{len(paths)}')
        rows=score_existing(probe_cfg.out_dir,args.gt_dir,args.img_dir,None,args.min_gain)
    rows=rows[:max(0,args.num)]
    if not rows:
        print('[INFO] No improvements meeting threshold; lower --min-gain or increase --limit.');return
    export_selected(src,args,rows,root)
    if fallback_used:
        import shutil
        shutil.rmtree(root/'ranking_run',ignore_errors=True)
    print('[DONE] Figure-ready cases in:',root)

if __name__=='__main__': main()
