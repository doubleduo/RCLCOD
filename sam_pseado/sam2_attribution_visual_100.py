#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled SAM2 qualitative attribution: random 100 images, geometry / prompts / memory.

Uses sam2_refine_core.py (copied from user's uploaded baseline) in same directory.
Every contrast uses matching initial prompts and only the named switch changes.
GT is read ONLY for evaluation/selection of qualitative examples.
"""
import argparse, csv, json, os, random, math
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sam2_refine_core import (Config, DataLoader, SAM2Engine, union_box,
    centered_square_crop, transform_box_to_model, resize_prob, expand_box,
    MetricsEvaluator, sigmoid)

def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name, default in [('img-dir','./data/Train/Imgs'),('scribble-dir','./data/Train/Scribble'),('gt-dir','./data/Train/GT'),('json-dir','./data/Train/box_label'),('sam2-checkpoint','./sam_pseado/weights/sam2_hiera_large.pt'),('sam2-config','sam2_hiera_l.yaml'),('output-dir','./softresult/attribution_random100')]:
        p.add_argument('--'+name,default=default)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--top-k',type=int,default=12)
    p.add_argument('--min-gain',type=float,default=0.003)
    p.add_argument('--pos-points',type=int,default=3)
    p.add_argument('--neg-points',type=int,default=3)
    p.add_argument('--size',type=int,default=1024)
    p.add_argument('--crop-thr',type=float,default=.10)
    p.add_argument('--crop-context',type=float,default=.35)
    p.add_argument('--perbox-margin',type=int,default=25)
    p.add_argument('--save-all-panels',action='store_true')
    p.add_argument('--no-stage-panels',action='store_true',help='Disable saving panels; by default always save all panels')
    return p.parse_args()

def xy_image_frame(model_xy, region, n):
    x0,y0,x1,y1=region
    return [(float(x0+(x+.5)*(x1-x0)/n-.5),float(y0+(y+.5)*(y1-y0)/n-.5)) for x,y in model_xy]

def make_input(rgb, sc, box, cfg, crop):
    h,w=rgb.shape[:2]; region=(0,0,w,h)
    area=float((box[2]-box[0])*(box[3]-box[1])/(w*h))
    if crop and area<=cfg.small_box_area_thr:
        region=centered_square_crop(box,w,h,cfg.crop_context,cfg.crop_min_context,cfg.crop_square)
    x0,y0,x1,y1=region
    im=cv2.resize(rgb[y0:y1,x0:x1],(cfg.size,cfg.size),interpolation=cv2.INTER_LINEAR)
    scrib=cv2.resize(sc[y0:y1,x0:x1],(cfg.size,cfg.size),interpolation=cv2.INTER_NEAREST)
    mb=transform_box_to_model(box,region,cfg.size,cfg.box_prompt_margin)
    fg,bg=DataLoader.parse_scribble(scrib)
    # faithful to source: ignore positive strokes outside current box
    xlo,ylo,xhi,yhi=np.round(mb).astype(int)
    roi=np.zeros((cfg.size,cfg.size),bool);roi[max(0,ylo):min(cfg.size,yhi+1),max(0,xlo):min(cfg.size,xhi+1)]=True
    scrib=scrib.copy();scrib[fg & ~roi]=0
    pts,labs=DataLoader.points_from_scribble(scrib,cfg.max_initial_per_class)
    return im,scrib,mb,region,pts,labs

def distribute(mask, score, oldpts, k, min_dist=22, candidate_limit=20000):
    """Greedy reliability-then-coverage; avoids any point already supplied."""
    if not k or not mask.any():return []
    work=np.where(mask,np.asarray(score,np.float32),-1e6).copy()
    out=[]
    radius=int(min_dist)
    for x,y in ([] if oldpts is None else np.asarray(oldpts).reshape(-1,2)):
        cv2.circle(work,(int(round(x)),int(round(y))),radius,-1e6,-1)
    for _ in range(k):
        y,x=np.unravel_index(np.argmax(work),work.shape)
        if work[y,x]<-1e5:break
        out.append((float(x),float(y)))
        cv2.circle(work,(int(x),int(y)),radius,-1e6,-1)
    return out

def get_active_points(prob,sc,box,initial,kp,kn):
    """Fixed reliable positive & negative candidate sets reused by all paired comparisons.
    Positive: known scribble mistakes first, else eroded foreground away from known negatives.
    Negative: known scribble mistakes first, else strictly outside original box.
    Unlabeled interior never assigned a negative label.
    """
    fg,bg=DataLoader.parse_scribble(sc)
    h,w=prob.shape
    old=initial if initial is not None else np.empty((0,2),np.float32)
    x1,y1,x2,y2=np.round(box).astype(int)
    inbox=np.zeros((h,w),bool);inbox[max(0,y1):min(h,y2+1),max(0,x1):min(w,x2+1)]=True
    p=prob.astype(np.float32)
    badpos=fg & (p<.5) & inbox
    badneg=bg & (p>=.5)
    # build positive mask strictly inside eroded predicted foreground where confidence high
    consensus_safe=cv2.erode((p>.85).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    safe_pos=consensus_safe & inbox & ~bg
    pos=distribute(badpos,1-p,old,kp,min_dist=16)
    if len(pos)<kp:
        cur=np.concatenate((old,np.asarray(pos,np.float32).reshape(-1,2)),axis=0)
        pos+=distribute(safe_pos,p,cur,kp-len(pos),min_dist=24)
    # known negatives anywhere; otherwise outside annotated box (more reliable than inside-box guesses)
    cur=np.concatenate((old,np.asarray(pos,np.float32).reshape(-1,2)),axis=0)
    neg=distribute(badneg,p,cur,kn,min_dist=16)
    if len(neg)<kn:
        outside=~inbox
        radius=max(8,int(min(x2-x1,y2-y1)*.06))
        # prefer close but not adjacent border; avoid image-wide distant pixels
        expanded=cv2.dilate(inbox.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
        ring=outside & expanded
        yy,xx=np.indices((h,w));cx=(x1+x2)/2;cy=(y1+y2)/2
        cur=np.concatenate((cur,np.asarray(neg,np.float32).reshape(-1,2)),axis=0)
        neg+=distribute(ring,1/(1+np.sqrt((xx-cx)**2+(yy-cy)**2)),cur,kn-len(neg),min_dist=32)
    return np.asarray(pos,np.float32).reshape(-1,2),np.asarray(neg,np.float32).reshape(-1,2)

def do_predict(engine,im,pts,labs,box,memory=None):
    engine.set_image(im)
    return engine._predict_once(pts,labs,box,memory)

def infer_box(engine,rgb,sc,box,cfg,crop=False,second='none',kp=3,kn=3):
    im,msc,mb,region,pts,labs=make_input(rgb,sc,box,cfg,crop)
    p0,low=do_predict(engine,im,pts,labs,mb)
    extras={'pos':[],'neg':[],'old':xy_image_frame(pts,region,cfg.size) if pts is not None else [],'crop_rect':list(region),'box':[float(v) for v in box]}
    if second!='none':
        pos,neg=get_active_points(p0,msc,mb,pts,kp,kn)
        # enable toggles independently: first pass p0 never changes
        takepos=pos if second in ('p', 'pn', 'pnm') else np.empty((0,2),np.float32)
        takeneg=neg if second in ('n', 'pn', 'pnm') else np.empty((0,2),np.float32)
        ext=np.concatenate((takepos,takeneg),axis=0)
        elab=np.concatenate((np.ones(len(takepos),np.int32),np.zeros(len(takeneg),np.int32)))
        p2=pts if not len(ext) else (np.concatenate((pts,ext)) if pts is not None else ext)
        l2=labs if not len(ext) else (np.concatenate((labs,elab)) if labs is not None else elab)
        memory=low if second in ('m', 'pnm') else None
        p0,_=do_predict(engine,im,p2,l2,mb,memory)
        extras['pos']=xy_image_frame(takepos,region,cfg.size)
        extras['neg']=xy_image_frame(takeneg,region,cfg.size)
    x0,y0,x1,y1=region;h,w=rgb.shape[:2]
    pred=np.zeros((h,w),np.float32)
    pred[y0:y1,x0:x1]=resize_prob(p0,x1-x0,y1-y0)
    return pred,extras

def perbox(engine,rgb,sc,boxes,cfg,crop=False,second='none',kp=3,kn=3,margin=25):
    h,w=rgb.shape[:2]; merged=np.zeros((h,w),np.float32);info={'pos':[],'neg':[],'old':[],'boxes':[],'crop_rects':[]}
    for b in boxes:
        p,ext=infer_box(engine,rgb,sc,b,cfg,crop,second,kp,kn)
        m=max(2,round(margin*max(h,w)/cfg.size));gb=expand_box(b,m,w,h)
        x1,y1,x2,y2=np.round(gb).astype(int)
        gate=np.zeros((h,w),np.float32);gate[y1:y2+1,x1:x2+1]=1
        merged=np.maximum(merged,p*gate)
        for key in ('pos','neg','old'):info[key].extend(ext[key])
        info['boxes'].append(ext['box']);info['crop_rects'].append(ext['crop_rect'])
    return merged,info

def binary_mae(p,gt):
    return float(np.mean((p>=.5)!=(gt>128)))

def choose_stages(engine,rgb,sc,boxes,cfg,a):
    # Geometry is *strictly* first-pass only; other variables fixed.
    b=union_box(boxes)
    o={};d={}
    o['G0_Box'],d['G0_Box']=infer_box(engine,rgb,sc,b,cfg,crop=False)
    o['G1_Crop'],d['G1_Crop']=infer_box(engine,rgb,sc,b,cfg,crop=True)
    # only apply per-box on images containing multiple boxes
    if len(boxes)>1:
        o['G2_PerBox'],d['G2_PerBox']=perbox(engine,rgb,sc,boxes,cfg,False,margin=a.perbox_margin)
        o['G3_PerBoxCrop'],d['G3_PerBoxCrop']=perbox(engine,rgb,sc,boxes,cfg,True,margin=a.perbox_margin)
    else:
        o['G2_PerBox']=o['G0_Box'].copy();d['G2_PerBox']=d['G0_Box']
        o['G3_PerBoxCrop']=o['G1_Crop'].copy();d['G3_PerBoxCrop']=d['G1_Crop']
    # Hold geometry fixed at G3, then toggle individual prompt and memory factors.
    # Independent baselines expose real contributions rather than silently mixing them.
    def run(mode):
        if len(boxes)>1:return perbox(engine,rgb,sc,boxes,cfg,True,second=mode,kp=a.pos_points,kn=a.neg_points,margin=a.perbox_margin)
        return infer_box(engine,rgb,sc,b,cfg,True,second=mode,kp=a.pos_points,kn=a.neg_points)
    for key,mode in [('P0_Repeat','repeat'),('P1_Pos','p'),('P2_Neg','n'),('P3_PosNeg','pn'),('M0_Memory','m'),('M1_PosNegMemory','pnm')]:
        o[key],d[key]=run(mode)
    return o,d

def panel(rgb,gt,preds,diagnostic,label,width=260):
    h,w=rgb.shape[:2]; scale=width/w;Ht=max(1,int(round(h*scale)))
    col=[]
    rgb_vis=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(width,Ht))
    gt_bin=gt>128
    rows=[('RGB',rgb_vis),('GT',cv2.resize((gt_bin.astype('uint8')*255),(width,Ht),interpolation=cv2.INTER_NEAREST))]
    for nm in ('G0_Box','G1_Crop','G2_PerBox','G3_PerBoxCrop','P0_Repeat','P1_Pos','P2_Neg','P3_PosNeg','M0_Memory','M1_PosNegMemory'):
        if nm not in preds:continue
        pred=(preds[nm]>=.5)
        vis=cv2.cvtColor((pred.astype(np.uint8)*255),cv2.COLOR_GRAY2BGR)
        gt_here=gt_bin
        if vis.shape[:2]!=gt_here.shape:gt_here=cv2.resize(gt_here.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)>0
        # TP grey/white, FP red, FN blue: color by overlay
        vis[(pred)&(~gt_here)]=(55,55,235)
        vis[(~pred)&(gt_here)]=(220,105,40)
        vis=cv2.resize(vis,(width,Ht),interpolation=cv2.INTER_NEAREST)
        rows.append((nm,vis))
    for nm,img in rows:
        im=img if img.ndim==3 else cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        im=cv2.copyMakeBorder(im,30,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
        if nm in preds:
            mae=binary_mae(preds[nm],gt)
            caption=f'{nm}   MAE {mae:.4f}'
        else:caption=nm
        cv2.putText(im,caption,(6,21),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
        if nm=='M1_PosNegMemory':
            for field,color in [('pos',(30,220,30)),('neg',(30,30,235))]:
                for x,y in diagnostic[nm].get(field,[]):
                    yy=int(round(y*Ht/h))+30;xx=int(round(x*width/w))
                    cv2.circle(im,(xx,yy),4,color,-1);cv2.circle(im,(xx,yy),4,(255,255,255),1)
        col.append(im)
    # contact sheet, 4 columns
    tile_h=max(i.shape[0] for i in col);tile_w=width
    canv=np.full((math.ceil(len(col)/4)*tile_h,4*tile_w,3),245,np.uint8)
    for idx,im in enumerate(col):
        rr,cc=divmod(idx,4);canv[rr*tile_h:rr*tile_h+im.shape[0],cc*tile_w:(cc+1)*tile_w]=im
    cv2.putText(canv,'Red = FP    Blue = FN',(6,canv.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,.40,(0,0,0),1)
    return canv

def main():
    a=args_parser();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=Config(sam2_checkpoint=a.sam2_checkpoint,sam2_config=a.sam2_config,img_dir=a.img_dir,scribble_dir=a.scribble_dir,gt_dir=a.gt_dir,json_dir=a.json_dir,out_dir=a.output_dir)
    cfg.size=a.size;cfg.small_box_area_thr=a.crop_thr;cfg.crop_context=a.crop_context
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    allimgs=DataLoader.list_images(a.img_dir)
    rng=random.Random(a.seed);images=rng.sample(allimgs,min(a.limit,len(allimgs)))
    (out/'sampled_files.txt').write_text('\n'.join(images)+'\n',encoding='utf8')
    engine=SAM2Engine(cfg)
    pairs=[('Crop_effect','G0_Box','G1_Crop'),('PerBox_effect','G0_Box','G2_PerBox'),
           ('Crop_after_PerBox','G2_PerBox','G3_PerBoxCrop'),('PerBox_after_Crop','G1_Crop','G3_PerBoxCrop'),
           ('Repeat_effect','G3_PerBoxCrop','P0_Repeat'),('Pos_effect','P0_Repeat','P1_Pos'),
           ('Neg_effect','P0_Repeat','P2_Neg'),('PosNeg_effect','P0_Repeat','P3_PosNeg'),
           ('Memory_effect','P0_Repeat','M0_Memory'),('Memory_on_PosNeg','P3_PosNeg','M1_PosNegMemory'),
           ('Points_on_Memory','M0_Memory','M1_PosNegMemory'),('Overall','G0_Box','M1_PosNegMemory')]
    records=[]; failures=[];summaries=[]
    for imagepath in tqdm(images,desc='SAM2 attribution'):
        stem=Path(imagepath).stem
        try:
            im=cv2.imread(imagepath);assert im is not None
            rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
            scp=DataLoader.find_file(a.scribble_dir,stem,('.png','.jpg'))
            gtp=DataLoader.find_file(a.gt_dir,stem,('.png','.jpg'))
            bp=DataLoader.find_file(a.json_dir,stem,('.json',))
            if not scp or not gtp:raise ValueError('missing Scribble or GT')
            sc=cv2.imread(scp,cv2.IMREAD_GRAYSCALE);gt=cv2.imread(gtp,cv2.IMREAD_GRAYSCALE)
            if sc is None or gt is None:raise ValueError('unreadable Scribble/GT')
            if sc.shape!=(h,w):sc=cv2.resize(sc,(w,h),interpolation=cv2.INTER_NEAREST)
            boxes=DataLoader.load_boxes(bp,w,h)
            if not boxes:raise ValueError('no boxes')
            preds,diagnostics=choose_stages(engine,rgb,sc,boxes,cfg,a)
            data={'image':Path(imagepath).name,'num_boxes':len(boxes),'box_area_ratio':float((union_box(boxes)[2]-union_box(boxes)[0])*(union_box(boxes)[3]-union_box(boxes)[1])/(w*h))}
            for name,mask in preds.items():data['mae_'+name]=binary_mae(mask,gt)
            for title,base,changed in pairs:data['gain_'+title]=data['mae_'+base]-data['mae_'+changed]
            data['num_pos']=len(diagnostics['M1_PosNegMemory'].get('pos',[]))
            data['num_neg']=len(diagnostics['M1_PosNegMemory'].get('neg',[]))
            records.append(data)
            summaries.append((stem,im,gt,preds,diagnostics))
            if not a.no_stage_panels:
                (out/'all_panels').mkdir(exist_ok=True)
                cv2.imwrite(str(out/'all_panels'/f'{stem}.jpg'),panel(rgb,gt,preds,diagnostics,stem))
        except Exception as e:
            failures.append({'image':Path(imagepath).name,'error':repr(e)})
            print('[WARN]',stem,repr(e))
    if records:
        with open(out/'per_image_metrics.csv','w',newline='',encoding='utf-8-sig') as f:
            wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
        lookup={r['image'].rsplit('.',1)[0]:r for r in records}
        for title,base,changed in pairs:
            group=out/'best_examples'/title;group.mkdir(parents=True,exist_ok=True)
            ranked=sorted(records,key=lambda r:r['gain_'+title],reverse=True)
            winners=[r for r in ranked if r['gain_'+title]>=a.min_gain][:a.top_k]
            # always also save a ranking CSV even if nothing passes threshold
            with open(group/'ranking.csv','w',newline='',encoding='utf-8-sig') as f:
                wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(ranked)
            wanted=set(r['image'].rsplit('.',1)[0] for r in winners)
            for stem,im,gt,preds,diag in summaries:
                if stem in wanted:
                    rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                    gain=lookup[stem]['gain_'+title]
                    cv2.imwrite(str(group/f'{gain:+.4f}_{stem}.jpg'),panel(rgb,gt,preds,diag,stem))
        report={}
        for title,base,changed in pairs:
            vals=np.asarray([r['gain_'+title] for r in records]);report[title]={
                'baseline':base,'variant':changed,'mean_gain_mae':float(vals.mean()),
                'improved':int((vals>1e-8).sum()),'worsened':int((vals<-1e-8).sum()),
                'equal':int((abs(vals)<=1e-8).sum()),
                'gain_gt_0_003':int((vals>.003).sum()),'gain_gt_0_01':int((vals>.01).sum()),
                'gain_gt_0_05':int((vals>.05).sum()),'hurt_gt_0_01':int((vals<-.01).sum())}
        json.dump({'sample_count':len(images),'evaluated':len(records),'seed':a.seed,'notes':['Crop contrast includes changed model input scale','Per-box contrast valid only for multi-box images','All prompts originate from Scribble, predicted mask, or box geometry; GT used only for evaluation','P0 Repeat performs two decodings, no memory, no new points','The cropped and per-box branches may use different SAM2 embeddings','Binary MAE is pixel error rate at threshold 0.5'],'effects':report,'failures':failures},open(out/'report.json','w'),indent=2,ensure_ascii=False)
        print(json.dumps(report,indent=2))
    else:print('No valid images processed; see warnings')
if __name__=='__main__':main()
