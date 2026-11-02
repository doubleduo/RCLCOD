#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-ready four-case visualizer for sam2_refine_crop_flipgate.py.
Put this script beside the ORIGINAL sam2_refine_crop_flipgate.py. No original file edits.
Uses its actual SAM2, box/crop, multi-object and active-point code; runs controlled
no-memory / no-new-points ablations to avoid confounding contributions.
Requires GT solely for case selection and MAE reporting, NEVER for prompts.
"""
import argparse, csv, json, os, importlib.util, sys
from pathlib import Path
from dataclasses import dataclass
import cv2
import numpy as np
from tqdm import tqdm

LABELS = {'small': 'Small object / Crop', 'multi': 'Multiple objects / Per-box',
          'points': 'Active pos/neg points', 'memory': 'Previous mask logits'}

def load_original(path):
    spec = importlib.util.spec_from_file_location('sam2_refine_crop_flipgate_original', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def mask(prob):
    return (np.asarray(prob) >= .5).astype(np.uint8) * 255

def mae(prob, gt):
    return float(np.mean((mask(prob) > 0) != (gt > 128)))

def read_data(core, cfg, image):
    base = Path(image).stem
    bgr = cv2.imread(str(image))
    if bgr is None: return None
    h, w = bgr.shape[:2]
    sp = core.DataLoader.find_file(cfg.scribble_dir, base, ('.png','.jpg'))
    gp = core.DataLoader.find_file(cfg.gt_dir, base, ('.png','.jpg'))
    jp = core.DataLoader.find_file(cfg.json_dir, base, ('.json',))
    if not sp or not gp or not jp: return None
    sc, gt = cv2.imread(sp, 0), cv2.imread(gp, 0)
    if sc is None or gt is None: return None
    if sc.shape != (h,w): sc = cv2.resize(sc, (w,h), interpolation=cv2.INTER_NEAREST)
    if gt.shape != (h,w): gt = cv2.resize(gt, (w,h), interpolation=cv2.INTER_NEAREST)
    boxes = core.DataLoader.load_boxes(jp, w, h)
    if not boxes: return None
    return dict(name=base,bgr=bgr,rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),sc=sc,gt=gt,boxes=boxes)

def predict_variants(core, pipe, sample, n):
    rgb, sc, boxes = sample['rgb'], sample['sc'], sample['boxes']
    cfg = pipe.cfg
    old = (cfg.refine_iters, cfg.active_points, cfg.use_crop)
    real_predict = pipe.engine._predict_once
    real_sample = pipe.engine.sample_active_point
    trace, recorded = [], []
    state = {'disable_memory':False, 'trace':False, 'step':0}
    def predict_wrapper(points, labels, box, memory):
        if state['trace']:
            if memory is not None and not state['disable_memory']:
                # Low-resolution logits supplied as historical SAM input.
                if not recorded: recorded.append(np.asarray(memory).copy())
            state['step'] += 1
        return real_predict(points, labels, box, None if state['disable_memory'] else memory)
    def sample_wrapper(prob, scribble, existing, roi=None):
        pt, lb, why = real_sample(prob, scribble, existing, roi=roi)
        if state['trace'] and pt is not None:
            region = pipe._cached_region or (0,0,rgb.shape[1],rgb.shape[0])
            x0,y0,x1,y1 = region
            x = x0 + (float(pt[0,0])+.5)*(x1-x0)/cfg.size-.5
            y = y0 + (float(pt[0,1])+.5)*(y1-y0)/cfg.size-.5
            trace.append(dict(x=x,y=y,label=int(lb[0]),reason=why,step=state['step']))
        return pt,lb,why
    pipe.engine._predict_once = predict_wrapper
    pipe.engine.sample_active_point = sample_wrapper
    def run(*, use_crop, iters, active, no_memory, refined, trace_on=False):
        cfg.use_crop, cfg.refine_iters, cfg.active_points = use_crop,iters,active
        state.update(disable_memory=no_memory,trace=trace_on,step=0)
        # Do not permit stale image embedding across variant/crop changes.
        pipe._cached_rgb = None
        pipe._cached_region = None
        if refined: return pipe.infer_refined(rgb,sc,boxes)[0]
        return pipe._infer_baseline(rgb,sc,boxes)
    try:
        base = run(use_crop=False,iters=0,active=False,no_memory=True,refined=False)
        geometry = run(use_crop=True,iters=0,active=False,no_memory=True,refined=True)
        repeat = run(use_crop=True,iters=n,active=False,no_memory=True,refined=True)
        point_only = run(use_crop=True,iters=n,active=True,no_memory=True,refined=True,trace_on=True)
        point_only_points = list(trace)
        trace.clear()
        memory_only = run(use_crop=True,iters=n,active=False,no_memory=False,refined=True,trace_on=True)
        memory_only_logits = recorded[0] if recorded else None
        recorded.clear()
        full = run(use_crop=True,iters=n,active=True,no_memory=False,refined=True,trace_on=True)
    finally:
        pipe.engine._predict_once, pipe.engine.sample_active_point = real_predict, real_sample
        cfg.refine_iters,cfg.active_points,cfg.use_crop = old
    return dict(base=base,geometry=geometry,repeat=repeat,point_only=point_only,
                memory_only=memory_only,full=full,active_points=point_only_points,full_active_points=trace,
                memory_logits=memory_only_logits)

def display_style(image, panel_width=420, panel_height=180):
    """Stroke and dot sizes chosen for the FINAL figure, not source resolution."""
    h, w = image.shape[:2]
    scale = min(panel_width / w, panel_height / h)
    # After downsizing, borders >= 3px and points >= 6px in the fitted image.
    stroke = max(3, int(np.ceil(3.5 / scale)))
    dot = max(5, int(np.ceil(7.0 / scale)))
    halo = max(2, int(np.ceil(2.0 / scale)))
    label_scale = max(.7, 0.75 / scale)
    return stroke, dot, halo, label_scale


def draw_boxes(image, boxes, color=(0, 165, 255), thickness=None,
               numbered=False, panel_width=420, panel_height=180):
    out = image.copy()
    stroke, _, halo, label_scale = display_style(out, panel_width, panel_height)
    thickness = stroke if thickness is None else max(stroke, thickness)
    for i, b in enumerate(boxes):
        x0,y0,x1,y1 = np.round(b).astype(int)
        # Black outer rim makes the box legible against sand / grass / bright RGB.
        cv2.rectangle(out,(x0,y0),(x1,y1),(0,0,0), thickness+2*halo, cv2.LINE_AA)
        cv2.rectangle(out,(x0,y0),(x1,y1),color,thickness,cv2.LINE_AA)
        if numbered:
            tx,ty=x0+4*halo,max(y0+int(22/ min(panel_width/out.shape[1],panel_height/out.shape[0])),20)
            cv2.putText(out,str(i+1),(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,
                        label_scale,(0,0,0),max(4,stroke+halo),cv2.LINE_AA)
            cv2.putText(out,str(i+1),(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,
                        label_scale,(255,255,255),max(2,stroke//2),cv2.LINE_AA)
    return out


def memory_heatmap(logits):
    if logits is None: return None
    z=np.asarray(logits,np.float32).squeeze()
    if z.ndim!=2: return None
    z=1/(1+np.exp(-np.clip(z,-25,25)))
    return cv2.applyColorMap((z*255).astype(np.uint8),cv2.COLORMAP_TURBO)


def annotate(bgr,boxes,kind,points,core,cfg,memory_logits=None,
             panel_width=420,panel_height=180):
    img=draw_boxes(bgr,boxes,numbered=kind=='multi',
                   panel_width=panel_width,panel_height=panel_height)
    h,w=img.shape[:2]
    stroke,rad,halo,_=display_style(img,panel_width,panel_height)
    if kind=='small':
        for b in boxes:
            c=core.centered_square_crop(b,w,h,cfg.crop_context,cfg.crop_min_context,cfg.crop_square)
            if tuple(c)!=(0,0,w,h):
                cv2.rectangle(img,(c[0],c[1]),(c[2]-1,c[3]-1),
                              (0,0,0),stroke+2*halo,cv2.LINE_AA)
                cv2.rectangle(img,(c[0],c[1]),(c[2]-1,c[3]-1),
                              (50,240,75),stroke,cv2.LINE_AA)
    elif kind=='points':
        for p in points:
            x,y=int(round(p['x'])),int(round(p['y']))
            color=(45,210,45) if p['label']==1 else (35,40,240)
            cv2.circle(img,(x,y),rad+halo+halo,(0,0,0),-1,cv2.LINE_AA)
            cv2.circle(img,(x,y),rad+halo,(255,255,255),-1,cv2.LINE_AA)
            cv2.circle(img,(x,y),rad,color,-1,cv2.LINE_AA)
            if p['label']==0:
                arm=max(3,rad//2)
                cv2.line(img,(x-arm,y-arm),(x+arm,y+arm),(255,255,255),max(2,halo),cv2.LINE_AA)
                cv2.line(img,(x-arm,y+arm),(x+arm,y-arm),(255,255,255),max(2,halo),cv2.LINE_AA)
    elif kind=='memory':
        heat=memory_heatmap(memory_logits)
        if heat is not None:
            iw=max(55,int(w*.31)); ih=max(55,int(h*.38))
            iw=min(iw,w-10);ih=min(ih,h-10)
            x0,y0=w-iw-7,7
            heat=cv2.resize(heat,(iw,ih),interpolation=cv2.INTER_LINEAR)
            img[y0:y0+ih,x0:x0+iw]=heat
            cv2.rectangle(img,(x0,y0),(x0+iw-1,y0+ih-1),(255,255,255),max(3,stroke),cv2.LINE_AA)
            text_scale=max(.65, .5/min(panel_width/w,panel_height/h))
            cv2.putText(img,'Prev logits',(max(2,x0),min(h-5,y0+ih+int(18/min(panel_width/w,panel_height/h)))),
                        cv2.FONT_HERSHEY_SIMPLEX,text_scale,(0,0,0),max(4,stroke+halo),cv2.LINE_AA)
            cv2.putText(img,'Prev logits',(max(2,x0),min(h-5,y0+ih+int(18/min(panel_width/w,panel_height/h)))),
                        cv2.FONT_HERSHEY_SIMPLEX,text_scale,(255,255,255),max(2,stroke//2),cv2.LINE_AA)
    return img


def panel_rect(image, width=420, height=180, is_mask=False):
    # Fit, never distort spatial geometry, letterbox to a FLAT rectangle.
    h,w=image.shape[:2]
    scale=min(width/w,height/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    bg=0 if is_mask else 242
    out=np.full((height,width,3),bg,dtype=np.uint8)
    interp=cv2.INTER_NEAREST if is_mask else (cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    sub=cv2.resize(image,(nw,nh),interpolation=interp)
    yy=(height-nh)//2;xx=(width-nw)//2
    out[yy:yy+nh,xx:xx+nw]=sub
    return out


def export_case(root,kind,sample,res,core,cfg,comparison,gain, panel_w, panel_h):
    name=sample['name']; folder=root/kind/name
    folder.mkdir(exist_ok=True,parents=True)
    before_key,after_key=comparison
    before,after=res[before_key],res[after_key]
    boxes=sample['boxes']
    rgb_box=draw_boxes(sample['bgr'], boxes, panel_width=panel_w,panel_height=panel_h)
    annotated=annotate(sample['bgr'],boxes,kind,res['active_points'],core,cfg,res['memory_logits'],panel_w,panel_h)
    assets={'RGB.png':sample['bgr'], 'RGB_Boxes.png':rgb_box,
            'GT.png':mask(sample['gt']/255.0),
            'Before_Mask.png':mask(before),'After_Mask.png':mask(after),
            'Strategy_RGB.png':annotated}
    heat=memory_heatmap(res['memory_logits']) if kind=='memory' else None
    if heat is not None: assets['Memory_Probability_Heatmap.png']=heat
    for fn,im in assets.items(): cv2.imwrite(str(folder/fn),im)
    info={'image':name,'kind':kind,'before':before_key,'after':after_key,
          'before_mae':mae(before,sample['gt']),'after_mae':mae(after,sample['gt']),
          'gain':gain,'boxes':[list(map(float,b)) for b in boxes],
          'active_points':res['active_points'] if kind=='points' else [],
          'note':'No flip. Paired SAM2 controlled comparison; GT for ranking only.'}
    (folder/'info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    cells=[]
    for im,ismask in [(rgb_box,False),(cv2.cvtColor(mask(before),cv2.COLOR_GRAY2BGR),True),
                      (annotated,False),(cv2.cvtColor(mask(after),cv2.COLOR_GRAY2BGR),True)]:
        cells.append(panel_rect(im,panel_w,panel_h,ismask))
    return np.concatenate(cells,axis=1)


def make_overview(out, rows, best, comparisons, panel_w, panel_h):
    # Consistent four-column, horizontal rectangular panels. Export all images
    # independently too, so captions can be finalized in Illustrator/PPT.
    canvas_w=4*panel_w
    header=np.full((48,canvas_w,3),255,np.uint8)
    font=cv2.FONT_HERSHEY_SIMPLEX
    for i,heading in enumerate(('RGB + Boxes','Before Mask','Strategy on RGB','After Mask')):
        cv2.putText(header,heading,(i*panel_w+15,31),font,.7,(45,45,45),2,cv2.LINE_AA)
    pieces=[header]
    for kind in comparisons:
        if kind not in rows:continue
        labelbar=np.full((43,canvas_w,3),248,np.uint8)
        gain=best[kind]['gain'] if isinstance(best[kind],dict) else best[kind][0]
        text=f'{kind.upper()}  |  {LABELS[kind]}  |  MAE gain: {gain:.4f}'
        cv2.putText(labelbar,text,(15,29),font,.72,(35,35,35),2,cv2.LINE_AA)
        pieces.extend([labelbar, rows[kind]])
        pieces.append(np.full((8,canvas_w,3),255,np.uint8))
    cv2.imwrite(str(out/'Four_Case_Overview.png'),np.concatenate(pieces,axis=0))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default='sam2_refine_crop_flipgate.py')
    ap.add_argument('--output-dir',default='paper_four_cases')
    ap.add_argument('--img-dir',default='./data/Train/Imgs')
    ap.add_argument('--scribble-dir',default='./data/Train/Scribble')
    ap.add_argument('--gt-dir',default='./data/Train/GT')
    ap.add_argument('--json-dir',default='./data/Train/box_label')
    ap.add_argument('--sam2-checkpoint',default='./sam_pseado/weights/sam2_hiera_large.pt')
    ap.add_argument('--sam2-config',default='sam2_hiera_l.yaml')
    ap.add_argument('--limit',type=int,default=100, help='Uniformly sample this many images before case selection (default: random 100)')
    ap.add_argument('--refine-iters',type=int,default=2)
    ap.add_argument('--min-gain',type=float,default=.003)
    ap.add_argument('--topk-each',type=int,default=10,help='Save up to this many distinct improving cases per strategy (default 10)')
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--device',default=None)
    ap.add_argument('--panel-width',type=int,default=420)
    ap.add_argument('--panel-height',type=int,default=180)
    args=ap.parse_args()
    if args.refine_iters<1: ap.error('--refine-iters must be >= 1')
    source = Path(args.source).expanduser()
    if not source.is_absolute():
        candidates = [Path.cwd() / source,
                      Path(__file__).resolve().parent / source,
                      Path.cwd() / 'sam_mask' / source]
        source = next((p for p in candidates if p.is_file()), candidates[0])
    if not source.is_file():
        ap.error(f"Original script not found: {source}. Use --source with the correct file path.")
    core=load_original(source.resolve())
    cfg=core.Config(sam2_checkpoint=args.sam2_checkpoint,sam2_config=args.sam2_config,
                    img_dir=args.img_dir,scribble_dir=args.scribble_dir,gt_dir=args.gt_dir,
                    json_dir=args.json_dir,out_dir=args.output_dir,
                    refine_iters=args.refine_iters,flip_gate=False)
    if args.device: cfg.device=args.device
    # Note: constructing Pipeline also creates its standard directories, but this
    # visualizer does not call process_image and does not write original CSVs.
    pipe=core.Pipeline(cfg)
    rng=np.random.default_rng(args.seed)
    paths=core.DataLoader.list_images(cfg.img_dir)
    rng.shuffle(paths)
    paths=paths[:args.limit] if args.limit>0 else paths
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'random_sampled_files.txt').write_text('\n'.join(map(str,paths))+'\n',encoding='utf-8')
    print(f'[Sample] Randomly selected {len(paths)} images (seed={args.seed}).')
    comparisons={'small':('base','geometry'),'multi':('base','geometry'),
                 'points':('repeat','point_only'),'memory':('repeat','memory_only')}
    # Select by improvement only; original size, aspect ratio and foreground fraction
    # are recorded for reference but NEVER used for filtering or ranking.
    errors=[]; ranking={k:[] for k in comparisons}
    for path in tqdm(paths,desc='Find comparable SAM2 cases'):
        try:
            sample=read_data(core,cfg,path)
            if sample is None: continue
            h,w=sample['bgr'].shape[:2]
            aspect=w/h
            res=predict_variants(core,pipe,sample,args.refine_iters)
            ratios=[float((b[2]-b[0])*(b[3]-b[1])/(w*h)) for b in sample['boxes']]
            fg_fraction=float(np.mean(sample['gt']>128))
            if fg_fraction <= 0: continue
            for kind,pair in comparisons.items():
                if kind=='small' and not (len(ratios)==1 and ratios[0]<=cfg.small_box_area_thr): continue
                if kind=='multi' and len(ratios)<2: continue
                if kind=='points' and not res['active_points']: continue
                if kind=='memory' and res['memory_logits'] is None: continue
                gain=mae(res[pair[0]],sample['gt'])-mae(res[pair[1]],sample['gt'])
                if gain < args.min_gain: continue
                ranking[kind].append(dict(name=sample['name'],path=str(path),gain=float(gain),
                                          aspect=float(aspect),foreground=float(fg_fraction),
                                          width=w,height=h,box_fraction=float(sum(ratios))))
        except Exception as ex:
            errors.append({'image':str(path),'error':repr(ex)})
            print(f'[WARN] {Path(path).name}: {ex}')
    for kind in ranking:
        ranking[kind].sort(key=lambda x:x['gain'],reverse=True)
    # Save up to top-k independent cases PER category; no cross-category size
    # restrictions and no global four-image requirement. The same source image
    # is allowed in different categories, because the interventions differ.
    available={k:len(v) for k,v in ranking.items()}
    print('[Candidates passing improvement/category filters]',available)
    if args.topk_each < 1: ap.error('--topk-each must be >= 1')
    kinds=list(comparisons)
    selected={k: ranking[k][:args.topk_each] for k in kinds}
    rows={}; overview_best={}; selection={}
    for kind in kinds:
        selection[kind]=[]
        for rank, c in enumerate(selected[kind],1):
            try:
                sample=read_data(core,cfg,c['path'])
                if sample is None:
                    print(f'[WARN] {kind}/{c["name"]}: input files missing on second read')
                    continue
                res=predict_variants(core,pipe,sample,args.refine_iters)
                panel=export_case(out,kind,sample,res,core,cfg,comparisons[kind],c['gain'],args.panel_width,args.panel_height)
                entry={'rank':rank,'name':c['name'],'gain_mae':round(c['gain'],6),
                       'original_size':[c['width'],c['height']],
                       'aspect_ratio':round(c['aspect'],3),
                       'fg_fraction':round(c['foreground'],4),
                       'comparison':list(comparisons[kind]),
                       'output_dir':str(out/kind/c['name'])}
                selection[kind].append(entry)
                if kind not in rows:
                    rows[kind]=panel
                    overview_best[kind]=c
            except Exception as ex:
                errors.append({'image':c['path'],'category':kind,'error':repr(ex)})
                print(f'[WARN] Export {kind}/{c["name"]}: {ex}')
        print(f'[Saved] {kind}: {len(selection[kind])} cases in {out/kind}')
    if rows:
        make_overview(out,rows,overview_best,comparisons,args.panel_width,args.panel_height)
    (out/'selected_cases.json').write_text(json.dumps({
        'selected':selection,'candidate_counts':available,
        'topk_each':args.topk_each,'ranking':ranking,'errors':errors},
        indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved count:',{k:len(v) for k,v in selection.items()})
    print('Output:',out.resolve())

if __name__=='__main__':main()
