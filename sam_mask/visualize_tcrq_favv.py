#!/usr/bin/env python3
"""Render saved three-candidate SIM, evidence effects and FAVV decisions; no SAM/GPU needed."""
import argparse
import csv
import html
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

GROUP_COLORS={'clean':'#21835b','noisy':'#cf9b25','unvalue':'#c4484d'}


def number(row,key,default=float('nan')):
    try: return float(row.get(key,default))
    except (TypeError,ValueError): return default


def flag(row,key):
    return number(row,key,0)==1


def load_cache(path):
    with np.load(path,allow_pickle=False) as z:
        shape=tuple(int(x) for x in z['shape'])
        if len(shape)!=3 or shape[0]!=3 or min(shape)<=0:
            raise ValueError(f'Invalid candidate shape: {shape}')
        masks=np.unpackbits(z['masks_bits'],count=int(np.prod(shape))).reshape(shape).astype(bool)
        low=np.unpackbits(z['low_bits'],count=int(np.prod(shape[1:]))).reshape(shape[1:]).astype(bool)
        high=np.unpackbits(z['high_bits'],count=int(np.prod(shape[1:]))).reshape(shape[1:]).astype(bool)
        return masks,low,high,z['scores'].copy(),z['stability'].copy(),int(z['selected'])


def resolve(row,key,root):
    if not row.get(key): return None
    p=Path(row[key])
    return p if p.is_absolute() else root/p


def source_image(row,args,root):
    if args.img_dir:
        choices=sorted(p for p in Path(args.img_dir).glob(row['image']+'.*')
                       if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))
        if len(choices)!=1: raise FileNotFoundError(f"Expected one image for {row['image']}, got {choices}")
        return choices[0]
    path=resolve(row,'image_path',root)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Image unavailable for {row['image']}; use --img-dir")
    return path


def categories(r, score_pass=.8, stability_pass=.9):
    result=['sim_three_views']
    if flag(r,'score_effect'): result.append('score_effect')
    if flag(r,'stability_effect'): result.append('stability_effect')
    if flag(r,'sim_to_tcrq_changed'): result.append('sim_to_tcrq_changed')
    if r.get('conditional_group')!=r.get('primary_group'):
        result.append('favv_caught')
        if number(r,'sam_score')>=score_pass and number(r,'sam_stability')>=stability_pass:
            result.append('score_stability_pass_favv_caught')
        if r.get('primary_group')=='clean': result.append('favv_caught_clean')
        if flag(r,'score_effect') or flag(r,'stability_effect'):
            result.append('cue_effect_and_favv_caught')
    if flag(r,'favv_retained_vs_hard'):
        result.append('favv_retained_vs_hard')
        if flag(r,'score_effect') or flag(r,'stability_effect'):
            result.append('cue_effect_and_favv_retained')
    return result


def box_on(ax,box):
    x1,y1,x2,y2=box
    ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#ffd34e',linewidth=1.5))


def show(ax,image,title,box=None):
    ax.imshow(image,cmap='gray',vmin=0,vmax=255 if image.dtype==np.uint8 else 1)
    ax.set_title(title,fontsize=10,pad=7)
    ax.axis('off')
    if box is not None: box_on(ax,box)


def overlay(rgb,mask,color=(64,213,159),alpha=.5):
    out=rgb.astype(float).copy()
    out[mask]=out[mask]*(1-alpha)+np.asarray(color)*alpha
    return np.clip(out,0,255).astype(np.uint8)


def pair_image(a,b):
    out=np.zeros((*a.shape,3),np.uint8)
    out[a&b]=(235,239,242)
    out[a&~b]=(245,158,65)
    out[b&~a]=(54,190,224)
    return out


def draw_case(r,args,root,out):
    cache=resolve(r,'candidate_cache_path',root)
    if cache is None or not cache.exists():
        raise FileNotFoundError('Three-candidate cache missing. Old final-mask-only output cannot reconstruct SIM views; rerun generation.')
    masks,low,high,scores,stability,idx=load_cache(cache)
    rgb=np.asarray(Image.open(source_image(r,args,root)).convert('RGB'))
    if rgb.shape[:2]!=masks.shape[1:]: raise ValueError('Image/cache shape mismatch')
    if idx!=int(number(r,'selected_candidate_index')):
        raise ValueError('CSV/cache selected candidate mismatch')
    box=[number(r,'box_'+k) for k in ('x1','y1','x2','y2')]
    selected=masks[idx]
    h,w=selected.shape
    bm=np.zeros((h,w),bool)
    x1,y1,x2,y2=box
    bm[max(0,int(np.floor(y1))):min(h,int(np.ceil(y2))),
       max(0,int(np.floor(x1))):min(w,int(np.ceil(x2)))]=True
    gt_path=Path(args.gt_dir)/(r['image']+'.png') if args.gt_dir else resolve(r,'gt_path',root)
    gt=None
    if gt_path is not None and gt_path.exists():
        gt=np.asarray(Image.open(gt_path).convert('L').resize((w,h),Image.Resampling.NEAREST))>128
    directory=out/'cases'/r['image']
    directory.mkdir(parents=True,exist_ok=True)
    for i,m in enumerate(masks,1): Image.fromarray(m.astype(np.uint8)*255).save(directory/f'M{i}.png')
    Image.fromarray(selected.astype(np.uint8)*255).save(directory/'selected.png')
    # Compact SIM triptych plus image/box.
    fig,ax=plt.subplots(1,4,figsize=(16,4.7),facecolor='white')
    show(ax[0],rgb,'Input + box',box)
    for i in range(3):
        show(ax[i+1],masks[i],f'M{i+1}' + ('  [SELECTED]' if i==idx else '')+
             f'\nScore {scores[i]:.4f} | Stability {stability[i]:.4f}')
    sims=' | '.join(f'SIM{k}={number(r,"candidate_sim_"+k):.4f}' for k in ('12','13','23'))
    fig.suptitle(f'{r["image"]}   |   {sims}   |   mean SIM={number(r,"candidate_consensus_s"):.4f}',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(directory/'SIM_three_views.png',dpi=args.dpi,bbox_inches='tight'); plt.close(fig)
    # Detailed evidence sheet.
    fig,axs=plt.subplots(3,4,figsize=(16,12),facecolor='#fafbfc')
    ax=axs.ravel()
    show(ax[0],rgb,'Input + prompt box',box)
    for i in range(3):
        show(ax[i+1],overlay(rgb,masks[i]),f'M{i+1}'+(' [SELECTED]' if i==idx else '')+
             f'\nScore {scores[i]:.4f} | Stability {stability[i]:.4f}',box)
    show(ax[4],selected,'Selected binary mask',box)
    for j,(a,b,key) in enumerate(((0,1,'12'),(0,2,'13'),(1,2,'23')),5):
        show(ax[j],pair_image(masks[a],masks[b]),f'M{a+1} vs M{b+1} | SIM={number(r,"candidate_sim_"+key):.4f}'+
             f'\nIoU={number(r,"candidate_iou_"+key):.4f} | SSIM={number(r,"candidate_ssim_"+key):.4f}')
    band=np.zeros((*selected.shape,3),np.uint8)
    band[high]=(64,213,159); band[low&~high]=(245,158,65)
    show(ax[8],band,f'Stability={number(r,"sam_stability"):.4f}\nGreen: high-threshold | orange: disputed',box)
    geometry=overlay(rgb,selected,(64,213,159))
    geometry=overlay(geometry,selected&~bm,(255,45,65),.8)
    show(ax[9],geometry,f'FAVV | Outside={number(r,"outside_mask_ratio"):.3f}\n'+
         f'Keff={number(r,"effective_num_components",0):.0f} | Largest={number(r,"largest_effective_component_ratio"):.3f}',box)
    if gt is not None:
        show(ax[10],gt,f'GT (evaluation only)\nIoU={number(r,"IoU"):.4f} | {r.get("gt_grade","unknown")}')
    else:
        ax[10].text(.5,.5,'GT unavailable\nDecision example only',ha='center',va='center',transform=ax[10].transAxes)
        ax[10].axis('off')
    p=[number(r,k) for k in ('p_score','p_stability','p_sim')]
    ax[11].barh(['Score','Stability','SIM'],p,color=['#5289c7','#8c71b5','#269f91'])
    ax[11].set_xlim(0,1); ax[11].set_xlabel('Dataset percentile'); ax[11].invert_yaxis()
    ax[11].set_title(f'TCRQ={number(r,"tcrq_score"):.4f}',fontsize=11)
    for i,val in enumerate(p):
        if np.isfinite(val): ax[11].text(min(val+.02,.85),i,f'{val:.3f}',va='center',fontsize=9)
    primary=r.get('primary_group','?'); final=r.get('final_group','?')
    fig.suptitle(f'{r["image"]}   |   {primary.upper()}  ->  {final.upper()}   |   {r.get("favv_reason","")}',fontsize=17,y=.985)
    details=(f'Raw Score={number(r,"sam_score"):.4f} | Stability={number(r,"sam_stability"):.4f} | SIM={number(r,"candidate_consensus_s"):.4f}\n'
             f'SIM-only: {r.get("group_sim")} | no Score: {r.get("group_no_score")} | no Stability: {r.get("group_no_stability")} | TCRQ: {r.get("group_tcrq")}\n'
             f'Blanket rejection: {r.get("hard_all_group")} | Conditional FAVV: {r.get("conditional_group")} | mode: {r.get("favv_mode")}')
    fig.text(.5,.055,details,ha='center',va='center',fontsize=10,linespacing=1.7)
    fig.legend(handles=[Patch(color='#ebeff2',label='Pair: common foreground'),Patch(color='#f59e41',label='Pair: first only'),
                        Patch(color='#36bee0',label='Pair: second only')],loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.95),h_pad=2,w_pad=2)
    fig.savefig(directory/'evidence.png',dpi=args.dpi,bbox_inches='tight')
    if args.pdf: fig.savefig(directory/'evidence.pdf',bbox_inches='tight')
    plt.close(fig)
    return str(directory.relative_to(out)/'evidence.png')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--out-dir',default=None)
    p.add_argument('--img-dir',help='Override original image location after moving outputs')
    p.add_argument('--gt-dir',help='Optional GT location override')
    p.add_argument('--cases',default='all',help='Comma-separated categories; default all')
    p.add_argument('--per-category',type=int,default=12,help='0 = all matched samples')
    p.add_argument('--ids',nargs='+',help='Explicit image stems; overrides category sampling')
    p.add_argument('--only-gt',choices=['any','good','middle','bad'],default='any',
                   help='Optional display-only GT filter; default no GT-based selection')
    p.add_argument('--score-pass',type=float,default=.8,help='Display-only old raw-score gate, not TCRQ threshold')
    p.add_argument('--stability-pass',type=float,default=.9,help='Display-only old raw-stability gate')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--dpi',type=int,default=150)
    p.add_argument('--pdf',action='store_true')
    args=p.parse_args()
    if args.per_category<0: p.error('--per-category must be >=0')
    root=Path(args.run_dir).resolve()
    out=Path(args.out_dir).resolve() if args.out_dir else root/'Visualizations'
    out.mkdir(parents=True,exist_ok=True)
    with (root/'csv/image_level_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    names=['sim_three_views','score_effect','stability_effect','sim_to_tcrq_changed','favv_caught',
           'favv_caught_clean','score_stability_pass_favv_caught','cue_effect_and_favv_caught','favv_retained_vs_hard','cue_effect_and_favv_retained']
    wanted=names if args.cases=='all' else args.cases.split(',')
    if any(x not in names for x in wanted): p.error('Categories: '+','.join(names))
    # Group membership is read from generation, never inferred from GT.
    membership={r['image']:categories(r,args.score_pass,args.stability_pass) for r in rows}
    rng=np.random.default_rng(args.seed)
    chosen=set(); counts={}; chosen_by={}
    for name in wanted:
        candidates=sorted(r['image'] for r in rows if name in membership[r['image']] and
                          (args.only_gt=='any' or r.get('gt_grade')==args.only_gt))
        counts[name]=len(candidates)
        if args.ids: selected=sorted(set(candidates)&set(args.ids))
        elif args.per_category and len(candidates)>args.per_category:
            selected=sorted(rng.choice(candidates,args.per_category,replace=False).tolist())
        else: selected=candidates
        chosen.update(selected); chosen_by[name]=selected
    if args.ids:
        known={r['image'] for r in rows}
        missing=set(args.ids)-known
        if missing: p.error('Unknown IDs: '+','.join(sorted(missing)))
        chosen.update(r['image'] for r in rows if r['image'] in args.ids and
                      (args.only_gt=='any' or r.get('gt_grade')==args.only_gt))
    index=[]; errors=[]; assets={}
    for r in rows:
        if r['image'] not in chosen: continue
        try:
            assets[r['image']]=draw_case(r,args,root,out)
            index.append(dict(image=r['image'],categories=';'.join(membership[r['image']]),
                              primary=r.get('primary_group'),final=r.get('final_group'),
                              reason=r.get('favv_reason'),IoU=r.get('IoU'),path=assets[r['image']]))
        except Exception as exc:
            errors.append(dict(image=r['image'],error=str(exc)))
    for name,data in [('selected_cases',index),('visualization_errors',errors)]:
        with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]) if data else ['image'])
            w.writeheader(); w.writerows(data)
    report=dict(eligible_counts=counts,rendered=len(index),errors=errors,seed=args.seed,
                per_category=args.per_category,gt_filter=args.only_gt,selected_by_category=chosen_by,
                display_score_pass=args.score_pass,display_stability_pass=args.stability_pass)
    (out/'selection_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=[]
    for name in wanted:
        cards.append(f'<h2>{html.escape(name)} ({counts[name]} eligible)</h2>')
        for ident in chosen_by[name]:
            if ident in assets:
                link=html.escape(assets[ident],quote=True)
                cards.append(f'<article><h3>{html.escape(ident)}</h3><a href="{link}"><img loading="lazy" src="{link}"></a></article>')
        if not chosen_by[name]: cards.append('<p>No matching cases.</p>')
    if args.ids:
        cards.append('<h2>Requested IDs</h2>')
        for ident in sorted(chosen):
            if ident in assets:
                link=html.escape(assets[ident],quote=True)
                cards.append(f'<article><h3>{html.escape(ident)}</h3><a href="{link}"><img src="{link}"></a></article>')
    page='<!doctype html><meta charset="utf-8"><title>TCRQ / FAVV evidence</title><style>body{font:16px system-ui;max-width:1400px;margin:40px auto;background:#f2f5f7;color:#203040}article{background:white;padding:18px;margin:20px 0}img{width:100%}h2{margin-top:50px}</style>'
    page+='<h1>TCRQ / FAVV evidence gallery</h1><p>FAVV changes supervision groups, not masks. GT is evaluation-only. Empty categories are reported explicitly.</p>'
    (out/'index.html').write_text(page+''.join(cards),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f'Gallery: {out / "index.html"}')
    if errors: raise SystemExit(f'{len(errors)} case(s) could not be rendered; see visualization_errors.csv')


if __name__=='__main__': main()
