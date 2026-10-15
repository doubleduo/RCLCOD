#!/usr/bin/env python3
"""Per-image CAMO segmentation evaluation + reproducible qualitative figure selection.
Example:
 python camo_qualitative_select.py --pred-root data/else/camo --gt-root data/Test/CAMO-TE/Mask --image-root data/Test/CAMO-TE/Image --out qualitative_camo
"""
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}

def index_images(root):
    result=defaultdict(list)
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in EXTS:
            result[p.stem.lower()].append(p)
    return result

def read_mask(path, shape):
    im=Image.open(path).convert('L')
    if im.size!=(shape[1],shape[0]):
        im=im.resize((shape[1],shape[0]),Image.Resampling.BILINEAR)
    return np.asarray(im,dtype=np.float32)/255.0

def metrics(pred, gt):
    pb=pred>=0.5; gb=gt>=0.5
    tp=np.count_nonzero(pb & gb)
    union=np.count_nonzero(pb | gb)
    pn=np.count_nonzero(pb); gn=np.count_nonzero(gb)
    return {'MAE':float(np.mean(np.abs(pred-gt))),
            'IoU':float(tp/(union+1e-12)) if union else 1.0,
            'Dice':float(2*tp/(pn+gn+1e-12)) if pn+gn else 1.0}

def find_models(root, ignore_stem_set, model_paths):
    root=Path(root)
    if model_paths:
        out={}
        for spec in model_paths:
            if '=' not in spec: raise ValueError('--model requires Name=path')
            name,path=spec.split('=',1)
            path=Path(path)
            if not path.is_absolute() and not path.exists(): path=root/path
            if not path.exists(): raise FileNotFoundError(path)
            out[name]=path
        return out
    # Find every directory containing recognizable mask filenames; summarize by its topmost model dir.
    result={}
    for d in sorted(root.iterdir()):
        if not d.is_dir(): continue
        idx=index_images(d)
        n=sum(int(k in ignore_stem_set) for k in idx)
        if n: result[d.name]=d
    return result

def pick_match(paths, model_name):
    # If multiple versions of a sample exist, error rather than silently choosing a different run.
    if len(paths)>1:
        raise RuntimeError(f'Multiple predictions for {model_name}, image {paths[0].stem}: '+', '.join(map(str,paths[:8]))+'\nPlease use --model Name=exact/prediction/folder.')
    return paths[0]

def make_figure(sample_names, names, imgs, gts, model_indices, out_path, dpi=240, categories=None):
    """CVPR-style: cases along columns; methods down rows; no altered predictions."""
    import textwrap
    from matplotlib.patches import Rectangle
    categories = categories or {}
    rows = ['Image', 'GT'] + names
    ncol, nrow = len(sample_names), len(rows)
    fig = plt.figure(figsize=(max(9, 2.05*ncol+1.35), 1.54*nrow+1.0), facecolor='white')
    gs = fig.add_gridspec(nrow+1, ncol+1, left=.035, right=.99, top=.96, bottom=.03,
                          width_ratios=[1.18]+[1]*ncol,
                          height_ratios=[.72]+[1]*nrow, wspace=.018, hspace=.018)
    palette=['#E9F0E2','#EEE7F4','#E8EFF5','#FFF0DB','#E9EAF7','#F6E6E9','#E7F0EE','#F1E9E2']
    left_colors={'Image':'#E5ECF4', 'GT':'#E5ECF4','Ours':'#FFE1A0'}
    for j,stem in enumerate(sample_names):
        ax=fig.add_subplot(gs[0,j+1]);ax.set_axis_off()
        ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,facecolor=palette[j%len(palette)],edgecolor='none'))
        category=categories.get(stem, f'Case {j+1:02d}')
        ax.text(.5,.5,'\n'.join(textwrap.wrap(category, 15)) if len(category)>15 else category,
                ha='center',va='center',weight='bold',fontsize=11,color='#263343',transform=ax.transAxes)
    for i,name in enumerate(rows):
        ax=fig.add_subplot(gs[i+1,0]);ax.set_axis_off()
        ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,
                               facecolor=left_colors.get(name,'#EDF2F8'),edgecolor='none'))
        label=f'({chr(97+i)}) {name}' if i<26 else name
        ax.text(.5,.5,label,ha='center',va='center',weight='bold',fontsize=10.8,
                color='#273449',transform=ax.transAxes)
    for j,stem in enumerate(sample_names):
        gt_path=gts[stem][0]
        with Image.open(gt_path) as im: gt=im.convert('L'); size=gt.size
        paths=imgs.get(stem,[])
        for i,name in enumerate(rows):
            ax=fig.add_subplot(gs[i+1,j+1]);ax.set_axis_off()
            if name=='Image':
                if paths:
                    with Image.open(paths[0]) as im: display=im.convert('RGB').resize(size,Image.Resampling.BILINEAR)
                    ax.imshow(display,interpolation='nearest',aspect='auto')
                else: ax.text(.5,.5,'Missing RGB',ha='center',va='center',transform=ax.transAxes)
            elif name=='GT':ax.imshow(gt,cmap='gray',vmin=0,vmax=255,interpolation='nearest',aspect='auto')
            else:
                with Image.open(pick_match(model_indices[name][stem],name)) as im:
                    display=im.convert('L').resize(size,Image.Resampling.BILINEAR)
                ax.imshow(display,cmap='gray',vmin=0,vmax=255,interpolation='nearest',aspect='auto')
            for spine in ax.spines.values():spine.set_visible(False)
    fig.savefig(out_path,dpi=dpi,facecolor='white',pad_inches=.02)
    fig.savefig(out_path.with_suffix('.pdf'),facecolor='white',pad_inches=.02)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred-root',required=True)
    p.add_argument('--gt-root',required=True)
    p.add_argument('--image-root',required=True)
    p.add_argument('--out',default='camo_qualitative')
    p.add_argument('--ours',default='PvtV2B4_ZoomNeXt_BS8',help='Folder model name or --model alias')
    p.add_argument('--model',action='append',help='Explicit Name=path; repeat for each model. Best for nested duplicates')
    p.add_argument('--rows',type=int,default=5,help='Number of example columns')
    p.add_argument('--category',action='append',default=[],help='stem=Category label for chosen image; repeat as needed')
    p.add_argument('--select',nargs='+',help='Optional exact image stems to show, in your preferred order')
    p.add_argument('--min-baselines',type=int,default=2,help='Number of baselines that must lose to Ours by delta')
    p.add_argument('--mae-gap',type=float,default=.025)
    p.add_argument('--iou-gap',type=float,default=.08)
    p.add_argument('--max-ours-mae',type=float,default=.09)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    gts=index_images(a.gt_root);imgs=index_images(a.image_root)
    if not gts:raise RuntimeError(f'No ground-truth masks in {a.gt_root}')
    models=find_models(a.pred_root,set(gts),a.model)
    if a.ours not in models:raise RuntimeError(f'Ours name {a.ours!r} not found. Found: {list(models)}; use --ours MODELNAME or --model Ours=... --ours Ours')
    indices={('Ours' if name==a.ours else name):index_images(root) for name,root in models.items()}
    if len(indices)!=len(models):raise RuntimeError('Duplicate model names after renaming Ours')
    names=['Ours']+[n for n in indices if n!='Ours']
    print('Model roots:', {('Ours' if n==a.ours else n):str(root) for n,root in models.items()})
    rows=[]
    for stem,gt_paths in sorted(gts.items()):
        if len(gt_paths)!=1:
            raise RuntimeError(f'Duplicate GT for {stem}: {gt_paths}')
        gt=np.asarray(Image.open(gt_paths[0]).convert('L'),dtype=np.float32)/255
        shape=gt.shape
        for name in names:
            paths=indices[name].get(stem,[])
            if not paths:continue
            pred=read_mask(pick_match(paths,name),shape)
            rows.append(dict(image=stem,model=name,**metrics(pred,gt)))
    full=pd.DataFrame(rows)
    if full.empty:raise RuntimeError('No matching GT / prediction filenames. Inspect predictions and use --model explicitly.')
    full.to_csv(out/'camo_per_image_metrics_long.csv',index=False,encoding='utf-8-sig')
    wide=full.pivot(index='image',columns='model',values=['MAE','IoU','Dice'])
    wide.columns=[f'{name}_{metric}' for metric,name in wide.columns]
    wide=wide.reset_index()
    wide.to_csv(out/'camo_per_image_metrics_wide.csv',index=False,encoding='utf-8-sig')
    print('Prediction count:',full.groupby('model').size().to_dict())
    print('Dataset means (only available masks, may be incomplete):')
    print(full.groupby('model')[['MAE','IoU','Dice']].mean().round(4).to_string())
    baselines=[n for n in names if n!='Ours']
    qualifying=[]
    for _,row in wide.iterrows():
        stem=row['image'];o_mae=row.get('Ours_MAE',np.nan);o_iou=row.get('Ours_IoU',np.nan)
        if not np.isfinite(o_mae) or o_mae>a.max_ours_mae:continue
        if stem not in imgs:continue
        if any(stem not in indices[name] for name in names):continue # figure needs every column
        losses=[]
        for n in baselines:
            dm=row[f'{n}_MAE']-o_mae;di=o_iou-row[f'{n}_IoU']
            if dm>=a.mae_gap and di>=a.iou_gap: losses.append(n)
        if len(losses)<a.min_baselines: continue
        # Favor robust gains across methods, not an isolated failure of a single baseline.
        avg_gain=np.mean([max(0,row[f'{n}_MAE']-o_mae) + .25*max(0,o_iou-row[f'{n}_IoU']) for n in baselines])
        score=2*len(losses)+avg_gain*10+o_iou-o_mae
        qualifying.append(dict(image=stem,score=score,ours_MAE=o_mae,ours_IoU=o_iou,losing_count=len(losses),losing_models=', '.join(losses)))
    ranked=pd.DataFrame(qualifying)
    if len(ranked):ranked=ranked.sort_values('score',ascending=False)
    ranked.to_csv(out/'ranked_advantage_samples.csv',index=False,encoding='utf-8-sig')
    picks=ranked.head(a.rows)['image'].tolist() if len(ranked) else []
    if a.select:
        picks=[Path(x).stem.lower() for x in a.select]
        for stem in picks:
            if stem not in gts or stem not in imgs or any(stem not in indices[n] for n in names):
                raise ValueError(f'Missing RGB, GT or model prediction for manually selected sample: {stem}')
    categories={}
    for item in a.category:
        if '=' not in item: raise ValueError('--category requires image_stem=Category')
        stem,label=item.split('=',1);categories[Path(stem).stem.lower()]=label
    if picks:
        make_figure(picks,names,imgs,gts,indices,out/'CAMO_qualitative_comparison.png', categories=categories)
        print('\nFigure saved:',out/'CAMO_qualitative_comparison.png')
        print('Selected images:',picks)
    else:
        print('\nNo cases satisfied thresholds and full-model coverage. Check missing predictions or relax --mae-gap, --iou-gap, --max-ours-mae, --min-baselines.')
    print('CSV files saved in:',out)

if __name__=='__main__':main()
