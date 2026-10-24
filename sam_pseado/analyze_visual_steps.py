#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze completed SAM2 100-image attribution output, without rerunning SAM2.
Requires per_image_metrics.csv and all_panels/*.jpg created by the patched evaluator.
Outputs paired visual examples per effect, distributions, diagnosis table and markdown report.
"""
import argparse,csv,json,math,shutil
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAIRS=[('Crop_effect','G0_Box','G1_Crop'),('PerBox_effect','G0_Box','G2_PerBox'),('Crop_after_PerBox','G2_PerBox','G3_PerBoxCrop'),('PerBox_after_Crop','G1_Crop','G3_PerBoxCrop'),('Repeat_effect','G3_PerBoxCrop','P0_Repeat'),('Pos_effect','P0_Repeat','P1_Pos'),('Neg_effect','P0_Repeat','P2_Neg'),('PosNeg_effect','P0_Repeat','P3_PosNeg'),('Memory_effect','P0_Repeat','M0_Memory'),('Memory_on_PosNeg','P3_PosNeg','M1_PosNegMemory'),('Points_on_Memory','M0_Memory','M1_PosNegMemory'),('Overall','G0_Box','M1_PosNegMemory')]
ORDER=['RGB','GT','G0_Box','G1_Crop','G2_PerBox','G3_PerBoxCrop','P0_Repeat','P1_Pos','P2_Neg','P3_PosNeg','M0_Memory','M1_PosNegMemory']
def tile(panel,index):
    h,w=panel.shape[:2];tw=w//4;th=h//3;row,col=divmod(index,4)
    return panel[row*th:(row+1)*th,col*tw:(col+1)*tw].copy()
def make_pair(im, row, title, base, new, gain):
    keys=['RGB','GT',base,new]
    pieces=[tile(im,ORDER.index(k)) for k in keys]
    h=max(a.shape[0] for a in pieces); w=max(a.shape[1] for a in pieces)
    out=np.full((h+54, w*4,3),248,np.uint8)
    for i,(key,part) in enumerate(zip(keys,pieces)):
        out[54:54+part.shape[0],i*w:i*w+part.shape[1]]=part
        cv2.putText(out,key,(i*w+6,45),cv2.FONT_HERSHEY_SIMPLEX,.49,(35,35,35),1,cv2.LINE_AA)
    cap=f'{title} | {row["image"]} | MAE gain {gain:+.5f}'
    cv2.putText(out,cap,(7,19),cv2.FONT_HERSHEY_SIMPLEX,.47,(5,5,5),1,cv2.LINE_AA)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True,help='output directory from sam2_attribution_visual_100.py');ap.add_argument('--output',default=None);ap.add_argument('--top-k',type=int,default=12);ap.add_argument('--threshold',type=float,default=.003);a=ap.parse_args()
    source=Path(a.source);dest=Path(a.output) if a.output else source/'analysis_materials';dest.mkdir(parents=True,exist_ok=True)
    with (source/'per_image_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise RuntimeError('CSV has no data')
    names=[]; means=[]; counts=[]; summary=[]
    for title,b,v in PAIRS:
        vals=np.array([float(r['gain_'+title]) for r in rows]);names.append(title);means.append(vals.mean());counts.append([int((vals>1e-8).sum()),int((vals<-1e-8).sum()),int((abs(vals)<=1e-8).sum())])
        ranked=sorted(rows,key=lambda r:float(r['gain_'+title]),reverse=True)
        folder=dest/title;folder.mkdir(exist_ok=True)
        for label,seq in [('improved',[r for r in ranked if float(r['gain_'+title])>a.threshold][:a.top_k]),('worsened',[r for r in ranked[::-1] if float(r['gain_'+title])< -a.threshold][:a.top_k])]:
            sub=folder/label;sub.mkdir(exist_ok=True)
            for rank,r in enumerate(seq,1):
                stem=Path(r['image']).stem;inp=source/'all_panels'/f'{stem}.jpg'
                if not inp.exists():continue
                pic=cv2.imread(str(inp));
                if pic is None:continue
                gain=float(r['gain_'+title]);cv2.imwrite(str(sub/f'{rank:02d}_{gain:+.5f}_{stem}.jpg'),make_pair(pic,r,title,b,v,gain))
        with (folder/'ranking_all.csv').open('w',newline='',encoding='utf-8-sig') as f:
            fieldnames=['rank','image','gain','baseline_mae','variant_mae','num_boxes','num_pos','num_neg'];w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader()
            for rank,r in enumerate(ranked,1):w.writerow({'rank':rank,'image':r['image'],'gain':r['gain_'+title],'baseline_mae':r['mae_'+b],'variant_mae':r['mae_'+v],'num_boxes':r.get('num_boxes'),'num_pos':r.get('num_pos'),'num_neg':r.get('num_neg')})
        summary.append({'effect':title,'mean_gain':float(vals.mean()),'improved':int((vals>1e-8).sum()),'worsened':int((vals<-1e-8).sum()),'equal':int((abs(vals)<=1e-8).sum()),'gain_gt_0.01':int((vals>.01).sum()),'hurt_gt_0.01':int((vals<-.01).sum())})
    x=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(12,6));ax.barh(x,means);ax.set_yticks(x,names);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Mean MAE improvement (baseline - variant)');ax.set_title('Step-wise SAM2 attribution: mean MAE gain');fig.tight_layout();fig.savefig(dest/'01_mean_gain.png',dpi=180);plt.close(fig)
    counts=np.array(counts);fig,ax=plt.subplots(figsize=(12,6));left=np.zeros(len(names));
    for lab,j in [('Improved',0),('Worsened',1),('Equal',2)]:ax.barh(x,counts[:,j],left=left,label=lab);left+=counts[:,j]
    ax.set_yticks(x,names);ax.invert_yaxis();ax.set_xlabel('Number of images');ax.set_title('100-image sample: improved / worsened / equal');ax.legend();fig.tight_layout();fig.savefig(dest/'02_outcome_counts.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5));ax.boxplot([[float(r['gain_'+title]) for r in rows] for title,_,_ in PAIRS],tick_labels=names,showfliers=True);ax.axhline(0,color='gray',lw=.8);ax.tick_params(axis='x',rotation=70);ax.set_ylabel('Per-image MAE improvement');ax.set_title('Distribution of changes (not just mean)');fig.tight_layout();fig.savefig(dest/'03_gain_distributions.png',dpi=180);plt.close(fig)
    with (dest/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    lines=['# SAM2 random-100: qualitative attribution evidence','',f'Samples: {len(rows)}. Gain = MAE(baseline) - MAE(variant), positive is better.','', '## What to inspect','', '1. Crop and PerBox only have effects on images where the geometry branch is activated; verify actual crop and object counts.', '2. Pos-only should be checked for actual selected point counts and whether binary prediction changed.', '3. Memory-only is distinguished from Repeat (second decoding without logits).', '4. Review worsened alongside improved examples. Do not choose successful examples as the only evidence.','', '## Effect summary','','|Step|Mean gain|Improved|Worsened|Equal|','|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['effect']}|{r['mean_gain']:+.6f}|{r['improved']}|{r['worsened']}|{r['equal']}|")
    lines+=['','## How to use','', 'Each effect directory contains `ranking_all.csv`, plus `improved/` and `worsened/` paired 4-panel JPEGs (RGB, GT, before, after).', 'Red pixels in prediction panels: false positives; blue: false negatives.', '', '**Caution:** The previous script incorrectly detected positive points for `repeat` via substring matching (`p in "repeat"`). Only interpret these causal comparisons after rerunning with the corrected evaluator.', '', 'Per-image GT is only used to measure and select examples, never for prediction.']
    (dest/'analysis_report.md').write_text('\n'.join(lines),encoding='utf8')
    print('Created',dest,'for',len(rows),'images')
if __name__=='__main__':main()
