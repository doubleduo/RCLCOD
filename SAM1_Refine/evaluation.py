#!/usr/bin/env python3
"""Read A0..A4 evaluation_results.json, export a 5-run ablation comparison CSV."""
import argparse
import csv
import json
from pathlib import Path

FIELDS=['ablation','generated_count','failed_count','gt_evaluated_count','triggered_images',
        'accepted_images','S_before','S_after','Fw_before','Fw_after','E_before','E_after',
        'IoU_before','IoU_after','IoU_delta','MAE_before','MAE_after','MAE_delta',
        'strict_bad_before','strict_bad_after',
        'accepted_mean_iou_delta','accepted_mean_mae_delta','accepted_degraded_count']


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default='./SGT/SAM1_Refine_V1')
    args=ap.parse_args()
    root=Path(args.output_dir)
    rows=[]
    for abl in ['A0','A1','A2','A3','A4']:
        j=root/abl/'evaluation_results.json'
        if not j.exists():
            print('Missing',j)
            continue
        d=json.loads(j.read_text(encoding='utf-8'))
        before=d.get('sod_before',{})
        after=d.get('sod_after',{})
        accepted=d.get('accepted_only',{})
        before_iou=d.get('mean_iou_before')
        after_iou=d.get('mean_iou_after')
        before_mae=d.get('mean_mae_before')
        after_mae=d.get('mean_mae_after')
        rows.append({
            'ablation':abl,
            **{k:d.get(k,'') for k in ('generated_count','failed_count','gt_evaluated_count',
                                       'triggered_images','accepted_images','strict_bad_before','strict_bad_after')},
            'S_before':before.get('Smeasure',''),'S_after':after.get('Smeasure',''),
            'Fw_before':before.get('Fmeasure_w',''),'Fw_after':after.get('Fmeasure_w',''),
            'E_before':before.get('Emeasure_adp',''),'E_after':after.get('Emeasure_adp',''),
            'IoU_before':before_iou if before_iou is not None else '',
            'IoU_after':after_iou if after_iou is not None else '',
            'IoU_delta':(after_iou-before_iou) if before_iou is not None and after_iou is not None else '',
            'MAE_before':before_mae if before_mae is not None else '',
            'MAE_after':after_mae if after_mae is not None else '',
            'MAE_delta':(after_mae-before_mae) if before_mae is not None and after_mae is not None else '',
            'accepted_mean_iou_delta':accepted.get('mean_iou_delta',''),
            'accepted_mean_mae_delta':accepted.get('mean_mae_delta',''),
            'accepted_degraded_count':accepted.get('degraded_either_metric',''),
        })
    dest=root/'ablation_summary.csv'
    root.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print('Wrote',dest,'runs:',len(rows))

if __name__=='__main__':
    main()
