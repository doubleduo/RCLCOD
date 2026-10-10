"""Lightweight functional tests requiring no SAM weights and no GT metric package."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import Config,ablation_flags
from sam1_generate_refine import main
from sam_core import predict_initial,predict_with_prompts
from evidence_refine import (extract_reliable_points, LocalEvidence, crop_bounds,
                             refine_one_box, _recover_local_positive)
from quality_metrics import box_to_mask

class FakePredictor:
    def __init__(self):
        self.model=type('Model',(),{'mask_threshold':0.0})()
        self.is_image_set=False
        self.original_size=None
        self.encode_calls=0
    def set_image(self, image):
        assert image.dtype==np.uint8 and image.ndim==3
        self.original_size=image.shape[:2]
        self.is_image_set=True
        self.encode_calls+=1
    def predict(self,point_coords=None,point_labels=None,box=None,mask_input=None,
                multimask_output=True,return_logits=True):
        assert self.is_image_set and return_logits
        h,w=self.original_size
        x1,y1,x2,y2=box
        yy,xx=np.mgrid[:h,:w]
        cx=(x1+x2)/2;cy=(y1+y2)/2
        rx=max(3,(x2-x1)*.34);ry=max(3,(y2-y1)*.34)
        offset=2 if (point_coords is not None or mask_input is not None) else 0
        crop_gain=3 if (h,w)!=(120,160) else 0
        logits=[]
        num=3 if multimask_output else 1
        for k in range(num):
            jit=[-1.0,0.,1.0][k] if multimask_output else 0
            dist=((xx-cx-jit)/(rx+crop_gain))**2+((yy-cy)/(ry+crop_gain*0.6))**2
            lg=np.asarray((1.0-dist)*8.,dtype=np.float32)
            if offset:
                dist2=((xx-cx-jit)/float(rx+offset))**2+((yy-cy)/ry)**2
                lg=np.asarray((1.0-dist2)*8.,dtype=np.float32)
            logits.append(lg)
        scores=np.asarray([.80,.88,.82] if num==3 else [.90],dtype=np.float32)
        low=np.stack([cv2.resize(l,(256,256),interpolation=cv2.INTER_LINEAR) for l in logits])
        return np.stack(logits),scores,low


def _example(root):
    img=root/'Imgs';boxdir=root/'box_label';gt=root/'GT'
    for folder in [img,boxdir,gt]: folder.mkdir(exist_ok=True,parents=True)
    for idx in [1,2]:
        arr=np.full((120,160,3),120,dtype=np.uint8)
        cv2.circle(arr,(60,60),21,(70,90,110),-1)
        cv2.imwrite(str(img/f'sample{idx}.jpg'),arr)
        boxes=[[(30,24),(100,97)]]
        if idx==2: boxes.append([(107,25),(150,79)])
        data={'shapes':[{'shape_type':'rectangle','points':p} for p in boxes]}
        (boxdir/f'sample{idx}.json').write_text(json.dumps(data))
    return img,boxdir,gt


def test_sam_native_logit_prompt():
    p=FakePredictor();p.set_image(np.zeros((100,120,3),dtype=np.uint8))
    initial=predict_initial(p,[20,30,80,90])
    assert initial.low_res_logit.shape==(256,256)
    second=predict_with_prompts(p,[20,30,80,90],
           positives=[[48,53]],negatives=[[90,80]],mask_logits=initial.low_res_logit)
    assert second.mask.shape==initial.mask.shape
    assert second.score>initial.score


def test_points_and_crop():
    h,w=100,130
    mask=np.zeros((h,w),dtype=bool);mask[35:65,40:90]=1
    prob=np.where(mask,.99,.01).astype(np.float32)
    initial=type('Init',(),{'mask':mask,'prob':prob})()
    local=LocalEvidence(mask.copy(),prob.copy(),.9,.95,(20,20,110,85),np.zeros_like(mask))
    local.observed[20:85,20:110]=True
    cfg=Config(positive_points=2,negative_points=2,points_min_distance_px=3)
    pos,neg=extract_reliable_points(initial,local,[35,30,95,75],cfg)
    assert len(pos)>=1
    assert len(neg)>=1
    assert all(mask[y,x] for x,y in pos)
    outside=box_to_mask([35,30,95,75],h,w)
    assert all(not outside[y,x] for x,y in neg)
    assert crop_bounds((h,w),[35,30,95,75],.25,1.1)


def test_multiple_boxes_excluded_from_negative_points():
    """An adjacent target box must never be misidentified as reliable background."""
    h,w=90,140
    fg=np.zeros((h,w),bool);fg[30:47,27:45]=True
    ini_prob=np.where(fg,.99,.01).astype(np.float32)
    ini=type('Initial',(),{'mask':fg,'prob':ini_prob})()
    observed=np.zeros_like(fg)
    observed[15:65,15:105]=True
    local=LocalEvidence(fg.copy(),ini_prob.copy(),.95,.96,(15,15,105,65),observed)
    first=[20,20,52,55]
    second=[57,22,92,55]
    union=box_to_mask(first,h,w)|box_to_mask(second,h,w)
    cfg=Config(positive_points=2,negative_points=10,points_min_distance_px=2)
    pos,neg=extract_reliable_points(ini,local,first,cfg,all_boxes_mask=union)
    assert pos
    assert all(not union[y,x] for x,y in neg)


def test_bad_initial_uses_local_only_recovery_not_corrupt_logits():
    """When the first mask has collapsed, local evidence must not reuse its logits."""
    rgb=np.zeros((120,160,3),dtype=np.uint8)
    box=[30,24,100,97]
    p=FakePredictor();p.set_image(rgb)
    initial=predict_initial(p,box)
    initial.mask[:]=False
    initial.prob[:]=.01
    initial.score=.50
    initial.stability=.0
    cfg=Config(ablation='A4',risk_score_below=.8,points_min_distance_px=2)
    out=refine_one_box(p,rgb,box,initial,cfg)
    assert out.triggered
    assert out.recovered_from_local_only
    assert not out.used_native_logits
    assert out.points_pos
    assert out.accepted, out.rollback_reason


def test_fine_edits_are_not_rejected_by_full_image_fraction():
    cfg=Config()
    assert cfg.change_min_fraction == 0
    assert cfg.change_min_pixels <= 3


@pytest.mark.parametrize('abl',['A0','A1','A2','A3','A4'])
def test_all_ablations_generate_csv_and_masks(tmp_path,abl):
    img,boxdir,gt=_example(tmp_path)
    cfg=Config(img_dir=str(img),box_dir=str(boxdir),gt_dir=str(gt),
               output_dir=str(tmp_path/'results'),ablation=abl,
               use_gt_eval=False,save_visualizations='none',
               risk_score_below=.99)
    p=FakePredictor()
    res=main(cfg,predictor=p)
    assert res['generated_count']==2
    assert res['failed_count']==0
    out=tmp_path/'results'/abl
    assert len(list((out/'Pred_Masks').glob('*.png')))==2
    assert len(list((out/'LowRes_Logits').glob('*.npz')))==2
    import csv
    with (out/'csv'/'image_level_metrics.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    assert all(key in rows[0] for key in ('q_dual','candidate_consensus_s','recommended_pool',
                    'sam_score_selected','IoU','MAE','refine_accepted','negative_count'))
    assert int(rows[1]['num_boxes_in_json'])==2
    assert rows[1]['used_enclosing_union_box']=='0'
    native=np.load(out/'LowRes_Logits'/'sample2.npz')
    assert native['logits'].shape==(2,256,256)
    with (out/'csv'/'refinement_analysis.csv').open(encoding='utf-8-sig') as f:
        assert len(list(csv.DictReader(f)))==3
    if abl=='A0':
        assert res['triggered_images']==0
    else:
        assert res['triggered_images']==2
    if abl=='A4':
        assert res['accepted_images']>=1, res
        assert float(rows[0]['mask_change_ratio'])>0


def test_gt_eval_is_output_only_and_wfm_summary(tmp_path,monkeypatch):
    """Exercise the GT branch via a minimal metric API stub (NOT numerical metric validation)."""
    import sys
    from types import SimpleNamespace

    class Metric:
        def __init__(self):
            self.values=[]
        def step(self,pred,gt):
            self.values.append(float(np.mean(pred==gt)))
        def get_results(self):
            v=float(np.mean(self.values)) if self.values else 0.
            return {'sm':v,'em':{'adp':v},'fm':{'adp':v},'wfm':v,'mae':1-v}

    # All five metric types share the same API in this mock; no claim of numeric equivalence.
    monkeypatch.setitem(sys.modules,'py_sod_metrics',SimpleNamespace(
        Smeasure=Metric,Emeasure=Metric,Fmeasure=Metric,WeightedFmeasure=Metric,MAE=Metric))
    img,boxdir,gt=_example(tmp_path)
    g=np.zeros((120,160),dtype=np.uint8)
    cv2.ellipse(g,(65,59),(25,24),0,0,360,255,-1)
    cv2.imwrite(str(gt/'sample1.png'),g)
    cv2.imwrite(str(gt/'sample2.png'),g)
    cfg=Config(img_dir=str(img),box_dir=str(boxdir),gt_dir=str(gt),
               output_dir=str(tmp_path/'results'),ablation='A4',
               use_gt_eval=True,save_visualizations='all',risk_score_below=.99)
    res=main(cfg,predictor=FakePredictor())
    assert res['gt_evaluated_count']==2
    assert 'Fmeasure_w' in res['sod_before']
    assert 'Fmeasure_w' in res['sod_after']
    out=tmp_path/'results'/'A4'
    assert len(list((out/'Compare_Vis').glob('*.jpg')))==2
    import csv
    with (out/'csv'/'image_level_metrics.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    assert all(row['IoU']!='' and row['MAE']!='' and row['Fmeasure_w']!='' for row in rows)
