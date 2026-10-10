"""GT is isolated in this module and MUST NOT enter prompt generation or acceptance."""
from pathlib import Path
import cv2
import numpy as np
from quality_metrics import compute_iou


def load_gt(gt_dir, image_name, image_hw):
    path = Path(gt_dir) / f'{image_name}.png'
    if not path.is_file():
        return None
    gt = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    if gt.shape != tuple(image_hw):
        gt = cv2.resize(gt,(image_hw[1],image_hw[0]),interpolation=cv2.INTER_NEAREST)
    return gt > 128


def hard_metrics(mask, gt):
    return {'IoU':float(compute_iou(mask,gt)),
            'MAE':float(np.mean(mask != gt))}


def init_sod_group():
    from py_sod_metrics import Smeasure,Emeasure,Fmeasure,WeightedFmeasure,MAE
    return {'S':Smeasure(),'E':Emeasure(),'F':Fmeasure(),'Fw':WeightedFmeasure(),'MAE':MAE()}


def update_sod_group(group, pred, gt):
    p=(pred.astype(np.uint8)*255)
    g=(gt.astype(np.uint8)*255)
    group['S'].step(pred=p,gt=g)
    group['E'].step(pred=p,gt=g)
    group['F'].step(pred=p,gt=g)
    group['Fw'].step(pred=p,gt=g)
    group['MAE'].step(pred=p,gt=g)


def get_sod_group(group):
    return {
        'Smeasure':float(group['S'].get_results()['sm']),
        'Emeasure_adp':float(group['E'].get_results()['em']['adp']),
        'Fmeasure_adp':float(group['F'].get_results()['fm']['adp']),
        'Fmeasure_w':float(group['Fw'].get_results()['wfm']),
        'MAE':float(group['MAE'].get_results()['mae']),
    }


def sod_per_image(pred,gt):
    g=init_sod_group()
    update_sod_group(g,pred,gt)
    return get_sod_group(g)


def _blend(rgb,mask,color):
    ans=rgb.astype(np.float32)
    col=np.asarray(color,dtype=np.float32)
    ans[mask]=0.60*ans[mask]+0.40*col
    return ans.astype(np.uint8)


def save_comparison(image_rgb,boxes,initial,final,gt,path,labels=None):
    """RGB / initial / final / GT error map; GT omitted if unavailable."""
    h,w=image_rgb.shape[:2]
    left=image_rgb.copy()
    for b in boxes:
        x1,y1,x2,y2=[int(round(a)) for a in b]
        cv2.rectangle(left,(x1,y1),(max(x1,x2-1),max(y1,y2-1)),(80,200,80),2)
    panes=[left,_blend(image_rgb,initial,(220,80,45)),_blend(image_rgb,final,(50,200,80))]
    titles=['RGB+Boxes','Initial','Final']
    if gt is not None:
        tp=final & gt; fp=final & ~gt; fn=~final & gt
        err=np.zeros_like(image_rgb)
        err[tp]=(255,255,255)
        err[fp]=(255,75,40)
        err[fn]=(40,80,255)
        panes.append(err)
        titles.append('Final TP white / FP red / FN blue')
    side=max(1,min(640, w))
    new_w=min(w,side)
    new_h=max(1,round(h*new_w/w))
    panels=[]
    for pane,title in zip(panes,titles):
        pane=cv2.resize(pane,(new_w,new_h),interpolation=cv2.INTER_AREA)
        band=np.zeros((28,new_w,3),dtype=np.uint8)
        cv2.putText(band,title[:38],(4,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        panels.append(np.concatenate([band,pane],axis=0))
    result=np.concatenate(panels,axis=1)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),cv2.cvtColor(result,cv2.COLOR_RGB2BGR)):
        raise OSError(f'Cannot save comparison {path}')


def summarize_accepted(rows):
    selected=[r for r in rows if int(r.get('has_gt',0))]
    if not selected:
        return {'count':0}
    improves=sum(r['IoU']>r['IoU_before'] and r['MAE']<=r['MAE_before'] for r in selected)
    worse=sum(r['IoU']<r['IoU_before'] or r['MAE']>r['MAE_before'] for r in selected)
    return {'count':len(selected), 'improved_both_or_equal_mae':improves,
            'degraded_either_metric':worse,
            'mean_iou_delta':float(np.mean([r['IoU']-r['IoU_before'] for r in selected])),
            'mean_mae_delta':float(np.mean([r['MAE']-r['MAE_before'] for r in selected]))}
