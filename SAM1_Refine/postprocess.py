"""Non-GT risk triggering, crop evidence, point prompting, acceptance, and rollback."""
from dataclasses import dataclass, field
import cv2
import numpy as np
from config import ablation_flags
from quality_metrics import (box_to_mask, compute_iou, dilate_mask, erode_mask,
                             compute_boundary_uncertainty, effective_component_stats)
from sam_core import predict_initial, predict_with_prompts


@dataclass
class LocalEvidence:
    mask: np.ndarray
    prob: np.ndarray
    score: float
    stability: float
    crop_xyxy: tuple
    observed: np.ndarray


@dataclass
class RefineOutcome:
    mask: np.ndarray
    prob: np.ndarray
    triggered: bool = False
    accepted: bool = False
    trigger_reasons: list = field(default_factory=list)
    rollback_reason: str = ''
    points_pos: list = field(default_factory=list)
    points_neg: list = field(default_factory=list)
    used_local: bool = False
    used_native_logits: bool = False
    recovered_from_local_only: bool = False
    local_score: float = float('nan')
    refined_score: float = float('nan')
    candidate_iou: float = float('nan')
    change_ratio: float = 0.0
    crop_xyxy: tuple | None = None


def risk_reasons(initial, box, box_mask, cfg):
    h, w = initial.mask.shape
    area_box = max(1, int(box_mask.sum()))
    fill = float(initial.mask.sum() / area_box)
    boundary = compute_boundary_uncertainty(initial.prob, initial.mask)['boundary_uncertainty']
    s = float(initial.consensus.get('candidate_consensus_s', 0))
    outside = int((initial.mask & ~box_mask).sum()) / max(1, int(initial.mask.sum()))
    reasons = []
    if initial.score < cfg.risk_score_below:
        reasons.append('low_sam_score')
    if initial.stability < cfg.risk_stability_below:
        reasons.append('low_stability')
    if s < cfg.risk_consensus_below:
        reasons.append('low_candidate_consensus')
    if boundary > cfg.risk_boundary_above:
        reasons.append('boundary_uncertainty')
    if (box[2] - box[0]) * (box[3] - box[1]) / (h * w) < cfg.risk_small_box_fraction:
        reasons.append('small_box')
    if fill < cfg.risk_tiny_fill:
        reasons.append('tiny_mask')
    if fill > 0.96:
        reasons.append('box_like_mask')
    if outside > cfg.risk_outside_fraction:
        reasons.append('outside_leak')
    components=effective_component_stats(initial.mask)
    if (components['effective_num_components'] >= cfg.risk_fragment_components
            and components['largest_effective_component_ratio'] < cfg.risk_fragment_largest_ratio):
        # Only a *risk trigger*, never a standalone assertion that the mask is wrong.
        reasons.append('fragment_risk')
    return reasons


def crop_bounds(original_size, box, ratio, min_gain):
    h, w = original_size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2-x1, y2-y1
    x0 = max(0, int(np.floor(x1 - bw * ratio)))
    y0 = max(0, int(np.floor(y1 - bh * ratio)))
    x3 = min(w, int(np.ceil(x2 + bw * ratio)))
    y3 = min(h, int(np.ceil(y2 + bh * ratio)))
    if x3-x0 < 4 or y3-y0 < 4:
        return None
    if h * w / float((x3-x0) * (y3-y0)) < min_gain:
        return None
    return (x0, y0, x3, y3)


def predict_local_view(predictor, image_rgb, box, cfg):
    """Run SAM on a true RGB crop; local native logits are NEVER reused on full image."""
    h, w = image_rgb.shape[:2]
    rect = crop_bounds((h,w),box,cfg.crop_context_ratio,cfg.min_crop_coverage_gain)
    if rect is None:
        return None
    x1,y1,x2,y2 = rect
    crop_rgb = np.ascontiguousarray(image_rgb[y1:y2, x1:x2])
    crop_box = [max(0, box[0]-x1), max(0, box[1]-y1),
                min(x2-x1, box[2]-x1), min(y2-y1, box[3]-y1)]
    predictor.set_image(crop_rgb)
    pred = predict_initial(predictor, crop_box)
    mask = np.zeros((h,w),dtype=bool)
    prob = np.zeros((h,w),dtype=np.float32)
    observed = np.zeros((h,w),dtype=bool)
    mask[y1:y2,x1:x2] = pred.mask
    prob[y1:y2,x1:x2] = pred.prob
    observed[y1:y2,x1:x2] = True
    return LocalEvidence(mask,prob,pred.score,pred.stability,rect,observed)


def _pick_peaks(binary_region, scores, count, min_spacing, min_height=0.0):
    """Return reproducible pixel maxima (x,y) with suppression across selected points."""
    score = np.asarray(scores, dtype=np.float32).copy()
    score[~binary_region] = -np.inf
    results = []
    for _ in range(max(0,count)):
        idx = int(np.argmax(score))
        y,x = np.unravel_index(idx,score.shape)
        if not np.isfinite(score[y,x]) or score[y,x] < min_height:
            break
        results.append([int(x),int(y)])
        cv2.circle(score, (int(x),int(y)), int(min_spacing), float('-inf'), thickness=-1)
    return results


def extract_reliable_points(initial, local, box, cfg, all_boxes_mask=None):
    """Interior positive evidence is view intersection; negatives require BOTH views and observed pixels."""
    h,w=initial.mask.shape
    common_fg = initial.mask & local.mask & local.observed & box_to_mask(box,h,w)
    distance_fg = cv2.distanceTransform(common_fg.astype(np.uint8),cv2.DIST_L2,5)
    pos = _pick_peaks(common_fg,distance_fg,cfg.positive_points,
                      cfg.points_min_distance_px * 3,
                      min_height=cfg.points_min_distance_px)
    neg = []
    if cfg.enable_negative_points and cfg.negative_points > 0:
        h,w = initial.mask.shape
        boxmask = box_to_mask(box,h,w)
        # Critical for multiple annotated objects: a pixel outside *this* box
        # may lie INSIDE another target box and must never become a negative point.
        forbidden = boxmask if all_boxes_mask is None else np.asarray(all_boxes_mask,dtype=bool)
        if forbidden.shape != boxmask.shape:
            raise ValueError('all_boxes_mask shape differs from original image')
        yy,xx = np.ogrid[:h,:w]
        # Restrict to context ring outside supplied box, not arbitrary distant background.
        dx = np.maximum(np.maximum(box[0]-xx,xx-box[2]),0)
        dy = np.maximum(np.maximum(box[1]-yy,yy-box[3]),0)
        dist_box = np.sqrt(dx*dx + dy*dy)
        max_context = max(8.,0.25*min(box[2]-box[0],box[3]-box[1]))
        safe_bg = (local.observed & ~forbidden & ~initial.mask & ~local.mask
                   & (initial.prob <= cfg.background_prob_max)
                   & (local.prob <= cfg.background_prob_max)
                   & (dist_box <= max_context))
        d_to_foreground = cv2.distanceTransform((~(initial.mask | local.mask)).astype(np.uint8),
                                                  cv2.DIST_L2,5)
        neg = _pick_peaks(safe_bg,d_to_foreground,cfg.negative_points,
                          cfg.points_min_distance_px*3,
                          min_height=cfg.points_min_distance_px)
    return pos,neg


def _recover_local_positive(local, box, cfg):
    """Local-only rescue is allowed only when the original proposal is invalid.

    A point from this function is *not* cross-view confirmed, so use a stronger
    local quality gate and do not pass the invalid first-round logits along.
    """
    if local.score < cfg.local_recovery_score_min or local.stability < cfg.accept_min_stability:
        return []
    h,w = local.mask.shape
    valid_fg = local.mask & local.observed & box_to_mask(box,h,w)
    if valid_fg.sum() < max(3, int(.01 * box_to_mask(box,h,w).sum())):
        return []
    dist = cv2.distanceTransform(valid_fg.astype(np.uint8),cv2.DIST_L2,5)
    return _pick_peaks(valid_fg,dist,1,cfg.points_min_distance_px,
                       min_height=max(2.,cfg.points_min_distance_px * .5))


def _is_initial_invalid(initial, box_mask, cfg):
    fill = int(initial.mask.sum()) / max(1,int(box_mask.sum()))
    return fill < cfg.risk_tiny_fill or initial.score < 0.55


def _candidate_gate(initial, candidate, boxmask, cfg):
    """No GT. These checks avoid obvious catastrophic changes; they do not guarantee improvement."""
    if candidate is None or not bool(candidate.mask.any()):
        return False,'empty_candidate'
    if not np.isfinite(candidate.score) or candidate.score < initial.score - cfg.accept_score_tolerance:
        return False,'lower_sam_score'
    if candidate.stability < cfg.accept_min_stability:
        return False,'unstable_candidate'
    inside = int((candidate.mask & boxmask).sum())
    area = max(1,int(candidate.mask.sum()))
    fill = inside / max(1,int(boxmask.sum()))
    outside_ratio = (area-inside) / area
    if not cfg.accept_min_box_fill <= fill <= cfg.accept_max_box_fill:
        return False,'box_fill_out_of_range'
    if outside_ratio > cfg.accept_max_outside:
        return False,'outside_box_excess'
    invalid = _is_initial_invalid(initial,boxmask,cfg)
    orig_area = int(initial.mask.sum())
    if not invalid and orig_area:
        ratio = area / orig_area
        if not cfg.accept_min_area_multiplier <= ratio <= cfg.accept_max_area_multiplier:
            return False,'area_change_too_large'
        if compute_iou(initial.mask,candidate.mask) < cfg.accept_min_iou_with_initial:
            return False,'subject_drift'
    return True,''


def _selective_mask(initial, proposed, local, boxmask, cfg):
    """Only edits supported by local evidence; new distant thin structure can enter via confirmed add."""
    original = initial.mask
    p = proposed.mask
    l = local.mask
    band = dilate_mask(original,cfg.boundary_band_radius) & ~erode_mask(original,cfg.boundary_band_radius)
    supported_add = ((~original) & p & l & local.observed & boxmask
                     & (proposed.prob >= cfg.addition_support_prob_min)
                     & (local.prob >= cfg.addition_support_prob_min))
    supported_remove = (original & (~p) & (~l) & local.observed & band
                        & (proposed.prob <= cfg.removal_support_prob_max)
                        & (local.prob <= cfg.removal_support_prob_max))
    final = (original | supported_add) & (~supported_remove)
    prob = initial.prob.copy()
    edited = final != original
    prob[edited] = proposed.prob[edited]
    return final,prob


def refine_one_box(predictor, image_rgb, box, initial, cfg, all_boxes_mask=None):
    h,w = initial.mask.shape
    boxmask = box_to_mask(box,h,w)
    outcome=RefineOutcome(initial.mask.copy(),initial.prob.copy())
    flags = ablation_flags(cfg.ablation)
    reasons = risk_reasons(initial,box,boxmask,cfg)
    outcome.trigger_reasons = reasons
    if cfg.ablation == 'A0':
        outcome.rollback_reason='baseline'
        return outcome
    if cfg.refine_scope == 'risk' and not reasons:
        outcome.rollback_reason='no_risk_trigger'
        return outcome
    outcome.triggered=True
    local=None
    if flags['local']:
        local=predict_local_view(predictor,image_rgb,box,cfg)
        if local is None:
            outcome.rollback_reason='no_useful_local_crop'
            return outcome
        outcome.used_local=True
        outcome.crop_xyxy=local.crop_xyxy
        outcome.local_score=local.score
        # The crop changed the predictor image. Re-encode original before full-image prompting.
        if flags['points']:
            outcome.points_pos,outcome.points_neg=extract_reliable_points(
                initial,local,box,cfg,all_boxes_mask=all_boxes_mask)
    # Without an independent local view, A1 intentionally tests plain logit feedback.
    if cfg.ablation == 'A2':
        class _C: pass
        proposed=_C()
        proposed.mask=local.mask
        proposed.prob=local.prob
        proposed.score=local.score
        proposed.stability=local.stability
    else:
        if flags['local']:
            predictor.set_image(image_rgb)
        invalid = _is_initial_invalid(initial,boxmask,cfg)
        if flags['points'] and invalid and not outcome.points_pos:
            outcome.points_pos = _recover_local_positive(local,box,cfg)
            outcome.recovered_from_local_only = bool(outcome.points_pos)
        if flags['points'] and not outcome.points_pos:
            outcome.rollback_reason='no_reliable_positive_points'
            return outcome
        # A corrupted initial mask/logit should not overrule recovered local
        # evidence (especially for empty masks and completely missed objects).
        input_logits = (initial.low_res_logit if flags['logits']
                        and not outcome.recovered_from_local_only else None)
        proposed=predict_with_prompts(
            predictor,box,
            positives=outcome.points_pos if flags['points'] else None,
            negatives=outcome.points_neg if flags['points'] else None,
            mask_logits=input_logits,
        )
        outcome.used_native_logits=input_logits is not None
    outcome.refined_score=float(proposed.score)
    outcome.candidate_iou=compute_iou(initial.mask,proposed.mask)
    passed,why=_candidate_gate(initial,proposed,boxmask,cfg)
    if not passed:
        outcome.rollback_reason=why
        return outcome
    # A3/A4: verify the full-image prompted output is still compatible with
    # the *new* local-view evidence. This prevents a self-confirming logit loop.
    if local is not None and cfg.ablation in ('A3','A4'):
        local_agreement=compute_iou(proposed.mask & boxmask, local.mask & boxmask)
        if local_agreement < cfg.accept_min_iou_with_local:
            outcome.rollback_reason='local_evidence_disagrees'
            return outcome
    if flags['selective'] and local is not None and not _is_initial_invalid(initial,boxmask,cfg):
        final,prob=_selective_mask(initial,proposed,local,boxmask,cfg)
    elif cfg.ablation == 'A2' and not _is_initial_invalid(initial,boxmask,cfg):
        # A2 isolates local-view candidate. It gets a near-boundary edit restriction.
        band=dilate_mask(initial.mask,cfg.boundary_band_radius) & ~erode_mask(initial.mask,cfg.boundary_band_radius)
        final=np.where(band,local.mask,initial.mask)
        prob=np.where(final != initial.mask,local.prob,initial.prob)
    else:
        final=proposed.mask.copy()
        prob=proposed.prob.copy()
    if not final.any():
        outcome.rollback_reason='empty_final_mask'
        return outcome
    outcome.change_ratio=float(np.mean(final != initial.mask))
    changed_pixels=int(np.count_nonzero(final != initial.mask))
    if changed_pixels < cfg.change_min_pixels or outcome.change_ratio < cfg.change_min_fraction:
        outcome.rollback_reason='no_material_change'
        return outcome
    # Check the actual output too: A4 applies a local-edit fusion *after* the
    # candidate gate and the fused mask is not itself scored by SAM.
    area=int(final.sum())
    outside=int((final & ~boxmask).sum()) / max(1,area)
    fill=int((final & boxmask).sum()) / max(1,int(boxmask.sum()))
    if outside > cfg.accept_max_outside or not cfg.accept_min_box_fill <= fill <= cfg.accept_max_box_fill:
        outcome.rollback_reason='fused_mask_invalid'
        return outcome
    outcome.mask=final
    outcome.prob=np.asarray(prob,dtype=np.float32)
    outcome.accepted=True
    return outcome
