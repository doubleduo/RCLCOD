"""Configuration, presets, and reproducibility settings for the V1 experiments."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Config:
    checkpoint: str = './weights/sam_vit_h_4b8939.pth'
    sam_model_type: str = 'vit_h'
    device: str = 'cuda'
    img_dir: str = './data/Train/Imgs'
    gt_dir: str = './data/Train/GT'
    box_dir: str = './data/Train/box_label'
    output_dir: str = './SGT/SAM1_Refine_V1'
    ablation: str = 'A4'
    refine_scope: str = 'risk'  # risk | all
    use_gt_eval: bool = True
    save_sod_per_image: bool = True
    save_visualizations: str = 'changed'  # changed | all | none
    max_images: int = 0
    fail_fast: bool = False
    # Risk detection (heuristics, not learned or GT calibrated)
    risk_score_below: float = 0.80
    risk_stability_below: float = 0.90
    risk_consensus_below: float = 0.70
    risk_boundary_above: float = 0.50
    risk_small_box_fraction: float = 0.06
    risk_outside_fraction: float = 0.15
    risk_tiny_fill: float = 0.03
    risk_fragment_components: int = 3
    risk_fragment_largest_ratio: float = 0.75
    # Local view and point prompts
    crop_context_ratio: float = 0.25
    min_crop_coverage_gain: float = 1.10
    positive_points: int = 2
    negative_points: int = 2
    enable_negative_points: bool = True
    points_min_distance_px: int = 5
    background_prob_max: float = 0.20
    # Non-GT guard rails; conservative and tunable
    accept_score_tolerance: float = 0.06
    accept_min_stability: float = 0.75
    accept_min_iou_with_initial: float = 0.50
    accept_min_iou_with_local: float = 0.50
    accept_min_box_fill: float = 0.01
    accept_max_box_fill: float = 0.98
    accept_max_outside: float = 0.08
    accept_max_area_multiplier: float = 1.90
    accept_min_area_multiplier: float = 0.50
    # Thin limbs/antennae may alter a very small percentage of a large image.
    change_min_fraction: float = 0.0
    change_min_pixels: int = 3
    boundary_band_radius: int = 5
    addition_support_prob_min: float = 0.70
    removal_support_prob_max: float = 0.30
    local_recovery_score_min: float = 0.70
    # GT evaluation only; never used by the acceptance logic
    # Preserve legacy gt_bad_* semantics for downstream CSV consumers.
    bad_iou_threshold: float = 0.70
    bad_mae_threshold: float = 0.05

    @classmethod
    def load(cls, path=None):
        d = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
        extra = set(d) - set(cls.__dataclass_fields__)
        if extra:
            raise ValueError(f'Unknown config keys: {sorted(extra)}')
        obj = cls(**d)
        obj.validate()
        return obj

    def validate(self):
        if self.ablation not in {'A0', 'A1', 'A2', 'A3', 'A4'}:
            raise ValueError('ablation must be A0..A4')
        if self.refine_scope not in {'risk', 'all'}:
            raise ValueError('refine_scope must be risk or all')
        if self.save_visualizations not in {'changed', 'all', 'none'}:
            raise ValueError('save_visualizations must be changed, all or none')
        if self.crop_context_ratio < 0 or self.min_crop_coverage_gain < 1.0:
            raise ValueError('invalid crop parameters')
        if not 0.0 <= self.accept_min_iou_with_initial <= 1.0 or not 0.0 <= self.accept_min_iou_with_local <= 1.0:
            raise ValueError('invalid acceptance overlap')
        if not 0.0 <= self.background_prob_max <= 1.0:
            raise ValueError('background_prob_max must be in [0,1]')
        if self.change_min_pixels < 1 or not 0 <= self.change_min_fraction < 1:
            raise ValueError('invalid minimum edit size')
        for p in (self.addition_support_prob_min, self.removal_support_prob_max,
                  self.local_recovery_score_min):
            if not 0.0 <= p <= 1.0:
                raise ValueError('probability/score thresholds must be in [0,1]')

    def dump(self, filename):
        Path(filename).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding='utf-8')


def ablation_flags(abl):
    return {
        'A0': dict(local=False, points=False, logits=False, selective=False),
        'A1': dict(local=False, points=False, logits=True, selective=False),
        'A2': dict(local=True, points=False, logits=False, selective=False),
        'A3': dict(local=True, points=True, logits=False, selective=False),
        'A4': dict(local=True, points=True, logits=True, selective=True),
    }[abl]
