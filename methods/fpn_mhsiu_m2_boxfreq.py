# -*- coding: utf-8 -*-
"""Box-guided frequency ablation ON EXACT original 4-level MHSIU-AA-M2.

Parent model: PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M2
All four AA-M2 MHSIU units, original groupwise softmax, top-down FPN,
PVTv2-B4, input normalization, BCE, and optimizer groups are unchanged.

The NEW branch is applied AFTER MHSIU2 fusion only at P2/P3:
    F_out = F_M2 + tanh(freq_scale) * IDWT(0, gated LH/HL/HH).

The Box Prior Head predicts box support from RGB/PVT features at ALL times.
Ground-truth box masks are auxiliary loss TARGETS only (never input to body).


B1: M2 + weak box auxiliary head, zero frequency changes.
B2: B1 + local Haar high-frequency residual at P2/P3.
B3: B2 + ROI/ring spectral-energy difference as extra gate evidence.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .fpn_mhsiu_aa import PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M2


class HaarDWT(nn.Module):
    """One-level, orthonormal Haar; exact inverse via HaarIDWT."""

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode='replicate')
        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]
        ll = (a + b + c + d) * 0.5
        lh = (-a - b + c + d) * 0.5
        hl = (-a + b - c + d) * 0.5
        hh = (a - b - c + d) * 0.5
        return ll, lh, hl, hh


class HaarIDWT(nn.Module):
    def forward(self, ll, lh, hl, hh, shape):
        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5
        batch, channels, h, w = ll.shape
        result = ll.new_empty(batch, channels, 2 * h, 2 * w)
        result[..., 0::2, 0::2] = a
        result[..., 0::2, 1::2] = b
        result[..., 1::2, 0::2] = c
        result[..., 1::2, 1::2] = d
        return result[..., :shape[0], :shape[1]]


def _band_contrast(energy, roi, ring, eps=1e-5):
    """Per-image, per-band difference of interior and surrounding energy."""
    ri = roi.sum((-2, -1), keepdim=True)
    ro = ring.sum((-2, -1), keepdim=True)
    ei = (energy * roi).sum((-2, -1), keepdim=True) / ri.clamp_min(eps)
    eo = (energy * ring).sum((-2, -1), keepdim=True) / ro.clamp_min(eps)
    c = (ei - eo) / (ei + eo + eps)
    valid = (ri > 1) & (ro > 1)
    return torch.where(valid, c, torch.zeros_like(c)).clamp(-1, 1)


class BoxFrequencyResidual(nn.Module):
    """Frequency selection on already-fused MHSIU features (NOT a new encoder).

    Args:
        feat: [B,C,H,W] after inherited MHSIU2-AA-M2
        box_prior: [B,1,h,w] *predicted* from M2 original-scale features

    If freq_scale=0, forward returns feat exactly. This is useful for
    checking a newly inserted branch has not changed the existing baseline.
    """

    def __init__(self, channels=64, contrast=False):
        super().__init__()
        self.contrast = bool(contrast)
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        hidden = max(8, channels // 8)
        # 3 band energies + predicted box prior + optional 3 ROI/ring contrasts
        in_channels = 7 if self.contrast else 4
        self.band_gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )
        # Start at exact M2 segmentation features; this parameter learns first.
        self.freq_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer('_last_ratio', torch.tensor(0.0), persistent=False)

    def forward(self, feat, box_prior):
        if feat.ndim != 4 or box_prior.ndim != 4 or box_prior.shape[:2] != (feat.size(0), 1):
            raise ValueError('Expected feat B,C,H,W and predicted box_prior B,1,h,w')
        original_dtype = feat.dtype
        shape = feat.shape[-2:]
        # Haar with FP32 arithmetic under AMP.
        ll, lh, hl, hh = self.dwt(feat.float())
        bands = [lh, hl, hh]
        energies = [z.abs().mean(dim=1, keepdim=True) for z in bands]
        # Relative within-image energy is more stable than raw band magnitude.
        energies = [e / e.detach().mean((-2, -1), keepdim=True).clamp_min(1e-4)
                    for e in energies]
        hp_shape = ll.shape[-2:]
        box = F.interpolate(box_prior.float(), size=hp_shape, mode='bilinear', align_corners=False).clamp(0, 1)
        gate_features = [*energies, box]
        if self.contrast:
            # A soft expansion ring on the *same continuous features*.
            # No artificial high-frequency edges from zeroing out RGB at box boundary.
            ring = (F.max_pool2d(box, 7, stride=1, padding=3) - box).clamp(0, 1)
            cs = [_band_contrast(e, box, ring).expand_as(e) for e in energies]
            gate_features.extend(cs)
        gates = torch.sigmoid(self.band_gate(torch.cat(gate_features, dim=1)))
        # Box defines a coarse search range, not ground-truth foreground.
        # Do not set outside box strictly to zero: uncertain box head can be wrong.
        gates = gates * (0.20 + 0.80 * box)
        detail = self.idwt(
            torch.zeros_like(ll),
            *(band * gates[:, i:i+1] for i, band in enumerate(bands)),
            shape=shape,
        ).to(original_dtype)
        scale = torch.tanh(self.freq_scale)
        result = feat + scale * detail
        # Diagnostics, not part of autograd/training objective.
        with torch.no_grad():
            self._last_ratio.copy_((scale * detail).float().square().mean().sqrt() /
                                   feat.float().square().mean().sqrt().clamp_min(1e-6))
        return result


class _PvtV2B4_FPN_M2_BoxBase(PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M2):
    """M2 parent weights/architecture retained, with optional frequency bypass."""

    def __init__(self, enable_frequency=False, enable_contrast=False,
                 box_loss_weight=0.10, fpn_dim=64, **kwargs):
        super().__init__(fpn_dim=fpn_dim, **kwargs)
        self.enable_frequency = bool(enable_frequency)
        self.box_loss_weight = float(box_loss_weight)
        if self.box_loss_weight < 0:
            raise ValueError('box_loss_weight must be >=0')
        # Fork CPU/CUDA RNG so new branches do not perturb subsequent sampling.
        with torch.random.fork_rng(devices=[]):
            hidden = max(8, fpn_dim // 4)
            self.box_head = nn.Sequential(
                nn.Conv2d(fpn_dim, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, 1),
            )
            nn.init.constant_(self.box_head[-1].bias, -1.0)
            if self.enable_frequency:
                self.box_freq_2 = BoxFrequencyResidual(fpn_dim, contrast=enable_contrast)
                self.box_freq_3 = BoxFrequencyResidual(fpn_dim, contrast=enable_contrast)
        self._last_box_logits = None

    def body(self, data):
        # The parent's only difference is the TWO indicated lines adding
        # box_freq_2 and box_freq_3. All four SIU modules and FPN are shared.
        image_m = data['image_m']
        m2, m3, m4, m5 = self.normalize_encoder(image_m)
        z2 = self.lateral_2(m2)
        z3 = self.lateral_3(m3)
        z4 = self.lateral_4(m4)
        z5 = self.lateral_5(m5)
        image_l = data.get('image_l')
        if image_l is None:
            image_l = F.interpolate(image_m, scale_factor=1.5,
                                    mode='bilinear', align_corners=False)
        l2, l3, l4, l5 = self.normalize_encoder(image_l)

        # RGB/PVT-predicted box prior, including in the training step.
        self._last_box_logits = self.box_head(z3)
        box_prior = self._last_box_logits.sigmoid()

        z2 = self.siu_2(l=self.lateral_2(l2), m=z2)
        z3 = self.siu_3(l=self.lateral_3(l3), m=z3)
        z4 = self.siu_4(l=self.lateral_4(l4), m=z4)
        z5 = self.siu_5(l=self.lateral_5(l5), m=z5)

        if self.enable_frequency:
            z2 = self.box_freq_2(z2, box_prior)
            z3 = self.box_freq_3(z3, box_prior)

        p5 = self.smooth_5(z5)
        p4 = self.smooth_4(z4 + self._resize_like(p5, m4))
        p3 = self.smooth_3(z3 + self._resize_like(p4, m3))
        p2 = self.smooth_2(z2 + self._resize_like(p3, m2))
        logits = self.predictor(p2)
        return F.interpolate(logits, size=image_m.shape[-2:],
                             mode='bilinear', align_corners=False)

    def forward(self, data, iter_percentage=1, **kwargs):
        # Inherited original M2 forward computes the original BCE training
        # objective and output contract. Parent delegates to the body above.
        result = super().forward(data=data, iter_percentage=iter_percentage, **kwargs)
        if not self.training:
            return result
        if 'box_mask' not in data:
            raise KeyError('BoxFreq/BH need train.data.use_box=True and dataset.yaml box_json. '
                           'Use configs/curablation/sam2_real_m2_boxfreq.py.')
        box_target = data['box_mask'].float()
        if box_target.ndim != 4 or box_target.size(1) != 1:
            raise ValueError('box_mask must be [B,1,H,W]')
        box_target = F.interpolate(box_target, size=self._last_box_logits.shape[-2:], mode='nearest')
        loss_box = F.binary_cross_entropy_with_logits(self._last_box_logits, box_target)
        result['loss'] = result['loss'] + self.box_loss_weight * loss_box
        result['loss_items']['box_bce'] = loss_box.detach()
        result['loss_items']['total'] = result['loss'].detach()
        if self.enable_frequency:
            result['loss_items']['freq_scale_p2'] = torch.tanh(self.box_freq_2.freq_scale).detach()
            result['loss_items']['freq_scale_p3'] = torch.tanh(self.box_freq_3.freq_scale).detach()
            result['loss_items']['freq_ratio_p2'] = self.box_freq_2._last_ratio.detach()
            result['loss_items']['freq_ratio_p3'] = self.box_freq_3._last_ratio.detach()
        result['loss_str'] += f' Box:{loss_box.detach().item():.4f}'
        return result


class PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M2_BoxHead(_PvtV2B4_FPN_M2_BoxBase):
    """B1: M2 + box auxiliary supervision only, no frequency residual."""
    def __init__(self, **kwargs):
        super().__init__(enable_frequency=False, **kwargs)


class PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M2_BoxFreq(_PvtV2B4_FPN_M2_BoxBase):
    """B2: M2 + box-prior gated Haar high-frequency residual at P2/P3."""
    def __init__(self, **kwargs):
        super().__init__(enable_frequency=True, enable_contrast=False, **kwargs)


class PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M2_BoxContrast(_PvtV2B4_FPN_M2_BoxBase):
    """B3: M2 + B2 + ROI/ring band-energy contrast gating."""
    def __init__(self, **kwargs):
        super().__init__(enable_frequency=True, enable_contrast=True, **kwargs)


__all__ = [
    'HaarDWT', 'HaarIDWT', 'BoxFrequencyResidual',
    'PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M2_BoxHead',
    'PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M2_BoxFreq',
    'PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M2_BoxContrast',
]
