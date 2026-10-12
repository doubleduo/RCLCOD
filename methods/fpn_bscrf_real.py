# -*- coding: utf-8 -*-
"""Controlled BSCRF ablations ON the exact PVTv2-B4 single-scale FPN Baseline.

Trainer: basemain_continuous.py
Config:  csr_mhsiu_nc_real_bscrf.py (identical real schedule + use_box=True)

A0  = untouched original single-scale PVTv2-B4 FPN baseline
A0M = P2/P3 original ZoomNeXt MHSIU2; P4/P5 plain FPN
A1  = P2/P3 antialiased anchored spatial residual
A1P = A1 + weakly supervised predicted box prior head (frequency control)
A2  = A1P + box-conditioned Haar high-frequency residual
A3  = A2 + inside/outside local spectral contrast

All the variants share the original BCE segmentation loss and continuous
Clean/Noisy sampling: the *config filename* contains 'nc', but the actual
PvtV2B4_FPN_Baseline's forward implements BCE, not Noisy-COD NC loss.

All methods use RGB-only segmentation at test time. The weak box mask is
used ONLY as an auxiliary target during training for A1P/A2/A3.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .fpn_baseline import PvtV2B4_FPN_Baseline
from .zoomnext.layers import MHSIU2
from .bscrf_mhsiu import BSCRFMHSIU


class PvtV2B4_FPN_BSCRF_A0(PvtV2B4_FPN_Baseline):
    """Exact single-scale baseline, including original BCE loss."""
    pass


class _BSCRFRealBase(PvtV2B4_FPN_Baseline):
    def __init__(self, variant=1, use_box_head=False, use_original_mhsiu=False,
                 box_loss_weight=0.10, fpn_dim=64, **kwargs):
        super().__init__(fpn_dim=fpn_dim, **kwargs)
        self.variant = int(variant)
        self.use_box_head = bool(use_box_head)
        self.use_original_mhsiu = bool(use_original_mhsiu)
        self.box_loss_weight = float(box_loss_weight)
        if self.use_original_mhsiu:
            self.siu_2 = MHSIU2(fpn_dim, 4)
            self.siu_3 = MHSIU2(fpn_dim, 4)
        else:
            self.siu_2 = BSCRFMHSIU(channels=fpn_dim, variant=self.variant)
            self.siu_3 = BSCRFMHSIU(channels=fpn_dim, variant=self.variant)
        if self.use_box_head:
            hidden = max(8, fpn_dim // 4)
            self.box_head = nn.Sequential(
                nn.Conv2d(fpn_dim, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, 1),
            )
            nn.init.constant_(self.box_head[-1].bias, -1.0)
        self._last_box_logits = None

    def body(self, data):
        image_m = data['image_m']
        image_l = data.get('image_l')
        if image_l is None:
            # Matches existing MHSIU2 fallback; *true* large RGB supplied by
            # ImageTrainDataset/ImageTestDataset is preferred.
            image_l = F.interpolate(image_m, scale_factor=1.5,
                                    mode='bilinear', align_corners=False)
        m2, m3, m4, m5 = self.normalize_encoder(image_m)
        l2, l3, _, _ = self.normalize_encoder(image_l)
        z2 = self.lateral_2(m2)
        z3 = self.lateral_3(m3)
        z4 = self.lateral_4(m4)
        z5 = self.lateral_5(m5)
        zl2 = self.lateral_2(l2)
        zl3 = self.lateral_3(l3)

        box_prior = None
        if self.use_box_head:
            # IMPORTANT: do not use data['box_mask'] as model input, even during
            # training, preventing train/test prompt leakage.
            self._last_box_logits = self.box_head(z3)
            box_prior = self._last_box_logits.sigmoid()
        else:
            self._last_box_logits = None

        if self.use_original_mhsiu:
            z2 = self.siu_2(l=zl2, m=z2)
            z3 = self.siu_3(l=zl3, m=z3)
        else:
            z2 = self.siu_2(l=zl2, m=z2, box_prior=box_prior)
            z3 = self.siu_3(l=zl3, m=z3, box_prior=box_prior)

        # The P4/P5 feature projection, FPN top-down, smooth operators,
        # prediction head all remain identical to the original baseline.
        p5 = self.smooth_5(z5)
        p4 = self.smooth_4(z4 + self._resize_like(p5, m4))
        p3 = self.smooth_3(z3 + self._resize_like(p4, m3))
        p2 = self.smooth_2(z2 + self._resize_like(p3, m2))
        logits = self.predictor(p2)
        return F.interpolate(logits, size=image_m.shape[-2:],
                             mode='bilinear', align_corners=False)

    def forward(self, data, iter_percentage=1, **kwargs):
        # The parent forward handles original BCE loss and the runner output
        # contract. The continuous trainer handles Clean/Noisy batch sampling.
        output = super().forward(data=data, iter_percentage=iter_percentage,
                                 **kwargs)
        if not self.training or not self.use_box_head:
            return output
        if 'box_mask' not in data:
            raise KeyError('A1P/A2/A3 require data["box_mask"] during training. '
                           'Use csr_mhsiu_nc_real_bscrf.py (use_box=True), '
                           'and check combined_tr.box_json in dataset.yaml.')
        target = data['box_mask'].float()
        if target.ndim != 4 or target.size(1) != 1:
            raise ValueError('box_mask must be B,1,H,W.')
        target = F.interpolate(target, size=self._last_box_logits.shape[-2:],
                               mode='nearest').clamp(0., 1.)
        box_loss = F.binary_cross_entropy_with_logits(self._last_box_logits,
                                                        target)
        output['loss'] = output['loss'] + self.box_loss_weight * box_loss
        output['loss_items']['box_prior'] = box_loss.detach()
        output['loss_items']['total'] = output['loss'].detach()
        output['loss_str'] += f' Box:{box_loss.detach().item():.4f}'
        return output


class PvtV2B4_FPN_BSCRF_A0M(_BSCRFRealBase):
    """Reference old MHSIU2 at P2/P3 only; no added box supervision."""
    def __init__(self, **kwargs):
        super().__init__(use_original_mhsiu=True, use_box_head=False, **kwargs)


class PvtV2B4_FPN_BSCRF_A1(_BSCRFRealBase):
    def __init__(self, **kwargs):
        super().__init__(variant=1, use_box_head=False, **kwargs)


class PvtV2B4_FPN_BSCRF_A1P(_BSCRFRealBase):
    def __init__(self, **kwargs):
        super().__init__(variant=1, use_box_head=True, **kwargs)


class PvtV2B4_FPN_BSCRF_A2(_BSCRFRealBase):
    def __init__(self, **kwargs):
        super().__init__(variant=2, use_box_head=True, **kwargs)


class PvtV2B4_FPN_BSCRF_A3(_BSCRFRealBase):
    def __init__(self, **kwargs):
        super().__init__(variant=3, use_box_head=True, **kwargs)


__all__ = ['PvtV2B4_FPN_BSCRF_A0', 'PvtV2B4_FPN_BSCRF_A0M',
           'PvtV2B4_FPN_BSCRF_A1', 'PvtV2B4_FPN_BSCRF_A1P',
           'PvtV2B4_FPN_BSCRF_A2', 'PvtV2B4_FPN_BSCRF_A3']
