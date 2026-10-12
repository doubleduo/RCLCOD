# -*- coding: utf-8 -*-
"""Small, standalone MHSIU replacement for controlled RCLCOD ablations.

A1: anti-aliased large-to-medium alignment + anchored spatial residual.
A2: A1 + box-conditioned Haar high-frequency residual.
A3: A2 + inside-box vs surrounding-ring spectral contrast gate.

The box prior here is a *predicted* soft spatial map from the student model.
It is never the human annotation at inference (nor directly at train forward).
"""

import torch
from torch import nn
import torch.nn.functional as F


class AntiAliasAlign(nn.Module):
    """Fixed 3x3 Gaussian blur, then antialiased bilinear size alignment."""

    def __init__(self):
        super().__init__()
        k = torch.tensor([1., 2., 1.])
        kernel = (k[:, None] * k[None, :]) / 16.0
        self.register_buffer('kernel', kernel.reshape(1, 1, 3, 3))

    def forward(self, x, size):
        if x.shape[-2:] == tuple(size):
            return x
        channels = x.shape[1]
        ker = self.kernel.to(dtype=x.dtype).expand(channels, 1, 3, 3)
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), ker,
                     groups=channels)
        shrink = size[0] < x.shape[-2] or size[1] < x.shape[-1]
        return F.interpolate(x, size=size, mode='bilinear',
                             align_corners=False, antialias=shrink)


class HaarDWT(nn.Module):
    """Orthonormal single-level Haar DWT; LH/HL sign convention consistent with IDWT."""

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, w % 2, 0, h % 2), mode='replicate')
        a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
        c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
        return ((a + b + c + d) * .5,
                (-a - b + c + d) * .5,
                (-a + b - c + d) * .5,
                (a - b - c + d) * .5)


class HaarIDWT(nn.Module):
    def forward(self, ll, lh, hl, hh, output_size):
        a = (ll - lh - hl + hh) * .5
        b = (ll - lh + hl - hh) * .5
        c = (ll + lh - hl - hh) * .5
        d = (ll + lh + hl + hh) * .5
        n, ch, h, w = ll.shape
        result = ll.new_empty((n, ch, h * 2, w * 2))
        result[..., 0::2, 0::2] = a
        result[..., 0::2, 1::2] = b
        result[..., 1::2, 0::2] = c
        result[..., 1::2, 1::2] = d
        return result[..., :output_size[0], :output_size[1]]


def _energy_contrast(energy, roi, ring, eps=1e-5):
    """Signed, normalized ROI-vs-ring mean energy, Bx1x1x1."""
    roi_sum = roi.sum(dim=(-2, -1), keepdim=True)
    ring_sum = ring.sum(dim=(-2, -1), keepdim=True)
    e_roi = (energy * roi).sum(dim=(-2, -1), keepdim=True) / roi_sum.clamp_min(eps)
    e_ring = (energy * ring).sum(dim=(-2, -1), keepdim=True) / ring_sum.clamp_min(eps)
    contrast = (e_roi - e_ring) / (e_roi + e_ring + eps)
    valid = (roi_sum > 1.0) & (ring_sum > 1.0)
    return torch.where(valid, contrast, torch.zeros_like(contrast)).clamp(-1, 1)


class BSCRFMHSIU(nn.Module):
    """Small replacement for ZoomNeXt MHSIU2 on P2/P3 only.

    Input:
      l: [B,C,1.5H,1.5W] large-branch feature (projected by FPN lateral 1x1)
      m: [B,C,H,W] original-branch feature
      box_prior: [B,1,hp,wp] predicted probability map, A2/A3 only
    Output:
      [B,C,H,W], with m as a semantic anchor.

    variant=1: antialias + anchored spatial delta
    variant=2: variant1 + box-gated high-frequency residual
    variant=3: variant2 + ROI/ring frequency-energy contrast
    """

    def __init__(self, channels=64, variant=1):
        super().__init__()
        if variant not in (1, 2, 3):
            raise ValueError('variant must be 1, 2 or 3')
        self.variant = variant
        self.align = AntiAliasAlign()
        self.spatial_delta = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, max(8, channels // 4), 1),
            nn.GELU(),
            nn.Conv2d(max(8, channels // 4), 1, 1),
            nn.Sigmoid(),
        )
        self.gamma_spatial = nn.Parameter(torch.tensor(0.10))
        if variant >= 2:
            self.dwt = HaarDWT()
            self.idwt = HaarIDWT()
            hidden = 12
            # 3 spatial HF difference energies + box prior (+ 3 band contrasts in A3)
            num_gate_channels = 4 + (3 if variant == 3 else 0)
            self.frequency_gate = nn.Sequential(
                nn.Conv2d(num_gate_channels, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 3, 1),
                nn.Sigmoid(),
            )
            self.gamma_frequency = nn.Parameter(torch.tensor(0.05))

    def forward(self, l, m, box_prior=None):
        if l.ndim != 4 or m.ndim != 4 or l.shape[:2] != m.shape[:2]:
            raise ValueError('l and m must be BCHW and have matching batch/channels')
        large = self.align(l, m.shape[-2:])
        delta = self.spatial_delta(large - m)
        gate = self.spatial_gate(torch.cat([large, m], dim=1))
        result = m + self.gamma_spatial * (gate * delta)
        if self.variant == 1:
            return result

        if box_prior is None:
            raise ValueError('A2/A3 need a predicted box_prior tensor; not GT box_mask')
        if box_prior.ndim != 4 or box_prior.shape[:2] != (m.shape[0], 1):
            raise ValueError('box_prior must have shape [B,1,h,w]')

        # The wavelet computations are explicitly in fp32 for stability under AMP.
        out_dtype = m.dtype
        lll, *hl = self.dwt(large.float())
        llm, *hm = self.dwt(m.float())
        difference = [x - y for x, y in zip(hl, hm)]
        band_energy = [x.abs().mean(dim=1, keepdim=True) for x in difference]
        hp = band_energy[0].shape[-2:]
        soft_box = F.interpolate(box_prior.float(), size=hp,
                                 mode='bilinear', align_corners=False).clamp(0, 1)
        gate_features = band_energy + [soft_box]

        if self.variant == 3:
            # Wavelet transform is applied to continuous features FIRST.
            # This avoids spurious edges from zeroing pixels outside the box.
            # Ring is derived around the *predicted* soft box map.
            ring = (F.max_pool2d(soft_box, kernel_size=7, stride=1, padding=3)
                    - soft_box).clamp_min(0.)
            reference_energy = [band.abs().mean(dim=1, keepdim=True) for band in hm]
            contrast = [_energy_contrast(e, soft_box, ring).expand_as(e)
                        for e in reference_energy]
            gate_features.extend(contrast)

        band_gates = self.frequency_gate(torch.cat(gate_features, dim=1))
        # Gate affects only the predicted box support; not arbitrary background HF.
        band_gates = band_gates * soft_box
        diff_gated = [d * band_gates[:, i:i+1] for i, d in enumerate(difference)]
        high_only = self.idwt(torch.zeros_like(llm), *diff_gated,
                              output_size=m.shape[-2:])
        result = result + self.gamma_frequency * high_only.to(out_dtype)
        return result
