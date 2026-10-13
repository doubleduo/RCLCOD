# -*- coding: utf-8 -*-
"""Strict MHSIU2 pooling-only ablation for RCLCOD.

Parent: methods.fpn_zoomnext.PvtV2B4_FPN_ZoomNeXt_MHSIU

Everything in the parent is kept: dual-resolution PVTv2-B4, ALL FOUR
MHSIU levels (P2/P3/P4/P5), groupwise two-scale softmax, FPN, BCE,
optimizer parameter groups and the basemain_continuous forward API.

Only the 1.5x -> 1.0x downsampling in the inherited MHSIU2 is altered:
  M1:  2 * bilinear interpolation with antialias=True
  M2:  original_pool + tanh(alpha) * (AA_pool - original_pool)
       alpha=0 at initialization -> EXACT original MHSIU2 forward.

No Box, DWT, new PVT encoder, new loss, auxiliary head or RGPU.
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .fpn_zoomnext import PvtV2B4_FPN_ZoomNeXt_MHSIU
from .zoomnext.layers import MHSIU2


class MHSIU2_AA(MHSIU2):
    """Original MHSIU2; ONLY large-scale spatial alignment is modified.

    Compatible parameter names and tensors with original MHSIU2; in M2
    ``aa_mix`` is the only additional trainable parameter (per FPN level).
    """

    def __init__(self, in_dim: int, num_groups: int = 4, mode: str = "mix"):
        super().__init__(in_dim=in_dim, num_groups=num_groups)
        if mode not in ("replace", "mix"):
            raise ValueError("mode must be 'replace' (M1) or 'mix' (M2)")
        self.mode = mode
        if mode == "mix":
            # tanh(0)=0 and derivative at 0=1: exactly original at init,
            # but the AA path has a nonzero gradient on the first step.
            self.aa_mix = nn.Parameter(torch.zeros(()))

    @staticmethod
    def original_pool(l, tgt_size):
        return (
            F.adaptive_max_pool2d(l, tgt_size)
            + F.adaptive_avg_pool2d(l, tgt_size)
        )

    @staticmethod
    def antialias_pool(l, tgt_size):
        if l.shape[-2:] == tuple(tgt_size):
            # Preserve original max+avg (=2*l) for equal sizes.
            return 2.0 * l
        # One anti-aliased resampling step, unlike the prior BSCRF-A1
        # (which smoothed with Gaussian AND then used AA interpolate).
        # *2 keeps the response magnitude comparable with max+avg.
        return 2.0 * F.interpolate(
            l, size=tgt_size, mode="bilinear", align_corners=False,
            antialias=True,
        )

    def forward(self, l, m):
        tgt_size = m.shape[2:]
        l = self.conv_l_pre(l)

        # ===== The ONE ablated location in original MHSIU2 =====
        aa = self.antialias_pool(l, tgt_size)
        if self.mode == "replace":
            l = aa
        else:
            original = self.original_pool(l, tgt_size)
            alpha = torch.tanh(self.aa_mix)
            l = original + alpha * (aa - original)
        # ========================================================

        # Original MHSIU2 forward, otherwise unchanged:
        l = self.conv_l(l)
        m = self.conv_m(m)
        lm = torch.cat([l, m], dim=1)

        attn = self.conv_lm(lm)
        attn = rearrange(
            attn,
            "bt (nb ng d) h w -> (bt ng) (nb d) h w",
            nb=2, ng=self.num_groups,
        )
        attn = self.trans(attn)
        attn = attn.unsqueeze(2)

        x = self.initial_merge(lm)
        x = rearrange(
            x,
            "bt (nb ng d) h w -> (bt ng) nb d h w",
            nb=2, ng=self.num_groups,
        )
        x = (attn * x).sum(dim=1)
        x = rearrange(
            x,
            "(bt ng) d h w -> bt (ng d) h w",
            ng=self.num_groups,
        )
        return x


class _PvtV2B4_FPN_MHSIU_AA_Base(PvtV2B4_FPN_ZoomNeXt_MHSIU):
    """Reuses EXACT parent body()/forward()/loss() and replaces only SIUs."""

    def __init__(self, aa_mode="mix", **kwargs):
        # Superclass creates the original four MHSIU2 modules and disables RGPU.
        super().__init__(**kwargs)
        in_dim = self.lateral_2.out_channels
        num_groups = self.siu_2.num_groups
        if in_dim % num_groups:
            raise ValueError("FPN channels must be divisible by SIU groups")
        # Critical for a FAIR ablation:
        # 1) Preserve the exact initial weights of the original four SIUs.
        # 2) Construct replacements without changing global RNG state,
        #    so data-loader shuffling and subsequent RNG usage can match M0.
        with torch.random.fork_rng(devices=[]):
            for level in (2, 3, 4, 5):
                name = f"siu_{level}"
                original = getattr(self, name)
                replacement = MHSIU2_AA(
                    in_dim, num_groups=num_groups, mode=aa_mode
                )
                incompatible = replacement.load_state_dict(
                    original.state_dict(), strict=(aa_mode == "replace")
                )
                if aa_mode == "mix":
                    assert incompatible.missing_keys == ["aa_mix"]
                    assert not incompatible.unexpected_keys
                setattr(self, name, replacement)


class PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M1(_PvtV2B4_FPN_MHSIU_AA_Base):
    """4-level original MHSIU2 with direct antialias-pool replacement."""

    def __init__(self, **kwargs):
        super().__init__(aa_mode="replace", **kwargs)


class PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M2(_PvtV2B4_FPN_MHSIU_AA_Base):
    """4-level original MHSIU2 with learnable original/AA pool mixing."""

    def __init__(self, **kwargs):
        super().__init__(aa_mode="mix", **kwargs)


__all__ = [
    "MHSIU2_AA",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M1",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU_AA_M2",
]
