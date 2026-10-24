# -*- coding: utf-8 -*-
"""ESCNet-inspired lightweight edge/texture modules on the EXISTING PVTv2-B4 FPN.

For RCLCOD (commit b37a6b5...):
  - Inherits the exact PVTv2-B4 encoder, FPN projections, smoothing layers,
    predictor, image normalization, and grouped optimizer parameters.
  - AETP-Lite: semantic P2/P3/P4 edge prediction.
  - DSFA-Lite (optional): edge-conditioned single-point feature sampling on P3
    plus 3x3/5x5 depthwise texture extraction. No torchvision/DCN extension.
  - MFMM-Lite: coarse-mask + learned-edge guided residual P2 modulation.

IMPORTANT: This is a lightweight adaptation of ESCNet ideas, NOT a literal
replication of the original AETP / DSFA / MFMM architectures.

Ablation using cfg.model (basemain.py forwards cfg.model to constructor):
  A1: use_aetp=False, use_dsfa=False,
      use_mfmm_mask=True, use_mfmm_edge=False
  A2: use_aetp=True, use_dsfa=False,
      use_mfmm_mask=True, use_mfmm_edge=True
  A3: A2 + aux_edge_weight=0.05 (only when pseudo-boundaries are trustworthy)
  A4: A2 + use_dsfa=True

Evaluation: model(data) -> Bx1xHxW logits.
Training: dict(logits, vis, loss, loss_items, loss_str), same as baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline


def _gn_groups(channels, max_groups=8):
    for groups in range(min(channels, max_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvGNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, groups=groups, bias=False),
            nn.GroupNorm(_gn_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


def _up(x, ref):
    return F.interpolate(x, size=ref.shape[-2:],
                         mode="bilinear", align_corners=False)


class AETPLite(nn.Module):
    """P2 spatial detail + P3/P4 semantics -> learned edge logits on P2.

    Unlike ESCNet AETP this uses cheap depthwise convolution instead of
    cross-scale multi-head attention and deformable convolutions.
    """
    def __init__(self, channels):
        super().__init__()
        self.fuse = _ConvGNAct(channels * 3, channels, kernel_size=1)
        self.local = _ConvGNAct(channels, channels, kernel_size=3,
                                groups=channels)
        self.edge_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, p2_pre, p3, p4):
        x = torch.cat([p2_pre, _up(p3, p2_pre), _up(p4, p2_pre)], dim=1)
        x = self.fuse(x)
        x = x + self.local(x)
        return self.edge_head(x)


class DSFALite(nn.Module):
    """Edge-conditioned P3 sampling + lightweight multi-kernel texture refine.

    A single sampling point per output pixel is used; this is NOT a DCNv2
    convolution (DCNv2 has a separate offset for each kernel position).
    Zero offsets at initialization make the sampling start near identity.
    """
    def __init__(self, channels, max_shift_pixels=1.5):
        super().__init__()
        if max_shift_pixels <= 0:
            raise ValueError("max_shift_pixels must be positive")
        self.max_shift_pixels = float(max_shift_pixels)
        hidden = max(16, channels // 2)
        self.offset_net = nn.Sequential(
            _ConvGNAct(channels + 1, hidden, kernel_size=3),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1),
        )
        # Start with an identity sampling grid.
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)
        self.dw3 = _ConvGNAct(channels, channels, 3, groups=channels)
        self.dw5 = _ConvGNAct(channels, channels, 5, groups=channels)
        self.mix = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(channels + 1, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _sample(x, displacement):
        """displacement is (dx, dy) in pixels at feature-map resolution."""
        b, c, h, w = x.shape
        # Grid-sample in fp32 for reliable AMP behavior on CPU/CUDA.
        dy = displacement[:, 1].float()
        dx = displacement[:, 0].float()
        yy = torch.arange(h, dtype=torch.float32, device=x.device)
        xx = torch.arange(w, dtype=torch.float32, device=x.device)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        # align_corners=True: map pixel centers to [-1, 1].
        x_grid = 2.0 * (xx.unsqueeze(0) + dx) / max(w - 1, 1) - 1.0
        y_grid = 2.0 * (yy.unsqueeze(0) + dy) / max(h - 1, 1) - 1.0
        grid = torch.stack([x_grid, y_grid], dim=-1)
        sampled = F.grid_sample(x.float(), grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        return sampled.to(dtype=x.dtype)

    def forward(self, p3, edge_logits_p2):
        e3 = _up(edge_logits_p2, p3).sigmoid()
        guidance = torch.cat([p3, e3], dim=1)
        offsets = self.max_shift_pixels * self.offset_net(guidance).tanh()
        sampled = self._sample(p3, offsets)
        texture = self.mix(torch.cat([self.dw3(sampled),
                                      self.dw5(sampled)], dim=1))
        weight = self.gate(guidance).sigmoid()
        return p3 + self.scale * weight * texture


class MFMMMaskEdgeLite(nn.Module):
    """Mask/edge-conditioned residual modulation of P2.

    Both priors are model predictions, never GT/box/pseudo mask inputs.
    Keeping P2 as a residual identity branch prevents hard mask cutoffs.
    """
    def __init__(self, channels, use_mask=True, use_edge=True):
        super().__init__()
        if not (use_mask or use_edge):
            raise ValueError("At least one MFMM guidance branch is required")
        self.use_mask = bool(use_mask)
        self.use_edge = bool(use_edge)
        n = int(use_mask) + int(use_edge)
        if use_mask:
            self.mask_gate = nn.Conv2d(channels + 1, channels, 1)
            self.mask_local = _ConvGNAct(channels, channels, 3,
                                         groups=channels)
        if use_edge:
            self.edge_gate = nn.Conv2d(channels + 1, channels, 1)
            self.edge_local = _ConvGNAct(channels, channels, 3,
                                         groups=channels)
        self.project = _ConvGNAct(channels * n, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, p2, coarse_logits=None, edge_logits=None):
        refinements = []
        if self.use_mask:
            if coarse_logits is None:
                raise ValueError("coarse_logits missing for mask guidance")
            m = _up(coarse_logits, p2).sigmoid()
            g = self.mask_gate(torch.cat([p2, m], dim=1)).sigmoid()
            refinements.append((0.5 + g) * self.mask_local(p2))
        if self.use_edge:
            if edge_logits is None:
                raise ValueError("edge_logits missing for edge guidance")
            e = _up(edge_logits, p2).sigmoid()
            g = self.edge_gate(torch.cat([p2, e], dim=1)).sigmoid()
            refinements.append((0.5 + g) * self.edge_local(p2))
        return p2 + self.scale * self.project(torch.cat(refinements, dim=1))


class PvtV2B4_FPN_ESC_Lite(PvtV2B4_FPN_Baseline):
    """Training-compatible FPN + ESCNet-inspired modules.

    All FPN/backbone parameters and final predictor are inherited unchanged.
    Output shape and train/eval contract are exactly baseline-compatible.
    """
    def __init__(self, pretrained=True, input_norm=True, fpn_dim=64,
                 use_checkpoint=False,
                 use_aetp=True, use_dsfa=False,
                 use_mfmm_mask=True, use_mfmm_edge=True,
                 aux_coarse_weight=0.2, aux_edge_weight=0.0,
                 dsfa_shift_px=1.5, **kwargs):
        super().__init__(pretrained=pretrained, input_norm=input_norm,
                         fpn_dim=fpn_dim, use_checkpoint=use_checkpoint,
                         **kwargs)
        self.use_aetp = bool(use_aetp)
        self.use_dsfa = bool(use_dsfa)
        self.use_mfmm_mask = bool(use_mfmm_mask)
        self.use_mfmm_edge = bool(use_mfmm_edge)
        self.aux_coarse_weight = float(aux_coarse_weight)
        self.aux_edge_weight = float(aux_edge_weight)
        if self.use_dsfa and not self.use_aetp:
            raise ValueError("DSFA-lite requires AETP-lite edge guidance")
        if self.use_mfmm_edge and not self.use_aetp:
            raise ValueError("MFMM edge guidance requires AETP-lite")
        if self.aux_edge_weight > 0 and not self.use_aetp:
            raise ValueError("aux_edge_weight > 0 requires use_aetp=True")
        if self.aux_coarse_weight > 0 and not self.use_mfmm_mask:
            raise ValueError("aux_coarse_weight > 0 requires use_mfmm_mask=True")
        if self.aux_edge_weight < 0 or self.aux_coarse_weight < 0:
            raise ValueError("auxiliary loss weights must be nonnegative")
        if self.use_aetp:
            self.aetp = AETPLite(fpn_dim)
        if self.use_dsfa:
            self.dsfa_p3 = DSFALite(fpn_dim,
                                     max_shift_pixels=dsfa_shift_px)
        if self.use_mfmm_mask:
            self.coarse_head = nn.Sequential(
                _ConvGNAct(fpn_dim, fpn_dim, kernel_size=3),
                nn.Conv2d(fpn_dim, 1, kernel_size=1),
            )
        if self.use_mfmm_mask or self.use_mfmm_edge:
            self.mfmm = MFMMMaskEdgeLite(fpn_dim,
                                         use_mask=self.use_mfmm_mask,
                                         use_edge=self.use_mfmm_edge)

    def body(self, data, return_aux=False):
        image = data["image_m"]
        c2, c3, c4, c5 = self.normalize_encoder(image)

        # ORIGINAL top-down FPN: only P3 -> P2 can be optionally adjusted.
        p5 = self.smooth_5(self.lateral_5(c5))
        p4 = self.smooth_4(self.lateral_4(c4) + self._resize_like(p5, c4))
        p3 = self.smooth_3(self.lateral_3(c3) + self._resize_like(p4, c3))
        l2 = self.lateral_2(c2)
        p2_pre = l2 + self._resize_like(p3, c2)

        edge_logits = None
        if self.use_aetp:
            edge_logits = self.aetp(p2_pre, p3, p4)  # 1/4 image

        if self.use_dsfa:
            p3 = self.dsfa_p3(p3, edge_logits)  # 1/8 image

        # Original P2 smooth is executed exactly once.
        p2 = self.smooth_2(l2 + self._resize_like(p3, c2))

        coarse_logits = None
        if self.use_mfmm_mask:
            coarse_logits = self.coarse_head(p3)

        if self.use_mfmm_mask or self.use_mfmm_edge:
            p2 = self.mfmm(p2, coarse_logits=coarse_logits,
                           edge_logits=edge_logits)

        logits = self.predictor(p2)
        logits = F.interpolate(logits, size=image.shape[-2:],
                               mode="bilinear", align_corners=False)
        if return_aux:
            return logits, coarse_logits, edge_logits
        return logits

    @staticmethod
    def _boundary_target(mask, spatial_size):
        """Morphological mask boundary at edge-head resolution (pseudo-label)."""
        target = F.interpolate(mask, size=spatial_size, mode="nearest")
        dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(self, data, iter_percentage=1, **kwargs):
        del iter_percentage, kwargs
        if not self.training:
            return self.body(data)

        logits, coarse_logits, edge_logits = self.body(data, return_aux=True)
        mask = data["mask"].float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != logits.shape[-2:]:
            mask = F.interpolate(mask, size=logits.shape[-2:], mode="nearest")

        bce = F.binary_cross_entropy_with_logits(logits, mask)
        coarse_loss = bce.new_zeros(())
        edge_loss = bce.new_zeros(())

        if self.aux_coarse_weight > 0:
            target_coarse = F.interpolate(mask,
                                         size=coarse_logits.shape[-2:],
                                         mode="nearest")
            coarse_loss = F.binary_cross_entropy_with_logits(
                coarse_logits, target_coarse)

        if self.aux_edge_weight > 0:
            # Pseudo-mask boundary may be wrong! Keep disabled initially.
            target_edge = self._boundary_target(mask, edge_logits.shape[-2:])
            num_pos = target_edge.sum().detach()
            num_neg = (1.0 - target_edge).sum().detach()
            pos_weight = (num_neg / num_pos.clamp_min(1.0)).clamp(1.0, 8.0)
            edge_loss = F.binary_cross_entropy_with_logits(
                edge_logits, target_edge, pos_weight=pos_weight)

        total = (bce + self.aux_coarse_weight * coarse_loss
                 + self.aux_edge_weight * edge_loss)
        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid()},
            "loss": total,
            "loss_items": {
                "bce": bce.detach(),
                "coarse": coarse_loss.detach(),
                "edge": edge_loss.detach(),
                "total": total.detach(),
            },
            "loss_str": (
                f"L:{total.detach().item():.4f} "
                f"BCE:{bce.detach().item():.4f} "
                f"C:{coarse_loss.detach().item():.4f} "
                f"E:{edge_loss.detach().item():.4f}"
            ),
        }
