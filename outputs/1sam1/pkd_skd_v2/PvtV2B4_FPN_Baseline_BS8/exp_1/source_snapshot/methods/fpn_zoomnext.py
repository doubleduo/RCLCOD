# -*- coding: utf-8 -*-
"""PVTv2-B4 FPN + ZoomNeXt MHSIU2 / RGPU.

Optional Clean-only 1.5x training branch, on top of the original
RCLCOD fpn_zoomnext.py (commit d8f96615a1be05ab44d3792b5c32d981e65b8e9d).

With clean_only_highres=True:
  - Train / pool_id == 0 (Clean): image_m 1.0x + image_l 1.5x + MHSIU2.
  - Train / pool_id == 1 (Noisy): image_m 1.0x only, bypass MHSIU2.
  - Eval / inference: all samples use 1.0x + 1.5x + MHSIU2.
  - Both pools use the same FPN, optional RGPU, prediction head and BCE.

The original behavior is preserved when clean_only_highres=False (default).
No trainer or dataset changes are needed when using RCLCOD's existing
basemain_continuous.py and its ImageTrainDataset (which supplies pool_id).
"""

import torch
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline
from .zoomnext.layers import MHSIU2, RGPU


class PvtV2B4_FPN_ZoomNeXt(PvtV2B4_FPN_Baseline):
    """Dual-scale MHSIU2 + FPN top-down + RGPU, optionally quality-routed."""

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        num_frames=1,
        siu_groups=4,
        hmu_groups=6,
        use_mhsiu=True,
        use_rgpu=True,
        clean_only_highres=False,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.use_mhsiu = bool(use_mhsiu)
        self.use_rgpu = bool(use_rgpu)
        self.clean_only_highres = bool(clean_only_highres)

        if self.clean_only_highres and not self.use_mhsiu:
            raise ValueError("clean_only_highres=True requires use_mhsiu=True")

        if self.use_mhsiu:
            if siu_groups < 1 or fpn_dim % siu_groups != 0:
                raise ValueError(
                    f"MHSIU2 requires fpn_dim ({fpn_dim}) divisible by "
                    f"siu_groups ({siu_groups})"
                )
            self.siu_2 = MHSIU2(fpn_dim, siu_groups)
            self.siu_3 = MHSIU2(fpn_dim, siu_groups)
            self.siu_4 = MHSIU2(fpn_dim, siu_groups)
            self.siu_5 = MHSIU2(fpn_dim, siu_groups)

        if self.use_rgpu:
            if hmu_groups < 2:
                raise ValueError("RGPU requires hmu_groups >= 2")
            self.hmu_2 = RGPU(fpn_dim, hmu_groups, num_frames=num_frames)
            self.hmu_3 = RGPU(fpn_dim, hmu_groups, num_frames=num_frames)
            self.hmu_4 = RGPU(fpn_dim, hmu_groups, num_frames=num_frames)
            self.hmu_5 = RGPU(fpn_dim, hmu_groups, num_frames=num_frames)

    def _large_image(self, data, image_m):
        """Reuse dataset's genuine 1.5x image, with original fallback."""
        image_l = data.get("image_l")
        if image_l is None:
            image_l = F.interpolate(
                image_m, scale_factor=1.5, mode="bilinear", align_corners=False
            )
        return image_l

    def _fuse_large_subset(self, z_feats, image_l, indices):
        """Run the 1.5x encoder and MHSIU ONLY on Clean rows.

        index_copy restores the fused rows to their original mixed-batch
        positions, and remains differentiable for both feature branches.
        Every Noisy row remains identical to the original 1.0x lateral feature.
        """
        large = image_l.index_select(0, indices)
        l_feats = self.normalize_encoder(large)
        sius = (self.siu_2, self.siu_3, self.siu_4, self.siu_5)
        lateral = (self.lateral_2, self.lateral_3, self.lateral_4, self.lateral_5)

        result = []
        for z, lf, siu, lateral_layer in zip(z_feats, l_feats, sius, lateral):
            zm = z.index_select(0, indices)
            fused = siu(l=lateral_layer(lf), m=zm)
            # AMP 下保持回填特征 dtype 一致
            fused = fused.to(dtype=z.dtype)
            result.append(z.index_copy(0, indices, fused))
        return tuple(result)

    def body(self, data):
        image_m = data["image_m"]
        # Original full-batch 1.0x backbone and FPN lateral projections.
        m2, m3, m4, m5 = self.normalize_encoder(image_m)
        z2 = self.lateral_2(m2)
        z3 = self.lateral_3(m3)
        z4 = self.lateral_4(m4)
        z5 = self.lateral_5(m5)

        if self.use_mhsiu:
            if self.training and self.clean_only_highres:
                # The provided trainer's dataset already supplies pool_id:
                # Clean=0, Noisy=1. Do not route by data["pool_id"][0].
                pool_id = data.get("pool_id")
                if pool_id is None:
                    raise KeyError(
                        "clean_only_highres=True requires data['pool_id'] "
                        "during training (0=Clean, 1=Noisy)."
                    )
                pool_id = pool_id.reshape(-1).to(device=image_m.device)
                if pool_id.numel() != image_m.shape[0]:
                    raise ValueError("pool_id must have one value per batch sample")
                if not torch.all((pool_id == 0) | (pool_id == 1)):
                    raise ValueError("Training pool_id must contain only 0 (Clean) or 1 (Noisy)")

                clean_indices = torch.nonzero(pool_id == 0, as_tuple=True)[0]
                if clean_indices.numel() > 0:
                    image_l = self._large_image(data, image_m)
                    z2, z3, z4, z5 = self._fuse_large_subset(
                        (z2, z3, z4, z5), image_l, clean_indices
                    )
                # If batch is Noisy-only, bypass MHSIU completely.
            else:
                # Original behavior for normal models and all inference,
                # independent of pool_id (test loader has no pool_id).
                image_l = self._large_image(data, image_m)
                l2, l3, l4, l5 = self.normalize_encoder(image_l)
                z2 = self.siu_2(l=self.lateral_2(l2), m=z2)
                z3 = self.siu_3(l=self.lateral_3(l3), m=z3)
                z4 = self.siu_4(l=self.lateral_4(l4), m=z4)
                z5 = self.siu_5(l=self.lateral_5(l5), m=z5)

        # Original FPN top-down + optional RGPU: no changes for either pool.
        p5 = self.smooth_5(z5)
        if self.use_rgpu:
            p5 = self.hmu_5(p5)

        p4 = self.smooth_4(z4 + self._resize_like(p5, m4))
        if self.use_rgpu:
            p4 = self.hmu_4(p4)

        p3 = self.smooth_3(z3 + self._resize_like(p4, m3))
        if self.use_rgpu:
            p3 = self.hmu_3(p3)

        p2 = self.smooth_2(z2 + self._resize_like(p3, m2))
        if self.use_rgpu:
            p2 = self.hmu_2(p2)

        logits = self.predictor(p2)
        return F.interpolate(
            logits,
            size=image_m.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class PvtV2B4_FPN_ZoomNeXt_MHSIU(PvtV2B4_FPN_ZoomNeXt):
    """Ablation: FPN + MHSIU2, without RGPU; supports clean_only_highres."""

    def __init__(self, **kwargs):
        kwargs.pop("use_mhsiu", None)
        kwargs.pop("use_rgpu", None)
        super().__init__(use_mhsiu=True, use_rgpu=False, **kwargs)


class PvtV2B4_FPN_ZoomNeXt_RGPU(PvtV2B4_FPN_ZoomNeXt):
    """Ablation: single-scale FPN + RGPU, without MHSIU2."""

    def __init__(self, **kwargs):
        kwargs.pop("use_mhsiu", None)
        kwargs.pop("use_rgpu", None)
        super().__init__(use_mhsiu=False, use_rgpu=True, **kwargs)


__all__ = [
    "PvtV2B4_FPN_ZoomNeXt",
    "PvtV2B4_FPN_ZoomNeXt_MHSIU",
    "PvtV2B4_FPN_ZoomNeXt_RGPU",
]
