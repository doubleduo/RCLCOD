"""SAM-COD paper-form objective; does not call a model's legacy loss forward."""
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from utils.skd_transforms import SKDViewTransform, replay_geometry


@contextmanager
def preserve_bn_buffers(model):
    # The detached view shares weights and train-mode behavior; it must not
    # silently update running stats a second time. Dropout behavior is retained.
    saved = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            saved.append((module, module.track_running_stats))
            module.track_running_stats = False
    try:
        yield
    finally:
        for module, old in saved:
            module.track_running_stats = old


def prompt_weighted_bce(logits, target, band=None, extra_weight=1.0):
    if logits.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {logits.shape}, {target.shape}")
    if not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError("PKD expects labels in [0,1], not 1/2/255 encoded masks")
    pixels = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    base = pixels.mean()
    if band is None:
        return base, base * 0.0, base.new_zeros(())
    if band.shape != logits.shape:
        raise ValueError("pkd_band shape must match the prediction")
    region = (band > 0.5).float()
    extra = (pixels * region).mean() * extra_weight
    return base, extra, region.mean()


class SAMCODObjective:
    def __init__(self, cfg, augment_cfg):
        self.pkd = dict(cfg.get("pkd", {}))
        self.skd = dict(cfg.get("skd", {}))
        self.extra_weight = float(self.pkd.get("extra_weight", 1.0))
        self.skd_weight = float(self.skd.get("weight", 1.0))
        self.detach_view = self.skd.get("detach_view", "augmented")
        if self.detach_view not in ("original", "augmented"):
            raise ValueError("detach_view must be original or augmented")
        if self.extra_weight < 0 or self.skd_weight < 0:
            raise ValueError("Loss weights must be nonnegative")
        self.ignore_padding = bool(self.skd.get("ignore_padding", True))
        self.transforms = SKDViewTransform(augment_cfg)

    @staticmethod
    def augmented_data(data, image):
        # Rebuild every internal scale from the SAME augmented image.
        result = {"image_m": image}
        for key, factor in (("image_s", 0.5), ("image_l", 1.5)):
            result[key] = F.interpolate(image, scale_factor=factor, mode="bilinear", align_corners=False)
        if "pool_id" in data:
            result["pool_id"] = data["pool_id"]
        return result

    def __call__(self, model, data, iter_percentage=0.0):
        del iter_percentage  # No undocumented ramp in the paper-form default.
        if not hasattr(model, "body"):
            raise TypeError("samcod_paper engine needs model.body(data) -> logits")
        logits = model.body(data)
        target = data["mask"].float()
        band = None
        if self.pkd.get("enable", True):
            if "pkd_band" not in data:
                raise KeyError("Box PKD requires data['pkd_band']; enable train.data.box_prompt")
            band = data["pkd_band"]
        bce, extra, band_ratio = prompt_weighted_bce(logits, target, band, self.extra_weight)
        pkd = bce + extra
        skd = logits.sum() * 0.0
        stats = {"aug_s": 0.0, "aug_t": 0.0, "aug_f": 0.0, "aug_g": 0.0,
                 "aug_c": 0.0, "skd_valid_ratio": 0.0, "skd_scale": 1.0}
        if self.skd.get("enable", True) and self.skd_weight > 0:
            image, valid, record = self.transforms(data["image_m"])
            augmented = self.augmented_data(data, image)
            if self.detach_view == "augmented":
                with preserve_bn_buffers(model), torch.no_grad():
                    other_logits = model.body(augmented)
            else:
                other_logits = model.body(augmented)
            original_prob = replay_geometry(logits.float().sigmoid(), record)
            other_prob = F.interpolate(other_logits.float().sigmoid(),
                                       size=logits.shape[-2:], mode="bilinear",
                                       align_corners=self.transforms.align_corners)
            if self.detach_view == "augmented":
                other_prob = other_prob.detach()
            else:
                original_prob = original_prob.detach()
            if not self.ignore_padding:
                valid = torch.ones_like(valid)
            skd = ((original_prob - other_prob).abs() * valid).sum() / valid.sum().clamp_min(1)
            stats = {"aug_s": float(record.scale != 1), "aug_t": float(bool(record.dx or record.dy)),
                     "aug_f": float(record.flip != "none"), "aug_g": float(record.sigma > 0),
                     "aug_c": float(bool(record.crop)), "skd_valid_ratio": valid.mean().detach(),
                     "skd_scale": record.scale}
        total = pkd + self.skd_weight * skd
        items = dict(bce=bce.detach(), pkd_extra=extra.detach(), pkd=pkd.detach(),
                     skd=skd.detach(), skd_weighted=(self.skd_weight * skd).detach(),
                     pkd_band_ratio=band_ratio.detach(), total=total.detach(), **stats)
        return {"logits": logits, "loss": total, "loss_items": items,
                "vis": {"sal": logits.sigmoid()},
                "loss_str": f"L:{total.item():.4f} PKD:{pkd.item():.4f} SKD:{skd.item():.4f}"}
