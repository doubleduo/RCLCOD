"""Replayable SKD view transforms, on-device and independent of Dataset aug.

Defaults use the paper's best operator SET S/T/F/G, not a claimed recovery of
its unpublished exact sampling probabilities. one_of samples uniformly among
enabled T/F/G/C, then S resizes the result. compose applies each with its p.
Geometry is replayed on probabilities; blur is never replayed on targets.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ViewRecord:
    flip: str = "none"
    dx: int = 0
    dy: int = 0
    crop: tuple = ()  # top, left, height, width, in original-view coordinates
    sigma: float = 0.0
    scale: float = 1.0


def shift_zero(x, dx, dy):
    h, w = x.shape[-2:]
    if abs(dx) >= w or abs(dy) >= h:
        return x * 0.0
    sx0, sx1 = max(-dx, 0), min(w - dx, w)
    sy0, sy1 = max(-dy, 0), min(h - dy, h)
    cropped = x[..., sy0:sy1, sx0:sx1]
    return F.pad(cropped, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)))


def replay_geometry(x, record, mode="bilinear"):
    if record.flip == "horizontal":
        x = x.flip(-1)
    elif record.flip == "vertical":
        x = x.flip(-2)
    elif record.flip != "none":
        raise ValueError(f"Unknown flip: {record.flip}")
    x = shift_zero(x, record.dx, record.dy)
    if record.crop:
        size = x.shape[-2:]
        top, left, h, w = record.crop
        x = x[..., top:top + h, left:left + w]
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        x = F.interpolate(x, size=size, mode=mode, **kwargs)
    return x


def gaussian_blur(x, kernel_size, sigma):
    radius = kernel_size // 2
    # Float32 kernels also work under CPU test and CUDA autocast.
    pos = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    kernel = torch.exp(-pos.square() / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    kh = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1).to(x.dtype)
    kv = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1).to(x.dtype)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kh, groups=channels)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kv, groups=channels)


class SKDViewTransform:
    def __init__(self, cfg):
        self.policy = cfg.get("policy", "one_of")
        if self.policy not in ("one_of", "compose"):
            raise ValueError("SKD policy must be one_of or compose")
        self.scale = dict(cfg.get("scale", {}))
        self.ops = {name: dict(cfg.get(name, {})) for name in ("translate", "flip", "blur", "crop")}
        self.align_corners = bool(cfg.get("align_corners", False))
        factor = float(self.scale.get("factor", 0.5))
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("SKD scale.factor must be positive and finite")
        for name, op in self.ops.items():
            if not 0 <= float(op.get("p", 0.5)) <= 1:
                raise ValueError(f"{name}.p must be in [0, 1]")
        ratio = float(self.ops["translate"].get("ratio", 0.15))
        if not 0 <= ratio < 1:
            raise ValueError("translate.ratio must be in [0, 1)")
        kernel = int(self.ops["blur"].get("kernel_size", 5))
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("blur.kernel_size must be positive and odd")
        low, high = self.ops["blur"].get("sigma", (0.1, 2.0))
        if not 0 < low <= high:
            raise ValueError("blur.sigma must satisfy 0 < low <= high")
        if not 0 < float(self.ops["crop"].get("ratio", 0.7)) <= 1:
            raise ValueError("crop.ratio must be in (0, 1]")

    def sample(self, height, width):
        enabled = [n for n, op in self.ops.items() if op.get("enable", False)]
        if self.policy == "one_of":
            active = {enabled[int(torch.randint(len(enabled), ()).item())]} if enabled else set()
        else:
            active = {n for n in enabled if torch.rand(()).item() < self.ops[n].get("p", 0.5)}
        flip = "none"
        dx = dy = 0
        crop = ()
        sigma = 0.0
        if "flip" in active:
            axes = self.ops["flip"].get("axes", ["horizontal", "vertical"])
            if not axes or any(a not in ("horizontal", "vertical") for a in axes):
                raise ValueError("flip.axes must contain horizontal/vertical")
            flip = axes[int(torch.randint(len(axes), ()).item())]
        if "translate" in active:
            ratio = float(self.ops["translate"].get("ratio", 0.15))
            dx = int(width * ratio) * (1 if torch.rand(()).item() < 0.5 else -1)
            dy = int(height * ratio) * (1 if torch.rand(()).item() < 0.5 else -1)
        if "crop" in active:
            ratio = float(self.ops["crop"].get("ratio", 0.7))
            h, w = max(1, int(height * ratio)), max(1, int(width * ratio))
            crop = (int(torch.randint(height - h + 1, ()).item()),
                    int(torch.randint(width - w + 1, ()).item()), h, w)
        if "blur" in active:
            low, high = self.ops["blur"].get("sigma", (0.1, 2.0))
            sigma = float(low + (high - low) * torch.rand(()).item())
        scale = float(self.scale.get("factor", 0.5)) if self.scale.get("enable", True) else 1.0
        return ViewRecord(flip, dx, dy, crop, sigma, scale)

    def __call__(self, image, record=None):
        record = record or self.sample(*image.shape[-2:])
        out = replay_geometry(image, record)
        valid = replay_geometry(torch.ones_like(image[:, :1]), record, mode="nearest")
        if record.sigma > 0:
            out = gaussian_blur(out, int(self.ops["blur"].get("kernel_size", 5)), record.sigma)
        h, w = image.shape[-2:]
        size = (max(1, int(h * record.scale)), max(1, int(w * record.scale)))
        if size != (h, w):
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=self.align_corners)
        # Validity lives in the original-view canvas, where SKD is evaluated.
        return out, valid, record
