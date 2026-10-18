"""Noisy probability maps share the exact mask geometry and skip RGB jitter.

Adapted from ImageTrainDataset at RCLCOD b37a6b5. Clean calls the original
__getitem__ verbatim. Noisy uses that version's image/mask/box preprocessing.
"""
from pathlib import Path
import numpy as np
import torch
import cv2
import basemain as core

io, ops = core.io, core.ops
_read_labelme_box_mask = core._read_labelme_box_mask


def load_probability(path):
    """Read soft evidence; q is computed BEFORE resizing/augmentation."""
    p = np.load(path, allow_pickle=False).astype(np.float32)
    if p.ndim == 3 and p.shape[0] == 1:
        p = p[0]
    if p.ndim != 2 or not p.size:
        raise ValueError(f"Expected nonempty [H,W] probability in {path}, got {p.shape}")
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError(f"Expected finite probabilities in [0,1]: {path}")
    # x*log(x)=0 at endpoints, handled with a safe log argument.
    entropy = -(p*np.log(np.maximum(p, 1e-7)) + (1-p)*np.log(np.maximum(1-p, 1e-7))) / np.log(2.0)
    q = float(np.clip(1.0 - entropy.mean(), 0.0, 1.0))
    return p, q


class NoisyReliabilityDataset(core.ImageTrainDataset):
    def __init__(self, *args, probability_dir, reliability_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.probability_dir = Path(probability_dir)
        self.reliability_enabled = bool(reliability_enabled)
        self._pool_id()  # reject ambiguous pool names, never silently use all=Clean
        if self.pool_name == "noisy" and self.reliability_enabled:
            missing = [self.probability_dir / (Path(row[0]).stem + ".npy")
                       for row in self.total_data_paths
                       if not (self.probability_dir / (Path(row[0]).stem + ".npy")).is_file()]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} Noisy probability maps. First: {missing[0]}")
            # Treat as a MASK so brightness/HSV never change probability values.
            # Geometry uses the same mask interpolation as Y (nearest in this version).
            self.trains.add_targets({"sam_probability": "mask"})

    def _pool_id(self):
        if self.pool_name in ("clean", "clean_probe"):
            return 0
        if self.pool_name == "noisy":
            return 1
        raise ValueError(f"Expected Clean/Noisy pool, got {self.pool_name!r}")

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def __getitem__(self, index):
        if self.pool_name != "noisy" or not self.reliability_enabled:
            sample = super().__getitem__(index)
            values = sample["data"]
            values["pool_id"] = torch.tensor(self._pool_id(), dtype=torch.long)
            values["sam_q"] = torch.tensor(1.0, dtype=torch.float32)
            values["sam_probability"] = torch.ones_like(values["mask"], dtype=torch.float32)
            return sample
        image_path, mask_path, box_path = self.total_data_paths[index]
        image = io.read_color_array(image_path)
        mask = io.read_gray_array(mask_path, thr=0)
        probability, q = load_probability(self.probability_dir / (Path(image_path).stem + ".npy"))
        probability = cv2.resize(probability, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        box_mask = None
        if self.use_box:
            box_mask = _read_labelme_box_mask(
                box_path,
                height=image.shape[0],
                width=image.shape[1],
            )
        if image.shape[:2] != mask.shape:
            h, w = mask.shape
            image = ops.resize(image, height=h, width=w)
            if box_mask is not None:
                box_mask = cv2.resize(
                    box_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

        if self.augment:
            transform_inputs = {"image": image, "mask": mask, "sam_probability": probability}
            if box_mask is not None:
                transform_inputs["box_mask"] = box_mask
            transformed = self.trains(**transform_inputs)
            image = transformed["image"]
            mask = transformed["mask"]
            probability = transformed["sam_probability"]
            if box_mask is not None:
                box_mask = transformed["box_mask"]

        base_h = self.shape["h"]
        base_w = self.shape["w"]

        image = ops.resize(image, height=base_h, width=base_w)
        images = ops.ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        image_s = torch.from_numpy(images[0]).div(255).permute(2, 0, 1)
        image_m = torch.from_numpy(images[1]).div(255).permute(2, 0, 1)
        image_l = torch.from_numpy(images[2]).div(255).permute(2, 0, 1)

        mask = ops.resize(mask, height=base_h, width=base_w)
        mask = torch.from_numpy(mask).unsqueeze(0)

        sample_data = {
                "image_s": image_s,
                "image_m": image_m,
                "image_l": image_l,
                "mask": mask,
            }
        if box_mask is not None:
            box_mask = cv2.resize(
                box_mask,
                (base_w, base_h),
                interpolation=cv2.INTER_NEAREST,
            )
            sample_data["box_mask"] = torch.from_numpy(
                (box_mask > 0).astype(np.float32)
            ).unsqueeze(0)

        probability = cv2.resize(probability, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
        sample_data["sam_probability"] = torch.from_numpy(np.ascontiguousarray(probability)).unsqueeze(0)
        sample_data["sam_q"] = torch.tensor(q, dtype=torch.float32)
        sample_data["pool_id"] = torch.tensor(1, dtype=torch.long)
        return dict(data=sample_data)
