import argparse
import copy
import csv
import datetime
import inspect
import json
import logging
import os
import shutil
import time
from collections import defaultdict

import albumentations as A
import colorlog
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from mmengine import Config
from torch.utils import data
from tqdm import tqdm

import methods as model_zoo
from utils import io, ops, pipeline, pt_utils, py_utils, recorder

LOGGER = logging.getLogger("main")
LOGGER.propagate = False
LOGGER.setLevel(level=logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(colorlog.ColoredFormatter("%(log_color)s[%(filename)s] %(reset)s%(message)s"))
LOGGER.addHandler(stream_handler)


def _read_labelme_box_mask(path, height, width):
    """Read all LabelMe rectangles/polygons as one filled binary box mask."""
    with open(path, mode="r", encoding="utf-8") as file:
        payload = json.load(file)

    box_mask = np.zeros((height, width), dtype=np.float32)
    for shape in payload.get("shapes", []):
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.size == 0:
            continue
        points = points.reshape(-1, 2)
        x1 = int(np.floor(points[:, 0].min()))
        y1 = int(np.floor(points[:, 1].min()))
        x2 = int(np.ceil(points[:, 0].max()))
        y2 = int(np.ceil(points[:, 1].max()))

        x1 = int(np.clip(x1, 0, width))
        y1 = int(np.clip(y1, 0, height))
        x2 = int(np.clip(x2, 0, width))
        y2 = int(np.clip(y2, 0, height))
        if x2 > x1 and y2 > y1:
            box_mask[y1:y2, x1:x2] = 1.0

    if not np.any(box_mask):
        raise ValueError(f"No valid box found in {path}")
    return box_mask



class ImageTestDataset(data.Dataset):
    def __init__(self, dataset_info: dict, shape: dict):
        super().__init__()
        self.shape = shape

        image_path = os.path.join(dataset_info["root"], dataset_info["image"]["path"])
        image_suffix = dataset_info["image"]["suffix"]
        mask_path = os.path.join(dataset_info["root"], dataset_info["mask"]["path"])
        mask_suffix = dataset_info["mask"]["suffix"]

        image_names = [p[: -len(image_suffix)] for p in sorted(os.listdir(image_path)) if p.endswith(image_suffix)]
        mask_names = [p[: -len(mask_suffix)] for p in sorted(os.listdir(mask_path)) if p.endswith(mask_suffix)]
        valid_names = sorted(set(image_names).intersection(mask_names))
        self.total_data_paths = [
            (os.path.join(image_path, n) + image_suffix, os.path.join(mask_path, n) + mask_suffix) for n in valid_names
        ]

    def __getitem__(self, index):
        image_path, mask_path = self.total_data_paths[index]
        image = io.read_color_array(image_path)

        base_h = self.shape["h"]
        base_w = self.shape["w"]

        image = ops.resize(image, height=base_h, width=base_w)
        images = ops.ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        image_s = torch.from_numpy(images[0]).div(255).permute(2, 0, 1)
        image_m = torch.from_numpy(images[1]).div(255).permute(2, 0, 1)
        image_l = torch.from_numpy(images[2]).div(255).permute(2, 0, 1)

        return dict(
            data={"image_s": image_s, "image_m": image_m, "image_l": image_l},
            info=dict(mask_path=mask_path, group_name="image"),
        )

    def __len__(self):
        return len(self.total_data_paths)


def _normalize_sample_name(name):
    """Normalize names from txt/csv lists and dataset file stems for robust matching."""
    name = str(name).strip()
    name = os.path.basename(name)
    stem, _ = os.path.splitext(name)
    return stem


def _resolve_existing_path(path, cfg=None):
    """Resolve absolute/relative paths used in curriculum config."""
    if path is None:
        return None
    path = str(path)
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.append(path)
        if cfg is not None and hasattr(cfg, "proj_root"):
            candidates.append(os.path.join(cfg.proj_root, path))
            candidates.append(os.path.join(cfg.proj_root, "lists", path))
            candidates.append(os.path.join(cfg.proj_root, "data", path))
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Cannot find curriculum list: {path}. Tried: {candidates}")


def _load_name_set(list_path, cfg=None):
    """Load image stems from a txt/csv file. Txt is preferred for training lists."""
    if not list_path:
        return None
    list_path = _resolve_existing_path(list_path, cfg=cfg)
    names = set()
    if list_path.lower().endswith(".csv"):
        import csv
        with open(list_path, mode="r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            name_col = None
            for cand in ["image", "image_name", "filename", "name", "file", "img_name"]:
                if cand in fieldnames:
                    name_col = cand
                    break
            if name_col is None:
                name_col = fieldnames[0]
            for row in reader:
                names.add(_normalize_sample_name(row[name_col]))
    else:
        with open(list_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Support lines like "name" or "name label"; use first token.
                names.add(_normalize_sample_name(line.split()[0].split(",")[0]))
    LOGGER.info(f"Loaded curriculum list: {list_path}, samples={len(names)}")
    return names


class ImageTrainDataset(data.Dataset):
    def __init__(
        self,
        dataset_infos: dict,
        shape: dict,
        allowed_names=None,
        pool_name="all",
        augment=True,
        use_box=False,
    ):
        super().__init__()
        self.shape = shape
        self.pool_name = pool_name
        self.allowed_names = allowed_names
        self.augment = bool(augment)
        self.use_box = bool(use_box)

        self.total_data_paths = []
        for dataset_name, dataset_info in dataset_infos.items():
            image_path = os.path.join(dataset_info["root"], dataset_info["image"]["path"])
            image_suffix = dataset_info["image"]["suffix"]
            mask_path = os.path.join(dataset_info["root"], dataset_info["mask"]["path"])
            mask_suffix = dataset_info["mask"]["suffix"]

            box_path = None
            box_suffix = None
            if self.use_box:
                if "box_json" not in dataset_info:
                    raise KeyError(
                        f"Dataset '{dataset_name}' requires box_json when "
                        "train.data.use_box=True."
                    )
                box_path = os.path.join(
                    dataset_info["root"],
                    dataset_info["box_json"]["path"],
                )
                box_suffix = dataset_info["box_json"]["suffix"]

            image_names = [p[: -len(image_suffix)] for p in sorted(os.listdir(image_path)) if p.endswith(image_suffix)]
            mask_names = [p[: -len(mask_suffix)] for p in sorted(os.listdir(mask_path)) if p.endswith(mask_suffix)]
            valid_names = sorted(set(image_names).intersection(mask_names))
            if self.use_box:
                box_names = [
                    p[: -len(box_suffix)]
                    for p in sorted(os.listdir(box_path))
                    if p.endswith(box_suffix)
                ]
                missing_boxes = sorted(set(valid_names).difference(box_names))
                if missing_boxes:
                    preview = ", ".join(missing_boxes[:5])
                    raise FileNotFoundError(
                        f"Dataset '{dataset_name}' is missing box JSON for "
                        f"{len(missing_boxes)} image/mask pairs. First: {preview}"
                    )
            if self.allowed_names is not None:
                valid_names = [n for n in valid_names if _normalize_sample_name(n) in self.allowed_names]
            data_paths = [
                (
                    os.path.join(image_path, n) + image_suffix,
                    os.path.join(mask_path, n) + mask_suffix,
                    (
                        os.path.join(box_path, n) + box_suffix
                        if self.use_box
                        else None
                    ),
                )
                for n in valid_names
            ]
            LOGGER.info(f"Length of {dataset_name} [{self.pool_name}]: {len(data_paths)}")
            self.total_data_paths.extend(data_paths)

        transforms = [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            ]
        self.trains = A.Compose(
            transforms,
            additional_targets=(
                {"box_mask": "mask"}
                if self.use_box
                else {}
            ),
        )

    def __getitem__(self, index):
        image_path, mask_path, box_path = self.total_data_paths[index]
        image = io.read_color_array(image_path)
        mask = io.read_gray_array(mask_path, thr=0)
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
            transform_inputs = {"image": image, "mask": mask}
            if box_mask is not None:
                transform_inputs["box_mask"] = box_mask
            transformed = self.trains(**transform_inputs)
            image = transformed["image"]
            mask = transformed["mask"]
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

        return dict(data=sample_data)

    def __len__(self):
        return len(self.total_data_paths)


class Evaluator:
    def __init__(self, device, metric_names, clip_range=None):
        self.device = device
        self.clip_range = clip_range
        self.metric_names = metric_names

    @torch.no_grad()
    def eval(self, model, data_loader, save_path=""):
        model.eval()
        all_metrics = recorder.GroupedMetricRecorder(metric_names=self.metric_names)

        for batch in tqdm(data_loader, total=len(data_loader), ncols=79, desc="[EVAL]"):
            batch_images = pt_utils.to_device(batch["data"], device=self.device)
            logits = model(data=batch_images)  # B,1,H,W
            probs = logits.sigmoid()
            prob_min = probs.amin(dim=(2, 3), keepdim=True)
            prob_max = probs.amax(dim=(2, 3), keepdim=True)
            probs = (probs - prob_min) / (prob_max - prob_min + 1e-8)
            probs = probs.squeeze(1).cpu().detach().numpy()

            mask_paths = batch["info"]["mask_path"]
            group_names = batch["info"]["group_name"]
            for pred_idx, pred in enumerate(probs):
                mask_path = mask_paths[pred_idx]
                mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_array[mask_array > 0] = 255
                mask_h, mask_w = mask_array.shape
                pred = ops.resize(pred, height=mask_h, width=mask_w)

                if self.clip_range is not None:
                    pred = ops.clip_to_normalize(pred, clip_range=self.clip_range)

                group_name = group_names[pred_idx]
                if save_path:  # 这里的save_path包含了数据集名字
                    ops.save_array_as_image(
                        data_array=pred,
                        save_name=os.path.basename(mask_path),
                        save_dir=os.path.join(save_path, group_name),
                    )

                pred = (pred * 255).astype(np.uint8)
                all_metrics.step(group_name=group_name, pre=pred, gt=mask_array, gt_path=mask_path)
        return all_metrics.show()


def _append_cod_metrics(csv_path, epoch, metrics):
    """Append COD metrics to a dataset-specific CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["epoch", "S", "Fw", "E", "MAE"])

        writer.writerow(
            [
                "" if epoch is None else int(epoch),
                f"{metrics['S']:.4f}",
                f"{metrics['Fw']:.4f}",
                f"{metrics['E']:.4f}",
                f"{metrics['MAE']:.4f}",
            ]
        )



def test(model, cfg, epoch=None):
    """Evaluate each COD dataset and save metrics separately."""
    test_wrapper = Evaluator(
        device=cfg.device,
        metric_names=cfg.metric_names,
        clip_range=cfg.test.clip_range,
    )

    dataset_csv_names = {
        "chameleon": "CHAMELEON",
        "camo_te": "CAMO",
        "cod10k_te": "COD10K",
        "nc4k": "NC4K",
    }

    LOGGER.info(
        f"{'dataset':<12} | {'S':>6} | {'Fw':>6} | {'E':>6} | {'MAE':>6}"
    )

    for te_name in cfg.test.data.names:
        te_info = cfg.dataset_infos[te_name]

        te_dataset = ImageTestDataset(
            dataset_info=te_info,
            shape=cfg.test.data.shape,
        )

        te_loader = data.DataLoader(
            dataset=te_dataset,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            pin_memory=True,
        )

        if cfg.save_results:
            save_path = os.path.join(cfg.path.save, te_name)
        else:
            save_path = ""

        seg_results = test_wrapper.eval(
            model=model,
            data_loader=te_loader,
            save_path=save_path,
        )

        required_keys = ("sm", "wfm", "maxem", "mae")
        missing_keys = [
            key for key in required_keys
            if key not in seg_results
        ]

        if missing_keys:
            raise KeyError(
                f"Missing COD metrics {missing_keys}. "
                f"Available metrics: {list(seg_results.keys())}. "
                "Please ensure metric_names contains sm, wfm, mae and em."
            )

        cod_metrics = {
            "S": float(seg_results["sm"]),
            "Fw": float(seg_results["wfm"]),
            "E": float(seg_results["maxem"]),
            "MAE": float(seg_results["mae"]),
        }

        LOGGER.info(
            f"{te_name:<12} | "
            f"{cod_metrics['S']:>6.3f} | "
            f"{cod_metrics['Fw']:>6.3f} | "
            f"{cod_metrics['E']:>6.3f} | "
            f"{cod_metrics['MAE']:>6.3f}"
        )

        # 训练过程中验证时，每个数据集分别保存一个 CSV。
        # 单独执行 evaluate 时 epoch=None，只打印指标，不重复写入。
        if epoch is not None:
            csv_dataset_name = dataset_csv_names.get(
                te_name,
                te_name.upper(),
            )

            metrics_csv_path = os.path.join(
                cfg.path.pth_log,
                f"{csv_dataset_name}_metrics.csv",
            )

            _append_cod_metrics(
                csv_path=metrics_csv_path,
                epoch=epoch,
                metrics=cod_metrics,
            )


LOSS_CSV_COLUMNS = [
    "epoch",
    "stage",
    "total",
    "bce",
]


def _loss_items_to_floats(loss_items):
    """Detach structured loss values and convert them to plain floats."""
    values = {}
    for name, value in (loss_items or {}).items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(
                    f"Loss item '{name}' must be scalar, got shape={tuple(value.shape)}"
                )
            values[name] = float(value.detach().item())
        else:
            values[name] = float(value)
    return values


def _append_epoch_losses(csv_path, epoch, stage_name, loss_sums, num_batches):
    """Append one epoch-level average loss row to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    denom = max(int(num_batches), 1)

    row = {
        "epoch": int(epoch),
        "stage": str(stage_name),
    }
    for name in LOSS_CSV_COLUMNS[2:]:
        row[name] = f"{loss_sums.get(name, 0.0) / denom:.6f}"

    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOSS_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return {
        name: loss_sums.get(name, 0.0) / denom
        for name in LOSS_CSV_COLUMNS[2:]
    }


def _cfg_get(mapping, key, default=None):
    if mapping is None:
        return default
    return mapping.get(key, default) if hasattr(mapping, "get") else getattr(mapping, key, default)



def _build_train_subset(
    cfg,
    list_path=None,
    pool_name="all",
    augment=True,
    max_samples=None,
):
    allowed_names = _load_name_set(list_path, cfg=cfg) if list_path else None
    dataset = ImageTrainDataset(
        dataset_infos={
            data_name: cfg.dataset_infos[data_name]
            for data_name in cfg.train.data.names
        },
        shape=cfg.train.data.shape,
        allowed_names=allowed_names,
        pool_name=pool_name,
        augment=augment,
        use_box=bool(cfg.train.data.get("use_box", False)),
    )
    if max_samples is not None and len(dataset) > int(max_samples):
        dataset.total_data_paths = dataset.total_data_paths[: int(max_samples)]
        LOGGER.info(
            f"Truncated curriculum pool '{pool_name}' to "
            f"{len(dataset.total_data_paths)} samples."
        )
    if len(dataset) == 0:
        LOGGER.warning(
            f"Curriculum pool '{pool_name}' is empty. "
            f"Please check list_path={list_path}"
        )
    return dataset


def _build_standard_loader(cfg):
    tr_dataset = _build_train_subset(
        cfg,
        list_path=None,
        pool_name="all",
        augment=True,
    )
    LOGGER.info(f"Total Length of Image Trainset: {len(tr_dataset)}")
    tr_loader = data.DataLoader(
        dataset=tr_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=(
            pt_utils.customized_worker_init_fn
            if cfg.use_custom_worker_init
            else None
        ),
    )
    return tr_loader, {"all": tr_dataset}, None, None


def _build_curriculum_loader(
    cfg,
    pool_datasets,
    stage,
    num_samples_per_epoch,
):
    """Build a loader from pool-level probabilities.

    Stage weights are interpreted as desired pool masses, not per-sample
    weights. Dividing by pool size ensures Clean replay is not diluted merely
    because another pool contains more files.
    """
    weights_cfg = _cfg_get(stage, "weights", {})
    active_datasets = []
    sample_weights = []
    active_desc = []

    for pool_name, dataset in pool_datasets.items():
        pool_mass = float(_cfg_get(weights_cfg, pool_name, 0.0))
        if dataset is None or len(dataset) == 0 or pool_mass <= 0:
            continue

        per_sample_weight = pool_mass / float(len(dataset))
        active_datasets.append(dataset)
        sample_weights.extend([per_sample_weight] * len(dataset))
        active_desc.append(
            f"{pool_name}:N={len(dataset)},pool_p={pool_mass:.3f}"
        )

    if not active_datasets:
        raise RuntimeError(
            "No active curriculum dataset in stage: "
            f"{_cfg_get(stage, 'name', 'unknown')}"
        )

    train_dataset = (
        active_datasets[0]
        if len(active_datasets) == 1
        else data.ConcatDataset(active_datasets)
    )

    sampler = data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(num_samples_per_epoch),
        replacement=True,
    )

    LOGGER.info(
        f"Curriculum stage [{_cfg_get(stage, 'name', 'unnamed')}]: "
        + "; ".join(active_desc)
        + f"; sampled_per_epoch={num_samples_per_epoch}"
    )

    return data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=(
            pt_utils.customized_worker_init_fn
            if cfg.use_custom_worker_init
            else None
        ),
    )


def _build_clean_probe_loader(cfg, clean_list_path):
    probe_cfg = _cfg_get(cfg.train, "clean_probe", {})
    max_samples = int(_cfg_get(probe_cfg, "max_samples", 512))
    probe_dataset = _build_train_subset(
        cfg=cfg,
        list_path=clean_list_path,
        pool_name="clean_probe",
        augment=False,
        max_samples=max_samples,
    )
    if len(probe_dataset) == 0:
        raise RuntimeError("Clean probe set is empty.")

    return data.DataLoader(
        dataset=probe_dataset,
        batch_size=int(_cfg_get(probe_cfg, "batch_size", 8)),
        num_workers=int(_cfg_get(probe_cfg, "num_workers", 2)),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=(
            pt_utils.customized_worker_init_fn
            if cfg.use_custom_worker_init
            else None
        ),
    )


def _prepare_curriculum(cfg):
    curriculum = cfg.train.get("curriculum", None)
    if not curriculum or not _cfg_get(curriculum, "enable", False):
        return _build_standard_loader(cfg)

    stages = list(_cfg_get(curriculum, "stages", []))
    if len(stages) != 2:
        raise ValueError(
            "B2 saturation curriculum requires exactly two stages: "
            "stage1_clean and stage2_mixed."
        )

    pool_cfg = _cfg_get(curriculum, "pools", {})
    if "clean" not in pool_cfg:
        raise KeyError("curriculum.pools must contain a clean list.")

    pool_datasets = {
        pool_name: _build_train_subset(
            cfg=cfg,
            list_path=list_path,
            pool_name=pool_name,
            augment=True,
        )
        for pool_name, list_path in pool_cfg.items()
    }

    union_len = sum(len(ds) for ds in pool_datasets.values())
    if union_len == 0:
        raise RuntimeError("All curriculum pools are empty.")

    num_samples_per_epoch = int(
        _cfg_get(curriculum, "num_samples_per_epoch", union_len)
    )
    pool_msg = ", ".join(
        f"{name}={len(ds)}" for name, ds in pool_datasets.items()
    )
    LOGGER.info(
        f"Dynamic curriculum enabled: {pool_msg}, "
        f"num_samples_per_epoch={num_samples_per_epoch}"
    )

    first_stage = stages[0]
    tr_loader = _build_curriculum_loader(
        cfg=cfg,
        pool_datasets=pool_datasets,
        stage=first_stage,
        num_samples_per_epoch=num_samples_per_epoch,
    )

    clean_probe_loader = _build_clean_probe_loader(
        cfg=cfg,
        clean_list_path=pool_cfg["clean"],
    )

    curriculum_state = dict(
        stages=stages,
        pool_datasets=pool_datasets,
        num_samples_per_epoch=num_samples_per_epoch,
        stage_idx=0,
        stage_start_epoch=1,
        last_stage_name=_cfg_get(first_stage, "name", "stage1_clean"),
        rebuild_loader=False,
        loss_history=[],
        saturation_count=0,
        switched_epoch=None,
        kd_enabled=False,
    )
    return tr_loader, pool_datasets, curriculum_state, clean_probe_loader


@torch.no_grad()
def _update_ema_teacher(student, teacher, decay):
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher.named_parameters():
        teacher_param.mul_(decay).add_(
            student_params[name].detach(),
            alpha=1.0 - decay,
        )

    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        source = student_buffers[name].detach()
        if teacher_buffer.dtype.is_floating_point:
            teacher_buffer.mul_(decay).add_(
                source,
                alpha=1.0 - decay,
            )
        else:
            teacher_buffer.copy_(source)


def _high_confidence_ema_kd(
    student_logits,
    teacher_logits,
    temperature,
    confidence_threshold,
    confidence_gamma,
):
    """Bernoulli logit distillation on confident EMA pixels only."""
    with torch.no_grad():
        raw_teacher_prob = teacher_logits.sigmoid()
        teacher_target = torch.sigmoid(teacher_logits / temperature)

        confidence_prob = torch.maximum(
            raw_teacher_prob,
            1.0 - raw_teacher_prob,
        )
        valid = (
            confidence_prob >= confidence_threshold
        ).to(student_logits.dtype)
        confidence = (
            2.0 * confidence_prob - 1.0
        ).clamp_min(0.0).pow(confidence_gamma)
        weight = valid * confidence

    pixel_kd = F.binary_cross_entropy_with_logits(
        student_logits / temperature,
        teacher_target,
        reduction="none",
    )

    weight_sum = weight.sum()
    if weight_sum.detach().item() <= 0:
        return student_logits.new_zeros(())

    return (
        (pixel_kd * weight).sum()
        / weight_sum.clamp_min(1e-6)
        * (temperature ** 2)
    )


@torch.no_grad()
def _evaluate_student_ema_consistency(
    student,
    teacher,
    probe_loader,
    device,
):
    """Average per-image soft IoU between student and EMA predictions."""
    student_was_training = student.training
    student.eval()
    teacher.eval()

    score_sum = 0.0
    sample_count = 0

    for batch in probe_loader:
        batch_data = pt_utils.to_device(
            data=batch["data"],
            device=device,
        )
        student_prob = student(data=batch_data).sigmoid()
        teacher_prob = teacher(data=batch_data).sigmoid()

        intersection = (
            student_prob * teacher_prob
        ).flatten(1).sum(dim=1)
        union = (
            student_prob
            + teacher_prob
            - student_prob * teacher_prob
        ).flatten(1).sum(dim=1)

        soft_iou = (
            intersection + 1e-6
        ) / (
            union + 1e-6
        )
        score_sum += float(soft_iou.sum().item())
        sample_count += int(soft_iou.numel())

    if student_was_training:
        student.train()

    return score_sum / max(sample_count, 1)


def _relative_window_loss_drop(loss_history, window_size):
    if len(loss_history) < 2 * window_size:
        return None

    old_mean = float(np.mean(loss_history[-2 * window_size : -window_size]))
    new_mean = float(np.mean(loss_history[-window_size:]))
    return (old_mean - new_mean) / (abs(old_mean) + 1e-12)


def _saturation_step(
    curriculum_state,
    current_epoch,
    epoch_bce,
    ema_consistency,
    transition_cfg,
):
    """Update the Stage-1 saturation controller."""
    curriculum_state["loss_history"].append(float(epoch_bce))

    stage_age = (
        current_epoch
        - int(curriculum_state["stage_start_epoch"])
        + 1
    )
    window_size = int(_cfg_get(transition_cfg, "window_size", 5))
    loss_drop = _relative_window_loss_drop(
        curriculum_state["loss_history"],
        window_size,
    )

    min_stage_epochs = int(
        _cfg_get(transition_cfg, "min_stage_epochs", 20)
    )
    max_stage_epochs = int(
        _cfg_get(transition_cfg, "max_stage_epochs", 100)
    )
    loss_eps = float(_cfg_get(transition_cfg, "loss_eps", 0.005))
    max_regression = float(
        _cfg_get(transition_cfg, "max_loss_regression", 0.01)
    )
    consistency_threshold = float(
        _cfg_get(
            transition_cfg,
            "consistency_threshold",
            0.98,
        )
    )
    patience = int(_cfg_get(transition_cfg, "patience", 3))
    use_consistency = bool(
        _cfg_get(
            transition_cfg,
            "use_ema_consistency",
            True,
        )
    )

    if stage_age >= max_stage_epochs:
        return True, loss_drop, "max_stage_epochs"

    enough_history = loss_drop is not None
    loss_saturated = (
        enough_history
        and -max_regression <= loss_drop < loss_eps
    )
    prediction_stable = (
        (not use_consistency)
        or ema_consistency >= consistency_threshold
    )

    if (
        stage_age >= min_stage_epochs
        and loss_saturated
        and prediction_stable
    ):
        curriculum_state["saturation_count"] += 1
    else:
        curriculum_state["saturation_count"] = 0

    if curriculum_state["saturation_count"] >= patience:
        return True, loss_drop, "saturation"

    return False, loss_drop, ""


CURRICULUM_CSV_COLUMNS = [
    "epoch",
    "stage",
    "stage_age",
    "bce",
    "kd",
    "kd_coef",
    "loss_drop",
    "ema_consistency",
    "saturation_count",
    "switched",
    "switch_reason",
]


def _append_curriculum_metrics(
    csv_path,
    *,
    epoch,
    stage,
    stage_age,
    bce,
    kd,
    kd_coef,
    loss_drop,
    ema_consistency,
    saturation_count,
    switched,
    switch_reason,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    row = {
        "epoch": int(epoch),
        "stage": str(stage),
        "stage_age": int(stage_age),
        "bce": f"{float(bce):.6f}",
        "kd": f"{float(kd):.6f}",
        "kd_coef": f"{float(kd_coef):.6f}",
        "loss_drop": (
            "" if loss_drop is None else f"{float(loss_drop):.6f}"
        ),
        "ema_consistency": f"{float(ema_consistency):.6f}",
        "saturation_count": int(saturation_count),
        "switched": int(bool(switched)),
        "switch_reason": str(switch_reason),
    }

    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CURRICULUM_CSV_COLUMNS,
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _save_b2_state(
    save_path,
    *,
    epoch,
    ema_teacher,
    curriculum_state,
):
    state = {
        "epoch": int(epoch),
        "ema_teacher": ema_teacher.state_dict(),
        "curriculum": {
            "stage_idx": int(curriculum_state["stage_idx"]),
            "stage_start_epoch": int(
                curriculum_state["stage_start_epoch"]
            ),
            "loss_history": list(
                curriculum_state["loss_history"]
            ),
            "saturation_count": int(
                curriculum_state["saturation_count"]
            ),
            "switched_epoch": curriculum_state[
                "switched_epoch"
            ],
            "kd_enabled": bool(
                curriculum_state["kd_enabled"]
            ),
        },
    }
    torch.save(state, save_path)


def train(model, cfg):
    (
        tr_loader,
        pool_datasets,
        curriculum_state,
        clean_probe_loader,
    ) = _prepare_curriculum(cfg)

    # ------------------------------------------------------------------
    # Curriculum toggle
    # ------------------------------------------------------------------
    # curriculum.enable=True:
    #   original two-stage curriculum:
    #   Clean -> Mixed + saturation controller + EMA teacher/KD
    #
    # curriculum.enable=False:
    #   standard all-data training from epoch 1:
    #   - no clean/noisy pool sampling
    #   - no stage transition
    #   - no saturation detector
    #   - no clean probe
    #   - no EMA teacher
    #   - no EMA KD
    #
    # IMPORTANT:
    # This only disables the SAMPLE curriculum in this trainer.
    # Model-internal schedules (e.g. NCLoss q=2 -> q=1, UAL ramp)
    # are intentionally left unchanged for a fair curriculum ablation.
    curriculum_enabled = curriculum_state is not None

    if curriculum_enabled:
        LOGGER.info(
            "Curriculum mode: ENABLED "
            "(Clean -> Mixed, saturation switch, EMA KD)."
        )
    else:
        LOGGER.info(
            "Curriculum mode: DISABLED. "
            "Training on ALL samples from epoch 1; "
            "stage switching / clean probe / EMA KD are OFF."
        )

    counter = recorder.TrainingCounter(
        epoch_length=len(tr_loader),
        epoch_based=cfg.train.epoch_based,
        num_epochs=cfg.train.num_epochs,
        num_total_iters=cfg.train.num_iters,
    )
    optimizer = pipeline.construct_optimizer(
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        cfg=cfg.train.optimizer.cfg,
    )
    scheduler = pipeline.Scheduler(
        optimizer=optimizer,
        num_iters=counter.num_total_iters,
        epoch_length=counter.num_inner_iters,
        scheduler_cfg=cfg.train.scheduler,
        step_by_batch=cfg.train.sche_usebatch,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)
    scheduler.plot_lr_coef_curve(save_path=cfg.path.pth_log)
    scaler = pipeline.Scaler(
        optimizer,
        cfg.train.use_amp,
        set_to_none=cfg.train.optimizer.set_to_none,
    )

    LOGGER.info(f"Scheduler:\n{scheduler}\nOptimizer:\n{optimizer}")

    # EMA/KD hyperparameters are read for the curriculum-on path only.
    ema_cfg = cfg.train.get("ema_kd", {})
    ema_decay = float(_cfg_get(ema_cfg, "decay", 0.999))
    kd_lambda = float(_cfg_get(ema_cfg, "lambda_kd", 0.1))
    kd_temperature = float(
        _cfg_get(ema_cfg, "temperature", 2.0)
    )
    kd_conf_threshold = float(
        _cfg_get(ema_cfg, "confidence_threshold", 0.8)
    )
    kd_conf_gamma = float(
        _cfg_get(ema_cfg, "confidence_gamma", 1.0)
    )
    kd_warmup_epochs = max(
        int(_cfg_get(ema_cfg, "warmup_epochs", 5)),
        1,
    )

    # Do not even allocate an EMA copy when curriculum is disabled.
    ema_teacher = None
    if curriculum_enabled:
        ema_teacher = copy.deepcopy(model)
        ema_teacher.to(cfg.device)
        ema_teacher.eval()
        ema_teacher.requires_grad_(False)

    loss_recorder = recorder.HistoryBuffer()
    iter_time_recorder = recorder.HistoryBuffer()
    loss_csv_path = os.path.join(
        cfg.path.pth_log,
        "loss_metrics.csv",
    )
    curriculum_csv_path = os.path.join(
        cfg.path.pth_log,
        "curriculum_metrics.csv",
    )

    LOGGER.info(
        f"Image Mean: {model.normalizer.mean.flatten()}, "
        f"Image Std: {model.normalizer.std.flatten()}"
    )
    if cfg.train.bn.freeze_encoder:
        LOGGER.info(" >>> Freeze Backbone !!! <<< ")
        model.encoder.requires_grad_(False)

    train_start_time = time.perf_counter()

    for _ in range(counter.num_epochs):
        LOGGER.info(f"Exp_Name: {cfg.exp_name}")
        epoch_no = counter.curr_epoch + 1

        # --------------------------------------------------------------
        # Curriculum stage update / standard all-data mode
        # --------------------------------------------------------------
        if curriculum_enabled:
            if curriculum_state["rebuild_loader"]:
                current_stage = curriculum_state["stages"][
                    curriculum_state["stage_idx"]
                ]
                tr_loader = _build_curriculum_loader(
                    cfg=cfg,
                    pool_datasets=curriculum_state[
                        "pool_datasets"
                    ],
                    stage=current_stage,
                    num_samples_per_epoch=curriculum_state[
                        "num_samples_per_epoch"
                    ],
                )
                curriculum_state["last_stage_name"] = _cfg_get(
                    current_stage,
                    "name",
                    f"stage_{curriculum_state['stage_idx']}",
                )
                curriculum_state["rebuild_loader"] = False

            stage_idx = int(curriculum_state["stage_idx"])
            stage_name = curriculum_state["last_stage_name"]
            stage_age = (
                epoch_no
                - int(curriculum_state["stage_start_epoch"])
                + 1
            )

            if curriculum_state["kd_enabled"]:
                kd_coef = kd_lambda * min(
                    1.0,
                    stage_age / float(kd_warmup_epochs),
                )
            else:
                kd_coef = 0.0
        else:
            # Fixed all-data training: no curriculum stages.
            stage_idx = -1
            stage_name = "all_fixed"
            stage_age = epoch_no
            kd_coef = 0.0

        epoch_loss_sums = defaultdict(float)
        epoch_loss_count = 0

        model.train()
        if cfg.train.bn.freeze_status:
            pt_utils.frozen_bn_stats(
                model.encoder,
                freeze_affine=cfg.train.bn.freeze_affine,
            )

        for batch_idx, batch in enumerate(tr_loader):
            iter_start_time = time.perf_counter()
            scheduler.step(curr_idx=counter.curr_iter)

            data_batch = pt_utils.to_device(
                data=batch["data"],
                device=cfg.device,
            )

            with torch.cuda.amp.autocast(
                enabled=cfg.train.use_amp
            ):
                outputs = model(
                    data=data_batch,
                    iter_percentage=counter.curr_percent,
                )
                seg_loss = outputs["loss"]

                kd_loss = seg_loss.new_zeros(())

                # EMA-KD exists ONLY in curriculum stage 2.
                kd_active = (
                    curriculum_enabled
                    and curriculum_state["kd_enabled"]
                )
                if kd_active:
                    if "logits" not in outputs:
                        raise KeyError(
                            "The training model must return "
                            "outputs['logits'] for EMA KD."
                        )
                    with torch.no_grad():
                        teacher_logits = ema_teacher(
                            data=data_batch
                        )
                    kd_loss = _high_confidence_ema_kd(
                        student_logits=outputs["logits"],
                        teacher_logits=teacher_logits,
                        temperature=kd_temperature,
                        confidence_threshold=kd_conf_threshold,
                        confidence_gamma=kd_conf_gamma,
                    )

                total_loss = seg_loss + kd_coef * kd_loss

            base_loss_values = _loss_items_to_floats(
                outputs.get("loss_items", {})
            )
            bce_value = base_loss_values.get(
                "bce",
                float(seg_loss.detach().item()),
            )
            loss_values = {
                "bce": float(bce_value),
                "kd": float(kd_loss.detach().item()),
                "kd_coef": float(kd_coef),
                "total": float(total_loss.detach().item()),
            }

            loss = total_loss / cfg.train.grad_acc_step
            scaler.calculate_grad(loss=loss)
            if counter.every_n_iters(
                cfg.train.grad_acc_step
            ):
                scaler.update_grad()

                # In the curriculum-on path the EMA teacher is updated from
                # the beginning of training, even before KD is activated,
                # so it is ready when the Clean -> Mixed switch happens.
                if curriculum_enabled:
                    _update_ema_teacher(
                        student=model,
                        teacher=ema_teacher,
                        decay=ema_decay,
                    )

            item_loss = loss_values["total"]
            data_shape = tuple(data_batch["mask"].shape)
            loss_recorder.update(
                value=item_loss,
                num=data_shape[0],
            )

            for loss_name, loss_value in loss_values.items():
                epoch_loss_sums[loss_name] += loss_value
            epoch_loss_count += 1

            if cfg.log_interval > 0 and (
                counter.every_n_iters(cfg.log_interval)
                or counter.is_first_inner_iter()
                or counter.is_last_inner_iter()
                or counter.is_last_total_iter()
            ):
                gpu_mem_gb = (
                    torch.cuda.memory_reserved() / 1E9
                    if torch.cuda.is_available()
                    else 0.0
                )
                gpu_mem = f"{gpu_mem_gb:.3g}G"
                eta_seconds = (
                    iter_time_recorder.avg
                    * (
                        counter.num_total_iters
                        - counter.curr_iter
                        - 1
                    )
                )
                eta_string = (
                    f"ETA: "
                    f"{datetime.timedelta(seconds=int(eta_seconds))}"
                )
                progress = (
                    f"{counter.curr_iter}:"
                    f"{counter.num_total_iters} "
                    f"{batch_idx}/{counter.num_inner_iters} "
                    f"{counter.curr_epoch}/"
                    f"{counter.num_epochs}"
                )
                loss_info = (
                    f"L:{loss_values['total']:.4f} "
                    f"BCE:{loss_values['bce']:.4f} "
                    f"KD:{loss_values['kd']:.4f}"
                    f"x{loss_values['kd_coef']:.3f}"
                )
                lr_info = f"LR:{optimizer.lr_string()}"
                LOGGER.info(
                    f"{eta_string}({gpu_mem}) | "
                    f"{progress} | {stage_name} | "
                    f"{lr_info} | {loss_info}"
                )

                cfg.tb_logger.write_to_tb(
                    "lr",
                    optimizer.lr_groups(),
                    counter.curr_iter,
                )
                cfg.tb_logger.write_to_tb(
                    "iter_loss",
                    item_loss,
                    counter.curr_iter,
                )
                cfg.tb_logger.write_to_tb(
                    "avg_loss",
                    loss_recorder.global_avg,
                    counter.curr_iter,
                )
                for loss_name, loss_value in loss_values.items():
                    cfg.tb_logger.write_to_tb(
                        name=f"loss/{loss_name}",
                        data=loss_value,
                        curr_iter=counter.curr_iter,
                    )

            if counter.curr_iter < 3:
                recorder.plot_results(
                    dict(
                        img=data_batch["image_m"],
                        msk=data_batch["mask"],
                        **outputs["vis"],
                    ),
                    save_path=os.path.join(
                        cfg.path.pth_log,
                        "img",
                        f"iter_{counter.curr_iter}.png",
                    ),
                )

            iter_time_recorder.update(
                value=time.perf_counter() - iter_start_time
            )
            if counter.is_last_total_iter():
                break
            counter.update_iter_counter()

        recorder.plot_results(
            dict(
                img=data_batch["image_m"],
                msk=data_batch["mask"],
                **outputs["vis"],
            ),
            save_path=os.path.join(
                cfg.path.pth_log,
                "img",
                f"epoch_{counter.curr_epoch}.png",
            ),
        )

        io.save_weight(
            model=model,
            save_path=cfg.path.final_state_net,
        )

        current_epoch = counter.curr_epoch + 1
        epoch_loss_avg = _append_epoch_losses(
            csv_path=loss_csv_path,
            epoch=current_epoch,
            stage_name=stage_name,
            loss_sums=epoch_loss_sums,
            num_batches=epoch_loss_count,
        )

        epoch_kd = (
            epoch_loss_sums.get("kd", 0.0)
            / max(epoch_loss_count, 1)
        )
        epoch_kd_coef = (
            epoch_loss_sums.get("kd_coef", 0.0)
            / max(epoch_loss_count, 1)
        )

        switched_this_epoch = False
        switch_reason = ""
        loss_drop = None
        ema_consistency = None

        # --------------------------------------------------------------
        # Curriculum-only end-of-epoch controller
        # --------------------------------------------------------------
        if curriculum_enabled:
            ema_consistency = _evaluate_student_ema_consistency(
                student=model,
                teacher=ema_teacher,
                probe_loader=clean_probe_loader,
                device=cfg.device,
            )

            if curriculum_state["stage_idx"] == 0:
                transition_cfg = cfg.train.curriculum.transition
                (
                    should_switch,
                    loss_drop,
                    switch_reason,
                ) = _saturation_step(
                    curriculum_state=curriculum_state,
                    current_epoch=current_epoch,
                    epoch_bce=epoch_loss_avg["bce"],
                    ema_consistency=ema_consistency,
                    transition_cfg=transition_cfg,
                )

                if should_switch:
                    switched_this_epoch = True
                    curriculum_state["stage_idx"] = 1
                    curriculum_state["stage_start_epoch"] = (
                        current_epoch + 1
                    )
                    curriculum_state["switched_epoch"] = (
                        current_epoch
                    )
                    curriculum_state["kd_enabled"] = True
                    curriculum_state["rebuild_loader"] = True
                    curriculum_state["saturation_count"] = 0

                    LOGGER.info(
                        "=" * 20
                        + f" SWITCH Clean -> Mixed after epoch "
                        f"{current_epoch} ({switch_reason}) "
                        + "=" * 20
                    )

            LOGGER.info(
                f"Epoch {current_epoch:03d} | {stage_name} | "
                f"L:{epoch_loss_avg['total']:.4f} "
                f"BCE:{epoch_loss_avg['bce']:.4f} "
                f"KD:{epoch_kd:.4f}x{epoch_kd_coef:.3f} | "
                f"EMA-IoU:{ema_consistency:.4f} | "
                f"drop:{'NA' if loss_drop is None else f'{loss_drop:.5f}'}"
            )

            _append_curriculum_metrics(
                csv_path=curriculum_csv_path,
                epoch=current_epoch,
                stage=stage_name,
                stage_age=stage_age,
                bce=epoch_loss_avg["bce"],
                kd=epoch_kd,
                kd_coef=epoch_kd_coef,
                loss_drop=loss_drop,
                ema_consistency=ema_consistency,
                saturation_count=curriculum_state[
                    "saturation_count"
                ],
                switched=switched_this_epoch,
                switch_reason=switch_reason,
            )

            cfg.tb_logger.write_to_tb(
                "curriculum/ema_consistency",
                ema_consistency,
                current_epoch,
            )
            if loss_drop is not None:
                cfg.tb_logger.write_to_tb(
                    "curriculum/loss_drop",
                    loss_drop,
                    current_epoch,
                )
            cfg.tb_logger.write_to_tb(
                "curriculum/stage_idx",
                float(stage_idx),
                current_epoch,
            )
        else:
            # No fake EMA/saturation numbers are logged in the no-curriculum
            # ablation; this makes the experiment records unambiguous.
            LOGGER.info(
                f"Epoch {current_epoch:03d} | {stage_name} | "
                f"L:{epoch_loss_avg['total']:.4f} "
                f"BCE:{epoch_loss_avg['bce']:.4f} | "
                "Curriculum:OFF | EMA-KD:OFF"
            )

        val_start_epoch = cfg.train.get(
            "val_start_epoch",
            10,
        )
        val_interval = cfg.train.get(
            "val_interval",
            5,
        )
        save_val_ckpt = cfg.train.get(
            "save_val_ckpt",
            True,
        )

        should_validate = (
            current_epoch >= val_start_epoch
            and (
                current_epoch - val_start_epoch
            ) % val_interval == 0
        )

        # When curriculum is enabled, also validate exactly at the switch.
        if curriculum_enabled:
            should_validate = (
                should_validate or switched_this_epoch
            )

        if should_validate:
            LOGGER.info(
                "=" * 25
                + f" Validation Epoch {current_epoch} "
                + "=" * 25
            )

            if save_val_ckpt:
                epoch_weight_path = os.path.join(
                    cfg.path.pth,
                    f"state_epoch_{current_epoch:03d}.pth",
                )
                io.save_weight(
                    model=model,
                    save_path=epoch_weight_path,
                )

                # Curriculum state is meaningful only in curriculum mode.
                if curriculum_enabled:
                    _save_b2_state(
                        save_path=os.path.join(
                            cfg.path.pth,
                            f"b2_state_epoch_"
                            f"{current_epoch:03d}.pth",
                        ),
                        epoch=current_epoch,
                        ema_teacher=ema_teacher,
                        curriculum_state=curriculum_state,
                    )

            old_save_results = cfg.save_results
            cfg.save_results = False
            test(
                model=model,
                cfg=cfg,
                epoch=current_epoch,
            )
            cfg.save_results = old_save_results

        counter.update_epoch_counter()

    cfg.tb_logger.close_tb()
    io.save_weight(
        model=model,
        save_path=cfg.path.final_state_net,
    )

    if curriculum_enabled:
        _save_b2_state(
            save_path=os.path.join(
                cfg.path.pth,
                "b2_state_final.pth",
            ),
            epoch=counter.num_epochs,
            ema_teacher=ema_teacher,
            curriculum_state=curriculum_state,
        )

    total_train_time = (
        time.perf_counter() - train_start_time
    )
    total_other_time = datetime.timedelta(
        seconds=int(
            total_train_time
            - iter_time_recorder.global_sum
        )
    )
    LOGGER.info(
        f"Total Training Time: "
        f"{datetime.timedelta(seconds=int(total_train_time))} "
        f"({total_other_time} on others)"
    )


def parse_cfg():
    parser = argparse.ArgumentParser("Training and evaluation script")
    parser.add_argument("--config", default="configs/icod_train.py")
    parser.add_argument("--data-cfg", type=str, default="./dataset.yaml")
    parser.add_argument("--model-name", type=str, default="PvtV2B4_FPN_Baseline",choices=model_zoo.__dict__.keys())
    parser.add_argument("--output-dir", type=str, default="Stageout")
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--metric-names",
        nargs="+",
        type=str,
        default=["sm", "wfm", "mae", "em"],
        choices=recorder.GroupedMetricRecorder.supported_metrics,
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--info", type=str)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    with open(cfg.data_cfg, mode="r") as f:
        cfg.dataset_infos = yaml.safe_load(f)

    cfg.proj_root = os.path.dirname(os.path.abspath(__file__))
    cfg.exp_name = py_utils.construct_exp_name(model_name=cfg.model_name, cfg=cfg)
    cfg.output_dir = os.path.join(cfg.proj_root, cfg.output_dir)
    cfg.path = py_utils.construct_path(output_dir=cfg.output_dir, exp_name=cfg.exp_name)
    cfg.device = "cuda:0"

    py_utils.pre_mkdir(cfg.path)
    with open(cfg.path.cfg_copy, encoding="utf-8", mode="w") as f:
        f.write(cfg.pretty_text)
    shutil.copy(__file__, cfg.path.trainer_copy)

    file_handler = logging.FileHandler(cfg.path.log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(filename)s] %(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.info(cfg.pretty_text)

    cfg.tb_logger = recorder.TBLogger(tb_root=cfg.path.tb)
    return cfg


def main():
    cfg = parse_cfg()
    pt_utils.initialize_seed_cudnn(seed=cfg.base_seed, deterministic=cfg.deterministic)

    model_class = model_zoo.__dict__.get(cfg.model_name)
    assert model_class is not None, "Please check your --model-name"
    model_code = inspect.getsource(model_class)
    # Model/loss ablation switches live in cfg.model.  Without forwarding this
    # mapping, every ablation silently uses the constructor defaults.
    model_kwargs = dict(cfg.get("model", {}))
    model = model_class(
        num_frames=1,
        pretrained=cfg.pretrained,
        use_checkpoint=cfg.use_checkpoint,
        **model_kwargs,
    )
    LOGGER.info(model_code)
    model.to(cfg.device)

    if cfg.load_from:
        io.load_weight(model=model, load_path=cfg.load_from, strict=True)

    LOGGER.info(f"Number of Parameters: {sum((v.numel() for v in model.parameters(recurse=True)))}")
    if not cfg.evaluate:
        train(model=model, cfg=cfg)

    if cfg.evaluate or cfg.has_test:
        io.save_weight(model=model, save_path=cfg.path.final_state_net)
        test(model=model, cfg=cfg)

    LOGGER.info("End training...")


if __name__ == "__main__":
    main()
