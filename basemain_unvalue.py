# -*- coding: utf-8 -*-
"""Clean/Noisy/Unvalue trainer for APBOXNet.

The segmentation task never changes.  Clean and noisy samples keep dense-mask
supervision in every active batch.  Unvalue samples use their boxes from epoch
41, then receive EMA-teacher soft-mask supervision only on reliable pixels.

Run from the APBOXNet root::

    python basemain_unvalue.py \
      --config configs/curablation/csr_nc_unvalue.py \
      --model-name PvtV2B4_FPN_Unvalue
"""

import copy
import csv
import datetime
import inspect
import os
import shutil
import time
from collections import defaultdict

import torch
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Dataset

import basemain as core
from methods.fpn_unvalue_nc import PvtV2B4_FPN_Unvalue

# Registration happens before core.parse_cfg() builds argparse choices.
setattr(
    core.model_zoo,
    "PvtV2B4_FPN_Unvalue",
    PvtV2B4_FPN_Unvalue,
)


LOGGER = core.LOGGER


POOL_IDS = {"clean": 0, "noisy": 1, "unvalue": 2}
POOL_RELIABILITY = {"clean": 1.0, "noisy": 0.5, "unvalue": 0.0}


class PoolTaggedDataset(Dataset):
    """Inject pool identity without modifying the repository dataset class."""

    def __init__(self, dataset, pool_name):
        self.dataset = dataset
        self.pool_name = str(pool_name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        payload = sample["data"]
        pool_id = POOL_IDS[self.pool_name]
        payload["pool_id"] = torch.tensor(pool_id, dtype=torch.long)
        payload["pool_reliability"] = torch.tensor(
            POOL_RELIABILITY[self.pool_name], dtype=torch.float32
        )
        payload["has_mask_supervision"] = torch.tensor(
            0.0 if self.pool_name == "unvalue" else 1.0,
            dtype=torch.float32,
        )
        return sample


class ThreePoolBatchSampler(BatchSampler):
    """Fixed pool quota per batch, sampled with replacement."""

    def __init__(self, lengths, counts, num_batches, seed):
        self.names = ("clean", "noisy", "unvalue")
        self.lengths = {k: int(lengths[k]) for k in self.names}
        self.counts = {k: int(counts.get(k, 0)) for k in self.names}
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.offsets = {}
        offset = 0
        for name in self.names:
            self.offsets[name] = offset
            offset += self.lengths[name]
        if sum(self.counts.values()) <= 0:
            raise ValueError("At least one sample per batch is required.")
        for name in self.names:
            if self.counts[name] > 0 and self.lengths[name] == 0:
                raise RuntimeError(f"Pool '{name}' is active but empty.")

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.num_batches):
            indices = []
            for name in self.names:
                count = self.counts[name]
                if count <= 0:
                    continue
                local = torch.randint(
                    self.lengths[name], (count,), generator=generator
                ).tolist()
                indices.extend(self.offsets[name] + i for i in local)
            order = torch.randperm(len(indices), generator=generator).tolist()
            yield [indices[i] for i in order]

    def __len__(self):
        return self.num_batches


def _stage_for_epoch(cfg, epoch):
    for stage in cfg.train.curriculum.batch_schedule:
        if epoch <= int(stage.end_epoch):
            counts = {
                name: int(stage.batch.get(name, 0))
                for name in POOL_IDS
            }
            if sum(counts.values()) != int(cfg.train.batch_size):
                raise ValueError(
                    f"Stage {stage.name} batch counts must sum to "
                    f"batch_size={cfg.train.batch_size}, got {counts}."
                )
            return str(stage.name), counts
    raise ValueError(f"No batch_schedule stage covers epoch {epoch}.")


def _pool_probs(counts):
    total = float(sum(counts.values()))
    return {name: counts[name] / total for name in POOL_IDS}


def _append_schedule_csv(
    csv_path,
    epoch,
    stage_name,
    counts,
):
    """Save the exact per-batch composition for reproducibility."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)

    probs = _pool_probs(counts)

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "epoch",
                    "stage",
                    "clean_count",
                    "noisy_count",
                    "unvalue_count",
                    "clean_prob",
                    "noisy_prob",
                    "unvalue_prob",
                ]
            )
        writer.writerow(
            [
                int(epoch),
                stage_name,
                counts["clean"],
                counts["noisy"],
                counts["unvalue"],
                f"{probs['clean']:.6f}",
                f"{probs['noisy']:.6f}",
                f"{probs['unvalue']:.6f}",
            ]
        )


def _append_dynamic_epoch_losses(
    csv_path,
    epoch,
    stage_name,
    loss_sums,
    num_batches,
):
    """Write every model-provided scalar, including Z3 router diagnostics."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)
    denominator = max(int(num_batches), 1)
    averages = {
        name: float(value) / denominator
        for name, value in loss_sums.items()
    }

    preferred = ["total", "bce", "wbce", "nc", "q"]
    metric_names = [name for name in preferred if name in averages]
    metric_names.extend(
        sorted(name for name in averages if name not in metric_names)
    )
    fieldnames = ["epoch", "stage", *metric_names]

    if exists:
        with open(csv_path, mode="r", encoding="utf-8", newline="") as file:
            existing_fields = next(csv.reader(file), [])
        if existing_fields != fieldnames:
            raise RuntimeError(
                "loss_metrics.csv columns changed during one experiment. "
                f"Existing={existing_fields}, current={fieldnames}"
            )

    row = {"epoch": int(epoch), "stage": str(stage_name)}
    row.update({name: f"{averages[name]:.6f}" for name in metric_names})
    with open(csv_path, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return averages


def _prepare_pool_datasets(cfg):
    """Build and tag the three disjoint curriculum pools once."""
    curriculum = cfg.train.curriculum
    pool_cfg = curriculum.pools

    required = ("clean", "noisy", "unvalue")
    for name in required:
        if name not in pool_cfg:
            raise KeyError(
                f"train.curriculum.pools must contain '{name}'."
            )

    pool_datasets = {}
    for name in required:
        base_dataset = core._build_train_subset(
            cfg=cfg,
            list_path=pool_cfg[name],
            pool_name=name,
            augment=True,
        )
        pool_datasets[name] = PoolTaggedDataset(base_dataset, name)

    for name, dataset in pool_datasets.items():
        if len(dataset) == 0:
            raise RuntimeError(f"Curriculum pool '{name}' is empty.")

    LOGGER.info(
        "Mask/box curriculum pools: "
        + ", ".join(
            f"{name}={len(dataset)}"
            for name, dataset in pool_datasets.items()
        )
    )
    return pool_datasets


def _build_epoch_loader(
    cfg,
    pool_datasets,
    epoch,
):
    """Build a deterministic fixed-quota three-pool loader for one epoch."""
    stage_name, counts = _stage_for_epoch(cfg, epoch)
    names = ("clean", "noisy", "unvalue")
    joined = ConcatDataset([pool_datasets[name] for name in names])
    num_batches = (
        int(cfg.train.curriculum.num_samples_per_epoch)
        // int(cfg.train.batch_size)
    )
    batch_sampler = ThreePoolBatchSampler(
        lengths={name: len(pool_datasets[name]) for name in names},
        counts=counts,
        num_batches=num_batches,
        seed=int(cfg.base_seed) + int(epoch) * 1009,
    )
    loader = DataLoader(
        joined,
        batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        worker_init_fn=(
            core.pt_utils.customized_worker_init_fn
            if cfg.use_custom_worker_init
            else None
        ),
    )
    probs = _pool_probs(counts)

    LOGGER.info(
        "=" * 18
        + f" Epoch {epoch:03d} Mask/Box Curriculum "
        + "=" * 18
    )
    LOGGER.info(
        f"{stage_name} | "
        f"batch C:N:U={counts['clean']}:{counts['noisy']}:"
        f"{counts['unvalue']} | probs C={probs['clean']:.2%}, "
        f"N={probs['noisy']:.2%}, U={probs['unvalue']:.2%}"
    )

    return (
        loader,
        stage_name,
        counts,
    )


@torch.no_grad()
def _ema_update(teacher, student, decay):
    teacher_params = dict(teacher.named_parameters())
    for name, student_param in student.named_parameters():
        teacher_params[name].mul_(decay).add_(
            student_param.detach(), alpha=1.0 - decay
        )
    teacher_buffers = dict(teacher.named_buffers())
    for name, student_buffer in student.named_buffers():
        if name in teacher_buffers:
            teacher_buffers[name].copy_(student_buffer.detach())


@torch.no_grad()
def _attach_unvalue_teacher(data_batch, teacher, cfg, epoch_no):
    """Infer the Unvalue subset with aligned ZoomNeXt image scales."""
    start_epoch = int(cfg.train.unvalue_teacher.start_epoch)
    if epoch_no < start_epoch:
        return

    pool_id = data_batch["pool_id"].reshape(-1)
    indices = torch.nonzero(
        pool_id == POOL_IDS["unvalue"],
        as_tuple=False,
    ).flatten()
    if indices.numel() == 0:
        return

    required = ("image_l", "image_m")
    missing = [key for key in required if key not in data_batch]
    if missing:
        raise KeyError(
            "ZoomNeXt EMA teacher requires image_l and image_m; "
            f"missing={missing}, available={sorted(data_batch.keys())}."
        )

    # The current ZoomNeXt uses image_l + image_m. Forward image_s as well
    # when available, so this remains compatible with a three-scale variant.
    scale_keys = tuple(
        key
        for key in ("image_s", "image_m", "image_l")
        if key in data_batch
    )
    teacher_data = {
        key: data_batch[key].index_select(0, indices)
        for key in scale_keys
    }

    amp_enabled = bool(cfg.train.use_amp)
    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
        logits = teacher(data=teacher_data)
        probability = logits.sigmoid()

        if bool(cfg.train.unvalue_teacher.get("hflip", True)):
            flipped_data = {
                key: torch.flip(value, dims=(-1,))
                for key, value in teacher_data.items()
            }
            flipped_logits = teacher(data=flipped_data)
            flipped_probability = torch.flip(
                flipped_logits.sigmoid(),
                dims=(-1,),
            )
            disagreement = (
                probability - flipped_probability
            ).abs()
            probability = 0.5 * (
                probability + flipped_probability
            )
        else:
            disagreement = torch.zeros_like(probability)

    batch_size = int(data_batch["image_m"].shape[0])
    height, width = probability.shape[-2:]
    teacher_probability = torch.full(
        (batch_size, 1, height, width),
        0.5,
        device=probability.device,
        dtype=probability.dtype,
    )
    teacher_disagreement = torch.ones_like(teacher_probability)
    teacher_valid = torch.zeros(
        batch_size,
        device=probability.device,
        dtype=probability.dtype,
    )

    teacher_probability.index_copy_(0, indices, probability)
    teacher_disagreement.index_copy_(0, indices, disagreement)
    teacher_valid.index_fill_(0, indices, 1.0)

    data_batch["teacher_prob"] = teacher_probability.detach()
    data_batch["teacher_disagreement"] = (
        teacher_disagreement.detach()
    )
    data_batch["teacher_valid"] = teacher_valid.detach()


def train(model, cfg):
    """Train with mask anchors plus gradually introduced box-only samples."""
    curriculum = cfg.train.get("curriculum", None)
    if (
        not curriculum
        or not curriculum.get("enable", False)
        or not curriculum.get("continuous_schedule", {}).get(
            "enable", False
        )
    ):
        raise ValueError(
            "basemain_unvalue.py requires "
            "train.curriculum.enable=True and "
            "train.curriculum.continuous_schedule.enable=True."
        )

    pool_datasets = _prepare_pool_datasets(cfg)

    # Build epoch-1 loader first so TrainingCounter sees the correct,
    # constant epoch length.
    (
        tr_loader,
        stage_name,
        pool_counts,
    ) = _build_epoch_loader(
        cfg=cfg,
        pool_datasets=pool_datasets,
        epoch=1,
    )

    counter = core.recorder.TrainingCounter(
        epoch_length=len(tr_loader),
        epoch_based=cfg.train.epoch_based,
        num_epochs=cfg.train.num_epochs,
        num_total_iters=cfg.train.num_iters,
    )

    optimizer = core.pipeline.construct_optimizer(
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        cfg=cfg.train.optimizer.cfg,
    )

    scheduler = core.pipeline.Scheduler(
        optimizer=optimizer,
        num_iters=counter.num_total_iters,
        epoch_length=counter.num_inner_iters,
        scheduler_cfg=cfg.train.scheduler,
        step_by_batch=cfg.train.sche_usebatch,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)
    scheduler.plot_lr_coef_curve(save_path=cfg.path.pth_log)

    scaler = core.pipeline.Scaler(
        optimizer,
        cfg.train.use_amp,
        set_to_none=cfg.train.optimizer.set_to_none,
    )

    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    ema_decay = float(cfg.train.unvalue_teacher.get("ema_decay", 0.99))
    LOGGER.info(
        "Three-pool mask/box curriculum: ENABLED. "
        f"EMA teacher decay={ema_decay:.5f}."
    )
    LOGGER.info(f"Scheduler:\n{scheduler}\nOptimizer:\n{optimizer}")

    loss_recorder = core.recorder.HistoryBuffer()
    iter_time_recorder = core.recorder.HistoryBuffer()

    loss_csv_path = os.path.join(
        cfg.path.pth_log,
        "loss_metrics.csv",
    )
    schedule_csv_path = os.path.join(
        cfg.path.pth_log,
        "continuous_sampling_schedule.csv",
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
        epoch_no = counter.curr_epoch + 1
        LOGGER.info(f"Exp_Name: {cfg.exp_name}")

        # Rebuild only the sampler/loader. Dataset objects are reused.
        (
            tr_loader,
            stage_name,
            pool_counts,
        ) = _build_epoch_loader(
            cfg=cfg,
            pool_datasets=pool_datasets,
            epoch=epoch_no,
        )

        _append_schedule_csv(
            csv_path=schedule_csv_path,
            epoch=epoch_no,
            stage_name=stage_name,
            counts=pool_counts,
        )

        epoch_loss_sums = defaultdict(float)
        epoch_loss_count = 0

        model.train()
        if cfg.train.bn.freeze_status:
            core.pt_utils.frozen_bn_stats(
                model.encoder,
                freeze_affine=cfg.train.bn.freeze_affine,
            )

        for batch_idx, batch in enumerate(tr_loader):
            iter_start_time = time.perf_counter()

            scheduler.step(curr_idx=counter.curr_iter)

            data_batch = core.pt_utils.to_device(
                data=batch["data"],
                device=cfg.device,
            )

            teacher.eval()
            _attach_unvalue_teacher(
                data_batch=data_batch,
                teacher=teacher,
                cfg=cfg,
                epoch_no=epoch_no,
            )

            with torch.cuda.amp.autocast(
                enabled=cfg.train.use_amp
            ):
                outputs = model(
                    data=data_batch,
                    iter_percentage=counter.curr_percent,
                )
                total_loss = outputs["loss"]

            base_loss_values = core._loss_items_to_floats(
                outputs.get("loss_items", {})
            )
            bce_value = base_loss_values.get(
                "bce",
                float(total_loss.detach().item()),
            )

            loss_values = dict(base_loss_values)
            loss_values["bce"] = float(bce_value)
            loss_values["total"] = float(total_loss.detach().item())

            loss = total_loss / cfg.train.grad_acc_step
            scaler.calculate_grad(loss=loss)

            if counter.every_n_iters(cfg.train.grad_acc_step):
                scaler.update_grad()
                freeze_epoch = int(
                    cfg.train.unvalue_teacher.get("freeze_epoch", 120)
                )
                if epoch_no <= freeze_epoch:
                    _ema_update(
                        teacher=teacher,
                        student=model,
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
                    "ETA: "
                    + str(
                        datetime.timedelta(
                            seconds=int(eta_seconds)
                        )
                    )
                )

                probs = _pool_probs(pool_counts)

                progress = (
                    f"{counter.curr_iter}:"
                    f"{counter.num_total_iters} "
                    f"{batch_idx}/{counter.num_inner_iters} "
                    f"{counter.curr_epoch}/{counter.num_epochs}"
                )

                # Prefer the model's own loss breakdown.  NC models report
                # NC/WBCE/q here, while existing BCE models keep their
                # original readable string.
                loss_text = outputs.get(
                    "loss_str",
                    (
                        f"L:{loss_values['total']:.4f} "
                        f"BCE:{loss_values['bce']:.4f}"
                    ),
                )

                LOGGER.info(
                    f"{eta_string}({gpu_mem}) | "
                    f"{progress} | "
                    f"{stage_name} "
                    f"C:{probs['clean']:.3f}/N:{probs['noisy']:.3f}/"
                    f"U:{probs['unvalue']:.3f} | "
                    f"LR:{optimizer.lr_string()} | "
                    f"{loss_text}"
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
                for loss_name, loss_value in base_loss_values.items():
                    cfg.tb_logger.write_to_tb(
                        f"loss_item/{loss_name}",
                        loss_value,
                        counter.curr_iter,
                    )

            if counter.curr_iter < 3:
                core.recorder.plot_results(
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

        # One image snapshot per epoch, same behavior as the original trainer.
        core.recorder.plot_results(
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

        core.io.save_weight(
            model=model,
            save_path=cfg.path.final_state_net,
        )

        current_epoch = counter.curr_epoch + 1

        epoch_loss_avg = _append_dynamic_epoch_losses(
            csv_path=loss_csv_path,
            epoch=current_epoch,
            stage_name=stage_name,
            loss_sums=epoch_loss_sums,
            num_batches=epoch_loss_count,
        )

        probs = _pool_probs(pool_counts)

        epoch_details = " ".join(
            f"{name.upper()}:{epoch_loss_avg[name]:.4f}"
            for name in (
                "nc",
                "scale",
                "background",
                "q",
                "m5_large",
                "m5_medium",
                "m2_large",
                "m2_medium",
            )
            if name in epoch_loss_avg
        )
        LOGGER.info(
            f"Epoch {current_epoch:03d} | "
            f"{stage_name} | "
            f"C:{probs['clean']:.3f}/N:{probs['noisy']:.3f}/"
            f"U:{probs['unvalue']:.3f} | "
            f"L:{epoch_loss_avg['total']:.4f} "
            f"BCE:{epoch_loss_avg['bce']:.4f} | "
            f"{epoch_details} | "
            "EMA-KD:OFF"
        )

        cfg.tb_logger.write_to_tb(
            "curriculum/clean_prob",
            probs["clean"],
            current_epoch,
        )
        cfg.tb_logger.write_to_tb(
            "curriculum/noisy_prob",
            probs["noisy"],
            current_epoch,
        )
        cfg.tb_logger.write_to_tb(
            "curriculum/unvalue_prob",
            probs["unvalue"],
            current_epoch,
        )

        val_start_epoch = cfg.train.get(
            "val_start_epoch",
            30,
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

        # Also validate exact curriculum landmarks.
        if current_epoch in (50, 100):
            should_validate = True

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
                core.io.save_weight(
                    model=model,
                    save_path=epoch_weight_path,
                )

            old_save_results = cfg.save_results
            cfg.save_results = False

            core.test(
                model=model,
                cfg=cfg,
                epoch=current_epoch,
            )

            cfg.save_results = old_save_results

        counter.update_epoch_counter()

    cfg.tb_logger.close_tb()

    core.io.save_weight(
        model=model,
        save_path=cfg.path.final_state_net,
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
        "Total Training Time: "
        f"{datetime.timedelta(seconds=int(total_train_time))} "
        f"({total_other_time} on others)"
    )


def main():
    # Reuse repository argument/config/path handling.
    cfg = core.parse_cfg()

    # core.parse_cfg() copies basemain.py; overwrite trainer snapshot with
    # this actual continuous trainer for reproducibility.
    shutil.copy(__file__, cfg.path.trainer_copy)

    core.pt_utils.initialize_seed_cudnn(
        seed=cfg.base_seed,
        deterministic=cfg.deterministic,
    )

    model_class = core.model_zoo.__dict__.get(cfg.model_name)
    assert model_class is not None, "Please check your --model-name"

    model_code = inspect.getsource(model_class)

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
        core.io.load_weight(
            model=model,
            load_path=cfg.load_from,
            strict=True,
        )

    LOGGER.info(
        "Number of Parameters: "
        f"{sum(v.numel() for v in model.parameters(recurse=True))}"
    )

    if not cfg.evaluate:
        train(
            model=model,
            cfg=cfg,
        )

    if cfg.evaluate or cfg.has_test:
        core.io.save_weight(
            model=model,
            save_path=cfg.path.final_state_net,
        )
        core.test(
            model=model,
            cfg=cfg,
        )

    LOGGER.info("End training...")


if __name__ == "__main__":
    main()
