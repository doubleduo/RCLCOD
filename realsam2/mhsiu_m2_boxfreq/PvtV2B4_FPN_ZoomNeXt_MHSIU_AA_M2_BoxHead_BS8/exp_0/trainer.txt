# -*- coding: utf-8 -*-
"""
APBOXNet continuous Clean/Noisy curriculum trainer.

This file deliberately reuses the repository's existing datasets, evaluator,
optimizer/scheduler, metric code and model registry from basemain.py, while
removing the old two-stage saturation/EMA-KD controller.

The only experimental variable added here is the per-epoch Clean/Noisy
sampling trajectory:

    1-50   : 1.0 : 1.0
    51-100 : noisy cosine anneals from 1.0 to 0.3
    101-150: 1.0 : 0.3

Run:
    python basemain_continuous.py \
        --config configs/curablation/zoomnext_nc_continuous.py \
        --model-name PvtV2B4_ZoomNeXt_NC_Curriculum

or:
    python basemain_continuous.py \
        --config configs/icod_c3_continuous_clean_noisy.py \
        --model-name ConvNeXtB384_FPN_Baseline
"""

import csv
import datetime
import inspect
import math
import os
import shutil
import time
from collections import defaultdict

import torch

import basemain as core


LOGGER = core.LOGGER


def _schedule_weights(cfg, epoch):
    """Return (stage_name, clean_weight, noisy_weight) for one epoch."""
    curriculum = cfg.train.curriculum
    schedule = curriculum.continuous_schedule
    noisy_cfg = schedule.noisy

    clean_weight = float(schedule.get("clean_weight", 1.0))

    hold_end = int(noisy_cfg.get("hold_end_epoch", 50))
    anneal_start = int(noisy_cfg.get("anneal_start_epoch", hold_end + 1))
    anneal_end = int(noisy_cfg.get("anneal_end_epoch", 100))

    start_weight = float(noisy_cfg.get("start_weight", 1.0))
    end_weight = float(noisy_cfg.get("end_weight", 0.3))
    mode = str(noisy_cfg.get("mode", "cosine")).lower()

    if epoch <= hold_end:
        return "C3_broad_1to1", clean_weight, start_weight

    if epoch >= anneal_end:
        stage_name = (
            "C3_reliable_0p3"
            if epoch > anneal_end
            else "C3_anneal"
        )
        return stage_name, clean_weight, end_weight

    # epoch in [anneal_start, anneal_end)
    denom = max(anneal_end - anneal_start, 1)
    progress = (epoch - anneal_start) / float(denom)
    progress = min(max(progress, 0.0), 1.0)

    if mode == "cosine":
        # start at 1.0, smoothly approach 0.3
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        noisy_weight = end_weight + (start_weight - end_weight) * coeff
    elif mode == "linear":
        noisy_weight = start_weight + (end_weight - start_weight) * progress
    else:
        raise ValueError(
            f"Unknown continuous noisy schedule mode: {mode}. "
            "Use 'cosine' or 'linear'."
        )

    return "C3_anneal", clean_weight, float(noisy_weight)


def _normalized_pool_probs(clean_weight, noisy_weight):
    total = clean_weight + noisy_weight
    if total <= 0:
        raise ValueError("clean_weight + noisy_weight must be > 0.")
    return clean_weight / total, noisy_weight / total


def _append_schedule_csv(
    csv_path,
    epoch,
    stage_name,
    clean_weight,
    noisy_weight,
):
    """Save the exact planned pool masses/probabilities for reproducibility."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)

    clean_prob, noisy_prob = _normalized_pool_probs(
        clean_weight,
        noisy_weight,
    )

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "epoch",
                    "stage",
                    "clean_weight",
                    "noisy_weight",
                    "clean_prob",
                    "noisy_prob",
                ]
            )
        writer.writerow(
            [
                int(epoch),
                stage_name,
                f"{clean_weight:.6f}",
                f"{noisy_weight:.6f}",
                f"{clean_prob:.6f}",
                f"{noisy_prob:.6f}",
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
    """Build Clean and Noisy datasets once. Only sampler weights change."""
    curriculum = cfg.train.curriculum
    pool_cfg = curriculum.pools

    required = ("clean", "noisy")
    for name in required:
        if name not in pool_cfg:
            raise KeyError(
                f"train.curriculum.pools must contain '{name}'."
            )

    # Intentionally ignore unvalue/camo even if someone accidentally leaves
    # them in another config. This experiment is Clean+Noisy only.
    pool_datasets = {
        name: core._build_train_subset(
            cfg=cfg,
            list_path=pool_cfg[name],
            pool_name=name,
            augment=True,
        )
        for name in required
    }

    for name, dataset in pool_datasets.items():
        if len(dataset) == 0:
            raise RuntimeError(f"Curriculum pool '{name}' is empty.")

    LOGGER.info(
        "Continuous C3 pools: "
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
    """Rebuild WeightedRandomSampler once per epoch with scheduled masses."""
    stage_name, clean_weight, noisy_weight = _schedule_weights(
        cfg=cfg,
        epoch=epoch,
    )

    stage = dict(
        name=stage_name,
        weights=dict(
            clean=clean_weight,
            noisy=noisy_weight,
        ),
    )

    loader = core._build_curriculum_loader(
        cfg=cfg,
        pool_datasets=pool_datasets,
        stage=stage,
        num_samples_per_epoch=int(
            cfg.train.curriculum.num_samples_per_epoch
        ),
    )

    clean_prob, noisy_prob = _normalized_pool_probs(
        clean_weight,
        noisy_weight,
    )

    LOGGER.info(
        "=" * 18
        + f" Epoch {epoch:03d} Continuous C3 "
        + "=" * 18
    )
    LOGGER.info(
        f"{stage_name} | "
        f"weights C:N={clean_weight:.4f}:{noisy_weight:.4f} | "
        f"expected probs C={clean_prob:.2%}, N={noisy_prob:.2%}"
    )

    return (
        loader,
        stage_name,
        clean_weight,
        noisy_weight,
    )


def train(model, cfg):
    """Train with per-epoch continuous Clean/Noisy sampling."""
    curriculum = cfg.train.get("curriculum", None)
    if (
        not curriculum
        or not curriculum.get("enable", False)
        or not curriculum.get("continuous_schedule", {}).get(
            "enable", False
        )
    ):
        raise ValueError(
            "basemain_continuous.py requires "
            "train.curriculum.enable=True and "
            "train.curriculum.continuous_schedule.enable=True."
        )

    pool_datasets = _prepare_pool_datasets(cfg)

    # Build epoch-1 loader first so TrainingCounter sees the correct,
    # constant epoch length.
    (
        tr_loader,
        stage_name,
        clean_weight,
        noisy_weight,
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

    LOGGER.info(
        "Continuous C3 mode: ENABLED. "
        "Saturation detector: OFF. EMA teacher: OFF. EMA-KD: OFF."
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
            clean_weight,
            noisy_weight,
        ) = _build_epoch_loader(
            cfg=cfg,
            pool_datasets=pool_datasets,
            epoch=epoch_no,
        )

        _append_schedule_csv(
            csv_path=schedule_csv_path,
            epoch=epoch_no,
            stage_name=stage_name,
            clean_weight=clean_weight,
            noisy_weight=noisy_weight,
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

                clean_prob, noisy_prob = _normalized_pool_probs(
                    clean_weight,
                    noisy_weight,
                )

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
                    f"C:{clean_prob:.3f}/N:{noisy_prob:.3f} | "
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

        clean_prob, noisy_prob = _normalized_pool_probs(
            clean_weight,
            noisy_weight,
        )

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
            f"C:{clean_prob:.3f}/N:{noisy_prob:.3f} | "
            f"L:{epoch_loss_avg['total']:.4f} "
            f"BCE:{epoch_loss_avg['bce']:.4f} | "
            f"{epoch_details} | "
            "EMA-KD:OFF"
        )

        cfg.tb_logger.write_to_tb(
            "curriculum/clean_prob",
            clean_prob,
            current_epoch,
        )
        cfg.tb_logger.write_to_tb(
            "curriculum/noisy_prob",
            noisy_prob,
            current_epoch,
        )
        cfg.tb_logger.write_to_tb(
            "curriculum/noisy_weight",
            noisy_weight,
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
