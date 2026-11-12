"""One runner for FPN/ZoomNeXt paper-form PKD + SKD.

Legacy basemain and basemain_continuous configurations keep their old runner.
This runner owns losses; model.body owns only prediction.
"""
import csv
import inspect
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from engine.curriculum import build_epoch_loader, prepare_pools
from losses.samcod_objective import SAMCODObjective


def append_csv(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    fields = list(row)
    if exists:
        with open(path, newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream)) != fields:
                raise ValueError(f"CSV columns changed: {path}. Use a new output experiment.")
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def snapshot_sources(cfg, model):
    root = Path(cfg.proj_root)
    dst = Path(cfg.path.pth_log) / "source_snapshot"
    files = ["basemain.py", "basemain_continuous.py", "engine/trainer.py",
             "engine/curriculum.py", "losses/samcod_objective.py", "utils/skd_transforms.py",
             "utils/box_prompt.py", "utils/augmentation_ablation.py", "methods/fpn_baseline.py",
             "methods/fpn_zoomnext.py"]
    model_source = inspect.getsourcefile(type(model))
    if model_source and Path(model_source).is_relative_to(root):
        files.append(str(Path(model_source).relative_to(root)))
    for relative in set(files):
        out = dst / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, out)
    shutil.copy2(__file__, cfg.path.trainer_copy)


def train(model, cfg, core):
    if not cfg.train.get("epoch_based", True):
        raise ValueError("samcod_paper runner currently requires epoch_based=True")
    if cfg.train.get("ema_kd", {}).get("enable", False):
        raise ValueError("EMA-KD is outside the paper-form objective; disable it")
    if cfg.train.get("sbft", {}).get("enable", False):
        raise ValueError("Disable SBFT for the paper-form comparison")
    objective = SAMCODObjective(cfg.objective, cfg.skd_augment)
    pools = prepare_pools(cfg, core)
    loader, _, _ = build_epoch_loader(cfg, core, pools, 1)
    epoch_length = len(loader)
    epochs = int(cfg.train.num_epochs)
    total_iters = epoch_length * epochs
    acc = int(cfg.train.grad_acc_step)
    if epochs < 1 or acc < 1:
        raise ValueError("num_epochs and grad_acc_step must be positive")
    optimizer = core.pipeline.construct_optimizer(
        model=model, initial_lr=cfg.train.lr, mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode, cfg=cfg.train.optimizer.cfg,
    )
    scheduler_cfg = cfg.train.scheduler.copy()
    if "milestones_epochs" in scheduler_cfg.get("cfg", {}):
        scheduler_cfg = dict(scheduler_cfg)
        scheduler_cfg["cfg"] = dict(scheduler_cfg["cfg"])
        milestones = scheduler_cfg["cfg"].pop("milestones_epochs")
        scheduler_cfg["cfg"]["milestones"] = [int(e) * epoch_length for e in milestones]
    scheduler = core.pipeline.Scheduler(
        optimizer=optimizer, num_iters=total_iters, epoch_length=epoch_length,
        scheduler_cfg=scheduler_cfg, step_by_batch=True,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)
    scaler = core.pipeline.Scaler(optimizer, cfg.train.use_amp, set_to_none=cfg.train.optimizer.set_to_none)
    optimizer.zero_grad(set_to_none=cfg.train.optimizer.set_to_none)
    if cfg.train.bn.freeze_encoder:
        model.encoder.requires_grad_(False)
    snapshot_sources(cfg, model)
    core.LOGGER.info("samcod_paper runner | detach=%s | sampling=%s | gradient accumulation=%s",
                     objective.detach_view, objective.transforms.policy, acc)
    # Fail early on unsupported objectives rather than silently using model BCE.
    loss_path = os.path.join(cfg.path.pth_log, "loss_metrics.csv")
    schedule_path = os.path.join(cfg.path.pth_log, "continuous_sampling_schedule.csv")
    global_step = 0
    for epoch in range(1, epochs + 1):
        loader, stage, masses = build_epoch_loader(cfg, core, pools, epoch)
        if len(loader) != epoch_length:
            raise ValueError("Epoch length changed; use fixed num_samples_per_epoch")
        model.train()
        if cfg.train.bn.freeze_status:
            core.pt_utils.frozen_bn_stats(model.encoder, freeze_affine=cfg.train.bn.freeze_affine)
        sums, counts = defaultdict(float), defaultdict(int)
        for batch_index, batch in enumerate(loader):
            scheduler.step(curr_idx=global_step)
            data = core.pt_utils.to_device(batch["data"], device=cfg.device)
            with torch.cuda.amp.autocast(enabled=cfg.train.use_amp):
                out = objective(model, data, global_step / max(total_iters, 1))
            if not torch.isfinite(out["loss"]):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, step {global_step}")
            # Correctly normalize the final incomplete accumulation window.
            window_start = (batch_index // acc) * acc
            window_size = min(acc, epoch_length - window_start)
            scaler.calculate_grad(out["loss"] / window_size)
            if (batch_index + 1) % acc == 0 or batch_index + 1 == epoch_length:
                scaler.update_grad()
            values = core._loss_items_to_floats(out["loss_items"])
            for name, value in values.items():
                sums[name] += value
            ids = data.get("pool_id")
            if ids is not None:
                counts["clean"] += int((ids == 0).sum().item())
                counts["noisy"] += int((ids == 1).sum().item())
                counts["all"] += int(ids.numel())
            if batch_index == 0 or (cfg.log_interval > 0 and (batch_index + 1) % cfg.log_interval == 0):
                core.LOGGER.info("Epoch %03d %s batch %d/%d | %s", epoch, stage, batch_index + 1,
                                 epoch_length, out["loss_str"])
                for name, value in values.items():
                    cfg.tb_logger.write_to_tb(f"loss/{name}", value, global_step)
            global_step += 1
        averages = {k: v / epoch_length for k, v in sorted(sums.items())}
        append_csv(loss_path, dict(epoch=epoch, stage=stage, **averages))
        total_mass = sum(masses.values()) or 1.0
        append_csv(schedule_path, dict(
            epoch=epoch, stage=stage, clean_weight=masses.get("clean", 0.0),
            noisy_weight=masses.get("noisy", 0.0),
            clean_prob=masses.get("clean", 0.0) / total_mass,
            noisy_prob=masses.get("noisy", 0.0) / total_mass,
            actual_clean=counts["clean"], actual_noisy=counts["noisy"], actual_total=counts["all"],
        ))
        core.LOGGER.info("Epoch %03d | %s | PKD %.5f SKD %.5f | actual Clean=%d Noisy=%d",
                         epoch, stage, averages["pkd"], averages["skd"], counts["clean"], counts["noisy"])
        core.io.save_weight(model=model, save_path=cfg.path.final_state_net)
        val_start = int(cfg.train.get("val_start_epoch", 20))
        val_interval = int(cfg.train.get("val_interval", 5))
        if val_interval > 0 and epoch >= val_start and (epoch - val_start) % val_interval == 0:
            if cfg.train.get("save_val_ckpt", True):
                core.io.save_weight(model=model, save_path=os.path.join(cfg.path.pth, f"state_epoch_{epoch:03d}.pth"))
            old = cfg.save_results
            cfg.save_results = False
            try:
                core.test(model=model, cfg=cfg, epoch=epoch)
            finally:
                cfg.save_results = old
    cfg.tb_logger.close_tb()
