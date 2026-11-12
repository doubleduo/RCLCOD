"""Explicit Clean/Noisy pool masses; no implicit EMA or loss schedule."""
import math


def schedule_weights(curriculum, epoch):
    if epoch < 1:
        raise ValueError("Epochs are one-based")
    if not curriculum.get("enable", False):
        return "all", {}
    if "final_start_epoch" in curriculum:
        raise ValueError("Use clean_only_start_epoch explicitly, not final_start_epoch")
    clean = float(curriculum.get("clean_weight", 1.0))
    final = curriculum.get("clean_only_start_epoch")
    mode = curriculum.get("mode", "continuous")
    if mode not in ("fixed", "continuous"):
        raise ValueError("curriculum.mode must be fixed or continuous")
    if final is not None and epoch >= int(final):
        weights = {"clean": clean, "noisy": 0.0}
        stage = "clean_only"
    elif mode == "fixed":
        stage, weights = "fixed_mix", {"clean": clean, "noisy": float(curriculum.get("noisy_weight", 1.0))}
    else:
        noisy = curriculum["noisy"]
        hold = int(noisy.get("hold_end_epoch", 60))
        start = int(noisy.get("anneal_start_epoch", hold + 1))
        end = int(noisy.get("anneal_end_epoch", 120))
        if not 0 <= hold < start < end:
            raise ValueError("Require 0 <= hold_end < anneal_start < anneal_end")
        hi, lo = float(noisy.get("start_weight", 1.0)), float(noisy.get("end_weight", 0.4))
        if min(hi, lo) < 0:
            raise ValueError("Noisy weights must be nonnegative")
        if epoch < start:
            stage, mass = "broad", hi
        elif epoch > end:
            stage, mass = "reliable", lo
        else:
            progress = (epoch - start) / (end - start)
            curve = noisy.get("mode", "cosine")
            if curve == "cosine":
                mass = lo + (hi - lo) * (1 + math.cos(math.pi * progress)) / 2
            elif curve == "linear":
                mass = hi + (lo - hi) * progress
            else:
                raise ValueError("noisy.mode must be cosine or linear")
            stage = "anneal"
        weights = {"clean": clean, "noisy": mass}
    if any(not math.isfinite(v) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Pool masses must be finite, nonnegative, with positive sum")
    return stage, weights


def prepare_pools(cfg, core):
    curriculum = cfg.train.get("curriculum", {})
    if not curriculum.get("enable", False):
        return {"all": core._build_train_subset(cfg)}
    pools = {}
    for name in ("clean", "noisy"):
        pools[name] = core._build_train_subset(cfg, curriculum["pools"][name], pool_name=name)
        if not len(pools[name]):
            raise ValueError(f"Empty curriculum pool: {name}")
    # A sample must not be simultaneously treated as Clean and Noisy.
    clean = {p[0] for p in pools["clean"].total_data_paths}
    noisy = {p[0] for p in pools["noisy"].total_data_paths}
    if clean & noisy:
        raise ValueError(f"Clean/Noisy lists overlap in {len(clean & noisy)} images")
    return pools


def build_epoch_loader(cfg, core, pools, epoch):
    from torch.utils.data import DataLoader
    curriculum = cfg.train.get("curriculum", {})
    stage, weights = schedule_weights(curriculum, epoch)
    for dataset in pools.values():
        dataset.set_epoch(epoch)
    if weights:
        loader = core._build_curriculum_loader(
            cfg, pools, dict(name=stage, weights=weights),
            int(curriculum["num_samples_per_epoch"]),
        )
    else:
        loader = DataLoader(
            pools["all"], batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers, shuffle=True, drop_last=True,
            pin_memory=True,
            worker_init_fn=core.pt_utils.customized_worker_init_fn if cfg.use_custom_worker_init else None,
        )
    if not len(loader):
        raise ValueError("Training loader is empty; check pool size and batch_size")
    return loader, stage, weights
