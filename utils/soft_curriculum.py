"""Soft pool scheduling and balanced mini-batch sampling.

This module is independent from the training entrypoint so it can be tested in
isolation.  It provides two pieces used by ``soft_main.py``:

1. a continuous, epoch-wise pool-ratio schedule;
2. a batch sampler that enforces those ratios inside every mini-batch while
   traversing each pool without replacement until that pool is exhausted.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

from torch.utils.data import Sampler


def cfg_get(mapping, key: str, default=None):
    """Read from dict-like or attribute-style config objects."""
    if mapping is None:
        return default
    if hasattr(mapping, "get"):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _normalise_ratios(ratios: Mapping[str, float]) -> Dict[str, float]:
    cleaned = {name: max(0.0, float(value)) for name, value in ratios.items()}
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError(f"At least one pool ratio must be positive, got {ratios}")
    return {name: value / total for name, value in cleaned.items()}


def _cosine_progress(epoch: int, start_epoch: int, end_epoch: int) -> float:
    """Cosine interpolation coefficient in [0, 1]."""
    if end_epoch <= start_epoch:
        return 1.0
    raw = (float(epoch) - float(start_epoch)) / float(end_epoch - start_epoch)
    raw = min(max(raw, 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * raw))


def _lerp_ratios(
    start: Mapping[str, float],
    end: Mapping[str, float],
    coefficient: float,
    pool_names: Sequence[str],
) -> Dict[str, float]:
    coefficient = min(max(float(coefficient), 0.0), 1.0)
    values = {
        name: (1.0 - coefficient) * float(start.get(name, 0.0))
        + coefficient * float(end.get(name, 0.0))
        for name in pool_names
    }
    return _normalise_ratios(values)


def resolve_soft_pool_ratios(curriculum_cfg, epoch: int) -> Dict[str, float]:
    """Resolve continuous pool ratios for one epoch.

    The default trajectory is:

    - clean warm-up;
    - cosine transition to 55/40/5 clean/noisy/camo;
    - cosine transition to 35/60/5;
    - cosine consolidation to 45/50/5.

    All endpoints are configurable through ``curriculum.soft_schedule``.
    Pools absent from an endpoint are treated as zero.
    """

    schedule = cfg_get(curriculum_cfg, "soft_schedule", {})
    pools_cfg = cfg_get(curriculum_cfg, "pools", {})
    pool_names = list(pools_cfg.keys())
    if not pool_names:
        raise ValueError("curriculum.pools is empty")

    warmup_end = int(cfg_get(schedule, "warmup_end", 20))
    transition_end = int(cfg_get(schedule, "transition_end", 60))
    noisy_end = int(cfg_get(schedule, "noisy_end", 100))
    total_epochs = int(cfg_get(schedule, "total_epochs", noisy_end))

    warmup = cfg_get(schedule, "warmup_ratios", {"clean": 1.0})
    transition = cfg_get(
        schedule,
        "transition_ratios",
        {"clean": 0.55, "noisy": 0.40, "camo": 0.05},
    )
    noisy = cfg_get(
        schedule,
        "noisy_ratios",
        {"clean": 0.35, "noisy": 0.60, "camo": 0.05},
    )
    consolidation = cfg_get(
        schedule,
        "consolidation_ratios",
        {"clean": 0.45, "noisy": 0.50, "camo": 0.05},
    )

    warmup = _normalise_ratios({name: float(warmup.get(name, 0.0)) for name in pool_names})

    if epoch <= warmup_end:
        return warmup

    if epoch <= transition_end:
        coef = _cosine_progress(epoch, warmup_end, transition_end)
        return _lerp_ratios(warmup, transition, coef, pool_names)

    if epoch <= noisy_end:
        coef = _cosine_progress(epoch, transition_end, noisy_end)
        return _lerp_ratios(transition, noisy, coef, pool_names)

    coef = _cosine_progress(epoch, noisy_end, total_epochs)
    return _lerp_ratios(noisy, consolidation, coef, pool_names)


def format_ratios(ratios: Mapping[str, float]) -> str:
    return ", ".join(f"{name}={value:.3f}" for name, value in ratios.items())


@dataclass
class _CyclingIndexStream:
    """Shuffle a pool, consume it once, then reshuffle for the next cycle."""

    indices: Sequence[int]
    rng: random.Random

    def __post_init__(self):
        if not self.indices:
            raise ValueError("A cycling index stream cannot be empty")
        self._buffer = list(self.indices)
        self._cursor = 0
        self.rng.shuffle(self._buffer)

    def _restart(self) -> None:
        self._buffer = list(self.indices)
        self.rng.shuffle(self._buffer)
        self._cursor = 0

    def draw(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count > len(self.indices):
            raise ValueError(
                f"Cannot draw {count} unique samples from a pool of {len(self.indices)} "
                "inside one batch. Reduce batch quota or enlarge the pool."
            )

        selected: list[int] = []
        selected_set = set()
        while len(selected) < count:
            if self._cursor >= len(self._buffer):
                self._restart()

            candidate = self._buffer[self._cursor]
            self._cursor += 1
            if candidate in selected_set:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
        return selected


class BalancedPoolBatchSampler(Sampler[list[int]]):
    """Yield mini-batches with controlled pool composition.

    ``pool_lengths`` follows the same order as the datasets in the surrounding
    ``ConcatDataset``. Ratios are enforced with residual carry, so a 5% pool at
    batch size 16 alternates between zero and one sample instead of being
    rounded to 6.25% in every batch.
    """

    def __init__(
        self,
        pool_lengths: Mapping[str, int],
        pool_ratios: Mapping[str, float],
        batch_size: int,
        num_samples: int,
        seed: int,
        epoch: int,
        min_clean_per_batch: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_samples < batch_size:
            raise ValueError(
                f"num_samples ({num_samples}) must be at least one batch ({batch_size})"
            )

        self.batch_size = int(batch_size)
        self.num_batches = int(num_samples) // self.batch_size
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.min_clean_per_batch = max(0, int(min_clean_per_batch))

        active_lengths: MutableMapping[str, int] = {}
        active_ratios: MutableMapping[str, float] = {}
        for name, length in pool_lengths.items():
            length = int(length)
            ratio = max(0.0, float(pool_ratios.get(name, 0.0)))
            if length > 0 and ratio > 0:
                active_lengths[name] = length
                active_ratios[name] = ratio

        if not active_lengths:
            raise ValueError(
                f"No active non-empty pool. lengths={dict(pool_lengths)}, "
                f"ratios={dict(pool_ratios)}"
            )

        self.pool_lengths = dict(active_lengths)
        self.pool_ratios = _normalise_ratios(active_ratios)
        self.pool_names = list(self.pool_lengths.keys())

        if self.min_clean_per_batch > 0 and "clean" not in self.pool_names:
            raise ValueError("min_clean_per_batch is positive but clean pool is inactive")

        offsets: Dict[str, int] = {}
        cursor = 0
        for name, length in pool_lengths.items():
            if name in self.pool_lengths:
                offsets[name] = cursor
            cursor += int(length)
        self.offsets = offsets

    def __len__(self) -> int:
        return self.num_batches

    def _batch_counts(self, credits: MutableMapping[str, float]) -> Dict[str, int]:
        """Allocate one safe integer quota for the next mini-batch.

        This uses smooth weighted round-robin rather than floor-plus-residual
        apportionment.  The previous residual implementation could make a tiny
        pool's carry negative after granting it one remainder slot.  At the
        next batch ``floor(ratio * batch_size + residual)`` could therefore be
        ``-1`` (most visible during the first soft-transition epochs when CAMO
        has a very small ratio).

        Credits are updated once per batch slot.  The selected pool spends one
        credit, keeping long-run counts close to the requested ratios while
        guaranteeing every per-pool quota is a non-negative integer.
        """
        counts: Dict[str, int] = {name: 0 for name in self.pool_names}

        # Smooth weighted round-robin: add each pool's desired share for every
        # slot, then allocate the slot to the pool with the largest credit.
        for _ in range(self.batch_size):
            for name in self.pool_names:
                credits[name] += self.pool_ratios[name]

            chosen = max(
                self.pool_names,
                key=lambda name: (credits[name], self.pool_ratios[name]),
            )
            counts[chosen] += 1
            credits[chosen] -= 1.0

        # Keep reliable clean anchors when requested.  Credit correction makes
        # later batches compensate for this forced transfer instead of causing
        # a permanent ratio drift.
        if self.min_clean_per_batch > 0:
            deficit = self.min_clean_per_batch - counts.get("clean", 0)
            while deficit > 0:
                donors = [
                    name
                    for name in self.pool_names
                    if name != "clean" and counts.get(name, 0) > 0
                ]
                if not donors:
                    raise ValueError(
                        "Unable to satisfy min_clean_per_batch with the current ratios"
                    )

                # Prefer the pool that is most over-represented in this batch.
                donor = max(
                    donors,
                    key=lambda name: (
                        counts[name] - self.pool_ratios[name] * self.batch_size,
                        counts[name],
                    ),
                )
                counts[donor] -= 1
                counts["clean"] = counts.get("clean", 0) + 1

                # Undo the donor allocation and record the forced clean slot in
                # the credit state, so future batches naturally compensate.
                credits[donor] += 1.0
                credits["clean"] -= 1.0
                deficit -= 1

        if any(count < 0 for count in counts.values()):
            raise RuntimeError(f"Negative batch quota generated: {counts}")
        if sum(counts.values()) != self.batch_size:
            raise RuntimeError(f"Invalid batch quota: {counts}")
        return counts

    def __iter__(self) -> Iterator[list[int]]:
        master_rng = random.Random(self.seed + 100_003 * self.epoch)
        streams: Dict[str, _CyclingIndexStream] = {}
        for pool_index, name in enumerate(self.pool_names):
            offset = self.offsets[name]
            length = self.pool_lengths[name]
            pool_rng = random.Random(master_rng.randint(0, 2**31 - 1) + pool_index)
            streams[name] = _CyclingIndexStream(
                indices=list(range(offset, offset + length)),
                rng=pool_rng,
            )

        credits = {name: 0.0 for name in self.pool_names}
        for _ in range(self.num_batches):
            counts = self._batch_counts(credits)
            batch: list[int] = []
            for name in self.pool_names:
                batch.extend(streams[name].draw(counts[name]))
            master_rng.shuffle(batch)
            yield batch