# -*- coding: utf-8 -*-
"""Feedback-guided Clean/Noisy curriculum for RCLCOD.

Drop-in companion to the existing basemain_continuous.py trainer.
No changes to PvtV2B4_FPN_Baseline, its loss, optimizer or validation.

    python basemain_feedback.py --config configs/sam2_real_feedback.py \
        --model-name PvtV2B4_FPN_Baseline --output-dir realsam2/base_feedback

The original trainer calls _build_epoch_loader and _append_dynamic_epoch_losses
through module globals. We replace only these two extension points while
calling its unmodified train routine, then restore them in a finally block.
"""
import csv
import json
import math
import os
import random
import inspect
import shutil
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Subset

import basemain as core
import basemain_continuous as legacy

LOGGER = core.LOGGER
BASE_TRAIN = legacy.train


class PersistentPoolStream:
    """Replay every source sample before reshuffling a pool (across epochs)."""
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError('Every pool must contain at least one sample')
        self.size = int(size)
        self.rng = random.Random(int(seed))
        self.order = list(range(self.size))
        self.rng.shuffle(self.order)
        self.pos = 0
        self.cycles = 0

    def draw(self, k):
        # Avoid repeats inside a single batch. To use replacement, a separate
        # matched-sampler ablation must be performed.
        if k > self.size:
            raise ValueError(f'Cannot draw {k} distinct examples from pool of {self.size}')
        selected = []
        used = set()
        while len(selected) < k:
            if self.pos >= self.size:
                self.rng.shuffle(self.order)
                self.pos = 0
                self.cycles += 1
            idx = self.order[self.pos]
            self.pos += 1
            if idx in used:
                continue
            selected.append(idx)
            used.add(idx)
        return selected


class FeedbackBatchSampler(BatchSampler):
    """Exact-size two-pool batches; residual rounding; persistent old-data replay."""
    def __init__(self, streams, noisy_prob, batch_size, num_batches,
                 min_clean_per_batch, seed):
        self.streams = streams
        self.p_noisy = float(noisy_prob)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.min_clean = int(min_clean_per_batch)
        self.seed = int(seed)
        self.counts = defaultdict(int)
        self.seen = {'clean': set(), 'noisy': set()}
        assert 0 <= self.p_noisy <= 1
        assert 0 <= self.min_clean <= self.batch_size
        assert self.batch_size > 0 and self.num_batches > 0
        if self.p_noisy * self.batch_size > self.batch_size - self.min_clean + 1e-7:
            raise ValueError('Noisy ratio violates guaranteed clean replay quota')

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.counts.clear()
        for pool in self.seen:
            self.seen[pool].clear()
        rng = random.Random(self.seed)
        credit = 0.0
        clean_offset = 0
        noisy_offset = self.streams['clean'].size
        for _ in range(self.num_batches):
            credit += self.p_noisy * self.batch_size
            n_noisy = math.floor(credit + 1e-9)
            credit -= n_noisy
            n_noisy = min(n_noisy, self.batch_size - self.min_clean)
            n_clean = self.batch_size - n_noisy
            c = self.streams['clean'].draw(n_clean)
            n = self.streams['noisy'].draw(n_noisy)
            self.counts['clean'] += n_clean
            self.counts['noisy'] += n_noisy
            self.seen['clean'].update(c)
            self.seen['noisy'].update(n)
            batch = [clean_offset + i for i in c] + [noisy_offset + i for i in n]
            rng.shuffle(batch)
            yield batch


class FeedbackController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.p_noisy = float(cfg.initial_noisy_prob)
        self.last_action = 'WARMUP'
        self.ema = None
        self.previous = None
        self.best_dice = None
        self.plateau_count = 0
        self.bad_count = 0
        self.cooldown = 0

    def observe(self, metrics, epoch):
        beta = float(self.cfg.ema_beta)
        keys = ('clean_loss', 'clean_dice', 'consistency', 'fg_area_ratio')
        smoothed = {k: float(metrics[k]) for k in keys}
        if self.ema is not None:
            smoothed = {k: beta * self.ema[k] + (1-beta) * smoothed[k] for k in keys}
        prev = self.ema
        self.ema = smoothed
        self.best_dice = max(self.best_dice or 0.0, smoothed['clean_dice'])
        reason = 'first probe / warmup'
        action = 'WARMUP' if epoch < int(self.cfg.warmup_epochs) else 'HOLD'
        delta_loss, delta_stab = 0.0, 0.0
        if prev is not None:
            delta_loss = (prev['clean_loss'] - smoothed['clean_loss']) / max(prev['clean_loss'], 1e-6)
            delta_stab = smoothed['consistency'] - prev['consistency']
            area = smoothed['fg_area_ratio']
            severe = area < float(self.cfg.fg_ratio_min) or area > float(self.cfg.fg_ratio_max)
            dice_drop = smoothed['clean_dice'] < self.best_dice - float(self.cfg.dice_drop_trigger)
            weak_dice = smoothed['clean_dice'] < self.best_dice - float(self.cfg.dice_safe_margin)
            unstable = delta_stab < -float(self.cfg.stability_drop_trigger)
            worsening = delta_loss < -float(self.cfg.loss_bad_threshold)
            healthy = not severe and not weak_dice and delta_stab >= -float(self.cfg.stability_safe_margin)
            plateau = abs(delta_loss) < float(self.cfg.loss_good_threshold) and healthy
            self.plateau_count = self.plateau_count + 1 if plateau else 0
            self.bad_count = self.bad_count + 1 if (unstable or worsening or dice_drop) else 0

            if epoch >= int(self.cfg.warmup_epochs) and bool(getattr(self.cfg, 'enable', True)):
                if severe or self.bad_count >= int(self.cfg.bad_patience):
                    action = 'RECOVER'
                    reason = 'area collapse' if severe else 'persistent clean deterioration'
                    self.p_noisy -= float(self.cfg.recover_step)
                    self.cooldown = int(self.cfg.cooldown_checks)
                    self.bad_count = 0
                    self.plateau_count = 0
                elif self.cooldown > 0:
                    action = 'HOLD'
                    reason = 'recovery cooldown'
                    self.cooldown -= 1
                elif healthy and (delta_loss >= float(self.cfg.loss_good_threshold)
                                  or self.plateau_count >= int(self.cfg.plateau_patience)):
                    action = 'ADVANCE'
                    reason = 'clean improved' if delta_loss >= float(self.cfg.loss_good_threshold) else 'stable plateau'
                    self.p_noisy += float(self.cfg.advance_step)
                    self.plateau_count = 0
                else:
                    action = 'HOLD'
                    reason = 'insufficient evidence'
            else:
                if not bool(getattr(self.cfg, 'enable', True)):
                    action, reason = 'STATIC', 'feedback disabled'
                else:
                    reason = 'warmup'
        if not bool(getattr(self.cfg, 'enable', True)):
            action, reason = 'STATIC', 'feedback disabled'
        self.p_noisy = min(max(self.p_noisy, float(self.cfg.noisy_min)),
                           min(float(self.cfg.noisy_max), 1.0 - float(self.cfg.clean_min)))
        self.last_action = action
        out = dict(metrics)
        out.update({f'ema_{k}':v for k,v in smoothed.items()})
        out.update(action=action, reason=reason, delta_clean_loss=delta_loss,
                   delta_consistency=delta_stab, next_noisy_prob=self.p_noisy)
        return out

    def state_dict(self):
        return dict(p_noisy=self.p_noisy, last_action=self.last_action,
                    ema=self.ema, previous=self.previous, best_dice=self.best_dice,
                    plateau_count=self.plateau_count, bad_count=self.bad_count,
                    cooldown=self.cooldown)


def make_probe_loader(cfg, pool_name, max_items, seed):
    """Deterministic nonaugmented subset. Probe GT is training pseudo-GT only."""
    pool_path = cfg.train.curriculum.pools[pool_name]
    full = core._build_train_subset(cfg, list_path=pool_path,
                                    pool_name=pool_name, augment=False)
    n = min(max_items, len(full))
    if n < 1:
        raise ValueError(f'{pool_name} probe contains no samples')
    idx = sorted(random.Random(seed).sample(range(len(full)), n))
    return DataLoader(Subset(full, idx), batch_size=int(cfg.train.feedback.probe_batch_size),
                      shuffle=False, num_workers=int(cfg.train.feedback.probe_workers),
                      pin_memory=True, drop_last=False)


@torch.no_grad()
def evaluate_feedback_probe(model, cfg, clean_loader, noisy_loader):
    """Run identical data every probe; two-flip predictions share all model scales."""
    was_training = model.training
    model.eval()
    results = {}
    try:
        for name, loader in [('clean', clean_loader), ('noisy', noisy_loader)]:
            sums = defaultdict(float)
            count = 0
            for batch in loader:
                data = core.pt_utils.to_device(batch['data'], device=cfg.device)
                with torch.cuda.amp.autocast(enabled=bool(cfg.train.use_amp)):
                    pred = model(data=data)
                    flip_data = {
                        k:(torch.flip(v, dims=(-1,)) if k in ('image_s','image_m','image_l') else v)
                        for k,v in data.items()
                    }
                    pred_flip = torch.flip(model(data=flip_data), dims=(-1,))
                p = torch.sigmoid(pred.float())
                q = torch.sigmoid(pred_flip.float())
                y = (data['mask'].float() > 0.5).float()
                if y.shape[-2:] != p.shape[-2:]:
                    y = F.interpolate(y, p.shape[-2:], mode='nearest')
                b = p.shape[0]
                roi = F.max_pool2d(y, kernel_size=17, stride=1, padding=8)
                roi = torch.where(roi.flatten(1).sum(1)[:,None,None,None] > 0,
                                  roi, torch.ones_like(roi))
                denom = roi.flatten(1).sum(1).clamp_min(1)
                consistency = 1.0 - ((p-q).abs()*roi).flatten(1).sum(1)/denom
                fg_pred = (p>0.5).flatten(1).sum(1).float()
                fg_target = y.flatten(1).sum(1).clamp_min(1)
                fg_ratio = (fg_pred/fg_target).clamp(max=10)
                if name == 'clean':
                    bce = F.binary_cross_entropy_with_logits(pred.float(), y, reduction='none').flatten(1).mean(1)
                    dice = (2*(p*y).flatten(1).sum(1)+1) / ((p+y).flatten(1).sum(1)+1)
                    loss = bce + (1-dice)
                    sums['clean_loss'] += loss.sum().item()
                    sums['clean_dice'] += dice.sum().item()
                    sums['consistency'] += consistency.sum().item()
                    sums['fg_area_ratio'] += fg_ratio.sum().item()
                else:
                    confidence = (p>=float(cfg.train.feedback.pixel_confidence)) | (p<=1-float(cfg.train.feedback.pixel_confidence))
                    stable = (p-q).abs() <= float(cfg.train.feedback.pixel_disagreement)
                    agree = (p>=0.5) == (y>=0.5)
                    valid = confidence & stable & agree
                    pixel_coverage = (valid.float()*roi).flatten(1).sum(1)/denom
                    fg_denom = y.flatten(1).sum(1).clamp_min(1)
                    fg_coverage = (valid.float()*y).flatten(1).sum(1)/fg_denom
                    sums['noisy_pixel_coverage'] += pixel_coverage.sum().item()
                    sums['noisy_fg_coverage'] += fg_coverage.sum().item()
                count += b
            for metric, total in sums.items():
                results[metric] = total/max(count,1)
        return results
    finally:
        model.train(was_training)


def append_row(path, row, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    first = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if first:
            writer.writeheader()
        writer.writerow(row)


def train_feedback(model, cfg):
    if not cfg.train.curriculum.get('enable', False):
        raise ValueError('Feedback version requires curriculum.enable=True')
    feedback_cfg = cfg.train.feedback
    controller = FeedbackController(feedback_cfg)
    probe_clean = make_probe_loader(cfg, 'clean', int(feedback_cfg.clean_probe_size), int(cfg.base_seed) + 123)
    probe_noisy = make_probe_loader(cfg, 'noisy', int(feedback_cfg.noisy_probe_size), int(cfg.base_seed) + 456)
    streams = None
    sampler_for_epoch = {}
    epoch_proportions = {}
    original_train = BASE_TRAIN
    original_build = legacy._build_epoch_loader
    original_write = legacy._append_dynamic_epoch_losses
    num_requested = int(cfg.train.curriculum.num_samples_per_epoch)
    bs = int(cfg.train.batch_size)
    if num_requested % bs:
        raise ValueError('num_samples_per_epoch must be divisible by batch_size')

    def build_loader(cfg, pool_datasets, epoch):
        nonlocal streams
        if streams is None:
            streams = {name:PersistentPoolStream(len(pool_datasets[name]), int(cfg.base_seed)+j*10007)
                       for j,name in enumerate(('clean','noisy'))}
        for ds in pool_datasets.values():
            ds.set_epoch(epoch)
        pN = float(controller.p_noisy)
        sampler = FeedbackBatchSampler(
            streams=streams, noisy_prob=pN, batch_size=bs,
            num_batches=num_requested//bs,
            min_clean_per_batch=int(feedback_cfg.min_clean_per_batch),
            seed=int(cfg.base_seed)+epoch*1009)
        sampler_for_epoch[epoch] = sampler
        epoch_proportions[epoch] = pN
        joined = torch.utils.data.ConcatDataset([pool_datasets['clean'],pool_datasets['noisy']])
        loader = DataLoader(joined, batch_sampler=sampler,
                            num_workers=int(cfg.train.num_workers), pin_memory=True,
                            worker_init_fn=(core.pt_utils.customized_worker_init_fn
                                            if cfg.use_custom_worker_init else None))
        return loader, 'FGAC_'+controller.last_action, 1.0-pN, pN

    def write_epoch(csv_path, epoch, stage_name, loss_sums, num_batches):
        averages = original_write(csv_path=csv_path, epoch=epoch,
                                  stage_name=stage_name, loss_sums=loss_sums,
                                  num_batches=num_batches)
        s = sampler_for_epoch[epoch]
        counts = dict(s.counts)
        actual_n = counts.get('noisy',0)/max(sum(counts.values()),1)
        schedule_row = dict(epoch=epoch, action_used=stage_name,
                            target_clean=1-epoch_proportions[epoch], target_noisy=epoch_proportions[epoch],
                            actual_clean=1-actual_n, actual_noisy=actual_n,
                            clean_exposures=counts.get('clean',0), noisy_exposures=counts.get('noisy',0),
                            clean_unique=len(s.seen['clean']), noisy_unique=len(s.seen['noisy']),
                            clean_cycles=streams['clean'].cycles, noisy_cycles=streams['noisy'].cycles)
        append_row(os.path.join(cfg.path.pth_log,'feedback_sampling_actual.csv'), schedule_row, list(schedule_row))
        LOGGER.info('FGAC Epoch %03d exposure C=%d (%d unique) N=%d (%d unique) pN=%.4f',
                    epoch, counts.get('clean',0),len(s.seen['clean']), counts.get('noisy',0),
                    len(s.seen['noisy']), actual_n)
        if epoch % int(feedback_cfg.probe_interval) == 0:
            metrics = evaluate_feedback_probe(model, cfg, probe_clean, probe_noisy)
            row = controller.observe(metrics,epoch)
            row['epoch'] = epoch
            append_row(os.path.join(cfg.path.pth_log,'feedback_decisions.csv'), row,
                       ['epoch','clean_loss','clean_dice','consistency','fg_area_ratio',
                        'noisy_pixel_coverage','noisy_fg_coverage','ema_clean_loss',
                        'ema_clean_dice','ema_consistency','ema_fg_area_ratio',
                        'delta_clean_loss','delta_consistency','action','reason','next_noisy_prob'])
            LOGGER.info('FGAC Probe Epoch %03d loss=%.4f dice=%.4f stab=%.4f area=%.3f '
                        'N_fg=%.3f -> %s next pN=%.3f',
                        epoch, metrics['clean_loss'],metrics['clean_dice'],metrics['consistency'],
                        metrics['fg_area_ratio'],metrics['noisy_fg_coverage'],
                        row['action'],controller.p_noisy)
            cfg.tb_logger.write_to_tb('feedback/clean_dice',metrics['clean_dice'],epoch)
            cfg.tb_logger.write_to_tb('feedback/consistency',metrics['consistency'],epoch)
            cfg.tb_logger.write_to_tb('feedback/next_noisy_prob',controller.p_noisy,epoch)
            with open(os.path.join(cfg.path.pth_log,'feedback_controller_state.json'),'w',encoding='utf-8') as f:
                json.dump(dict(epoch=epoch,controller=controller.state_dict()),f,indent=2)
        return averages

    legacy._build_epoch_loader = build_loader
    legacy._append_dynamic_epoch_losses = write_epoch
    try:
        return original_train(model, cfg)
    finally:
        legacy._build_epoch_loader = original_build
        legacy._append_dynamic_epoch_losses = original_write


def main():
    # Same argument parsing, model instantiation and evaluation as the original.
    cfg = core.parse_cfg()
    shutil.copy(__file__, cfg.path.trainer_copy)
    core.pt_utils.initialize_seed_cudnn(
        seed=cfg.base_seed, deterministic=cfg.deterministic)

    model_class = core.model_zoo.__dict__.get(cfg.model_name)
    assert model_class is not None, "Please check your --model-name"
    model = model_class(
        num_frames=1,
        pretrained=cfg.pretrained,
        use_checkpoint=cfg.use_checkpoint,
        **dict(cfg.get('model', {})),
    )
    LOGGER.info(inspect.getsource(model_class))
    model.to(cfg.device)
    if cfg.load_from:
        # This loads model weights ONLY (not optimizer or feedback state).
        core.io.load_weight(model=model, load_path=cfg.load_from, strict=True)
    LOGGER.info('Number of Parameters: %d',
                sum(p.numel() for p in model.parameters(recurse=True)))
    if not cfg.evaluate:
        train_feedback(model, cfg)
    if cfg.evaluate or cfg.has_test:
        core.io.save_weight(model=model, save_path=cfg.path.final_state_net)
        core.test(model=model, cfg=cfg)
    LOGGER.info('End training...')


if __name__ == '__main__':
    main()
