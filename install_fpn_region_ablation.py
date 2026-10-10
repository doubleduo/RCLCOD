#!/usr/bin/env python3
# coding: utf-8
"""Install minimal FPN region curriculum ablations in an existing RCLCOD tree.

Usage:
    python install_fpn_region_ablation.py --epochs 40
    python install_fpn_region_ablation.py --epochs 150

Run from RCLCOD root, with fpn_region_ablation.py beside this installer.
Idempotent: does NOT modify existing fpn_baseline.py or training engine.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import shutil

IMPORT_LINE = 'from .fpn_region_ablation import PvtV2B4_FPN_RegionAblation\n'


def build_config(mode: str, epochs: int, high: str | None, middle: str | None) -> str:
    pools = ''
    if high or middle:
        parts = []
        if high:
            parts.append('clean=' + repr(high))
        if middle:
            parts.append('noisy=' + repr(middle))
        pools = '        pools=dict(' + ', '.join(parts) + '),\n'
    return (
        '# Minimal High/Middle FPN loss ablation. Middle has no UAL.\n'
        "_base_ = '../sam2_real.py'\n"
        'model = dict(\n'
        f'    region_mode={mode!r},\n'
        '    region_kernel=5,\n'
        '    box_margin_kernel=7,\n'
        '    adaptive_scale=0.25,\n'
        ')\n'
        'train = dict(\n'
        f'    num_epochs={epochs},\n'
        '    data=dict(use_box=True),\n'
        '    curriculum=dict(\n'
        + pools +
        '        continuous_schedule=dict(\n'
        '            enable=True, clean_weight=1.0,\n'
        '            noisy=dict(\n'
        f'                hold_end_epoch={epochs},\n'
        f'                anneal_start_epoch={epochs + 1},\n'
        f'                anneal_end_epoch={epochs + 2},\n'
        '                start_weight=1.0, end_weight=1.0, mode="cosine",\n'
        '            ),\n'
        f'            final_start_epoch={epochs + 1},\n'
        '        ),\n'
        '    ),\n'
        ')\n'
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.', help='RCLCOD repo root')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--high-list', default=None, help='optional override: High filenames list')
    p.add_argument('--middle-list', default=None, help='optional override: Middle filenames list')
    args = p.parse_args()
    if args.epochs < 1:
        p.error('--epochs must be positive')
    root = Path(args.root).resolve()
    src = Path(__file__).with_name('fpn_region_ablation.py')
    dest = root / 'methods' / 'fpn_region_ablation.py'
    init = root / 'methods' / '__init__.py'
    base = root / 'configs' / 'sam2_real.py'
    if not src.is_file():
        p.error('fpn_region_ablation.py must be next to installer')
    if not init.is_file() or not base.is_file() or not (root / 'methods' / 'fpn_baseline.py').is_file():
        p.error('RCLCOD tree not found; run in RCLCOD repository root')
    ast.parse(src.read_text(encoding='utf-8'))
    dest.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
    source = init.read_text(encoding='utf-8')
    if IMPORT_LINE.strip() not in source:
        init.write_text(source.rstrip() + '\n\n' + IMPORT_LINE, encoding='utf-8')
    out = root / 'configs' / 'rcl_fpn_region'
    out.mkdir(exist_ok=True, parents=True)
    for mode in ('baseline', 'seed', 'adaptive'):
        config = build_config(mode, args.epochs, args.high_list, args.middle_list)
        ast.parse(config)
        (out / f'fpn_{mode}.py').write_text(config, encoding='utf-8')
    print('Installed model:', dest)
    print('Registered model in:', init)
    print('Saved configs:', out)
    print('Epochs:', args.epochs)
    print('Sampler: High:Middle = 1:1 (each epoch, with replacement)')
    print('UAL: absent in FPN baseline, zero for Middle in all modes')
    print('Run: bash run_fpn_region_ablation.sh')

if __name__ == '__main__':
    main()
