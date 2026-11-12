#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate fair GT/pseudo-label subset controls for RCLCOD.

Run from the RCLCOD repository root.  No original files are modified.

Examples
--------
python make_gt_coverage_experiment.py --size 3000 --seed 112358
python make_gt_coverage_experiment.py --size 3000 --seed 112358 \
    --sam-list data/pseudo_pool/sam1/high.txt \
    --sam-list data/pseudo_pool/sam1/middle.txt --stats

Why two identical lists? basemain_continuous.py insists on clean/noisy pools;
when their name sets are identical, changing pool weights DOES NOT change
the probability assigned to any individual image.  The sampler still takes
4040 draws (with replacement) per epoch as in the original configuration.
"""

import argparse
import csv
import json
import random
import shlex
from pathlib import Path

import yaml


def absolute_path(path):
    return Path(path).expanduser().resolve()


def normalize_name(text):
    """Compatible with the repository's _normalize_sample_name for txt lists."""
    return Path(str(text).strip().split()[0].split(',')[0].replace('\\', '/')).stem


def read_name_list(path):
    path = absolute_path(path)
    if not path.is_file():
        raise FileNotFoundError(f'SAM selection list not found: {path}')
    names = set()
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        names.add(normalize_name(line))
    if not names:
        raise ValueError(f'No usable sample names in {path}')
    return names


def write_list(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(name + '\n' for name in sorted(names)), encoding='utf-8')


def collect_stems(directory, suffix):
    if not directory.is_dir():
        raise FileNotFoundError(f'Directory not found: {directory}')
    return {p.stem for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() == suffix.lower()}


def dataset_directory(repo_root, dataset_entry, key):
    root = Path(dataset_entry.get('root', '.'))
    path = Path(dataset_entry[key]['path'])
    if path.is_absolute():
        return path.resolve()
    return (repo_root / root / path).resolve()


def dump_dataset_yaml(original, path, mask_dir):
    output = dict(original)
    output['combined_tr'] = dict(original['combined_tr'])
    output['combined_tr']['mask'] = dict(original['combined_tr']['mask'])
    output['combined_tr']['mask']['path'] = str(mask_dir)
    output['combined_tr']['mask']['suffix'] = '.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(output, sort_keys=False, allow_unicode=True), encoding='utf-8')


def make_config(base_text, config_path, list_path, draws_per_epoch):
    # Copy the full original config, and OVERRIDE just the sampling list and draw count.
    # Both pools contain the same names. For any positive pool masses pC, pN:
    # P(image i) = pC/N + pN/N = 1/N after normalizing by pC+pN.
    extra = ("\n\n# GT/coverage experiment (automatically generated)\n"
             "# Identical lists for both pools => uniform image sampling.\n"
             f"train['curriculum']['pools'] = dict(clean={str(list_path)!r}, "
             f"noisy={str(list_path)!r})\n"
             f"train['curriculum']['num_samples_per_epoch'] = {draws_per_epoch}\n")
    config_path.write_text(base_text.rstrip() + extra, encoding='utf-8')


def mask_features(gt_dir, name):
    # cv2 is already a training dependency in the RCLCOD repository.
    import cv2
    import numpy as np

    arr = cv2.imread(str(gt_dir / f'{name}.png'), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f'Unreadable GT: {gt_dir / (name + ".png")}')
    mask = (arr > 0).astype('uint8')
    h, w = mask.shape
    area = int(mask.sum())
    if area == 0:
        return dict(fg_ratio=0.0, bbox_ratio=0.0, center_offset=0.0,
                    num_components=0, boundary_complexity=0.0)
    ys, xs = np.where(mask != 0)
    bbox_ratio = ((int(xs.max() - xs.min() + 1) * int(ys.max() - ys.min() + 1))
                  / float(h * w))
    # 0 = image center, 1 ~= image corner.
    center_offset = (float(((xs.mean()/w - 0.5)**2 + (ys.mean()/h - 0.5)**2)**0.5)
                     / (2**0.5 * 0.5))
    n_components, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Suppress tiny GT speckles: a significant component >=5% of FG or 16px.
    significant = max(16, area * 0.05)
    components = sum(int(stats[j, cv2.CC_STAT_AREA]) >= significant
                     for j in range(1, n_components))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    complexity = perimeter / max(2.0 * (3.1415926535 * area)**0.5, 1.0)
    return dict(fg_ratio=area / float(h * w), bbox_ratio=bbox_ratio,
                center_offset=center_offset, num_components=components,
                boundary_complexity=complexity)


def write_coverage_stats(directory, gt_dir, all_names, subsets):
    from statistics import mean, median
    feature_file = directory / 'gt_per_image_features.csv'
    all_features = {}
    with feature_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'fg_ratio', 'bbox_ratio',
            'center_offset', 'num_components', 'boundary_complexity'])
        writer.writeheader()
        for idx, name in enumerate(sorted(all_names), 1):
            row = mask_features(gt_dir, name)
            all_features[name] = row
            writer.writerow({'name': name, **row})
            if idx % 1000 == 0:
                print(f'[GT features] {idx}/{len(all_names)}')

    report_file = directory / 'coverage_group_summary.csv'
    with report_file.open('w', newline='', encoding='utf-8') as f:
        fields = ['group','n','fg_mean','fg_median','small_fg_under_3pct',
                  'center_offset_mean','off_center_over_035',
                  'multi_significant_components','complexity_mean']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group, names in subsets.items():
            rows = [all_features[n] for n in names]
            writer.writerow({
                'group': group, 'n': len(rows),
                'fg_mean': round(mean(r['fg_ratio'] for r in rows), 6),
                'fg_median': round(median(r['fg_ratio'] for r in rows), 6),
                'small_fg_under_3pct': round(mean(r['fg_ratio'] < .03 for r in rows), 6),
                'center_offset_mean': round(mean(r['center_offset'] for r in rows), 6),
                'off_center_over_035': round(mean(r['center_offset'] > .35 for r in rows), 6),
                'multi_significant_components': round(mean(r['num_components'] >= 2 for r in rows), 6),
                'complexity_mean': round(mean(r['boundary_complexity'] for r in rows), 6),
            })
    print('Saved GT coverage features:', feature_file)
    print('Saved GT group summary:', report_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-yaml', default='dataset.yaml')
    parser.add_argument('--base-config', default='configs/sam1/sam1candidate.py')
    parser.add_argument('--gt-dir', default='data/Train/GT')
    parser.add_argument('--pseudo-dir', default=None,
                        help='Default: the original combined_tr.mask path in dataset.yaml')
    parser.add_argument('--sam-list', action='append', default=[],
                        help='SAM kept-name txt list; repeat for multiple kept groups')
    parser.add_argument('--size', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=112358)
    parser.add_argument('--draws-per-epoch', type=int, default=4040)
    parser.add_argument('--out-dir', default='experiments/gt_coverage')
    parser.add_argument('--stats', action='store_true',
                        help='Measure GT object scale/position/components across subsets')
    args = parser.parse_args()

    if args.size <= 0 or args.draws_per_epoch <= 0:
        parser.error('--size and --draws-per-epoch must be positive')

    repo_root = Path.cwd().resolve()
    dataset_path = absolute_path(args.dataset_yaml)
    base_config_path = absolute_path(args.base_config)
    gt_dir = absolute_path(args.gt_dir)
    out_dir = absolute_path(args.out_dir)
    if not dataset_path.is_file() or not base_config_path.is_file():
        raise FileNotFoundError('Run from RCLCOD root; check --dataset-yaml and --base-config')
    dataset = yaml.safe_load(dataset_path.read_text(encoding='utf-8'))
    tr = dataset['combined_tr']
    image_dir = dataset_directory(repo_root, tr, 'image')
    image_suffix = tr['image']['suffix']
    if args.pseudo_dir:
        pseudo_dir = absolute_path(args.pseudo_dir)
    else:
        pseudo_dir = dataset_directory(repo_root, tr, 'mask')

    image_stems = collect_stems(image_dir, image_suffix)
    gt_stems = collect_stems(gt_dir, '.png')
    universe = image_stems & gt_stems
    if len(universe) < args.size:
        raise ValueError(f'Only {len(universe)} matching RGB+GT images, need {args.size}')
    print(f'RGB={len(image_stems)}, GT={len(gt_stems)}, matched={len(universe)}')
    rng = random.Random(args.seed)
    random_subset = set(rng.sample(sorted(universe), args.size))
    subsets = {'full_gt': universe, 'random_gt': random_subset}
    analysis_groups = {f'full_{len(universe)}': universe, f'random_{args.size}': random_subset}

    if args.sam_list:
        sam_names = set().union(*(read_name_list(p) for p in args.sam_list))
        missing = sam_names - universe
        if missing:
            raise ValueError(f'{len(missing)} SAM-list names lack RGB+GT. Examples: {sorted(missing)[:5]}')
        if len(sam_names) < args.size:
            raise ValueError(f'SAM kept union has only {len(sam_names)} names < requested {args.size}. '
                             'Use --size <= kept count for a matched-size control.')
        selected = set(random.Random(args.seed).sample(sorted(sam_names), args.size))
        subsets['selected_gt'] = selected
        excluded = universe - sam_names
        analysis_groups[f'sam_selected_{args.size}'] = selected
        analysis_groups['sam_kept_all'] = sam_names
        if excluded:
            analysis_groups['sam_excluded'] = excluded
            write_list(out_dir / 'sam_excluded_names.txt', excluded)
        print(f'SAM-kept union={len(sam_names)}; selected subset={len(selected)}')
        print(f'Random/selected overlap: {len(random_subset & selected)}/{args.size}')

    # Optional pseudo-label variants, only if every selected sample has a mask.
    if pseudo_dir.is_dir():
        pseudo_stems = collect_stems(pseudo_dir, '.png')
        for prefix in ['random', 'selected']:
            gt_key = prefix + '_gt'
            if gt_key not in subsets:
                continue
            missing = subsets[gt_key] - pseudo_stems
            if missing:
                print(f'[SKIP] {prefix}_pseudo: {len(missing)} pseudo masks missing; '
                      f'examples={sorted(missing)[:3]}')
            else:
                subsets[prefix + '_pseudo'] = subsets[gt_key]
    else:
        print(f'[SKIP] Pseudo mask directory not found: {pseudo_dir}')

    lists_dir = out_dir / 'lists'
    configs_dir = out_dir / 'configs'
    lists_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    for name, names in subsets.items():
        write_list(lists_dir / f'{name}_n{len(names)}_seed{args.seed}.txt', names)

    gt_yaml_path = out_dir / 'dataset_gt.yaml'
    pseudo_yaml_path = out_dir / 'dataset_pseudo.yaml'
    dump_dataset_yaml(dataset, gt_yaml_path, gt_dir)
    if pseudo_dir.is_dir():
        dump_dataset_yaml(dataset, pseudo_yaml_path, pseudo_dir)

    base_text = base_config_path.read_text(encoding='utf-8')
    commands = []
    for name, names in subsets.items():
        list_path = lists_dir / f'{name}_n{len(names)}_seed{args.seed}.txt'
        config_path = configs_dir / f'{name}_n{len(names)}_seed{args.seed}.py'
        make_config(base_text, config_path, list_path, args.draws_per_epoch)
        data_cfg = gt_yaml_path if name.endswith('_gt') else pseudo_yaml_path
        # Explicitly pass --data-cfg so masks cannot accidentally come from SAM.
        output_dir = f'outputs/gt_coverage/{name}_n{len(names)}_seed{args.seed}'
        flags = ['python', 'basemain_continuous.py', '--config', str(config_path),
                 '--data-cfg', str(data_cfg), '--model-name',
                 'PvtV2B4_FPN_Baseline', '--output-dir', output_dir]
        commands.append((name, shlex.join(flags)))

    run_sh = out_dir / 'run_commands.sh'
    run_sh.write_text('#!/usr/bin/env bash\nset -e\n\n' +
        '\n\n'.join(f'# {name}\n{command}' for name, command in commands) + '\n',
        encoding='utf-8')
    run_sh.chmod(0o755)

    summary = {'seed': args.seed, 'size': args.size,
               'draws_per_epoch': args.draws_per_epoch,
               'images_with_gt': len(universe), 'base_config': str(base_config_path),
               'gt_dir': str(gt_dir), 'pseudo_dir': str(pseudo_dir),
               'sam_lists': [str(absolute_path(p)) for p in args.sam_list],
               'groups': {k: len(v) for k, v in subsets.items()},
               'overlap_random_selected': len(subsets['random_gt'] & subsets['selected_gt'])
                   if 'selected_gt' in subsets else None}
    (out_dir / 'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    if args.stats:
        write_coverage_stats(out_dir, gt_dir, universe, analysis_groups)

    print('\nGenerated commands (choose the experiments you need):')
    for name, command in commands:
        print(f'\n[{name}]\n{command}')
    print(f'\nAll commands saved at: {run_sh}')
    print('IMPORTANT: In each generated config clean/noisy lists are intentionally identical.')


if __name__ == '__main__':
    main()
