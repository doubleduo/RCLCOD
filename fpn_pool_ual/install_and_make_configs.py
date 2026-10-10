#!/usr/bin/env python3
"""Run at RCLCOD repository root. Installs isolated FPN UAL class, writes configs.

No user files are overwritten except the dedicated model file. Existing
methods/__init__.py is backed up before first modification and appended to once.
"""
import argparse
from pathlib import Path
import shutil


TRIALS = {
    "a0_no_ual":      dict(high_ual_weight=0.0, middle_ual_mode="none", middle_ual_weight=0.0),
    "a1_high_only":   dict(high_ual_weight=0.2, middle_ual_mode="none", middle_ual_weight=0.0),
    "a2_both_full":   dict(high_ual_weight=0.2, middle_ual_mode="full", middle_ual_weight=0.2),
    "a3_middle_low":  dict(high_ual_weight=0.2, middle_ual_mode="low",  middle_ual_weight=0.05),
    "a4_middle_edge": dict(high_ual_weight=0.2, middle_ual_mode="edge", middle_ual_weight=0.05),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--high-lambda", type=float, default=0.2)
    ap.add_argument("--low-lambda", type=float, default=0.05)
    ap.add_argument("--edge-kernel", type=int, default=7)
    ap.add_argument("--pool-high", default="./data/pseudo_pool/sam2refine/high.txt")
    ap.add_argument("--pool-middle", default="./data/pseudo_pool/sam2refine/middle.txt")
    args=ap.parse_args()
    root=Path.cwd()
    assert (root/"methods/fpn_baseline.py").exists(), "Run from RCLCOD root"
    assert (root/"configs/sam2_real.py").exists(), "Missing configs/sam2_real.py"
    src=Path(__file__).resolve().parent / "fpn_pool_ual.py"
    dst=root/"methods/fpn_pool_ual.py"
    shutil.copyfile(src,dst)
    init=root/"methods/__init__.py"
    content=init.read_text(encoding="utf-8")
    declaration="from .fpn_pool_ual import PvtV2B4_FPN_PoolUAL"
    if declaration not in content:
        backup=root/"methods/__init__.py.before_fpn_pool_ual"
        if not backup.exists(): shutil.copyfile(init,backup)
        init.write_text(content.rstrip()+"\n\n"+declaration+"\n",encoding="utf-8")
    base=(root/"configs/sam2_real.py").read_text(encoding="utf-8")
    output=root/"configs/fpn_pool_ual"
    output.mkdir(parents=True,exist_ok=True)
    for name,spec in TRIALS.items():
        p=dict(spec)
        if name!="a0_no_ual": p["high_ual_weight"]=args.high_lambda
        if name=="a2_both_full":p["middle_ual_weight"]=args.high_lambda
        elif name in ("a3_middle_low","a4_middle_edge"):p["middle_ual_weight"]=args.low_lambda
        code=f"""\n# === generated pool-aware UAL ablation; same base FPN and full BCE ===
model = {dict(p,edge_kernel=args.edge_kernel,ual_schedule='cosine')!r}
train['num_epochs'] = {args.epochs}
train['curriculum']['pools']['clean'] = {args.pool_high!r}
train['curriculum']['pools']['noisy'] = {args.pool_middle!r}
train['curriculum']['continuous_schedule']['clean_weight'] = 1.0
train['curriculum']['continuous_schedule']['noisy'].update(dict(
    start_weight=1.0, end_weight=1.0,
    hold_end_epoch={args.epochs}, anneal_start_epoch={args.epochs+1},
    anneal_end_epoch={args.epochs+2}, mode='cosine',
))
train['curriculum']['continuous_schedule']['final_start_epoch'] = {args.epochs+1}
train['data']['use_box'] = False
"""
        path=output/f"{name}.py"
        path.write_text(base+"\n"+code,encoding="utf-8")
        print("Wrote",path)
    print("Installed model",dst)
    print("Registered",declaration)
    print("Run with: bash run_fpn_pool_ual.sh a1_high_only")


if __name__=="__main__":
    main()
