#!/usr/bin/env python3
"""Apply the two small edits required by the reliability loss patch.

Run from the RCLCOD repository root after copying this patch's new files:
    python apply_patch.py
"""
from pathlib import Path


def patch_basemain(path: Path):
    text = path.read_text(encoding="utf-8")
    marker = '        return dict(data=sample_data)\n'
    if 'sample_data["pool_id"]' in text:
        print("[skip] basemain.py already contains pool_id")
        return
    if marker not in text:
        raise RuntimeError("Cannot find ImageTrainDataset return marker in basemain.py")

    insert = '''        # Reliability supervision ID used by PvtV2B4_FPN_Reliable.\n        # 0 = Clean, 1 = Noisy, -1 = unknown/all.\n        if self.pool_name in ("clean", "clean_probe"):\n            pool_id = 0\n        elif self.pool_name == "noisy":\n            pool_id = 1\n        else:\n            pool_id = -1\n        sample_data["pool_id"] = torch.tensor(pool_id, dtype=torch.long)\n\n        return dict(data=sample_data)\n'''
    text = text.replace(marker, insert, 1)
    path.write_text(text, encoding="utf-8")
    print("[ok] patched basemain.py with pool_id")


def patch_registry(path: Path):
    text = path.read_text(encoding="utf-8")
    line = 'from .fpn_reliable_loss import PvtV2B4_FPN_Reliable\n'
    if line in text:
        print("[skip] methods/__init__.py already imports PvtV2B4_FPN_Reliable")
        return
    if not text.endswith("\n"):
        text += "\n"
    text += "\n" + line
    path.write_text(text, encoding="utf-8")
    print("[ok] registered PvtV2B4_FPN_Reliable")


def main():
    root = Path.cwd()
    basemain = root / "basemain.py"
    registry = root / "methods" / "__init__.py"
    if not basemain.exists() or not registry.exists():
        raise SystemExit("Run this script from the RCLCOD repository root.")
    patch_basemain(basemain)
    patch_registry(registry)
    print("\nPatch applied. New files should be copied to:")
    print("  methods/fpn_reliable_loss.py")
    print("  configs/curablation/fpn_rnc_a2_real.py")
    print("  configs/curablation/fpn_rnc_rsc_a3_real.py")
    print("  configs/curablation/fpn_rnc_rsc_bual_a4_real.py")


if __name__ == "__main__":
    main()
