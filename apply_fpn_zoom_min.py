# -*- coding: utf-8 -*-
"""Run ONCE from the RCLCOD repository root, after unzipping this archive.

Safely registers the new FPN ZoomMini class and fixes the existing two-step
384->576 resize in the basemain.py image loaders. Other code is untouched.
The script is idempotent and verifies all expected match counts beforehand.
"""
from pathlib import Path

root = Path.cwd()
model_file = root / "methods/fpn_zoom_min.py"
conf_file = root / "configs/curablation/fpn_zoom_384.py"
if not model_file.is_file() or not conf_file.is_file():
    raise SystemExit("Run in RCLCOD root AFTER extracting ZIP with folder structure.")

loader = root / "basemain.py"
registry = root / "methods/__init__.py"
if not loader.is_file() or not registry.is_file():
    raise SystemExit("Cannot find RCLCOD basemain.py / methods/__init__.py")

source = loader.read_text(encoding="utf-8")
entry = '        image = ops.resize(image, height=base_h, width=base_w)\n'
count = source.count(entry)
if count not in (0, 2):
    raise SystemExit(f"Unexpected image resize pattern count {count}, expect 2 or 0. No changes made.")

source_reg = registry.read_text(encoding="utf-8")
add_line = "from .fpn_zoom_min import PvtV2B4_FPN_ZoomMini"
if add_line not in source_reg and 'PvtV2B4_FPN_ZoomMini' in source_reg:
    raise SystemExit("ZoomMini seems registered differently; check methods/__init__.py manually.")

if count == 2:
    # Before: original -> 384 -> ms_resize(192,384,576).
    # After : original -> ms_resize(192,384,576) directly in both datasets.
    loader.write_text(source.replace(entry, ""), encoding="utf-8")
    print("Updated basemain.py: independent native-size multiscale inputs for train + test")
else:
    print("basemain.py resizing already updated")

if add_line not in source_reg:
    registry.write_text(source_reg.rstrip() + "\n\n" + add_line + "\n", encoding="utf-8")
    print("Registered PvtV2B4_FPN_ZoomMini in methods/__init__.py")
else:
    print("ZoomMini already registered")
print("Installation ready")
