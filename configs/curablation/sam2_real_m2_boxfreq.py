# -*- coding: utf-8 -*-
"""Exact inheritance of existing configs/sam2_real.py, with box-mask loading.

Use the SAME config for M2, M2_BoxHead, M2_BoxFreq, M2_BoxContrast.
The MMEngine config system recursively merges train.data.use_box=True.
All other properties (pools, schedule, LR, epochs, validation sets) inherit.
"""
_base_ = '../sam2_real.py'
train = dict(data=dict(use_box=True))
