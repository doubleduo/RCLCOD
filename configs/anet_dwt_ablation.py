# -*- coding: utf-8 -*-
"""
Noisy-COD ANet ratio config.

Default is the released F20 setting. The ratio can be overridden at runtime:

    python anet_main.py --config configs/anet_noisycod_ratio.py --ratio 10

Released split rule (4040 total):
    labeled_count = int(ratio * 400 / 10) = ratio * 40

So:
    F1  -> 40 GT,  4000 pseudo candidates
    F5  -> 200 GT, 3840 pseudo candidates
    F10 -> 400 GT, 3640 pseudo candidates
    F20 -> 800 GT, 3240 pseudo candidates
"""

cfg = dict(
    experiment=dict(
        name="NoisyCOD_ANet",
        seed=2024,

        # Only this value controls the GT ratio.
        # Can be overridden by --ratio from bash.
        ratio=20,

        expected_total=4040,
        strict_total=True,

        # Keep True for faithful Noisy-COD experiments.
        # When True, only 1/5/10/20 are accepted.
        official_ratio_only=True,
    ),

    data=dict(
        image_dir="./data/Train/Imgs",
        image_suffix=".jpg",

        mask_dir="./data/Train/GT",
        mask_suffix=".png",

        # LabelMe box json by default.
        box_dir="./data/Train/box_label",
        box_suffix=".json",
        box_format="labelme_json",

        # Optional precomputed edge target folder.
        # If None, edge target is generated from GT online.
        edge_dir=None,
        edge_suffix=".png",
    ),

    model=dict(
        channels=64,
        backbone_name="convnext_base.fb_in22k_ft_in1k_384",

        # ----------------------------------------------------------
        # DWT ablation
        # A0: original Noisy-COD: HH->F1/F2, LL->F3/F4
        # A1: mean(LL,LH,HL,HH) for all stages
        # A2: adaptive spatial softmax router over LL/LH/HL/HH
        # ----------------------------------------------------------
        ablation="A0",
        router_hidden=32,
        router_temperature=1.0,
    ),

    train=dict(
        image_size=384,
        epochs=100,
        batch_size=16,
        num_workers=6,

        # --------------------------------------------------------------
        # Clean pseudo-label training mode.
        # Leave sample_list=None to use the original ratio/GT split.
        # Set these two paths to train only on samples listed in a TXT.
        # TXT may contain stems (CAMO_xxx), image filenames, or full paths.
        # --------------------------------------------------------------
        sample_list=None,
        target_mask_dir=None,
        target_mask_suffix=".png",

        optimizer="adam",
        init_lr=1e-7,
        top_epoch=10,
        top_lr=5e-4,
        min_lr=1e-7,

        edge_loss_weight=4.0,
        ual_loss_weight=2.0,

        amp=True,
        augment=True,
        strict_albumentations_v1=False,

        validate_every=10,
        save_last_epochs=10,
    ),

    generate=dict(
        image_size=384,
        batch_size=12,
        num_workers=4,
    ),

    output=dict(
        # This is the default final output directory.
        # It can be completely overridden by --output-root.
        root="./ANet_outputs/NoisyCOD_ANet_F20",
        split_dir="splits",
        checkpoint_dir="checkpoints",
        pseudo_mask_dir="pseudo_mask",
        pseudo_edge_dir="pseudo_edge",
        log_file="train.log",
    ),
)
