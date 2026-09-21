# -*- coding: utf-8 -*-
"""
Noisy-COD ANet A3 configuration.

Ablations:
    A0: original Noisy-COD routing
        HH -> F1/F2, LL -> F3/F4
    A1: equal four-band fusion
        mean(LL, LH, HL, HH) -> all stages
    A2: stage-wise spatial softmax routing over LL/LH/HL/HH
    A3: prior-preserving residual routing
        shallow: HH + alpha * (Adaptive - HH)
        deep:    LL + alpha * (Adaptive - LL)

A3 starts exactly from A0 because alpha is initialized to 0.
"""

cfg = dict(
    experiment=dict(
        name="NoisyCOD_ANet_A3_Top400",
        seed=2024,
        ratio=20,
        expected_total=4040,
        strict_total=True,
        official_ratio_only=True,
    ),

    data=dict(
        image_dir="./data/Train/Imgs",
        image_suffix=".jpg",
        mask_dir="./data/Train/GT",
        mask_suffix=".png",
        box_dir="./data/Train/box_label",
        box_suffix=".json",
        box_format="labelme_json",
        edge_dir=None,
        edge_suffix=".png",
    ),

    model=dict(
        channels=64,
        backbone_name="convnext_base.fb_in22k_ft_in1k_384",

        # ----------------------------------------------------------
        # Frequency ablation
        # ----------------------------------------------------------
        ablation="A3",
        router_hidden=32,
        router_temperature=1.0,

        # A3: four independent stage residual strengths.
        # 0.0 => the network starts exactly as A0.
        router_alpha_init=0.0,
    ),

    train=dict(
        image_size=384,
        epochs=100,
        batch_size=16,
        num_workers=6,

        # Clean pseudo-label mode. Keep None for released ratio split.
        sample_list="data/pseudo_pool/Top400.txt",
        target_mask_dir="data/Train/S1_GT",
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


    logging=dict(
        # Also write step summaries into train.log every N train iterations.
        # 0 disables step-level file logging; tqdm still updates every step.
        print_freq=20,

        # Router statistics are aggregated over the TRAIN batches of the epoch.
        frequency_interval=1,
        save_frequency_csv=True,
        stats_dir="frequency_stats",
    ),

    output=dict(
        root="./ANet_outputs/DWT_A3_Top400",
        split_dir="splits",
        checkpoint_dir="checkpoints",
        pseudo_mask_dir="pseudo_mask",
        pseudo_edge_dir="pseudo_edge",
        log_file="train.log",
    ),
)
