# ConvNeXtB_ZoomNeXt_ANet training config.
#
# Important:
#   ANet training is NOT B100 box-only training.
#   It needs a fully annotated subset: RGB + dense GT + box.
#   After training, use `anet_main.py --generate` on RGB + box samples to
#   generate dense pseudo labels.

base_seed = 112358
deterministic = True
use_custom_worker_init = True
log_interval = 20
model_type = "anet"



model = dict(
    mid_dim=64,
    hmu_groups=6,

    # Noisy-COD-style ANet loss weights.
    edge_loss_weight=4.0,
    ual_loss_weight=2.0,
    ual_start=0.0,
    ual_full=1.0,
)

train = dict(
    # Two ConvNeXt-B encoders are heavy. Start from BS=4 on a 48GB GPU.
    batch_size=4,
    num_workers=8,
    use_amp=True,

    # Noisy-COD ANet is trained strongly; 200 is the faithful starting point.
    num_epochs=40,
    grad_acc_step=1,

    # Set to 0.01 / 0.05 / 0.10 / 0.20 to reproduce F1/F5/F10/F20-like
    # fully annotated subsets. 1.0 uses every image with a dense mask.
    # For clean-pseudo ANet, usually keep all samples in clean_list and let
    # clean_list control the subset. Set <1 only if you intentionally want
    # another random subsample.
    sample_ratio=0.6,

    # Supervision target. Add `pseudo_mask` to dataset.yaml.
    target_key="pseudo_mask",

    # Only these pseudo labels are treated as ANet supervision.
    

    # False for binary/hard pseudo masks; True for 0..255 soft probability maps.
    soft_target=False,

    augment=True,
    box_format="xyxy",

    lr=1e-4,
    backbone_lr_factor=0.1,
    weight_decay=1e-4,
    optimizer="adamw",

    # Per-iteration warmup + cosine decay.
    warmup_steps=500,
    min_lr_ratio=0.01,

    save_interval=5,

    data=dict(
        shape=dict(h=384, w=384),
        names=["combined_tr"],
    ),
)

generate = dict(
    batch_size=8,
    num_workers=4,
    box_format="xyxy",
    data=dict(
        shape=dict(h=384, w=384),
        names=["combined_tr"],
    ),
)


# ANet validation:
# Standard COD test GT is used ONLY to derive the box prompt and compute metrics.
# The model still receives only RGB + box_mask during validation.
val = dict(
    enable=True,
    start_epoch=5,
    interval=5,
    batch_size=8,
    num_workers=4,

    # Keep validation light for fast model selection.
    # Add "chameleon" / "nc4k" later if desired.
    data=dict(
        shape=dict(h=384, w=384),
        names=["camo_te", "cod10k_te"],
    ),

    # Save a few predicted masks for visual inspection each validation epoch.
    save_vis_n=12,

    # Ignore tiny connected components when turning test GT into box prompts.
    min_component_area=4,
)

# Metrics follow the repository evaluator naming.
metric_names = ["sm", "wfm", "mae", "em"]
