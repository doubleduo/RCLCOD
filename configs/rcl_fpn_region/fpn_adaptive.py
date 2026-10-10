# Minimal High/Middle FPN loss ablation. Middle has no UAL.
_base_ = '../sam2_real.py'
model = dict(
    region_mode='adaptive',
    region_kernel=5,
    box_margin_kernel=7,
    adaptive_scale=0.25,
)
train = dict(
    num_epochs=40,
    data=dict(use_box=True),
    curriculum=dict(
        continuous_schedule=dict(
            enable=True, clean_weight=1.0,
            noisy=dict(
                hold_end_epoch=40,
                anneal_start_epoch=41,
                anneal_end_epoch=42,
                start_weight=1.0, end_weight=1.0, mode="cosine",
            ),
            final_start_epoch=41,
        ),
    ),
)
