_base_ = ["./csr_mhsiu_nc_real.py"]

# A2: window-size ablation; compare with A1, especially on Thin samples.
train = dict(
    scnp=dict(
        enable=True,
        apply_to=["clean"],
        kernel_size=5,
        large_value=1e4,
        start_epoch=11,
        ramp_epochs=10,
        weight=0.25,
    ),
)

