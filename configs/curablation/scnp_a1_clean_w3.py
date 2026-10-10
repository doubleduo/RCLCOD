_base_ = ["./csr_mhsiu_nc_real.py"]

# A1 (recommended): delayed, weak SCNP only on Clean pseudo labels.
train = dict(
    scnp=dict(
        enable=True,
        apply_to=["clean"],
        kernel_size=3,
        large_value=1e4,
        start_epoch=11,
        ramp_epochs=10,
        weight=0.25,
    ),
)

