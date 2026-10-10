_base_ = ["./csr_mhsiu_nc_real.py"]

# A3 (diagnostic, not recommended): tests whether Noisy labels are amplified.
train = dict(
    scnp=dict(
        enable=True,
        apply_to=["clean", "noisy"],
        kernel_size=3,
        large_value=1e4,
        start_epoch=11,
        ramp_epochs=10,
        weight=0.25,
    ),
)

