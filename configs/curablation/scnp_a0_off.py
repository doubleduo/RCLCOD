_base_ = ["./csr_mhsiu_nc_real.py"]

# A0: exact original curriculum baseline.
train = dict(
    scnp=dict(enable=False),
)

