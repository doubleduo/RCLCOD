# -*- coding: utf-8 -*-
"""
PVTv2-B4 + FPN + COD-adapted RFGM (C4 only).

Ablation principle
------------------
Keep everything in PvtV2B4_FPN_Baseline unchanged except:

    lateral_4(c4) -> CODRFGM -> original FPN fusion

This isolates whether frequency-domain residual preservation helps COD.

Important implementation details
--------------------------------
1. FFT is computed in float32 for AMP compatibility.
2. Amplitude stays non-negative through multiplicative residual modulation.
3. Phase is represented by sin/cos and only receives a bounded residual.
4. Frequency branch uses a bounded learnable residual scale initialized to 0.1.
5. No extra edge loss / NC loss / cross-stage top-k guidance is added here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline


class CODRFGM(nn.Module):
    """Stable COD adaptation of a residual frequency guidance module.

    Input / output:
        x: [B, C, H, W]
        y: [B, C, H, W]

    The module separates FFT amplitude and phase:
      - amplitude: bounded multiplicative modulation
      - phase: bounded residual correction based on sin/cos encoding
      - output: residual reconstruction with a bounded learnable scale
    """

    def __init__(
        self,
        channels: int,
        amplitude_gain: float = 0.5,
        max_phase_ratio: float = 0.25,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive.")
        if amplitude_gain < 0:
            raise ValueError("amplitude_gain must be >= 0.")
        if not 0.0 <= max_phase_ratio <= 1.0:
            raise ValueError("max_phase_ratio must be in [0, 1].")
        if not 0.0 < residual_scale_init < 1.0:
            raise ValueError("residual_scale_init must be in (0, 1).")

        self.channels = int(channels)
        self.amplitude_gain = float(amplitude_gain)
        self.max_phase_ratio = float(max_phase_ratio)

        self.in_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        # Operate on log-amplitude for better numerical range.
        self.mag_net = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )

        # sin(phi) + cos(phi) => 2C channels.
        self.phase_net = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )

        self.out_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        # Bounded scale in (0, 1), initialized to residual_scale_init.
        init_logit = math.log(
            residual_scale_init / (1.0 - residual_scale_init)
        )
        self.residual_scale_logit = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Keep the frequency branch close to identity at initialization.
        nn.init.zeros_(self.mag_net[-1].weight)
        nn.init.zeros_(self.mag_net[-1].bias)
        nn.init.zeros_(self.phase_net[-1].weight)
        nn.init.zeros_(self.phase_net[-1].bias)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"CODRFGM expects BCHW input, got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"CODRFGM expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )

        identity = x
        input_dtype = x.dtype

        # Projection can stay in the current autocast dtype.
        x_proj = self.in_proj(x)

        # FFT is safer in float32 under AMP.
        x_fft = x_proj.float()

        freq = torch.fft.rfft2(
            x_fft,
            dim=(-2, -1),
            norm="ortho",
        )

        mag = torch.abs(freq)
        phase = torch.angle(freq)

        # ---------------------------------------------------------
        # 1) Amplitude residual modulation
        # ---------------------------------------------------------
        mag_feature = torch.log1p(mag)

        mag_delta = self.mag_net(
            mag_feature
        )

        # scale range:
        #   [1-amplitude_gain, 1+amplitude_gain]
        # amplitude therefore remains non-negative.
        mag_scale = (
            1.0
            + self.amplitude_gain
            * torch.tanh(mag_delta)
        )

        mag_new = mag * mag_scale

        # ---------------------------------------------------------
        # 2) Bounded phase residual
        # ---------------------------------------------------------
        phase_feature = torch.cat(
            [
                torch.sin(phase),
                torch.cos(phase),
            ],
            dim=1,
        )

        phase_delta = (
            self.max_phase_ratio
            * math.pi
            * torch.tanh(
                self.phase_net(phase_feature)
            )
        )

        phase_new = phase + phase_delta

        # ---------------------------------------------------------
        # 3) Reconstruct spatial feature
        # ---------------------------------------------------------
        real = mag_new * torch.cos(phase_new)
        imag = mag_new * torch.sin(phase_new)

        freq_new = torch.complex(real, imag)

        restored = torch.fft.irfft2(
            freq_new,
            s=x_proj.shape[-2:],
            dim=(-2, -1),
            norm="ortho",
        )

        restored = restored.to(dtype=input_dtype)
        restored = self.out_proj(restored)

        alpha = torch.sigmoid(
            self.residual_scale_logit
        ).to(
            device=restored.device,
            dtype=restored.dtype,
        )

        return identity + alpha * restored


class PvtV2B4_FPN_RFGM(PvtV2B4_FPN_Baseline):
    """Plain PVTv2-B4 FPN with CODRFGM inserted at C4 lateral feature only.

    Baseline path:
        C5 -> P5
        C4 + up(P5) -> P4
        C3 + up(P4) -> P3
        C2 + up(P3) -> P2

    This ablation changes only:
        L4 = lateral_4(C4)
        L4 = CODRFGM(L4)

    Loss, predictor, optimizer grouping and curriculum remain inherited
    from PvtV2B4_FPN_Baseline.
    """

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        rfgm_amplitude_gain=0.5,
        rfgm_max_phase_ratio=0.25,
        rfgm_residual_scale_init=0.1,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

        self.rfgm_4 = CODRFGM(
            channels=fpn_dim,
            amplitude_gain=rfgm_amplitude_gain,
            max_phase_ratio=rfgm_max_phase_ratio,
            residual_scale_init=rfgm_residual_scale_init,
        )

    def body(self, data):
        image = data["image_m"]
        c2, c3, c4, c5 = self.normalize_encoder(image)

        # Keep C2/C3/C5 exactly identical to the FPN baseline.
        l2 = self.lateral_2(c2)
        l3 = self.lateral_3(c3)
        l5 = self.lateral_5(c5)

        # The only architectural change in this ablation.
        l4 = self.lateral_4(c4)
        l4 = self.rfgm_4(l4)

        p5 = self.smooth_5(l5)

        p4 = self.smooth_4(
            l4
            + self._resize_like(
                p5,
                c4,
            )
        )

        p3 = self.smooth_3(
            l3
            + self._resize_like(
                p4,
                c3,
            )
        )

        p2 = self.smooth_2(
            l2
            + self._resize_like(
                p3,
                c2,
            )
        )

        logits = self.predictor(p2)

        return F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


__all__ = [
    "CODRFGM",
    "PvtV2B4_FPN_RFGM",
]
