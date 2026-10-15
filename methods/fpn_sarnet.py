# -*- coding: utf-8 -*-
"""SARNet OAA/FGC ports for the RCLCOD PVTv2-B4 FPN baseline.

This file keeps RCLCOD's model interface, backbone, optimizer grouping, and
final BCE supervision unchanged.  Only the decoder topology is varied.

The four registered variants provide a controlled ablation:

* ``PvtV2B4_FPN_SARPlain``: resolution/channel-matched plain FPN control.
* ``PvtV2B4_FPN_OAA``: official-style OAA amplify pyramid, no FGC.
* ``PvtV2B4_FPN_FGC``: plain FPN amplify pyramid followed by FGC.
* ``PvtV2B4_FPN_OAA_FGC``: official-style OAA followed by FGC.

Two deliberate engineering fixes are made to the public SARNet code:

1. OpenCV/CPU dilation is replaced with equivalent PyTorch max pooling.
2. Fixed scale factors are replaced with explicit target sizes.

The dilation branch is detached before max pooling, matching the gradient
semantics of the original ``detach().cpu().numpy()`` implementation while
remaining device agnostic and avoiding CPU/GPU synchronization.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline


Size2D = Tuple[int, int]


class ConvBNAct(nn.Sequential):
    """Conv-BN-activation used by the SARNet decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        act: str = "relu",
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        if act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "gelu":
            activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act}")

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            activation,
        )


class OAA(nn.Module):
    """Object Area Amplification from SARNet.

    Each input is first projected independently, both are resized to one
    target grid with bicubic interpolation, and the concatenated result is
    fused by a 3x3 Conv-BN-GELU block.

    Explicit ``target_size`` is preferable to the original fixed scale factor:
    it preserves the intended alignment for odd or non-square input sizes.
    """

    def __init__(
        self,
        current_channels: int,
        deeper_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.current_proj = ConvBNAct(
            current_channels,
            out_channels,
            kernel_size=3,
            act="gelu",
        )
        self.deeper_proj = ConvBNAct(
            deeper_channels,
            out_channels,
            kernel_size=3,
            act="gelu",
        )
        self.fuse = ConvBNAct(
            2 * out_channels,
            out_channels,
            kernel_size=3,
            act="gelu",
        )

    @staticmethod
    def _resize(x: torch.Tensor, target_size: Size2D) -> torch.Tensor:
        return F.interpolate(
            x,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )

    def forward(
        self,
        current: torch.Tensor,
        deeper: torch.Tensor,
        target_size: Size2D | None = None,
    ) -> torch.Tensor:
        if target_size is None:
            target_size = current.shape[-2:]

        current = self._resize(
            self.current_proj(current),
            target_size,
        )
        deeper = self._resize(
            self.deeper_proj(deeper),
            target_size,
        )
        return self.fuse(torch.cat((current, deeper), dim=1))


def morphological_dilation(
    probability: torch.Tensor,
    kernel_size: int,
    iterations: int,
) -> torch.Tensor:
    """Device-safe rectangular grayscale dilation.

    ``max_pool2d`` is equivalent to OpenCV dilation with an all-ones square
    structuring element for non-negative probability maps.
    """

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    output = probability
    padding = kernel_size // 2
    for _ in range(iterations):
        output = F.max_pool2d(
            output,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
    return output


class FGC(nn.Module):
    """Figure-Ground Conversion from SARNet.

    Background mode uses the outer ring ``dilate(P) - P`` to retrieve regions
    missed by the deeper prediction.  Foreground mode uses ``P`` directly to
    reconfirm the predicted object.  The original learnable scalar ``beta`` is
    retained without an extra sigmoid constraint.
    """

    def __init__(
        self,
        current_channels: int,
        deeper_channels: int,
        focus_background: bool,
        dilation_kernel: int = 7,
        dilation_iterations: int = 1,
    ) -> None:
        super().__init__()
        if dilation_kernel < 1 or dilation_kernel % 2 == 0:
            raise ValueError("dilation_kernel must be a positive odd integer")
        if dilation_iterations < 1:
            raise ValueError("dilation_iterations must be >= 1")

        self.focus_background = bool(focus_background)
        self.dilation_kernel = int(dilation_kernel)
        self.dilation_iterations = int(dilation_iterations)

        # Official SARNet applies the 7x7 projection before upsampling.
        self.deeper_proj = ConvBNAct(
            deeper_channels,
            current_channels,
            kernel_size=7,
            act="relu",
        )
        self.attended_proj = nn.Conv2d(
            current_channels,
            current_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.refine = nn.Sequential(
            ConvBNAct(
                2 * current_channels,
                current_channels,
                kernel_size=3,
                act="relu",
            ),
            ConvBNAct(
                current_channels,
                current_channels,
                kernel_size=3,
                act="relu",
            ),
            ConvBNAct(
                current_channels,
                current_channels,
                kernel_size=3,
                act="relu",
            ),
        )
        self.prediction = nn.Conv2d(
            current_channels,
            1,
            kernel_size=7,
            padding=3,
            bias=True,
        )
        self.beta = nn.Parameter(torch.ones(1))

    @staticmethod
    def _resize_bilinear(
        x: torch.Tensor,
        target_size: Size2D,
    ) -> torch.Tensor:
        # nn.UpsamplingBilinear2d in the public implementation uses
        # align_corners=True.  Keep it here for reproduction fidelity.
        return F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=True,
        )

    def _attention_from_prediction(
        self,
        deeper_logits: torch.Tensor,
        target_size: Size2D,
    ) -> torch.Tensor:
        probability = torch.sigmoid(
            self._resize_bilinear(deeper_logits, target_size)
        )
        if not self.focus_background:
            return probability

        # The public implementation detaches only the OpenCV dilation branch.
        # Subtracting the live probability preserves its original gradient path.
        with torch.no_grad():
            dilated = morphological_dilation(
                probability.detach().float(),
                kernel_size=self.dilation_kernel,
                iterations=self.dilation_iterations,
            ).to(dtype=probability.dtype)

        return (dilated - probability).clamp(min=0.0, max=1.0)

    def forward(
        self,
        current: torch.Tensor,
        deeper: torch.Tensor,
        deeper_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_size = current.shape[-2:]

        deeper = self._resize_bilinear(
            self.deeper_proj(deeper),
            target_size,
        )
        attention = self._attention_from_prediction(
            deeper_logits,
            target_size,
        )
        attended = self.attended_proj(current * attention)

        refined = self.refine(
            torch.cat((deeper, self.beta * attended), dim=1)
        )
        logits = self.prediction(refined)
        return refined, logits


class _PvtV2B4_FPNSAR(PvtV2B4_FPN_Baseline):
    """Internal configurable implementation for the four ablations."""

    def __init__(
        self,
        pretrained: bool = True,
        input_norm: bool = True,
        fpn_dim: int = 64,
        use_checkpoint: bool = False,
        *,
        use_oaa: bool,
        use_fgc: bool,
        fgc_dilation_kernel: int = 7,
        fgc_dilation_iterations: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

        if fpn_dim < 4 or fpn_dim % 4 != 0:
            raise ValueError("fpn_dim must be divisible by 4")

        self.use_oaa = bool(use_oaa)
        self.use_fgc = bool(use_fgc)
        self.deep_channels = int(fpn_dim)
        self.shallow_channels = int(fpn_dim // 2)
        self.fine_channels = int(fpn_dim // 4)

        # The inherited predictor is never used by the resolution-matched SAR
        # variants.  Removing it keeps parameter counts honest.
        del self.predictor

        if self.use_oaa:
            # Exact SARNet mapping for PVT stages C2/C3/C4/C5.  PVTv2-B3 and
            # B4 share [64, 128, 320, 512] stage widths, so only depth changes.
            self.oaa_s2 = OAA(
                self.embed_dims[0],
                self.embed_dims[1],
                self.shallow_channels,
            )
            self.oaa_s3 = OAA(
                self.embed_dims[1],
                self.embed_dims[2],
                self.deep_channels,
            )
            self.oaa_s4 = OAA(
                self.embed_dims[2],
                self.embed_dims[3],
                self.deep_channels,
            )
            self.deep_reduce = ConvBNAct(
                self.embed_dims[3],
                self.deep_channels,
                kernel_size=3,
                act="relu",
            )
            self.oaa_s1 = OAA(
                self.shallow_channels,
                self.deep_channels,
                self.fine_channels,
            )

            # These conventional FPN layers are unused in OAA variants.
            for module_name in (
                "lateral_2",
                "lateral_3",
                "lateral_4",
                "lateral_5",
                "smooth_2",
                "smooth_3",
                "smooth_4",
                "smooth_5",
            ):
                delattr(self, module_name)
        else:
            # Topology-matched plain path for the OAA ablation.  It retains the
            # original FPN and creates S2/S1 only with resize + convolution.
            self.plain_s2 = ConvBNAct(
                self.deep_channels,
                self.shallow_channels,
                kernel_size=3,
                act="relu",
            )
            self.plain_s1 = ConvBNAct(
                self.shallow_channels,
                self.fine_channels,
                kernel_size=3,
                act="relu",
            )

        if self.use_fgc:
            # Official order: three background conversions, then one
            # foreground conversion at the finest level.
            fgc_kwargs = dict(
                dilation_kernel=fgc_dilation_kernel,
                dilation_iterations=fgc_dilation_iterations,
            )
            self.coarse_prediction = nn.Conv2d(
                self.deep_channels,
                1,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            self.fgc_s4 = FGC(
                self.deep_channels,
                self.deep_channels,
                focus_background=True,
                **fgc_kwargs,
            )
            self.fgc_s3 = FGC(
                self.deep_channels,
                self.deep_channels,
                focus_background=True,
                **fgc_kwargs,
            )
            self.fgc_s2 = FGC(
                self.shallow_channels,
                self.deep_channels,
                focus_background=True,
                **fgc_kwargs,
            )
            self.fgc_s1 = FGC(
                self.fine_channels,
                self.shallow_channels,
                focus_background=False,
                **fgc_kwargs,
            )
        else:
            self.final_prediction = nn.Conv2d(
                self.fine_channels,
                1,
                kernel_size=3,
                padding=1,
                bias=True,
            )

    @staticmethod
    def _resize(
        source: torch.Tensor,
        target_size: Size2D,
        mode: str,
    ) -> torch.Tensor:
        return F.interpolate(
            source,
            size=target_size,
            mode=mode,
            align_corners=False,
        )

    @staticmethod
    def _fine_size(
        image_size: Sequence[int],
        s2_size: Sequence[int],
    ) -> Size2D:
        # Official S1 is 2x S2 (1/2 input for a standard PVT pyramid).
        return (
            min(int(image_size[0]), int(s2_size[0]) * 2),
            min(int(image_size[1]), int(s2_size[1]) * 2),
        )

    def _build_oaa_pyramid(
        self,
        image: torch.Tensor,
        features: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        c2, c3, c4, c5 = features
        s2 = self.oaa_s2(c2, c3, target_size=c2.shape[-2:])
        s3 = self.oaa_s3(c3, c4, target_size=c3.shape[-2:])
        s4 = self.oaa_s4(c4, c5, target_size=c4.shape[-2:])
        s5 = self.deep_reduce(c5)
        s1 = self.oaa_s1(
            s2,
            s5,
            target_size=self._fine_size(
                image.shape[-2:],
                s2.shape[-2:],
            ),
        )
        return s1, s2, s3, s4, s5

    def _build_plain_pyramid(
        self,
        image: torch.Tensor,
        features: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        c2, c3, c4, c5 = features

        p5 = self.smooth_5(self.lateral_5(c5))
        p4 = self.smooth_4(
            self.lateral_4(c4)
            + self._resize(p5, c4.shape[-2:], mode="bilinear")
        )
        p3 = self.smooth_3(
            self.lateral_3(c3)
            + self._resize(p4, c3.shape[-2:], mode="bilinear")
        )
        p2 = self.smooth_2(
            self.lateral_2(c2)
            + self._resize(p3, c2.shape[-2:], mode="bilinear")
        )

        s5, s4, s3 = p5, p4, p3
        s2 = self.plain_s2(p2)
        s1 = self.plain_s1(
            self._resize(
                s2,
                self._fine_size(image.shape[-2:], s2.shape[-2:]),
                mode="bicubic",
            )
        )
        return s1, s2, s3, s4, s5

    def _decode_fgc(
        self,
        pyramid: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        s1, s2, s3, s4, s5 = pyramid

        logits_s5 = self.coarse_prediction(s5)
        refined_s4, logits_s4 = self.fgc_s4(s4, s5, logits_s5)
        refined_s3, logits_s3 = self.fgc_s3(
            s3,
            refined_s4,
            logits_s4,
        )
        refined_s2, logits_s2 = self.fgc_s2(
            s2,
            refined_s3,
            logits_s3,
        )
        _, logits_s1 = self.fgc_s1(
            s1,
            refined_s2,
            logits_s2,
        )
        return logits_s1

    def body(self, data):
        image = data["image_m"]
        features = self.normalize_encoder(image)

        if self.use_oaa:
            pyramid = self._build_oaa_pyramid(image, features)
        else:
            pyramid = self._build_plain_pyramid(image, features)

        if self.use_fgc:
            logits = self._decode_fgc(pyramid)
        else:
            logits = self.final_prediction(pyramid[0])

        return F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class PvtV2B4_FPN_SARPlain(_PvtV2B4_FPNSAR):
    """Resolution/channel control: plain FPN + simple S2/S1 expansion."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            use_oaa=False,
            use_fgc=False,
            **kwargs,
        )


class PvtV2B4_FPN_OAA(_PvtV2B4_FPNSAR):
    """PVTv2-B4 + official-style OAA pyramid, without FGC."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            use_oaa=True,
            use_fgc=False,
            **kwargs,
        )


class PvtV2B4_FPN_FGC(_PvtV2B4_FPNSAR):
    """Plain FPN pyramid + official-order FGC refinement."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            use_oaa=False,
            use_fgc=True,
            **kwargs,
        )


class PvtV2B4_FPN_OAA_FGC(_PvtV2B4_FPNSAR):
    """PVTv2-B4 + OAA amplify pyramid + FGC recognize decoder."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            use_oaa=True,
            use_fgc=True,
            **kwargs,
        )


__all__ = [
    "OAA",
    "FGC",
    "PvtV2B4_FPN_SARPlain",
    "PvtV2B4_FPN_OAA",
    "PvtV2B4_FPN_FGC",
    "PvtV2B4_FPN_OAA_FGC",
]
