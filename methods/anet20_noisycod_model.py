# -*- coding: utf-8 -*-
"""
Faithful reimplementation of the released Noisy-COD ANet.

Network:
    RGB ---------> ConvNeXt-B -> GCM3 -> DWT -> HH/LL routing --\
                                                                +-> GPM + 4x REU
    RGB * Box ---> ConvNeXt-B -> GCM3 -> DWT -> HH/LL routing --/

Key released-code details preserved:
  * ConvNeXt-B depths [3, 3, 27, 3], dims [128, 256, 512, 1024]
  * fixed Haar-like DWT
  * HH injected into shallow F1/F2
  * LL injected into deep F3/F4
  * nearest x2 upsample for HH -> F1
  * GCM3 bilinear scale factors 2/4/8 with align_corners=True
  * GPM input = concat of the two deepest ConvNeXt features (2048 ch)
  * four reverse-enhancement units (REU)
  * boundary edge_enhance operation
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ConvNeXt-B backbone, using module/key names compatible with the official
# Facebook ConvNeXt checkpoint used by the released Noisy-COD code.
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        if self.data_format != "channels_first":
            raise ValueError(f"Unsupported data_format={self.data_format}")
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        )

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ConvNeXtBase(nn.Module):
    """ConvNeXt-Base with checkpoint-compatible key names."""

    def __init__(self):
        super().__init__()
        depths = [3, 3, 27, 3]
        dims = [128, 256, 512, 1024]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList(
            [
                nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
                for i in range(4)
            ]
        )

        # These are unused by ANet forward, but are kept so the released
        # 22K->1K checkpoint can load strictly.
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], 1000)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x) -> List[torch.Tensor]:
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        return outs


def load_convnext_base_384(
    model_name: str = "convnext_base.fb_in22k_ft_in1k_384",
):
    """Build the exact tagged timm ConvNeXt-B 22K->1K 384 backbone.

    timm/Hugging Face downloads the pretrained weights on first use and then
    reuses its local cache. features_only=True returns the four feature maps
    required by Noisy-COD ANet: [128, 256, 512, 1024] channels.
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "timm is required for the automatic ConvNeXt-B pretrained download. "
            "Install it with: pip install -U timm huggingface_hub"
        ) from e

    model = timm.create_model(
        model_name,
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )

    channels = list(model.feature_info.channels())
    expected = [128, 256, 512, 1024]
    if channels != expected:
        raise RuntimeError(
            f"Unexpected ConvNeXt feature channels: {channels}; expected {expected}. "
            f"model_name={model_name!r}"
        )
    return model


# ---------------------------------------------------------------------------
# Noisy-COD ANet blocks
# ---------------------------------------------------------------------------

class BasicConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, need_relu=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x) if self.need_relu else x


class BasicDeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, out_padding=0, need_relu=True
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            output_padding=out_padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x) if self.need_relu else x


class ETM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = BasicConv2d(in_channels, out_channels, 3, 1, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, (3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, (5, 1), padding=(2, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, (5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, (7, 1), padding=(3, 0)),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, (7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, (1, 9), padding=(0, 4)),
            BasicConv2d(out_channels, out_channels, (9, 1), padding=(4, 0)),
        )

        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        return self.relu(x_cat + self.conv_res(x))


class DWT(nn.Module):
    """Fixed Haar-like DWT used by the released ANet."""

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2

        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        # For the released 384 input all sizes are even; these calls preserve
        # the original implementation's behavior.
        target = (x.shape[2] // 2, x.shape[3] // 2)
        x1 = F.interpolate(x1, size=target, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=target, mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=target, mode="bilinear", align_corners=False)
        x4 = F.interpolate(x4, size=target, mode="bilinear", align_corners=False)

        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh


class GCM3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.T1 = ETM(in_channels[0], out_channels)
        self.T2 = ETM(in_channels[1], out_channels)
        self.T3 = ETM(in_channels[2], out_channels)
        self.T4 = ETM(in_channels[3], out_channels)
        self.decoder = nn.Conv2d(out_channels * 4, out_channels, 3, 1, 1)
        self.DWT = DWT()

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)
        f2 = self.T2(f2)
        f3 = self.T3(f3)
        f4 = self.T4(f4)

        camo = self.decoder(
            torch.cat(
                [
                    f1,
                    F.interpolate(
                        f2, scale_factor=2, mode="bilinear",
                        align_corners=True
                    ),
                    F.interpolate(
                        f3, scale_factor=4, mode="bilinear",
                        align_corners=True
                    ),
                    F.interpolate(
                        f4, scale_factor=8, mode="bilinear",
                        align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        ll, lh, hl, hh = self.DWT(camo)
        return ll, lh, hl, hh, f1, f2, f3, f4


class Branch(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_base.fb_in22k_ft_in1k_384",
        channels=64,
    ):
        super().__init__()
        self.shared_encoder = load_convnext_base_384(backbone_name)
        self.GCM3 = GCM3([128, 256, 512, 1024], channels)

        self.LL_down3 = nn.Sequential(
            BasicConv2d(channels, channels, 3, stride=2, padding=1)
        )
        self.LL_down4 = nn.Sequential(
            BasicConv2d(channels, channels, 3, stride=2, padding=1),
            BasicConv2d(channels, channels, 3, stride=2, padding=1),
        )

        # Released code uses nn.Upsample(scale_factor=2) -> nearest.
        self.dePixelShuffle = nn.Upsample(scale_factor=2)

        self.one_conv_f4_ll = ETM(channels * 2, channels)
        self.one_conv_f3_ll = ETM(channels * 2, channels)
        self.one_conv_f1_hh = ETM(channels * 2, channels)
        self.one_conv_f2_hh = ETM(channels * 2, channels)

    def forward(self, x):
        x1, x2, x3, x4 = self.shared_encoder(x)
        ll, lh, hl, hh, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        # Original routing uses only HH and LL. LH/HL are intentionally unused.
        hh_up = self.dePixelShuffle(hh)
        f1_hh = self.one_conv_f1_hh(torch.cat([hh_up, f1], dim=1))
        f2_hh = self.one_conv_f2_hh(torch.cat([hh, f2], dim=1))

        ll_down3 = self.LL_down3(ll)
        f3_ll = self.one_conv_f3_ll(torch.cat([ll_down3, f3], dim=1))

        ll_down4 = self.LL_down4(ll)
        f4_ll = self.one_conv_f4_ll(torch.cat([ll_down4, f4], dim=1))

        return f1_hh, f2_hh, f3_ll, f4_ll, x4


class GPM(nn.Module):
    def __init__(
        self, in_c=128,
        dilation_series=(6, 12, 18),
        padding_series=(6, 12, 18),
        depth=32
    ):
        super().__init__()

        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(in_c, depth, 1, stride=1),
        )
        self.branch0 = BasicConv2d(in_c, depth, 1, stride=1)
        self.branch1 = BasicConv2d(
            in_c, depth, 3, stride=1,
            padding=padding_series[0], dilation=dilation_series[0]
        )
        self.branch2 = BasicConv2d(
            in_c, depth, 3, stride=1,
            padding=padding_series[1], dilation=dilation_series[1]
        )
        self.branch3 = BasicConv2d(
            in_c, depth, 3, stride=1,
            padding=padding_series[2], dilation=dilation_series[2]
        )
        self.head = nn.Sequential(
            BasicConv2d(depth * 5, depth, 1, padding=0)
        )
        self.out = nn.Sequential(
            nn.Conv2d(depth, depth, 3, padding=1),
            nn.BatchNorm2d(depth),
            nn.PReLU(),
            nn.Conv2d(depth, 1, 3, 1, 1),
        )

        # Preserve released GPM initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = F.interpolate(
            self.branch_main(x), size=size,
            mode="bilinear", align_corners=True
        )
        out = torch.cat(
            [
                branch_main,
                self.branch0(x),
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
            ],
            dim=1,
        )
        return self.out(self.head(out))


class REUBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, 3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, 3, padding=1),
        )
        self.out_B = nn.Sequential(
            BasicDeConv2d(
                in_channels, mid_channels // 2,
                kernel_size=3, stride=2, padding=1, out_padding=1
            ),
            BasicConv2d(mid_channels // 2, mid_channels // 4, 3, padding=1),
            nn.Conv2d(mid_channels // 4, 1, 3, padding=1),
        )
        self.ode = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 3, padding=1),
            BasicConv2d(in_channels, in_channels, 3, padding=1),
        )

    @staticmethod
    def edge_enhance(img):
        gradient = img.clone()
        gradient[:, :, :-1, :] = torch.abs(
            gradient[:, :, :-1, :] - gradient[:, :, 1:, :]
        )
        gradient[:, :, :, :-1] = torch.abs(
            gradient[:, :, :, :-1] - gradient[:, :, :, 1:]
        )
        return torch.clamp(img - gradient, 0, 1)

    def forward(self, x, prior_cam):
        prior_cam = F.interpolate(
            prior_cam, size=x.size()[2:],
            mode="bilinear", align_corners=True
        )

        yt = self.conv(
            torch.cat(
                [x, prior_cam.expand(-1, x.size(1), -1, -1)],
                dim=1
            )
        )
        ode_out = self.ode(yt)

        bound = self.edge_enhance(self.out_B(ode_out))

        reverse_prior = 1 - torch.sigmoid(prior_cam)
        y = reverse_prior.expand(-1, x.size(1), -1, -1) * x
        y = self.out_y(torch.cat([y, ode_out], dim=1))
        y = y + prior_cam

        return y, bound


class UNetDecoderWithEdges(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.REU_f1 = REUBlock(in_channels, mid_channels)
        self.REU_f2 = REUBlock(in_channels, mid_channels)
        self.REU_f3 = REUBlock(in_channels, mid_channels)
        self.REU_f4 = REUBlock(in_channels, mid_channels)

    @staticmethod
    def _up(x, size):
        return F.interpolate(
            x, size=size, mode="bilinear", align_corners=True
        )

    def forward(self, feats, prior_0, pic):
        f1, f2, f3, f4 = feats
        full_size = pic.size()[2:]

        f4_out, bound_f4 = self.REU_f4(f4, prior_0)
        f4_full = self._up(f4_out, full_size)
        bound_f4 = self._up(bound_f4, full_size)

        f3_out, bound_f3 = self.REU_f3(f3, f4_out)
        f3_full = self._up(f3_out, full_size)
        bound_f3 = self._up(bound_f3, full_size)

        f2_out, bound_f2 = self.REU_f2(f2, f3_out)
        f2_full = self._up(f2_out, full_size)
        bound_f2 = self._up(bound_f2, full_size)

        f1_out, bound_f1 = self.REU_f1(f1, f2_out)
        f1_full = self._up(f1_out, full_size)
        bound_f1 = self._up(bound_f1, full_size)

        return (
            f4_full, f3_full, f2_full, f1_full,
            bound_f4, bound_f3, bound_f2, bound_f1
        )


class NoisyCODANet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_base.fb_in22k_ft_in1k_384",
        channels=64,
    ):
        super().__init__()
        self.net1 = Branch(backbone_name=backbone_name, channels=channels)
        self.net2 = Branch(backbone_name=backbone_name, channels=channels)
        self.GPM = GPM(2048)
        self.decoder = UNetDecoderWithEdges(channels * 2, channels * 2)

    def forward(self, x, x_box):
        f1_hh1, f2_hh1, f3_ll1, f4_ll1, x41 = self.net1(x)
        f1_hh2, f2_hh2, f3_ll2, f4_ll2, x42 = self.net2(x_box)

        f1_hh = torch.cat([f1_hh1, f1_hh2], dim=1)
        f2_hh = torch.cat([f2_hh1, f2_hh2], dim=1)
        f3_ll = torch.cat([f3_ll1, f3_ll2], dim=1)
        f4_ll = torch.cat([f4_ll1, f4_ll2], dim=1)
        x4 = torch.cat([x41, x42], dim=1)

        prior_cam = self.GPM(x4)
        pred_0 = F.interpolate(
            prior_cam, size=x.size()[2:],
            mode="bilinear", align_corners=False
        )

        (
            f4, f3, f2, f1,
            bound_f4, bound_f3, bound_f2, bound_f1
        ) = self.decoder(
            [f1_hh, f2_hh, f3_ll, f4_ll], prior_cam, x
        )

        return (
            pred_0, f4, f3, f2, f1,
            bound_f4, bound_f3, bound_f2, bound_f1
        )


# ---------------------------------------------------------------------------
# Released ANet losses
# ---------------------------------------------------------------------------

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(
        pred, mask, reduction="none"
    )
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)

    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)

    num = torch.sum(predict * target * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(
        (predict.pow(p) + target.pow(p)) * valid_mask,
        dim=1
    ) + smooth
    return (1 - num / den).mean()


def cal_ual(seg_logits, seg_gts):
    if seg_logits.shape != seg_gts.shape:
        raise ValueError(
            f"UAL shape mismatch: logits={seg_logits.shape}, gt={seg_gts.shape}"
        )
    sigmoid_x = seg_logits.sigmoid()
    return (1 - (2 * sigmoid_x - 1).abs().pow(2)).mean()


def get_ual_coef(iter_percentage: float):
    # Released get_coef(..., method='cos').
    return float((1 - torch.cos(
        torch.tensor(iter_percentage * torch.pi)
    ).item()) / 2)
