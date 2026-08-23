from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BIT_M_R50X1_URL = (
    "https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz"
)


# ============================================================
# 1) BiT building blocks
# ============================================================

class WeightStandardizedConv2d(nn.Conv2d):
    """Conv2d with per-output-channel weight standardization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        var, mean = torch.var_mean(
            weight,
            dim=(1, 2, 3),
            keepdim=True,
            unbiased=False,
        )

        weight = (weight - mean) / torch.sqrt(var + 1e-10)

        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def ws_conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> WeightStandardizedConv2d:

    return WeightStandardizedConv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


def ws_conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> WeightStandardizedConv2d:

    return WeightStandardizedConv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _np_weight_to_torch(
    array: np.ndarray,
) -> torch.Tensor:

    # BiT convolution weights:
    # HWIO -> OIHW
    if array.ndim == 4:
        array = array.transpose(3, 2, 0, 1)

    array = np.ascontiguousarray(array)

    return torch.from_numpy(array)


def _copy_np_param(
    parameter: torch.Tensor,
    array: np.ndarray,
) -> None:

    tensor = _np_weight_to_torch(array).to(
        dtype=parameter.dtype,
        device=parameter.device,
    )

    if parameter.shape != tensor.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"model={tuple(parameter.shape)}, "
            f"weight={tuple(tensor.shape)}"
        )

    parameter.copy_(tensor)


class BiTPreActBottleneck(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        self.gn1 = nn.GroupNorm(
            32,
            in_channels,
        )

        self.relu = nn.ReLU(
            inplace=True,
        )

        self.conv1 = ws_conv1x1(
            in_channels,
            mid_channels,
        )

        self.gn2 = nn.GroupNorm(
            32,
            mid_channels,
        )

        self.conv2 = ws_conv3x3(
            mid_channels,
            mid_channels,
            stride=stride,
        )

        self.gn3 = nn.GroupNorm(
            32,
            mid_channels,
        )

        self.conv3 = ws_conv1x1(
            mid_channels,
            out_channels,
        )

        self.downsample: Optional[nn.Module] = None

        if stride != 1 or in_channels != out_channels:

            self.downsample = ws_conv1x1(
                in_channels,
                out_channels,
                stride=stride,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        preact = self.relu(
            self.gn1(x)
        )

        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(
                preact
            )

        out = self.conv1(
            preact
        )

        out = self.conv2(
            self.relu(
                self.gn2(out)
            )
        )

        out = self.conv3(
            self.relu(
                self.gn3(out)
            )
        )

        return out + residual

    def load_from_npz(
        self,
        weights,
        prefix: str,
    ) -> None:

        conv_key = "standardized_conv2d"

        with torch.no_grad():

            _copy_np_param(
                self.conv1.weight,
                weights[
                    f"{prefix}a/{conv_key}/kernel"
                ],
            )

            _copy_np_param(
                self.conv2.weight,
                weights[
                    f"{prefix}b/{conv_key}/kernel"
                ],
            )

            _copy_np_param(
                self.conv3.weight,
                weights[
                    f"{prefix}c/{conv_key}/kernel"
                ],
            )

            _copy_np_param(
                self.gn1.weight,
                weights[
                    f"{prefix}a/group_norm/gamma"
                ],
            )

            _copy_np_param(
                self.gn1.bias,
                weights[
                    f"{prefix}a/group_norm/beta"
                ],
            )

            _copy_np_param(
                self.gn2.weight,
                weights[
                    f"{prefix}b/group_norm/gamma"
                ],
            )

            _copy_np_param(
                self.gn2.bias,
                weights[
                    f"{prefix}b/group_norm/beta"
                ],
            )

            _copy_np_param(
                self.gn3.weight,
                weights[
                    f"{prefix}c/group_norm/gamma"
                ],
            )

            _copy_np_param(
                self.gn3.bias,
                weights[
                    f"{prefix}c/group_norm/beta"
                ],
            )

            if self.downsample is not None:

                _copy_np_param(
                    self.downsample.weight,
                    weights[
                        f"{prefix}a/proj/"
                        f"{conv_key}/kernel"
                    ],
                )


# ============================================================
# 2) BiT-M-R50x1 backbone
# ============================================================

class BiTMR50x1Backbone(nn.Module):

    out_channels = 2048

    def __init__(self):
        super().__init__()

        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        WeightStandardizedConv2d(
                            3,
                            64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),

                    (
                        "pad",
                        nn.ConstantPad2d(
                            1,
                            0,
                        ),
                    ),

                    (
                        "pool",
                        nn.MaxPool2d(
                            kernel_size=3,
                            stride=2,
                            padding=0,
                        ),
                    ),
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        self._make_stage(
                            in_channels=64,
                            out_channels=256,
                            mid_channels=64,
                            num_units=3,
                            first_stride=1,
                        ),
                    ),

                    (
                        "block2",
                        self._make_stage(
                            in_channels=256,
                            out_channels=512,
                            mid_channels=128,
                            num_units=4,
                            first_stride=2,
                        ),
                    ),

                    (
                        "block3",
                        self._make_stage(
                            in_channels=512,
                            out_channels=1024,
                            mid_channels=256,
                            num_units=6,
                            first_stride=2,
                        ),
                    ),

                    (
                        "block4",
                        self._make_stage(
                            in_channels=1024,
                            out_channels=2048,
                            mid_channels=512,
                            num_units=3,
                            first_stride=2,
                        ),
                    ),
                ]
            )
        )

        self.final_gn = nn.GroupNorm(
            32,
            2048,
        )

        self.final_relu = nn.ReLU(
            inplace=True,
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        num_units: int,
        first_stride: int,
    ) -> nn.Sequential:

        units = [
            (
                "unit01",
                BiTPreActBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    mid_channels=mid_channels,
                    stride=first_stride,
                ),
            )
        ]

        for index in range(
            2,
            num_units + 1,
        ):
            units.append(
                (
                    f"unit{index:02d}",
                    BiTPreActBottleneck(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        mid_channels=mid_channels,
                        stride=1,
                    ),
                )
            )

        return nn.Sequential(
            OrderedDict(units)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.root(x)

        x = self.body(x)

        x = self.final_gn(x)

        x = self.final_relu(x)

        return x

    def load_from_npz(
        self,
        weights,
        prefix: str = "resnet/",
    ) -> None:

        # Original ImageNet-21k classification head
        # is intentionally NOT loaded.

        with torch.no_grad():

            _copy_np_param(
                self.root.conv.weight,
                weights[
                    f"{prefix}root_block/"
                    "standardized_conv2d/kernel"
                ],
            )

            _copy_np_param(
                self.final_gn.weight,
                weights[
                    f"{prefix}group_norm/gamma"
                ],
            )

            _copy_np_param(
                self.final_gn.bias,
                weights[
                    f"{prefix}group_norm/beta"
                ],
            )

        for block_name, block in self.body.named_children():

            for unit_name, unit in block.named_children():

                unit.load_from_npz(
                    weights,
                    prefix=(
                        f"{prefix}"
                        f"{block_name}/"
                        f"{unit_name}/"
                    ),
                )


# ============================================================
# 3) Conv Adapter
# ============================================================

class ConvAdapter(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gamma: int = 4,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()

        if gamma <= 0:
            raise ValueError(
                "gamma must be > 0"
            )

        if in_channels % gamma != 0:
            raise ValueError(
                f"in_channels={in_channels} "
                f"must be divisible by gamma={gamma}"
            )

        if (
            kernel_size <= 0
            or kernel_size % 2 == 0
        ):
            raise ValueError(
                "adapter kernel_size must "
                "be a positive odd integer"
            )

        width = (
            in_channels // gamma
        )

        #
        # channel reduction
        #
        self.reduce = nn.Conv2d(
            in_channels,
            width,
            kernel_size=1,
            bias=False,
        )

        self.act1 = nn.ReLU(
            inplace=True,
        )

        #
        # spatial depthwise conv
        #
        self.depthwise = nn.Conv2d(
            width,
            width,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=width,
            bias=False,
        )

        self.act2 = nn.ReLU(
            inplace=True,
        )

        #
        # expand to residual output channels
        #
        self.expand = nn.Conv2d(
            width,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        #
        # adapter initialization
        #
        nn.init.kaiming_normal_(
            self.reduce.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        nn.init.kaiming_normal_(
            self.depthwise.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        #
        # Important:
        # adapter initially produces delta = 0
        #
        nn.init.zeros_(
            self.expand.weight
        )

        nn.init.zeros_(
            self.expand.bias
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.reduce(x)

        x = self.act1(x)

        x = self.depthwise(x)

        x = self.act2(x)

        x = self.expand(x)

        return x


# ============================================================
# 4) BiT bottleneck + parallel adapter
# ============================================================

class BiTBottleneckResidualParallelAdapter(nn.Module):

    def __init__(
        self,
        block: BiTPreActBottleneck,
        gamma: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()

        if not isinstance(
            block,
            BiTPreActBottleneck,
        ):
            raise TypeError(
                "Expected BiTPreActBottleneck, "
                f"got {type(block).__name__}"
            )

        self.block = block

        in_channels = (
            block.conv1.in_channels
        )

        out_channels = (
            block.conv3.out_channels
        )

        stride_value = (
            block.conv2.stride
        )

        if isinstance(
            stride_value,
            (tuple, list),
        ):
            stride = int(
                stride_value[0]
            )
        else:
            stride = int(
                stride_value
            )

        self.adapter = ConvAdapter(
            in_channels=in_channels,
            out_channels=out_channels,
            gamma=gamma,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        #
        # ResNet-v2 / BiT pre-activation
        #
        preact = self.block.relu(
            self.block.gn1(x)
        )

        #
        # residual / shortcut
        #
        if self.block.downsample is None:
            residual = x
        else:
            residual = self.block.downsample(
                preact
            )

        #
        # original frozen BiT branch
        #
        out = self.block.conv1(
            preact
        )

        out = self.block.conv2(
            self.block.relu(
                self.block.gn2(out)
            )
        )

        out = self.block.conv3(
            self.block.relu(
                self.block.gn3(out)
            )
        )

        #
        # trainable parallel adapter
        #
        delta = self.adapter(
            preact
        )

        #
        # original residual + BiT branch + adapter
        #
        return (
            out
            + residual
            + delta
        )


# ============================================================
# 5) Complete model:
#    BiT-M-R50x1 + Adapters + CAM
# ============================================================

class BiTMR50x1CAMWithAdapter(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        weights_path: Optional[
            Union[str, Path]
        ] = None,
        gamma: int = 4,
        adapter_kernel_size: int = 3,
        freeze_backbone: bool = True,
        download_if_missing: bool = True,
    ):
        super().__init__()

        if num_classes <= 0:
            raise ValueError(
                "num_classes must be > 0"
            )

        self.num_classes = num_classes

        self.freeze_backbone = (
            freeze_backbone
        )

        #
        # ----------------------------------------------------
        # 1) Build original BiT backbone
        # ----------------------------------------------------
        #
        backbone = (
            BiTMR50x1Backbone()
        )

        #
        # ----------------------------------------------------
        # 2) Load official BiT-M-R50x1 weights
        # ----------------------------------------------------
        #
        if pretrained:

            resolved_path = (
                self._resolve_weights_path(
                    weights_path=weights_path,
                    download_if_missing=(
                        download_if_missing
                    ),
                )
            )

            with np.load(
                resolved_path,
                allow_pickle=False,
            ) as weights:

                backbone.load_from_npz(
                    weights
                )

        #
        # ----------------------------------------------------
        # 3) Freeze ONLY original BiT
        # ----------------------------------------------------
        #
        # Adapters are added AFTER freezing.
        #
        if freeze_backbone:

            for parameter in backbone.parameters():

                parameter.requires_grad = False

        #
        # ----------------------------------------------------
        # 4) Add trainable adapters
        # ----------------------------------------------------
        #
        backbone.body = (
            self._wrap_body_with_adapters(
                body=backbone.body,
                gamma=gamma,
                kernel_size=(
                    adapter_kernel_size
                ),
            )
        )

        self.backbone = backbone

        #
        # ----------------------------------------------------
        # 5) CAM head
        # ----------------------------------------------------
        #
        self.cam_conv = nn.Conv2d(
            in_channels=2048,
            out_channels=num_classes,
            kernel_size=1,
            bias=True,
        )

        self.gap = (
            nn.AdaptiveAvgPool2d(1)
        )

    # ========================================================
    # BiT pretrained weights
    # ========================================================

    @staticmethod
    def _resolve_weights_path(
        weights_path: Optional[
            Union[str, Path]
        ],
        download_if_missing: bool,
    ) -> Path:

        #
        # User supplied .npz
        #
        if weights_path is not None:

            path = (
                Path(weights_path)
                .expanduser()
                .resolve()
            )

            if not path.exists():

                raise FileNotFoundError(
                    "BiT weights not found: "
                    f"{path}"
                )

            return path

        #
        # Default torch cache
        #
        cache_dir = (
            Path(torch.hub.get_dir())
            / "checkpoints"
        )

        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            cache_dir
            / "BiT-M-R50x1.npz"
        )

        #
        # Download if needed
        #
        if not path.exists():

            if not download_if_missing:

                raise FileNotFoundError(
                    "BiT-M-R50x1.npz was "
                    "not found and "
                    "download_if_missing=False."
                )

            print(
                "\nDownloading official "
                "BiT-M-R50x1 weights:\n"
                f"{path}\n"
            )

            torch.hub.download_url_to_file(
                BIT_M_R50X1_URL,
                str(path),
                progress=True,
            )

        return path

    # ========================================================
    # Add adapters
    # ========================================================

    @staticmethod
    def _wrap_body_with_adapters(
        body: nn.Sequential,
        gamma: int,
        kernel_size: int,
    ) -> nn.Sequential:

        wrapped_blocks = []

        for (
            block_name,
            block,
        ) in body.named_children():

            wrapped_units = []

            for (
                unit_name,
                unit,
            ) in block.named_children():

                wrapped_units.append(
                    (
                        unit_name,
                        BiTBottleneckResidualParallelAdapter(
                            block=unit,
                            gamma=gamma,
                            kernel_size=kernel_size,
                        ),
                    )
                )

            wrapped_blocks.append(
                (
                    block_name,
                    nn.Sequential(
                        OrderedDict(
                            wrapped_units
                        )
                    ),
                )
            )

        return nn.Sequential(
            OrderedDict(
                wrapped_blocks
            )
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ):

        #
        # [B, 3, H, W]
        #       ↓
        # [B, 2048, H/32, W/32]
        #
        features = self.backbone(
            x
        )

        #
        # CAM
        #
        cams = self.cam_conv(
            features
        )

        #
        # Classification logits
        #
        logits = self.gap(
            cams
        ).flatten(1)

        return logits, cams

    # ========================================================
    # Parameter utilities
    # ========================================================

    def trainable_parameter_count(
        self,
    ) -> int:

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def total_parameter_count(
        self,
    ) -> int:

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    def adapter_parameter_count(
        self,
    ) -> int:

        return sum(
            parameter.numel()
            for module in self.modules()
            if isinstance(
                module,
                ConvAdapter,
            )
            for parameter
            in module.parameters()
        )


# ============================================================
# 6) Test
# ============================================================

if __name__ == "__main__":

    NUM_CLASSES = 10

    model = (
        BiTMR50x1CAMWithAdapter(
            num_classes=NUM_CLASSES,

            # official ImageNet-21k BiT-M
            pretrained=True,

            # None => automatically download
            weights_path=None,

            gamma=4,

            adapter_kernel_size=3,

            # only Adapter + CAM train
            freeze_backbone=True,

            download_if_missing=True,
        )
    )

    print(model)

    print(
        "\nTrainable params:",
        f"{model.trainable_parameter_count():,}",
    )

    print(
        "Total params:",
        f"{model.total_parameter_count():,}",
    )

    print(
        "Adapter params:",
        f"{model.adapter_parameter_count():,}",
    )

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    logits, cams = model(x)

    print(
        "\nInput:",
        x.shape,
    )

    print(
        "Logits:",
        logits.shape,
    )

    print(
        "CAMs:",
        cams.shape,
    )