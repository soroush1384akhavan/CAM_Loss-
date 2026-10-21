from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file

from src.conv_adapter import ConvAdapter
from src.normalization import BIT_MEAN, BIT_STD

BIT_MODEL_NAME = "resnetv2_50x1_bit"
BIT_PRETRAIN_CLASSES = 21843


def _load_bit_m_weights(model: nn.Module, weights_path: Union[str, Path]) -> None:
    path = Path(weights_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"BiT-M weights not found: {path}")

    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"Expected a .safetensors checkpoint, got: {path}")

    state = load_file(str(path), device="cpu")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    required = {
        "stem.conv.weight",
        "stages.0.blocks.0.conv1.weight",
        "norm.weight",
        "head.fc.weight",
    }
    missing_required = sorted(required.difference(state))
    if missing_required:
        raise RuntimeError(
            "This is not the timm BiT-M-R50x1 ImageNet-21k checkpoint; "
            f"missing keys: {missing_required}"
        )

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The checkpoint does not exactly match timm/resnetv2_50x1_bit.goog_in21k. "
            "Download that repository's model.safetensors file."
        ) from error


class BiTBottleneckResidualParallelAdapter(nn.Module):
    def __init__(self, block: nn.Module, gamma: int = 4, kernel_size: int = 3):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be positive")

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("adapter kernel_size must be a positive odd number")

        required = ("conv1", "conv2", "conv3")
        if any(not hasattr(block, name) for name in required):
            raise TypeError("Expected a timm ResNetV2 PreActBottleneck block")

        in_channels = block.conv1.in_channels
        out_channels = block.conv3.out_channels
        if in_channels % gamma:
            raise ValueError(f"in_channels={in_channels} must be divisible by gamma={gamma}")

        width = in_channels // gamma
        stride = block.conv2.stride
        stride = stride[0] if isinstance(stride, tuple) else stride
        self.block = block

        self.adapter = ConvAdapter(
            inplanes=in_channels,
            outplanes=out_channels,
            width=width,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            groups=width,
            act_layer=nn.ReLU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.adapter(x)


class BiTMR50x1CAMWithAdapter(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        weights_path: Union[str, Path] = "weights/model.safetensors",
        gamma: int = 4,
        adapter_kernel_size: int = 3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.freeze_backbone = freeze_backbone
        self.backbone = timm.create_model(
            BIT_MODEL_NAME,
            pretrained=False,
            num_classes=BIT_PRETRAIN_CLASSES,
        )

        if pretrained:
            _load_bit_m_weights(self.backbone, weights_path)

        self.backbone.reset_classifier(0)

        if freeze_backbone:
            self.backbone.requires_grad_(False)

            for module in self.backbone.modules():
                if isinstance(module, nn.GroupNorm):
                    module.requires_grad_(True)

        self._add_adapters(gamma, adapter_kernel_size)
        self.cam_conv = nn.Conv2d(self.backbone.num_features, num_classes, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def _add_adapters(self, gamma: int, kernel_size: int) -> None:
        for stage in self.backbone.stages:
            stage.blocks = nn.Sequential(
                *(
                    BiTBottleneckResidualParallelAdapter(block, gamma, kernel_size)
                    for block in stage.blocks
                )
            )

    def forward(self, x: torch.Tensor):
        features = self.backbone.forward_features(x)
        cams = self.cam_conv(features)
        return self.gap(cams).flatten(1), cams

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def adapter_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in self.modules()
            if isinstance(module, ConvAdapter)
            for parameter in module.parameters()
        )
