"""Grid-preserving Ours modules adapted from the provided Ours source."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        kernel = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        sampling_points = kernel[0] * kernel[1]
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * sampling_points,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.modulator_conv = nn.Conv2d(
            in_channels,
            sampling_points,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.modulator_conv.weight)
        nn.init.zeros_(self.modulator_conv.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(inputs)
        modulation = 2.0 * torch.sigmoid(self.modulator_conv(inputs))
        return torchvision.ops.deform_conv2d(
            input=inputs,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=modulation,
        )


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.fc(self.avg_pool(inputs)) + self.fc(self.max_pool(inputs))
        )


class DeformableSpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.spatial_conv = DeformableConv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = torch.mean(inputs, dim=1, keepdim=True)
        maximum = torch.max(inputs, dim=1, keepdim=True).values
        return torch.sigmoid(self.spatial_conv(torch.cat((average, maximum), dim=1)))


class DeformableCBAM(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = DeformableSpatialAttention(spatial_kernel_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs * self.channel_attention(inputs)
        return outputs * self.spatial_attention(outputs)


class CBAMConvCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 5,
        qkv_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cbam = DeformableCBAM(
            embed_dim,
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=spatial_kernel_size,
        )
        padding = qkv_kernel_size // 2
        self.q_conv = nn.Conv2d(
            embed_dim, embed_dim, qkv_kernel_size, padding=padding, bias=True
        )
        self.k_conv = nn.Conv2d(
            embed_dim, embed_dim, qkv_kernel_size, padding=padding, bias=True
        )
        self.v_conv = nn.Conv2d(
            embed_dim, embed_dim, qkv_kernel_size, padding=padding, bias=True
        )
        self.out_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_feature.shape != student_feature.shape:
            raise ValueError(
                "Teacher and aligned student features must have the same shape: "
                f"teacher={tuple(teacher_feature.shape)} "
                f"student={tuple(student_feature.shape)}"
            )
        batch, channels, height, width = student_feature.shape
        enhanced_student = self.cbam(student_feature)
        query = self.q_conv(enhanced_student).flatten(2).transpose(1, 2)
        key = self.k_conv(teacher_feature).flatten(2).transpose(1, 2)
        value = self.v_conv(teacher_feature).flatten(2).transpose(1, 2)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, height * width, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        attention = torch.softmax(
            (query @ key.transpose(-2, -1)) * self.scale,
            dim=-1,
        )
        attention = self.dropout(attention)
        outputs = (attention @ value).transpose(1, 2).reshape(
            batch, height * width, channels
        )
        outputs = outputs.transpose(1, 2).reshape(batch, channels, height, width)
        return self.out_conv(outputs)


class TransformerAggregationPooling(nn.Module):
    """Learn one convex mixture of all student blocks for each CNN stage."""

    def __init__(self, num_transformer: int = 12, num_cnn: int = 3) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_cnn, num_transformer))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.weights.shape[1]:
            raise ValueError(
                f"Expected {self.weights.shape[1]} student features, got {len(features)}"
            )
        stacked = torch.stack(tuple(features), dim=1)
        normalized_weights = torch.softmax(self.weights, dim=-1)
        return torch.einsum("gl,bldhw->bgdhw", normalized_weights, stacked)

    def normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.weights.detach(), dim=-1)


class Ours(nn.Module):
    """Alignment, deformable enhancement, and grid-space cross-attention."""

    def __init__(
        self,
        student_channels: int = 192,
        teacher_channels: Sequence[int] = (16, 32, 64),
        num_student_blocks: int = 12,
        num_heads: int = 4,
        spatial_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.teacher_channels = tuple(int(value) for value in teacher_channels)
        self.aggregation = TransformerAggregationPooling(
            num_transformer=num_student_blocks,
            num_cnn=len(self.teacher_channels),
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(student_channels, channels, kernel_size=1)
            for channels in self.teacher_channels
        )
        self.fusion_blocks = nn.ModuleList(
            CBAMConvCrossAttention(
                channels,
                num_heads=num_heads,
                spatial_kernel_size=spatial_kernel_size,
                qkv_kernel_size=1,
            )
            for channels in self.teacher_channels
        )

    def forward(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        if len(teacher_features) != len(self.teacher_channels):
            raise ValueError(
                f"Expected {len(self.teacher_channels)} teacher stages, "
                f"got {len(teacher_features)}"
            )
        aggregated = self.aggregation(student_features)
        aligned_features: list[torch.Tensor] = []
        fused_features: list[torch.Tensor] = []
        alignment_loss = aggregated.new_zeros(())
        fusion_loss = aggregated.new_zeros(())
        for stage, (teacher_feature, projection, fusion) in enumerate(
            zip(teacher_features, self.projections, self.fusion_blocks, strict=True)
        ):
            aligned = projection(aggregated[:, stage])
            if aligned.shape[-2:] != teacher_feature.shape[-2:]:
                aligned = F.interpolate(
                    aligned,
                    size=teacher_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if aligned.shape[1] != teacher_feature.shape[1]:
                raise RuntimeError(
                    f"Stage {stage} channel mismatch after projection: "
                    f"aligned={aligned.shape[1]} teacher={teacher_feature.shape[1]}"
                )
            fused = fusion(aligned, teacher_feature)
            alignment_loss = alignment_loss + F.mse_loss(
                aligned.float(), teacher_feature.float()
            )
            fusion_loss = fusion_loss + F.mse_loss(
                fused.float(), teacher_feature.float()
            )
            aligned_features.append(aligned)
            fused_features.append(fused)
        return alignment_loss, fusion_loss, aligned_features, fused_features
