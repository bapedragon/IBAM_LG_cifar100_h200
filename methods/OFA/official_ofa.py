"""OFA-KD loss and DeiT projector matching the authors' official behavior.

The official repository is pinned in ``methods/OFA/README.md``. Its project at
that commit does not provide a license file, so this module is an independently
structured compatibility implementation rather than a verbatim source copy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ofa_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target_mask: torch.Tensor,
    eps: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Adaptive target-enhancement loss used by the official OFA distiller."""
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    product = (pred_teacher + target_mask) ** eps
    return torch.sum(
        -(product - target_mask) * torch.log(pred_student),
        dim=-1,
    ).mean()


class PatchMerging(nn.Module):
    """Official Swin-style 2x2 patch merger used by OFA's ViT projector."""

    def __init__(self, resolution: int, dim: int, out_dim: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch, length, channels = patches.shape
        height = width = self.resolution
        if length != height * width or channels != self.dim:
            raise RuntimeError(
                "OFA patch shape mismatch: "
                f"expected=(*,{height * width},{self.dim}) "
                f"actual={tuple(patches.shape)}"
            )
        if height % 2 or width % 2:
            raise RuntimeError(f"OFA patch grid must be even, got {height}x{width}")
        patches = patches.view(batch, height, width, channels)
        merged = torch.cat(
            (
                patches[:, 0::2, 0::2, :],
                patches[:, 1::2, 0::2, :],
                patches[:, 0::2, 1::2, :],
                patches[:, 1::2, 1::2, :],
            ),
            dim=-1,
        )
        merged = merged.reshape(batch, -1, 4 * channels)
        return self.activation(self.reduction(self.norm(merged)))


class OFAIntermediateHead(nn.Module):
    """Project one DeiT stage into the class-logit space."""

    def __init__(
        self,
        *,
        stage: int,
        embed_dim: int,
        patch_grid: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        from timm.models.vision_transformer import Block

        self.patch_merger = PatchMerging(patch_grid, embed_dim, embed_dim)
        self.blocks = nn.Sequential(
            *[
                Block(dim=embed_dim, num_heads=4)
                for _ in range(max(4 - stage, 1))
            ]
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != 197:
            raise RuntimeError(f"Unexpected DeiT token shape: {tuple(tokens.shape)}")
        class_token = tokens[:, :1, :]
        patch_tokens = self.patch_merger(tokens[:, 1:, :])
        tokens = torch.cat((class_token, patch_tokens), dim=1)
        tokens = self.blocks(tokens)
        return self.classifier(tokens[:, 0, :])


class OFAProjector(nn.Module):
    """The four official OFA intermediate logit projectors for DeiT-Ti."""

    def __init__(
        self,
        *,
        stages: tuple[int, ...],
        embed_dim: int,
        patch_grid: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("OFA requires at least one intermediate stage")
        self.stages = stages
        self.heads = nn.ModuleList(
            [
                OFAIntermediateHead(
                    stage=stage,
                    embed_dim=embed_dim,
                    patch_grid=patch_grid,
                    num_classes=num_classes,
                )
                for stage in stages
            ]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.heads):
            raise RuntimeError(
                f"OFA feature count mismatch: {len(features)} vs {len(self.heads)}"
            )
        return [head(feature) for head, feature in zip(self.heads, features)]
