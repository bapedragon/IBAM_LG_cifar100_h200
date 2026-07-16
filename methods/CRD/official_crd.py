"""CRD criterion ported from the authors' official RepDistiller repository.

Source:
    https://github.com/HobbitLong/RepDistiller
Pinned source commit:
    b84f547c5db6a35318d4671d7d5c4de74c822403

The CRDLoss, ContrastLoss, Embed, Normalize, ContrastMemory, and AliasMethod
algorithms follow the official implementation. Device handling was modernized
so registered buffers move with ``module.to(device)`` instead of calling
``.cuda()`` during construction.
"""

from __future__ import annotations

import math

import torch
from torch import nn


EPS = 1e-7


class Normalize(nn.Module):
    def __init__(self, power: int = 2) -> None:
        super().__init__()
        self.power = power

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        norm = value.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        return value.div(norm.clamp_min(1e-12))


class Embed(nn.Module):
    def __init__(self, dim_in: int = 1024, dim_out: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.view(value.shape[0], -1)
        return self.l2norm(self.linear(value))


class ContrastLoss(nn.Module):
    """Noise-contrastive estimation loss from the official CRD code."""

    def __init__(self, n_data: int) -> None:
        super().__init__()
        self.n_data = n_data

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch_size = value.shape[0]
        negative_count = value.size(1) - 1
        noise_probability = 1.0 / float(self.n_data)

        positive = value.select(1, 0)
        log_positive = torch.div(
            positive,
            positive.add(negative_count * noise_probability + EPS),
        ).log_()

        negative = value.narrow(1, 1, negative_count)
        log_negative = torch.div(
            negative.clone().fill_(negative_count * noise_probability),
            negative.add(negative_count * noise_probability + EPS),
        ).log_()

        return -(
            log_positive.sum(0) + log_negative.view(-1, 1).sum(0)
        ) / batch_size


class AliasMethod(nn.Module):
    """Alias sampler used by the official CRD memory bank."""

    def __init__(self, probabilities: torch.Tensor) -> None:
        super().__init__()
        probabilities = probabilities.clone().float()
        if probabilities.sum() > 1:
            probabilities.div_(probabilities.sum())

        count = len(probabilities)
        probability = torch.zeros(count)
        alias = torch.zeros(count, dtype=torch.long)
        smaller: list[int] = []
        larger: list[int] = []

        for index, item_probability in enumerate(probabilities):
            probability[index] = count * item_probability
            if probability[index] < 1.0:
                smaller.append(index)
            else:
                larger.append(index)

        while smaller and larger:
            small = smaller.pop()
            large = larger.pop()
            alias[small] = large
            probability[large] = (probability[large] - 1.0) + probability[small]
            if probability[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for remaining in smaller + larger:
            probability[remaining] = 1.0

        self.register_buffer("probability", probability)
        self.register_buffer("alias", alias)

    def draw(self, sample_count: int) -> torch.Tensor:
        count = self.alias.size(0)
        primary = torch.randint(count, (sample_count,), device=self.probability.device)
        probability = self.probability.index_select(0, primary)
        alias = self.alias.index_select(0, primary)
        choice = torch.bernoulli(probability)
        return primary.mul(choice.long()) + alias.mul((1 - choice).long())


class ContrastMemory(nn.Module):
    """Official two-sided non-parametric CRD memory bank."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        negative_count: int,
        temperature: float = 0.07,
        momentum: float = 0.5,
    ) -> None:
        super().__init__()
        self.negative_count = negative_count
        self.multinomial = AliasMethod(torch.ones(output_size))
        self.register_buffer(
            "params",
            torch.tensor([negative_count, temperature, -1.0, -1.0, momentum]),
        )

        standard_deviation = 1.0 / math.sqrt(input_size / 3)
        self.register_buffer(
            "memory_v1",
            torch.rand(output_size, input_size)
            .mul_(2 * standard_deviation)
            .add_(-standard_deviation),
        )
        self.register_buffer(
            "memory_v2",
            torch.rand(output_size, input_size)
            .mul_(2 * standard_deviation)
            .add_(-standard_deviation),
        )

    def forward(
        self,
        value1: torch.Tensor,
        value2: torch.Tensor,
        positive_index: torch.Tensor,
        sampled_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        negative_count = int(self.params[0].item())
        temperature = float(self.params[1].item())
        normalization_v1 = float(self.params[2].item())
        normalization_v2 = float(self.params[3].item())
        momentum = float(self.params[4].item())

        batch_size = value1.size(0)
        output_size = self.memory_v1.size(0)
        input_size = self.memory_v1.size(1)

        if sampled_index is None:
            sampled_index = self.multinomial.draw(
                batch_size * (negative_count + 1)
            ).view(batch_size, -1)
            sampled_index.select(1, 0).copy_(positive_index.data)

        sampled_index = sampled_index.long()
        positive_index = positive_index.long()

        weight_v1 = torch.index_select(
            self.memory_v1, 0, sampled_index.view(-1)
        ).detach()
        weight_v1 = weight_v1.view(
            batch_size, negative_count + 1, input_size
        )
        output_v2 = torch.bmm(
            weight_v1, value2.view(batch_size, input_size, 1)
        )
        output_v2 = torch.exp(torch.div(output_v2, temperature))

        weight_v2 = torch.index_select(
            self.memory_v2, 0, sampled_index.view(-1)
        ).detach()
        weight_v2 = weight_v2.view(
            batch_size, negative_count + 1, input_size
        )
        output_v1 = torch.bmm(
            weight_v2, value1.view(batch_size, input_size, 1)
        )
        output_v1 = torch.exp(torch.div(output_v1, temperature))

        if normalization_v1 < 0:
            self.params[2].copy_(
                (output_v1.mean() * output_size).detach()
            )
            normalization_v1 = float(self.params[2].item())
            print(
                f"[CRD] normalization constant Z_v1={normalization_v1:.1f}",
                flush=True,
            )
        if normalization_v2 < 0:
            self.params[3].copy_(
                (output_v2.mean() * output_size).detach()
            )
            normalization_v2 = float(self.params[3].item())
            print(
                f"[CRD] normalization constant Z_v2={normalization_v2:.1f}",
                flush=True,
            )

        output_v1 = torch.div(output_v1, normalization_v1).contiguous()
        output_v2 = torch.div(output_v2, normalization_v2).contiguous()

        with torch.no_grad():
            positive_v1 = torch.index_select(
                self.memory_v1, 0, positive_index.view(-1)
            )
            positive_v1.mul_(momentum)
            positive_v1.add_(value1 * (1 - momentum))
            updated_v1 = positive_v1.div(
                positive_v1.pow(2).sum(1, keepdim=True).pow(0.5).clamp_min(1e-12)
            )
            self.memory_v1.index_copy_(0, positive_index, updated_v1)

            positive_v2 = torch.index_select(
                self.memory_v2, 0, positive_index.view(-1)
            )
            positive_v2.mul_(momentum)
            positive_v2.add_(value2 * (1 - momentum))
            updated_v2 = positive_v2.div(
                positive_v2.pow(2).sum(1, keepdim=True).pow(0.5).clamp_min(1e-12)
            )
            self.memory_v2.index_copy_(0, positive_index, updated_v2)

        return output_v1, output_v2


class CRDLoss(nn.Module):
    """Official symmetric Contrastive Representation Distillation loss."""

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        feature_dim: int,
        n_data: int,
        negative_count: int,
        temperature: float,
        momentum: float,
    ) -> None:
        super().__init__()
        self.embed_s = Embed(student_dim, feature_dim)
        self.embed_t = Embed(teacher_dim, feature_dim)
        self.contrast = ContrastMemory(
            feature_dim,
            n_data,
            negative_count,
            temperature,
            momentum,
        )
        self.criterion_s = ContrastLoss(n_data)
        self.criterion_t = ContrastLoss(n_data)

    def forward(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
        positive_index: torch.Tensor,
        sampled_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        student_feature = self.embed_s(student_feature)
        teacher_feature = self.embed_t(teacher_feature)
        output_s, output_t = self.contrast(
            student_feature,
            teacher_feature,
            positive_index,
            sampled_index,
        )
        return self.criterion_s(output_s) + self.criterion_t(output_t)
