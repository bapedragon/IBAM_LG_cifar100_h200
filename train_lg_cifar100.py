#!/usr/bin/env python3
"""LG-style CIFAR-100 training scaffold for a ResNet56 teacher and DeiT-Ti.

This is intentionally a compact, runnable scaffold. It is not a claim of an
exact reproduction of the LG paper because several paper/repository details
are not fixed here (feature layer, loss weight, augmentation, and teacher
recipe in particular).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import InterpolationMode


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
NUM_CLASSES = 100


def log(message: str = "") -> None:
    """Print an H200/Jenkins-friendly, immediately flushed log line."""
    print(message, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class CIFARResNet56(nn.Module):
    """CIFAR-style ResNet56 (6n+2 with n=9), adapted to 224x224 inputs."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.in_channels = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(16, blocks=9, stride=1)
        self.stage2 = self._make_stage(32, blocks=9, stride=2)
        self.stage3 = self._make_stage(64, blocks=9, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.feature_channels = 64
        self._initialize_weights()

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(BasicBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        features = self.stage3(x)
        logits = self.fc(torch.flatten(self.pool(features), 1))
        if return_features:
            return logits, features
        return logits


class LGDistiller(nn.Module):
    """Align the final CNN feature map to DeiT patch tokens with an MSE loss."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.student = timm.create_model(
            "deit_tiny_patch16_224", pretrained=False, num_classes=num_classes
        )
        embed_dim = int(self.student.embed_dim)
        self.projector = nn.Conv2d(64, embed_dim, kernel_size=1)

        grid_size = self.student.patch_embed.grid_size
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        self.patch_grid = (int(grid_size[0]), int(grid_size[1]))
        self.num_patches = int(self.student.patch_embed.num_patches)
        self.embed_dim = embed_dim

    def forward(
        self, images: torch.Tensor, teacher_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_tokens = self.student.forward_features(images)
        logits = self.student.forward_head(all_tokens)

        if all_tokens.ndim != 3 or all_tokens.shape[1] < self.num_patches:
            raise RuntimeError(
                f"Unexpected DeiT feature shape: {tuple(all_tokens.shape)}; "
                f"expected at least {self.num_patches} tokens"
            )
        patch_tokens = all_tokens[:, -self.num_patches :, :]

        teacher_grid = F.adaptive_avg_pool2d(teacher_features, self.patch_grid)
        teacher_tokens = self.projector(teacher_grid).flatten(2).transpose(1, 2)

        # Per-token normalization stabilizes scale matching between CNN and ViT.
        student_for_loss = F.layer_norm(patch_tokens, (patch_tokens.shape[-1],))
        teacher_for_loss = F.layer_norm(teacher_tokens, (teacher_tokens.shape[-1],))
        return logits, student_for_loss, teacher_for_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet56 -> DeiT-Ti with LG-style feature distillation on CIFAR-100."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-epochs", type=int, default=300)
    parser.add_argument("--student-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--student-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--teacher-optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--teacher-lr", type=float, default=0.1)
    parser.add_argument("--teacher-weight-decay", type=float, default=5e-4)
    parser.add_argument("--teacher-warmup-epochs", type=int, default=5)

    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-samples", type=int, default=1024)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision (default: enabled).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = (
        "teacher_epochs",
        "student_epochs",
        "batch_size",
        "image_size",
        "smoke_train_samples",
        "smoke_test_samples",
    )
    for field in positive_int_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.image_size != 224:
        raise ValueError("This DeiT-Ti scaffold currently requires --image-size 224")
    if args.feature_weight < 0:
        raise ValueError("--feature-weight must be non-negative")


def make_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    resize_size = int(round(image_size / 0.875))
    test_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return train_transform, test_transform


def deterministic_subset(dataset: Dataset[Any], size: int, seed: int) -> Dataset[Any]:
    size = min(size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_loaders(
    args: argparse.Namespace, device: torch.device
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    train_transform, test_transform = make_transforms(args.image_size)
    train_dataset: Dataset[Any] = CIFAR100(
        root=args.data_dir, train=True, transform=train_transform, download=True
    )
    test_dataset: Dataset[Any] = CIFAR100(
        root=args.data_dir, train=False, transform=test_transform, download=True
    )

    if args.smoke:
        train_dataset = deterministic_subset(train_dataset, args.smoke_train_samples, args.seed)
        test_dataset = deterministic_subset(test_dataset, args.smoke_test_samples, args.seed + 1)

    generator = torch.Generator().manual_seed(args.seed)
    common: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, generator=generator, **common
    )
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common)

    log(f"[DATA] train_samples={len(train_dataset)} test_samples={len(test_dataset)}")
    log(
        f"[DATA] image_size={args.image_size} batch_size={args.batch_size} "
        f"num_workers={args.num_workers} smoke={args.smoke}"
    )
    return train_loader, test_loader


def create_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def effective_warmup(requested: int, total_epochs: int) -> int:
    # Keep 1-epoch smoke runs useful while preserving 20/300 for the full run.
    return min(requested, max(0, total_epochs // 5))


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer, total_epochs: int, requested_warmup: int
) -> Tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    warmup_epochs = effective_warmup(requested_warmup, total_epochs)

    def multiplier(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        decay_epochs = max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (epoch - warmup_epochs) / decay_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), warmup_epochs


def top1_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int(logits.argmax(dim=1).eq(targets).sum().item())


@torch.inference_mode()
def evaluate(model: nn.Module, loader: Iterable[Any], device: torch.device, amp: bool) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(amp):
            logits = model(images)
        correct += top1_correct(logits, targets)
        total += targets.size(0)
    return 100.0 * correct / max(1, total)


def checkpoint_payload(
    model: nn.Module, epoch: int, accuracy: float, args: argparse.Namespace, **extra: Any
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    payload.update(extra)
    return payload


def load_torch_checkpoint(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_teacher(
    model: CIFARResNet56,
    train_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: Path,
    amp: bool,
) -> float:
    if args.teacher_optimizer == "sgd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.teacher_lr,
            momentum=0.9,
            weight_decay=args.teacher_weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.teacher_lr, weight_decay=args.teacher_weight_decay
        )
    scheduler, warmup = make_cosine_scheduler(
        optimizer, args.teacher_epochs, args.teacher_warmup_epochs
    )
    scaler = create_grad_scaler(amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_accuracy = -1.0

    log(
        f"[TEACHER] optimizer={args.teacher_optimizer} lr={args.teacher_lr} "
        f"weight_decay={args.teacher_weight_decay} epochs={args.teacher_epochs} "
        f"effective_warmup={warmup}"
    )
    for epoch in range(1, args.teacher_epochs + 1):
        epoch_start = time.time()
        model.train()
        loss_sum = 0.0
        sample_count = 0
        correct = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            loss_sum += float(loss.detach()) * batch_size
            correct += top1_correct(logits.detach(), targets)
            sample_count += batch_size

        val_accuracy = evaluate(model, test_loader, device, amp)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                checkpoint_payload(model, epoch, best_accuracy, args), checkpoint_path
            )
            marker = " saved_best"
        else:
            marker = ""

        current_lr = optimizer.param_groups[0]["lr"]
        log(
            f"[TEACHER][{epoch:03d}/{args.teacher_epochs:03d}] "
            f"loss={loss_sum / max(1, sample_count):.4f} "
            f"train_acc={100.0 * correct / max(1, sample_count):.2f}% "
            f"val_acc={val_accuracy:.2f}% best={best_accuracy:.2f}% "
            f"lr={current_lr:.6g} time={time.time() - epoch_start:.1f}s{marker}"
        )
        scheduler.step()

    best_payload = load_torch_checkpoint(checkpoint_path, device)
    model.load_state_dict(best_payload["model"])
    log(f"[TEACHER] loaded_best checkpoint={checkpoint_path} top1={best_accuracy:.2f}%")
    return best_accuracy


def load_teacher_checkpoint(
    model: CIFARResNet56, path: Path, device: torch.device
) -> float:
    payload = load_torch_checkpoint(path, device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    accuracy = float(payload.get("accuracy", float("nan"))) if isinstance(payload, dict) else float("nan")
    log(f"[TEACHER] loaded_external checkpoint={path} recorded_top1={accuracy:.2f}%")
    return accuracy


def train_student(
    distiller: LGDistiller,
    teacher: CIFARResNet56,
    train_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: Path,
    amp: bool,
) -> float:
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()

    optimizer = torch.optim.AdamW(
        distiller.parameters(), lr=args.student_lr, weight_decay=args.weight_decay
    )
    scheduler, warmup = make_cosine_scheduler(
        optimizer, args.student_epochs, args.warmup_epochs
    )
    scaler = create_grad_scaler(amp)
    classification_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_accuracy = -1.0

    log(
        f"[STUDENT] model=deit_tiny_patch16_224 optimizer=adamw "
        f"lr={args.student_lr} weight_decay={args.weight_decay} "
        f"epochs={args.student_epochs} effective_warmup={warmup}"
    )
    log(
        f"[LG] teacher_layer=stage3 patch_grid={distiller.patch_grid} "
        f"projector=conv1x1 normalization=per_token_layer_norm "
        f"loss=classification+{args.feature_weight}*feature_mse"
    )

    for epoch in range(1, args.student_epochs + 1):
        epoch_start = time.time()
        distiller.train()
        teacher.eval()
        total_loss_sum = 0.0
        cls_loss_sum = 0.0
        feature_loss_sum = 0.0
        sample_count = 0
        correct = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(amp):
                with torch.no_grad():
                    _, teacher_features = teacher(images, return_features=True)
                logits, student_features, projected_teacher_features = distiller(
                    images, teacher_features
                )
                cls_loss = classification_criterion(logits, targets)
                feature_loss = F.mse_loss(student_features, projected_teacher_features)
                total_loss = cls_loss + args.feature_weight * feature_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            total_loss_sum += float(total_loss.detach()) * batch_size
            cls_loss_sum += float(cls_loss.detach()) * batch_size
            feature_loss_sum += float(feature_loss.detach()) * batch_size
            correct += top1_correct(logits.detach(), targets)
            sample_count += batch_size

        val_accuracy = evaluate(distiller.student, test_loader, device, amp)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                checkpoint_payload(
                    distiller.student,
                    epoch,
                    best_accuracy,
                    args,
                    projector=distiller.projector.state_dict(),
                ),
                checkpoint_path,
            )
            marker = " saved_best"
        else:
            marker = ""

        current_lr = optimizer.param_groups[0]["lr"]
        denominator = max(1, sample_count)
        log(
            f"[STUDENT][{epoch:03d}/{args.student_epochs:03d}] "
            f"loss={total_loss_sum / denominator:.4f} "
            f"cls={cls_loss_sum / denominator:.4f} "
            f"feat={feature_loss_sum / denominator:.4f} "
            f"train_acc={100.0 * correct / denominator:.2f}% "
            f"val_acc={val_accuracy:.2f}% best={best_accuracy:.2f}% "
            f"lr={current_lr:.6g} time={time.time() - epoch_start:.1f}s{marker}"
        )
        scheduler.step()

    log(f"[STUDENT] best_checkpoint={checkpoint_path} top1={best_accuracy:.2f}%")
    return best_accuracy


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.smoke:
        # `--smoke` alone should be safe; explicit epoch flags still take precedence.
        if args.teacher_epochs == 300:
            args.teacher_epochs = 1
        if args.student_epochs == 300:
            args.student_epochs = 1

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = "lg_cifar100_deit_tiny_smoke" if args.smoke else "lg_cifar100_deit_tiny_full"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    teacher_checkpoint = run_dir / "teacher_resnet56_best.pt"
    student_checkpoint = run_dir / "student_deit_tiny_lg_best.pt"

    log("=" * 72)
    log("LG-STYLE CIFAR-100 / RESNET56 -> DEIT-TI")
    log("=" * 72)
    log(f"[ENV] python={sys.version.split()[0]} torch={torch.__version__}")
    log(f"[ENV] torchvision={torchvision.__version__} timm={timm.__version__}")
    log(f"[ENV] cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        log(f"[ENV] gpu_name={torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        log(f"[ENV] gpu_memory_gib={props.total_memory / (1024**3):.2f}")
    log(f"[ENV] device={device} amp={amp} seed={args.seed}")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] run_dir={run_dir.resolve()}")
    log(f"[MODE] smoke={args.smoke} teacher_epochs={args.teacher_epochs} student_epochs={args.student_epochs}")
    log("[NOTE] This is an LG-style scaffold, not an exact official LG reproduction.")

    train_loader, test_loader = build_loaders(args, device)
    teacher = CIFARResNet56().to(device)
    distiller = LGDistiller().to(device)
    log(f"[MODEL] teacher_params={count_parameters(teacher):,}")
    log(
        f"[MODEL] student_params={count_parameters(distiller.student):,} "
        f"projector_params={count_parameters(distiller.projector):,}"
    )

    experiment_start = time.time()
    if args.teacher_checkpoint is not None:
        if not args.teacher_checkpoint.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")
        teacher_best = load_teacher_checkpoint(teacher, args.teacher_checkpoint, device)
        teacher_checkpoint_used = args.teacher_checkpoint
    else:
        teacher_best = train_teacher(
            teacher,
            train_loader,
            test_loader,
            device,
            args,
            teacher_checkpoint,
            amp,
        )
        teacher_checkpoint_used = teacher_checkpoint

    student_best = train_student(
        distiller,
        teacher,
        train_loader,
        test_loader,
        device,
        args,
        student_checkpoint,
        amp,
    )
    elapsed = time.time() - experiment_start

    summary = {
        "method": "LG-style",
        "dataset": "CIFAR-100",
        "teacher": "ResNet56",
        "student": "DeiT-Ti",
        "teacher_best_top1": teacher_best,
        "student_best_top1": student_best,
        "reference_lg_top1": 77.38,
        "smoke": args.smoke,
        "elapsed_seconds": elapsed,
        "teacher_checkpoint": str(teacher_checkpoint_used.resolve()),
        "student_checkpoint": str(student_checkpoint.resolve()),
    }
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    log("=" * 72)
    log(
        f"[FINAL_RESULT] student_best_top1={student_best:.2f}% "
        f"teacher_best_top1={teacher_best:.2f}% reference_lg_top1=77.38%"
    )
    log(f"[FINAL_RESULT] elapsed={elapsed / 60.0:.1f}min summary={summary_path.resolve()}")
    log(f"[FINAL_RESULT] best_checkpoint={student_checkpoint.resolve()}")
    log("[DONE] Training completed successfully; resources may be released.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Training did not complete.")
        raise
