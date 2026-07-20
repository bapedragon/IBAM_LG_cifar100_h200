#!/usr/bin/env python3
"""Train a CIFAR-100 ResNet56 teacher for downstream KD experiments.

Artifacts are written under the requested output directory. On the KAU H200
runner, use ``--output-dir /app/output`` so the runner collects the result after
the Pod is released.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
import traceback
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.datasets.utils import check_integrity, extract_archive
from torchvision.transforms import InterpolationMode


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
NUM_CLASSES = 100
REFERENCE_TEACHER_TOP1 = 70.43
CIFAR100_SOURCES = (
    (
        "Hugging Face mirror",
        "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/"
        "201a32345d2c6b970e1a36c582930c83e09c96d2/cifar-100-python.tar.gz",
    ),
    (
        "SJTU mirror",
        "https://scidata.sjtu.edu.cn/records/xk2s3-v1e12/files/"
        "cifar-100-python.tar.gz?download=1",
    ),
    ("Toronto official", CIFAR100.url),
)


def log(message: str = "") -> None:
    print(message, flush=True)


def install_signal_handlers() -> None:
    def handle_signal(signum: int, frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        log("=" * 72)
        log(f"[FATAL][SIGNAL] Received {signal_name}; external termination requested.")
        if frame is not None:
            traceback.print_stack(frame)
        log("[FATAL] Teacher training was interrupted before normal completion.")
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.fc(torch.flatten(self.pool(x), 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-100 ResNet56 teacher")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--teacher-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-train-samples", type=int, default=1024)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument(
        "--amp",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use CUDA autocast when CUDA is available.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for field in (
        "teacher_epochs",
        "batch_size",
        "image_size",
        "smoke_train_samples",
        "smoke_test_samples",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.image_size != 224:
        raise ValueError("This teacher scaffold currently expects --image-size 224")


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


def cifar100_files_ready(root: Path) -> bool:
    base = root / CIFAR100.base_folder
    required_files = list(CIFAR100.train_list) + list(CIFAR100.test_list)
    required_files.append((CIFAR100.meta["filename"], CIFAR100.meta["md5"]))
    return all(check_integrity(str(base / filename), md5) for filename, md5 in required_files)


def download_cifar100_archive(url: str, destination: Path, source_name: str) -> None:
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Ours-H200-CIFAR100/1.0)",
            "Accept-Encoding": "identity",
        },
    )
    digest = hashlib.md5()
    downloaded = 0
    next_report_percent = 10
    next_report_bytes = 32 * 1024 * 1024

    log(f"[DATA] Download source={source_name} url={url}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as file:
            total = int(response.headers.get("Content-Length", "0"))
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)

                if total > 0:
                    percent = int(downloaded * 100 / total)
                    if percent >= next_report_percent:
                        log(
                            f"[DATA] Download progress source={source_name} "
                            f"{min(percent, 100)}% ({downloaded / (1024**2):.1f} MiB)"
                        )
                        next_report_percent += 10
                elif downloaded >= next_report_bytes:
                    log(
                        f"[DATA] Download progress source={source_name} "
                        f"{downloaded / (1024**2):.1f} MiB"
                    )
                    next_report_bytes += 32 * 1024 * 1024

        actual_md5 = digest.hexdigest()
        if actual_md5 != CIFAR100.tgz_md5:
            raise RuntimeError(
                f"MD5 mismatch: expected={CIFAR100.tgz_md5} actual={actual_md5}"
            )
        partial.replace(destination)
        log(
            f"[DATA] Download verified source={source_name} "
            f"size={downloaded / (1024**2):.1f} MiB md5={actual_md5}"
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def ensure_cifar100_available(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if cifar100_files_ready(root):
        log("[DATA] Existing CIFAR-100 files passed integrity checks")
        return

    archive = root / CIFAR100.filename
    if check_integrity(str(archive), CIFAR100.tgz_md5):
        log(f"[DATA] Found verified archive; extracting {archive}")
        extract_archive(str(archive), str(root))
        if cifar100_files_ready(root):
            log("[DATA] CIFAR-100 extraction and integrity checks completed")
            return
    elif archive.exists():
        log(f"[DATA][WARN] Removing incomplete or invalid archive: {archive}")
        archive.unlink()

    failures = []
    for source_name, url in CIFAR100_SOURCES:
        for attempt in range(1, 3):
            try:
                log(f"[DATA] Attempt source={source_name} try={attempt}/2")
                download_cifar100_archive(url, archive, source_name)
                log(f"[DATA] Extracting verified archive from {source_name}")
                extract_archive(str(archive), str(root))
                if not cifar100_files_ready(root):
                    raise RuntimeError("extracted CIFAR-100 files failed integrity checks")
                log(f"[DATA] CIFAR-100 ready from {source_name}")
                return
            except Exception as error:
                message = f"{source_name} try={attempt}: {type(error).__name__}: {error}"
                failures.append(message)
                log(f"[DATA][WARN] {message}")
                archive.unlink(missing_ok=True)
                if attempt < 2:
                    time.sleep(3)

    details = " | ".join(failures)
    raise RuntimeError(f"All CIFAR-100 download sources failed: {details}")


def build_loaders(args: argparse.Namespace, device: torch.device) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    train_transform, test_transform = make_transforms(args.image_size)
    log(f"[DATA] CIFAR-100 root={args.data_dir.resolve()}")
    log("[DATA] Preparing CIFAR-100 with verified mirror fallback")
    ensure_cifar100_available(args.data_dir)
    log("[DATA] Preparing train split from verified local files")
    train_dataset: Dataset[Any] = CIFAR100(
        root=args.data_dir, train=True, transform=train_transform, download=False
    )
    log(f"[DATA] Train split ready: samples={len(train_dataset)}")
    log("[DATA] Preparing test split from verified local files")
    test_dataset: Dataset[Any] = CIFAR100(
        root=args.data_dir, train=False, transform=test_transform, download=False
    )
    log(f"[DATA] Test split ready: samples={len(test_dataset)}")

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


def public_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def checkpoint_payload(
    model: nn.Module,
    epoch: int,
    accuracy: float,
    args: argparse.Namespace,
    *,
    epoch_times: list[float],
    best_accuracy: float,
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "model_name": "CIFARResNet56",
        "dataset": "CIFAR-100",
        "reference_teacher_top1": REFERENCE_TEACHER_TOP1,
        "epoch_times": epoch_times,
        "args": public_args(args),
    }


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def write_summary(
    summary_path: Path,
    args: argparse.Namespace,
    *,
    best_accuracy: float,
    latest_accuracy: float,
    latest_epoch: int,
    epoch_times: list[float],
    elapsed_seconds: float,
    best_checkpoint: Path,
    latest_checkpoint: Path,
) -> None:
    average_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    estimated_300_seconds = average_epoch * 300 if average_epoch else 0.0
    summary = {
        "mode": "smoke" if args.smoke else "full",
        "model": "ResNet56",
        "dataset": "CIFAR-100",
        "teacher_epochs": args.teacher_epochs,
        "latest_epoch": latest_epoch,
        "best_top1": best_accuracy,
        "latest_top1": latest_accuracy,
        "reference_teacher_top1": REFERENCE_TEACHER_TOP1,
        "gap_to_reference": best_accuracy - REFERENCE_TEACHER_TOP1,
        "epoch_times": epoch_times,
        "avg_epoch_seconds": average_epoch,
        "estimated_300_seconds": estimated_300_seconds,
        "estimated_300_human": format_duration(estimated_300_seconds),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": format_duration(elapsed_seconds),
        "best_checkpoint": str(best_checkpoint.resolve()),
        "latest_checkpoint": str(latest_checkpoint.resolve()),
        "args": public_args(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def train_teacher(args: argparse.Namespace) -> None:
    install_signal_handlers()
    validate_args(args)
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    run_name = args.run_name or (
        "teacher_resnet56_cifar100_smoke" if args.smoke else "teacher_resnet56_cifar100_full"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = run_dir / "teacher_resnet56_best.pt"
    latest_checkpoint = run_dir / "teacher_resnet56_latest.pt"
    summary_path = run_dir / "summary.json"

    log("=" * 72)
    log("TRAIN CIFAR-100 RESNET56 TEACHER")
    log("=" * 72)
    log(f"[ENV] python={sys.version.split()[0]} torch={torch.__version__}")
    log(f"[ENV] torchvision={torchvision.__version__}")
    log(f"[ENV] cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        log(f"[ENV] gpu_name={torch.cuda.get_device_name(0)}")
        log(f"[ENV] gpu_memory_gib={torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}")
    log(f"[ENV] device={device} amp={amp_enabled} seed={args.seed}")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] run_dir={run_dir.resolve()}")
    log(f"[PATH] best_checkpoint={best_checkpoint.resolve()}")
    log(f"[MODE] smoke={args.smoke} teacher_epochs={args.teacher_epochs}")
    log(f"[REFERENCE] paper_teacher_top1={REFERENCE_TEACHER_TOP1:.2f}%")
    log("[NOTE] Teacher recipe is a scaffold choice because the paper does not specify it exactly.")

    train_loader, test_loader = build_loaders(args, device)
    model = CIFARResNet56().to(device)
    log(f"[MODEL] teacher_params={count_parameters(model):,}")

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, warmup_epochs = make_cosine_scheduler(optimizer, args.teacher_epochs, args.warmup_epochs)
    scaler = create_grad_scaler(amp_enabled)
    criterion = nn.CrossEntropyLoss()

    log(
        f"[TEACHER] optimizer={args.optimizer} lr={args.lr} momentum={args.momentum} "
        f"weight_decay={args.weight_decay} epochs={args.teacher_epochs} "
        f"effective_warmup={warmup_epochs}"
    )
    best_accuracy = 0.0
    latest_accuracy = 0.0
    epoch_times: list[float] = []
    start_time = time.time()
    last_completed_epoch = 0

    for epoch in range(1, args.teacher_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            correct += top1_correct(logits.detach(), targets)
            total += batch_size

        scheduler.step()
        latest_accuracy = evaluate(model, test_loader, device, amp_enabled)
        best_accuracy = max(best_accuracy, latest_accuracy)
        last_completed_epoch = epoch
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        average_epoch = sum(epoch_times) / len(epoch_times)
        estimated_300_seconds = average_epoch * 300
        elapsed = time.time() - start_time

        latest_payload = checkpoint_payload(
            model,
            epoch,
            latest_accuracy,
            args,
            epoch_times=epoch_times,
            best_accuracy=best_accuracy,
        )
        torch.save(latest_payload, latest_checkpoint)
        saved_best = latest_accuracy >= best_accuracy
        if saved_best:
            torch.save(latest_payload, best_checkpoint)

        write_summary(
            summary_path,
            args,
            best_accuracy=best_accuracy,
            latest_accuracy=latest_accuracy,
            latest_epoch=epoch,
            epoch_times=epoch_times,
            elapsed_seconds=elapsed,
            best_checkpoint=best_checkpoint,
            latest_checkpoint=latest_checkpoint,
        )

        log(
            f"[TEACHER][{epoch:03d}/{args.teacher_epochs:03d}] "
            f"loss={total_loss / max(1, total):.4f} "
            f"train_acc={100.0 * correct / max(1, total):.2f}% "
            f"val_acc={latest_accuracy:.2f}% best={best_accuracy:.2f}% "
            f"lr={scheduler.get_last_lr()[0]:.6g} time={epoch_time:.1f}s "
            f"avg_epoch={average_epoch:.1f}s "
            f"est_300={format_duration(estimated_300_seconds)} "
            f"elapsed={format_duration(elapsed)}"
            + (" saved_best" if saved_best else "")
        )

    total_elapsed = time.time() - start_time
    average_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    estimated_300_seconds = average_epoch * 300 if average_epoch else 0.0

    write_summary(
        summary_path,
        args,
        best_accuracy=best_accuracy,
        latest_accuracy=latest_accuracy,
        latest_epoch=last_completed_epoch,
        epoch_times=epoch_times,
        elapsed_seconds=total_elapsed,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
    )

    log("=" * 72)
    log(
        f"[FINAL_RESULT] teacher_best_top1={best_accuracy:.2f}% "
        f"reference_teacher_top1={REFERENCE_TEACHER_TOP1:.2f}% "
        f"gap_to_reference={best_accuracy - REFERENCE_TEACHER_TOP1:+.2f}pp"
    )
    log(
        f"[TIMING] teacher_avg_epoch={average_epoch:.1f}s "
        f"estimated_300_teacher={format_duration(estimated_300_seconds)} "
        f"elapsed={format_duration(total_elapsed)}"
    )
    log(f"[FINAL_RESULT] best_checkpoint={best_checkpoint.resolve()}")
    log(f"[FINAL_RESULT] latest_checkpoint={latest_checkpoint.resolve()}")
    log(f"[FINAL_RESULT] summary={summary_path.resolve()}")
    log("[DONE] Teacher training completed successfully; resources may be released.")


def main() -> None:
    try:
        train_teacher(parse_args())
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Teacher training did not complete.")
        raise


if __name__ == "__main__":
    main()
