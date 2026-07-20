#!/usr/bin/env python3
"""Train an Oxford Flowers ResNet56 teacher for downstream KD experiments.

This script follows the same H200-friendly pattern as train_teacher_cifar100.py:

- download/verify the dataset automatically,
- train a ResNet56 teacher from scratch,
- print concise epoch logs for GitHub Issue comments,
- save best/latest checkpoints and summary.json under the requested output dir.

The paper draft specifies Flowers + ResNet56 and reports a teacher Top-1 of
66.33%, but does not fully specify the teacher recipe.  Therefore the optimizer,
augmentation, seed, and checkpoint rule are scaffold choices kept consistent with
the CIFAR-100 teacher run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import Flowers102
from torchvision.datasets.utils import check_integrity, extract_archive
from torchvision.transforms import InterpolationMode

from train_teacher_cifar100 import CIFARResNet56


FLOWERS_MEAN = (0.485, 0.456, 0.406)
FLOWERS_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 102
REFERENCE_TEACHER_TOP1 = 66.33

FLOWERS_BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
FLOWERS_FILES = {
    "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
    "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
    "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
}


def log(message: str = "") -> None:
    print(message, flush=True)


def install_signal_handlers() -> None:
    def handle_signal(signum: int, frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        log("=" * 72)
        log(f"[FATAL][SIGNAL] Received {signal_name}; external termination requested.")
        if frame is not None:
            traceback.print_stack(frame)
        log("[FATAL] Flowers teacher training was interrupted before normal completion.")
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)


def ensure_python_package(module_name: str, pip_spec: str) -> None:
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        pass

    log(f"[BOOT] {module_name} not found; installing {pip_spec}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_spec])
    importlib.import_module(module_name)
    log(f"[BOOT] {module_name} installation completed")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Oxford Flowers ResNet56 teacher")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--teacher-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-train-samples", type=int, default=512)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    parser.add_argument(
        "--train-split",
        choices=("train", "trainval"),
        default="trainval",
        help="Oxford Flowers official split used for training. trainval uses train+val.",
    )
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
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
            transforms.Normalize(FLOWERS_MEAN, FLOWERS_STD),
        ]
    )
    resize_size = int(round(image_size / 0.875))
    test_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(FLOWERS_MEAN, FLOWERS_STD),
        ]
    )
    return train_transform, test_transform


def deterministic_subset(dataset: Dataset[Any], size: int, seed: int) -> Dataset[Any]:
    size = min(size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def flowers_base(root: Path) -> Path:
    return root / "flowers-102"


def flowers_files_ready(root: Path) -> bool:
    base = flowers_base(root)
    image_dir = base / "jpg"
    if not image_dir.is_dir():
        return False
    image_count = len(list(image_dir.glob("image_*.jpg")))
    if image_count != 8189:
        return False
    for key in ("label", "setid"):
        filename, md5 = FLOWERS_FILES[key]
        if not check_integrity(str(base / filename), md5):
            return False
    return True


def download_file(url: str, destination: Path, expected_md5: str, source_name: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Ours-H200-Flowers/1.0)",
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
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch: expected={expected_md5} actual={actual_md5}")
        partial.replace(destination)
        log(
            f"[DATA] Download verified source={source_name} "
            f"size={downloaded / (1024**2):.1f} MiB md5={actual_md5}"
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def ensure_flowers_available(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    base = flowers_base(root)
    base.mkdir(parents=True, exist_ok=True)
    if flowers_files_ready(root):
        log("[DATA] Existing Oxford Flowers files passed integrity checks")
        return

    image_archive, image_md5 = FLOWERS_FILES["image"]
    archive = base / image_archive
    if check_integrity(str(archive), image_md5):
        log(f"[DATA] Found verified image archive; extracting {archive}")
        extract_archive(str(archive), str(base))
    elif archive.exists():
        log(f"[DATA][WARN] Removing incomplete or invalid archive: {archive}")
        archive.unlink()

    if not (base / "jpg").is_dir():
        download_file(FLOWERS_BASE_URL + image_archive, archive, image_md5, "Oxford official images")
        log("[DATA] Extracting verified Oxford Flowers image archive")
        extract_archive(str(archive), str(base))

    for key, source_name in (("label", "Oxford official labels"), ("setid", "Oxford official splits")):
        filename, md5 = FLOWERS_FILES[key]
        path = base / filename
        if check_integrity(str(path), md5):
            log(f"[DATA] Existing {filename} passed integrity check")
            continue
        if path.exists():
            log(f"[DATA][WARN] Removing incomplete or invalid file: {path}")
            path.unlink()
        download_file(FLOWERS_BASE_URL + filename, path, md5, source_name)

    if not flowers_files_ready(root):
        raise RuntimeError("Oxford Flowers files failed integrity checks after download")
    log("[DATA] Oxford Flowers ready")


def build_loaders(args: argparse.Namespace, device: torch.device) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    ensure_python_package("scipy", "scipy>=1.10")
    train_transform, test_transform = make_transforms(args.image_size)
    log(f"[DATA] Oxford Flowers root={args.data_dir.resolve()}")
    log("[DATA] Preparing Oxford Flowers with official URL and MD5 checks")
    ensure_flowers_available(args.data_dir)

    if args.train_split == "trainval":
        train_parts = [
            Flowers102(root=args.data_dir, split="train", transform=train_transform, download=False),
            Flowers102(root=args.data_dir, split="val", transform=train_transform, download=False),
        ]
        train_dataset: Dataset[Any] = ConcatDataset(train_parts)
        log(
            f"[DATA] Train split ready: split=train+val samples="
            f"{len(train_parts[0])}+{len(train_parts[1])}={len(train_dataset)}"
        )
    else:
        train_dataset = Flowers102(
            root=args.data_dir, split="train", transform=train_transform, download=False
        )
        log(f"[DATA] Train split ready: split=train samples={len(train_dataset)}")

    test_dataset: Dataset[Any] = Flowers102(
        root=args.data_dir, split=args.eval_split, transform=test_transform, download=False
    )
    log(f"[DATA] Eval split ready: split={args.eval_split} samples={len(test_dataset)}")

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
        f"num_workers={args.num_workers} smoke={args.smoke} "
        f"train_split={args.train_split} eval_split={args.eval_split}"
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
    result: Dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


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
        "model_name": "ResNet56",
        "dataset": "Oxford Flowers 102",
        "num_classes": NUM_CLASSES,
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
        "dataset": "Oxford Flowers 102",
        "num_classes": NUM_CLASSES,
        "teacher_epochs": args.teacher_epochs,
        "latest_epoch": latest_epoch,
        "best_top1": best_accuracy,
        "latest_top1": latest_accuracy,
        "reference_teacher_top1": REFERENCE_TEACHER_TOP1,
        "gap_to_reference": best_accuracy - REFERENCE_TEACHER_TOP1,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
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
        "teacher_resnet56_flowers_smoke" if args.smoke else "teacher_resnet56_flowers_full"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = run_dir / "teacher_resnet56_flowers_best.pt"
    latest_checkpoint = run_dir / "teacher_resnet56_flowers_latest.pt"
    summary_path = run_dir / "summary.json"

    log("=" * 72)
    log("TRAIN OXFORD FLOWERS RESNET56 TEACHER")
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
    log("[NOTE] Matched to paper where explicit: Flowers, ResNet56 teacher, 224px, Top-1.")
    log("[NOTE] Teacher recipe/split details are scaffold choices because the paper does not specify them exactly.")

    train_loader, test_loader = build_loaders(args, device)
    model = CIFARResNet56(num_classes=NUM_CLASSES).to(device)
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
        previous_best = best_accuracy
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
        saved_best = latest_accuracy >= previous_best
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
    log("[DONE] Flowers teacher training completed successfully; resources may be released.")


def main() -> None:
    try:
        train_teacher(parse_args())
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Flowers teacher training did not complete.")
        raise


if __name__ == "__main__":
    main()
