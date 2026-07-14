#!/usr/bin/env python3
"""Train a Chaoyang ResNet56 teacher for downstream KD experiments.

The Chaoyang dataset is expected to be mounted by the H200 runner and must
contain the official train/test image folders plus train.json and test.json.
This script validates the official split before training, prints concise logs
for GitHub Issue comments, and saves reusable best/latest checkpoints.

The paper draft explicitly specifies Chaoyang, a ResNet56 teacher trained from
scratch, 224 x 224 input, and Top-1 evaluation.  It reports a teacher Top-1 of
77.20%, but does not fully specify the teacher training recipe.  The remaining
choices therefore match the CIFAR-100 and Flowers teacher scaffold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from train_teacher_cifar100 import CIFARResNet56


CHAOYANG_MEAN = (0.485, 0.456, 0.406)
CHAOYANG_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 4
REFERENCE_TEACHER_TOP1 = 77.20
CLASS_NAMES = {
    0: "normal",
    1: "serrated",
    2: "adenocarcinoma",
    3: "adenoma",
}
EXPECTED_COUNTS = {
    "train": {0: 1111, 1: 842, 2: 1404, 3: 664},
    "test": {0: 705, 1: 321, 2: 840, 3: 273},
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def log(message: str = "") -> None:
    print(message, flush=True)


def install_signal_handlers() -> None:
    def handle_signal(signum: int, frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        log("=" * 72)
        log(f"[FATAL][SIGNAL] Received {signal_name}; external termination requested.")
        if frame is not None:
            traceback.print_stack(frame)
        log("[FATAL] Chaoyang teacher training was interrupted before normal completion.")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Chaoyang ResNet56 teacher")
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data/chaoyang"))
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


def is_dataset_root(path: Path) -> bool:
    return all(
        (
            (path / "train").is_dir(),
            (path / "test").is_dir(),
            (path / "train.json").is_file(),
            (path / "test.json").is_file(),
        )
    )


def resolve_dataset_root(requested: Path) -> Path:
    requested = requested.expanduser().resolve()
    if not requested.exists():
        raise FileNotFoundError(f"Mounted Chaoyang path does not exist: {requested}")
    if not requested.is_dir():
        raise NotADirectoryError(f"Chaoyang --data-dir is not a directory: {requested}")
    if is_dataset_root(requested):
        return requested

    candidates = []
    for candidate in requested.rglob("train.json"):
        if "__MACOSX" in candidate.parts:
            continue
        parent = candidate.parent
        try:
            relative_depth = len(parent.relative_to(requested).parts)
        except ValueError:
            continue
        if relative_depth <= 3 and is_dataset_root(parent):
            candidates.append(parent.resolve())

    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    if not unique_candidates:
        raise FileNotFoundError(
            "Could not find train/, test/, train.json, and test.json under "
            f"{requested} (searched up to three nested levels)"
        )
    rendered = ", ".join(str(path) for path in unique_candidates)
    raise RuntimeError(f"Multiple Chaoyang dataset roots found under {requested}: {rendered}")


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
            transforms.Normalize(CHAOYANG_MEAN, CHAOYANG_STD),
        ]
    )
    resize_size = int(round(image_size / 0.875))
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CHAOYANG_MEAN, CHAOYANG_STD),
        ]
    )
    return train_transform, eval_transform


class ChaoyangDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, split: str, transform: Any) -> None:
        if split not in EXPECTED_COUNTS:
            raise ValueError(f"Unsupported Chaoyang split: {split}")
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = self._load_and_validate_records()

    def _load_and_validate_records(self) -> list[Tuple[Path, int]]:
        metadata_path = self.root / f"{self.split}.json"
        try:
            records = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid JSON metadata: {metadata_path}: {error}") from error
        if not isinstance(records, list):
            raise TypeError(f"Expected a JSON list in {metadata_path}")

        samples: list[Tuple[Path, int]] = []
        seen_names: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, dict) or "name" not in record or "label" not in record:
                raise ValueError(f"Invalid record at {metadata_path}[{index}]: {record!r}")
            relative_name = str(record["name"])
            label = int(record["label"])
            if label not in CLASS_NAMES:
                raise ValueError(f"Invalid label={label} at {metadata_path}[{index}]")
            if relative_name in seen_names:
                raise ValueError(f"Duplicate image record in {metadata_path}: {relative_name}")
            seen_names.add(relative_name)

            image_path = self.root / relative_name
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image listed in {metadata_path.name} is missing: {image_path}"
                )
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image extension: {image_path}")
            samples.append((image_path, label))

        expected_counts = EXPECTED_COUNTS[self.split]
        actual_counts = dict(sorted(Counter(label for _, label in samples).items()))
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Unexpected {self.split} class counts: expected={expected_counts} "
                f"actual={actual_counts}"
            )

        folder_images = {
            path.resolve()
            for path in (self.root / self.split).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        listed_images = {path.resolve() for path, _ in samples}
        missing_from_json = folder_images - listed_images
        missing_from_folder = listed_images - folder_images
        if missing_from_json or missing_from_folder:
            raise RuntimeError(
                f"{self.split} image/JSON mismatch: unlisted_images={len(missing_from_json)} "
                f"missing_images={len(missing_from_folder)}"
            )

        with Image.open(samples[0][0]) as image:
            first_size = image.size
            image.verify()
        if first_size != (512, 512):
            raise RuntimeError(
                f"Unexpected Chaoyang image size in {samples[0][0]}: {first_size}"
            )

        rendered_counts = " ".join(
            f"{CLASS_NAMES[label]}={actual_counts[label]}" for label in range(NUM_CLASSES)
        )
        log(
            f"[DATA] {self.split}_split verified samples={len(samples)} "
            f"{rendered_counts} first_image_size={first_size}"
        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def deterministic_subset(dataset: Dataset[Any], size: int, seed: int) -> Dataset[Any]:
    size = min(size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_loaders(
    args: argparse.Namespace, device: torch.device
) -> Tuple[DataLoader[Any], DataLoader[Any], Path]:
    dataset_root = resolve_dataset_root(args.data_dir)
    log(f"[DATA] requested_root={args.data_dir.expanduser().resolve()}")
    log(f"[DATA] resolved_root={dataset_root}")
    log("[DATA] Validating official Chaoyang train/test splits and labels")

    train_transform, eval_transform = make_transforms(args.image_size)
    train_dataset: Dataset[Any] = ChaoyangDataset(dataset_root, "train", train_transform)
    test_dataset: Dataset[Any] = ChaoyangDataset(dataset_root, "test", eval_transform)

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
    return train_loader, test_loader, dataset_root


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
    dataset_root: Path,
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "model_name": "ResNet56",
        "dataset": "Chaoyang",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "reference_teacher_top1": REFERENCE_TEACHER_TOP1,
        "dataset_root": str(dataset_root),
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
    dataset_root: Path,
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
        "dataset": "Chaoyang",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "dataset_root": str(dataset_root),
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
        "teacher_resnet56_chaoyang_smoke" if args.smoke else "teacher_resnet56_chaoyang_full"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = run_dir / "teacher_resnet56_chaoyang_best.pt"
    latest_checkpoint = run_dir / "teacher_resnet56_chaoyang_latest.pt"
    summary_path = run_dir / "summary.json"

    log("=" * 72)
    log("TRAIN CHAOYANG RESNET56 TEACHER")
    log("=" * 72)
    log(f"[ENV] python={sys.version.split()[0]} torch={torch.__version__}")
    log(f"[ENV] torchvision={torchvision.__version__}")
    log(f"[ENV] cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        log(f"[ENV] gpu_name={torch.cuda.get_device_name(0)}")
        log(f"[ENV] gpu_memory_gib={torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}")
    log(f"[ENV] device={device} amp={amp_enabled} seed={args.seed}")
    log(f"[PATH] requested_data_dir={args.data_dir.expanduser().resolve()}")
    log(f"[PATH] run_dir={run_dir.resolve()}")
    log(f"[PATH] best_checkpoint={best_checkpoint.resolve()}")
    log(f"[MODE] smoke={args.smoke} teacher_epochs={args.teacher_epochs}")
    log(f"[REFERENCE] paper_teacher_top1={REFERENCE_TEACHER_TOP1:.2f}%")
    log("[NOTE] Matched to paper where explicit: Chaoyang, ResNet56 teacher, 224px, Top-1.")
    log("[NOTE] Teacher recipe is a scaffold choice because the paper does not specify it exactly.")

    train_loader, test_loader, dataset_root = build_loaders(args, device)
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

            current_batch_size = targets.size(0)
            total_loss += float(loss.detach().item()) * current_batch_size
            correct += top1_correct(logits.detach(), targets)
            total += current_batch_size

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
            dataset_root=dataset_root,
        )
        torch.save(latest_payload, latest_checkpoint)
        saved_best = latest_accuracy >= previous_best
        if saved_best:
            torch.save(latest_payload, best_checkpoint)

        write_summary(
            summary_path,
            args,
            dataset_root=dataset_root,
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
        dataset_root=dataset_root,
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
    log("[DONE] Chaoyang teacher training completed successfully; resources may be released.")


def main() -> None:
    try:
        train_teacher(parse_args())
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Chaoyang teacher training did not complete.")
        raise


if __name__ == "__main__":
    main()
