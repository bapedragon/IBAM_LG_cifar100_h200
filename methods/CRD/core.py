#!/usr/bin/env python3
"""Train DeiT-Ti with the authors' official CRD objective."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.CRD.official_crd import CRDLoss
from methods.KD.core import (
    NUM_CLASSES,
    STUDENT_MODELS,
    VANILLA_TOP1,
    autocast_context,
    build_loaders as build_base_loaders,
    count_parameters,
    create_grad_scaler,
    create_scheduler,
    create_student,
    ensure_timm,
    evaluate,
    format_duration,
    install_signal_handlers,
    log,
    public_args,
    seed_everything,
    top1_correct,
)
from teacher_checkpoints import DEFAULT_CHECKPOINT_ROOT, load_teacher


OFFICIAL_REPOSITORY = "https://github.com/HobbitLong/RepDistiller"
OFFICIAL_COMMIT = "b84f547c5db6a35318d4671d7d5c4de74c822403"
STUDENT_DIM = 192
TEACHER_DIM = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(NUM_CLASSES), default="cifar100")
    parser.add_argument("--student", choices=("deit_ti",), default="deit_ti")
    parser.add_argument("--protocol-name", type=str, default="manual")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timing-run", action="store_true")
    parser.add_argument("--student-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-train-samples", type=int, default=1024)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--crd-weight", type=float, default=0.8)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--nce-k", type=int, default=16384)
    parser.add_argument("--nce-t", type=float, default=0.07)
    parser.add_argument("--nce-m", type=float, default=0.5)
    parser.add_argument("--mode", choices=("exact", "relax"), default="exact")
    parser.add_argument(
        "--amp",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    return parser.parse_args()


def finalize_args(args: argparse.Namespace) -> None:
    args.planned_epochs = args.student_epochs
    if args.timing_run:
        args.student_epochs = 2
    if args.data_dir is None:
        args.data_dir = (
            Path("/app/data/chaoyang")
            if args.dataset == "chaoyang"
            else Path("./data")
        )
    if args.run_name is None:
        suffix = (
            "timing_2ep"
            if args.timing_run
            else ("smoke" if args.smoke else f"{args.student_epochs}ep")
        )
        args.run_name = f"crd_{args.dataset}_{args.student}_{suffix}"

    positive_fields = (
        "student_epochs",
        "batch_size",
        "image_size",
        "smoke_train_samples",
        "smoke_test_samples",
        "lr",
        "feat_dim",
        "nce_k",
        "nce_t",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if args.crd_weight < 0:
        raise ValueError("--crd-weight must be non-negative")
    if not 0 <= args.nce_m <= 1:
        raise ValueError("--nce-m must be in [0, 1]")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.image_size != 224:
        raise ValueError("The fixed dataset protocols require --image-size 224")


def labels_from_dataset(dataset: Dataset[Any]) -> list[int]:
    if isinstance(dataset, Subset):
        base_labels = labels_from_dataset(dataset.dataset)
        return [int(base_labels[index]) for index in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        labels: list[int] = []
        for part in dataset.datasets:
            labels.extend(labels_from_dataset(part))
        return labels

    for attribute in ("targets", "_labels", "labels"):
        if hasattr(dataset, attribute):
            values = getattr(dataset, attribute)
            if isinstance(values, torch.Tensor):
                values = values.tolist()
            return [int(value) for value in values]

    if hasattr(dataset, "tensors") and len(getattr(dataset, "tensors")) >= 2:
        values = getattr(dataset, "tensors")[1]
        if isinstance(values, torch.Tensor):
            values = values.tolist()
        return [int(value) for value in values]

    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in getattr(dataset, "samples")]

    raise TypeError(
        f"Cannot extract labels without loading images from {type(dataset).__name__}"
    )


class ContrastiveSampleDataset(Dataset[Any]):
    """Add official CRD instance indexes and class-negative samples."""

    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        negative_count: int,
        mode: str,
        num_classes: int,
    ) -> None:
        self.dataset = dataset
        self.negative_count = negative_count
        self.mode = mode
        self.labels = np.asarray(labels_from_dataset(dataset), dtype=np.int64)
        if len(self.labels) != len(dataset):
            raise RuntimeError(
                f"CRD label count mismatch: labels={len(self.labels)} "
                f"dataset={len(dataset)}"
            )

        all_indexes = np.arange(len(dataset), dtype=np.int64)
        self.class_positive = [
            all_indexes[self.labels == class_index]
            for class_index in range(num_classes)
        ]
        self.class_negative = [
            all_indexes[self.labels != class_index]
            for class_index in range(num_classes)
        ]
        empty_classes = [
            class_index
            for class_index, indexes in enumerate(self.class_positive)
            if len(indexes) == 0
        ]
        if empty_classes:
            raise RuntimeError(
                f"CRD training split has empty classes: {empty_classes}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        image, target = self.dataset[index]
        target = int(target)
        if target != int(self.labels[index]):
            raise RuntimeError(
                f"CRD label mismatch at index={index}: "
                f"dataset={target} cached={self.labels[index]}"
            )

        if self.mode == "exact":
            positive_index = index
        else:
            positive_index = int(np.random.choice(self.class_positive[target], 1)[0])

        negative_pool = self.class_negative[target]
        replace = self.negative_count > len(negative_pool)
        negative_index = np.random.choice(
            negative_pool,
            self.negative_count,
            replace=replace,
        )
        sample_index = np.concatenate(
            (
                np.asarray([positive_index], dtype=np.int64),
                negative_index.astype(np.int64, copy=False),
            )
        )
        return image, target, index, torch.from_numpy(sample_index)


def seed_crd_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader[Any], DataLoader[Any], int]:
    base_train_loader, test_loader = build_base_loaders(args, device)
    train_dataset = ContrastiveSampleDataset(
        base_train_loader.dataset,
        negative_count=args.nce_k,
        mode=args.mode,
        num_classes=NUM_CLASSES[args.dataset],
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_crd_worker,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    return train_loader, test_loader, len(train_dataset)


def forward_teacher_with_rep(
    teacher: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature = teacher.stem(images)
    feature = teacher.stage1(feature)
    feature = teacher.stage2(feature)
    feature = teacher.stage3(feature)
    representation = torch.flatten(teacher.pool(feature), 1)
    logits = teacher.fc(representation)
    return representation, logits


def forward_student_with_rep(
    student: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_features = student.forward_features(images)
    representation = student.forward_head(token_features, pre_logits=True)
    logits = student.get_classifier()(representation)
    return representation, logits


def checkpoint_payload(
    student: torch.nn.Module,
    epoch: int,
    accuracy: float,
    best_accuracy: float,
    args: argparse.Namespace,
    teacher_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": student.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "method": "CRD",
        "student": args.student,
        "timm_model": STUDENT_MODELS[args.student],
        "dataset": args.dataset,
        "num_classes": NUM_CLASSES[args.dataset],
        "teacher": teacher_spec,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "args": public_args(args),
    }


def write_summary(
    path: Path,
    args: argparse.Namespace,
    teacher_spec: dict[str, Any],
    *,
    latest_epoch: int,
    best_accuracy: float,
    latest_accuracy: float,
    epoch_times: list[float],
    elapsed_seconds: float,
) -> None:
    average_epoch = sum(epoch_times) / max(1, len(epoch_times))
    summary = {
        "status": (
            "complete" if latest_epoch == args.student_epochs else "running"
        ),
        "method": "CRD",
        "dataset": args.dataset,
        "student": args.student,
        "timm_model": STUDENT_MODELS[args.student],
        "teacher": teacher_spec,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "student_epochs": args.student_epochs,
        "latest_epoch": latest_epoch,
        "best_top1": best_accuracy,
        "latest_top1": latest_accuracy,
        "vanilla_top1": VANILLA_TOP1[args.dataset][args.student],
        "gain_over_vanilla_pp": (
            best_accuracy - VANILLA_TOP1[args.dataset][args.student]
        ),
        "epoch_times": epoch_times,
        "avg_epoch_seconds": average_epoch,
        "planned_epochs": args.planned_epochs,
        "estimated_planned_seconds": average_epoch * args.planned_epochs,
        "estimated_planned_human": format_duration(
            average_epoch * args.planned_epochs
        ),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": format_duration(elapsed_seconds),
        "args": public_args(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    install_signal_handlers()
    args = parse_args()
    finalize_args(args)
    seed_everything(args.seed)
    timm = ensure_timm()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = run_dir / "student_best.pt"
    latest_checkpoint = run_dir / "student_latest.pt"
    summary_path = run_dir / "summary.json"

    log("=" * 72)
    log("OFFICIAL CRD / RESNET56 -> DEIT-TI")
    log("=" * 72)
    log(
        f"[ENV] python={platform.python_version()} torch={torch.__version__} "
        f"timm={timm.__version__}"
    )
    log(
        f"[ENV] cuda_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()}"
    )
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        log(f"[ENV] gpu_name={torch.cuda.get_device_name(0)}")
        log(
            f"[ENV] gpu_memory_gib={properties.total_memory / (1024**3):.2f}"
        )
    log(f"[ENV] device={device} amp={amp_enabled} seed={args.seed}")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] teacher_root={args.teacher_root.resolve()}")
    log(f"[PATH] run_dir={run_dir.resolve()}")
    log(
        f"[MODE] smoke={args.smoke} timing_run={args.timing_run} "
        f"student_epochs={args.student_epochs} "
        f"planned_epochs={args.planned_epochs}"
    )
    log(
        f"[PROTOCOL] name={args.protocol_name} optimizer=AdamW "
        f"lr={args.lr} weight_decay={args.weight_decay} "
        f"warmup={args.warmup_epochs} cosine batch={args.batch_size} "
        f"image={args.image_size}"
    )
    log(
        f"[CRD] loss=CE+{args.crd_weight}*CRD no_logit_KL "
        f"feat_dim={args.feat_dim} nce_k={args.nce_k} "
        f"nce_t={args.nce_t} nce_m={args.nce_m} mode={args.mode}"
    )
    log(
        f"[OFFICIAL] repository={OFFICIAL_REPOSITORY} "
        f"commit={OFFICIAL_COMMIT}"
    )
    log(
        "[ADAPTER] teacher=global-pooled stage3 (64d) "
        "student=DeiT CLS pre-logits (192d)"
    )

    train_loader, test_loader, n_data = build_loaders(args, device)
    teacher, teacher_payload, teacher_spec = load_teacher(
        args.dataset,
        device=device,
        checkpoint_root=args.teacher_root,
    )
    student = create_student(
        timm, args.student, NUM_CLASSES[args.dataset]
    ).to(device)
    criterion_crd = CRDLoss(
        student_dim=STUDENT_DIM,
        teacher_dim=TEACHER_DIM,
        feature_dim=args.feat_dim,
        n_data=n_data,
        negative_count=args.nce_k,
        temperature=args.nce_t,
        momentum=args.nce_m,
    ).to(device)

    with torch.no_grad():
        probe = torch.zeros(2, 3, args.image_size, args.image_size, device=device)
        teacher_rep, _ = forward_teacher_with_rep(teacher, probe)
        student_rep, _ = forward_student_with_rep(student, probe)
    if teacher_rep.shape[1] != TEACHER_DIM:
        raise RuntimeError(
            f"Unexpected teacher representation shape: {tuple(teacher_rep.shape)}"
        )
    if student_rep.shape[1] != STUDENT_DIM:
        raise RuntimeError(
            f"Unexpected student representation shape: {tuple(student_rep.shape)}"
        )

    log(
        f"[TEACHER] selected={teacher_spec['selected_kind']} "
        f"epoch={teacher_payload['epoch']} "
        f"top1={float(teacher_payload['accuracy']):.2f}% "
        f"sha256={teacher_spec['sha256']}"
    )
    log(
        f"[MODEL] teacher_params={count_parameters(teacher):,} "
        f"student_params={count_parameters(student):,} "
        f"crd_trainable_params={count_parameters(criterion_crd):,} "
        f"n_data={n_data}"
    )

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(criterion_crd.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler, effective_warmup = create_scheduler(
        optimizer,
        args.student_epochs,
        args.warmup_epochs,
    )
    scaler = create_grad_scaler(amp_enabled)
    log(
        f"[STUDENT] optimizer=adamw lr={args.lr} "
        f"weight_decay={args.weight_decay} epochs={args.student_epochs} "
        f"effective_warmup={effective_warmup}"
    )

    best_accuracy = 0.0
    latest_accuracy = 0.0
    epoch_times: list[float] = []
    training_start = time.time()

    for epoch_index in range(args.student_epochs):
        epoch = epoch_index + 1
        epoch_start = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        student.train()
        criterion_crd.train()
        teacher.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_crd = 0.0
        correct = 0
        total = 0

        for images, targets, indexes, sampled_indexes in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            indexes = indexes.to(device, non_blocking=True)
            sampled_indexes = sampled_indexes.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad(), autocast_context(amp_enabled):
                teacher_representation, _ = forward_teacher_with_rep(
                    teacher, images
                )
            with autocast_context(amp_enabled):
                student_representation, student_logits = (
                    forward_student_with_rep(student, images)
                )
                classification_loss = F.cross_entropy(
                    student_logits,
                    targets,
                    label_smoothing=args.label_smoothing,
                )

            contrastive_loss = criterion_crd(
                student_representation.float(),
                teacher_representation.float(),
                indexes,
                sampled_indexes,
            )
            loss = (
                classification_loss.float()
                + args.crd_weight * contrastive_loss
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            total += batch_size
            total_loss += float(loss.detach()) * batch_size
            total_ce += float(classification_loss.detach()) * batch_size
            total_crd += float(contrastive_loss.detach()) * batch_size
            correct += top1_correct(student_logits.detach(), targets)

        denominator = max(1, total)
        average_loss = total_loss / denominator
        average_ce = total_ce / denominator
        average_crd = total_crd / denominator
        train_accuracy = 100.0 * correct / denominator
        latest_accuracy = evaluate(
            student, test_loader, device, amp_enabled
        )
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        previous_best = best_accuracy
        best_accuracy = max(best_accuracy, latest_accuracy)

        payload = checkpoint_payload(
            student,
            epoch,
            latest_accuracy,
            best_accuracy,
            args,
            teacher_spec,
        )
        torch.save(payload, latest_checkpoint)
        saved_best = latest_accuracy >= previous_best
        if saved_best:
            torch.save(payload, best_checkpoint)

        elapsed = time.time() - training_start
        write_summary(
            summary_path,
            args,
            teacher_spec,
            latest_epoch=epoch,
            best_accuracy=best_accuracy,
            latest_accuracy=latest_accuracy,
            epoch_times=epoch_times,
            elapsed_seconds=elapsed,
        )
        average_epoch = sum(epoch_times) / len(epoch_times)
        suffix = " saved_best" if saved_best else ""
        log(
            f"[CRD][{epoch:03d}/{args.student_epochs:03d}] "
            f"loss={average_loss:.4f} ce={average_ce:.4f} "
            f"crd={average_crd:.4f} train_acc={train_accuracy:.2f}% "
            f"val_acc={latest_accuracy:.2f}% best={best_accuracy:.2f}% "
            f"lr={epoch_lr:.6g} time={epoch_seconds:.1f}s "
            f"avg_epoch={average_epoch:.1f}s "
            f"est_planned={format_duration(average_epoch * args.planned_epochs)} "
            f"elapsed={format_duration(elapsed)}{suffix}"
        )
        scheduler.step()

    elapsed = time.time() - training_start
    average_epoch = sum(epoch_times) / len(epoch_times)
    vanilla = VANILLA_TOP1[args.dataset][args.student]
    log("=" * 72)
    log(
        f"[FINAL_RESULT] crd_best_top1={best_accuracy:.2f}% "
        f"vanilla_top1={vanilla:.2f}% "
        f"gain_over_vanilla={best_accuracy - vanilla:+.2f}pp"
    )
    log(
        f"[TIMING] avg_epoch={average_epoch:.1f}s "
        f"planned_epochs={args.planned_epochs} "
        f"estimated_total={format_duration(average_epoch * args.planned_epochs)} "
        f"elapsed={format_duration(elapsed)}"
    )
    log(f"[FINAL_RESULT] best_checkpoint={best_checkpoint.resolve()}")
    log(f"[FINAL_RESULT] latest_checkpoint={latest_checkpoint.resolve()}")
    log(f"[FINAL_RESULT] summary={summary_path.resolve()}")
    log("[DONE] CRD training completed successfully; resources may be released.")


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] CRD training did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
