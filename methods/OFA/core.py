#!/usr/bin/env python3
"""Train DeiT-Ti with official-behavior-based One-for-All KD."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from teacher_checkpoints import DEFAULT_CHECKPOINT_ROOT, load_teacher
from methods.OFA.official_ofa import OFAProjector, ofa_loss


TIMM_VERSION = "1.0.27"
OFFICIAL_REPOSITORY = "https://github.com/Hao840/OFAKD"
OFFICIAL_COMMIT = "f7bb896cac9879040800bde08a8cc2057a904c52"
STUDENT_EMBED_DIM = 192
STAGE_TO_BLOCK = {1: 1, 2: 3, 3: 9, 4: 11}
STUDENT_MODELS = {
    "deit_ti": "deit_tiny_patch16_224",
    "convit": "convit_tiny",
    "pit": "pit_ti_224",
    "pvtv2": "pvt_v2_b0",
}
PENDING_OFFICIAL_INTEGRATION = ("cvt", "t2t_7", "t2t_14")
NUM_CLASSES = {"cifar100": 100, "flowers102": 102, "chaoyang": 4}
VANILLA_TOP1 = {
    "cifar100": {
        "deit_ti": 65.08,
        "convit": 74.87,
        "cvt": 74.29,
        "pit": 73.16,
        "pvtv2": 77.21,
        "t2t_7": 68.00,
        "t2t_14": 69.93,
    },
    "flowers102": {
        "deit_ti": 50.06,
        "convit": 57.45,
        "cvt": 60.82,
        "pit": 56.12,
        "pvtv2": 67.89,
        "t2t_7": 66.14,
        "t2t_14": 64.95,
    },
    "chaoyang": {
        "deit_ti": 82.00,
        "convit": 80.93,
        "cvt": 80.04,
        "pit": 81.53,
        "pvtv2": 82.52,
        "t2t_7": 78.78,
        "t2t_14": 75.04,
    },
}


def log(message: str = "") -> None:
    print(message, flush=True)


def ensure_timm() -> Any:
    try:
        timm = importlib.import_module("timm")
    except ModuleNotFoundError:
        log(f"[BOOT] timm not found; installing timm=={TIMM_VERSION}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", f"timm=={TIMM_VERSION}"]
        )
        timm = importlib.import_module("timm")
        log("[BOOT] timm installation completed")
    if timm.__version__ != TIMM_VERSION:
        log(
            f"[BOOT][WARN] expected timm={TIMM_VERSION}, found timm={timm.__version__}; "
            "the model API may differ"
        )
    return timm


def install_signal_handlers() -> None:
    def handle_signal(signum: int, frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        log("=" * 72)
        log(f"[FATAL][SIGNAL] Received {signal_name}; external termination requested.")
        if frame is not None:
            traceback.print_stack(frame)
        log("[FATAL] OFA training was interrupted before normal completion.")
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(NUM_CLASSES), default="cifar100")
    parser.add_argument("--student", choices=("deit_ti",), default="deit_ti")
    parser.add_argument(
        "--protocol-name",
        type=str,
        default="manual",
        help="Recorded name of the dataset-specific training protocol.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--timing-run",
        action="store_true",
        help="Use the full dataset for two epochs and estimate the planned full run.",
    )
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
    parser.add_argument("--ofa-eps", type=float, default=1.0)
    parser.add_argument("--ofa-stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--ofa-loss-weight", type=float, default=1.0)
    parser.add_argument("--kd-loss-weight", type=float, default=1.0)
    parser.add_argument("--gt-loss-weight", type=float, default=1.0)
    parser.add_argument("--ofa-temperature", type=float, default=1.0)
    parser.add_argument("--feature-grid", type=int, default=14)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--amp",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use CUDA autocast when CUDA is available.",
    )
    return parser.parse_args()


def finalize_args(args: argparse.Namespace) -> None:
    args.planned_epochs = args.student_epochs
    if args.timing_run:
        args.student_epochs = 2
    if args.data_dir is None:
        args.data_dir = (
            Path("/app/data/chaoyang") if args.dataset == "chaoyang" else Path("./data")
        )
    if args.run_name is None:
        suffix = (
            "timing_2ep"
            if args.timing_run
            else ("smoke" if args.smoke else f"{args.student_epochs}ep")
        )
        args.run_name = f"ofa_{args.dataset}_{args.student}_{suffix}"

    positive_fields = (
        "student_epochs",
        "batch_size",
        "image_size",
        "smoke_train_samples",
        "smoke_test_samples",
        "lr",
        "feature_grid",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if args.ofa_eps <= 0:
        raise ValueError("--ofa-eps must be positive")
    if args.ofa_temperature <= 0:
        raise ValueError("--ofa-temperature must be positive")
    if min(args.gt_loss_weight, args.kd_loss_weight, args.ofa_loss_weight) < 0:
        raise ValueError("OFA loss weights must be non-negative")
    args.ofa_stages = tuple(args.ofa_stages)
    if (
        not args.ofa_stages
        or tuple(sorted(args.ofa_stages)) != args.ofa_stages
        or len(set(args.ofa_stages)) != len(args.ofa_stages)
        or any(stage not in STAGE_TO_BLOCK for stage in args.ofa_stages)
    ):
        raise ValueError("--ofa-stages must be a sorted unique subset of 1 2 3 4")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.image_size != 224:
        raise ValueError("The fixed dataset protocols require --image-size 224")
    if args.feature_grid != 14:
        raise ValueError("DeiT-Ti patch features require --feature-grid 14")


def build_loaders(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    if args.dataset == "cifar100":
        from train_teacher_cifar100 import build_loaders as build_cifar100_loaders

        return build_cifar100_loaders(args, device)
    if args.dataset == "flowers102":
        from train_teacher_flowers import build_loaders as build_flowers_loaders

        args.train_split = "trainval"
        args.eval_split = "test"
        return build_flowers_loaders(args, device)

    from train_teacher_chaoyang import build_loaders as build_chaoyang_loaders

    train_loader, test_loader, _ = build_chaoyang_loaders(args, device)
    return train_loader, test_loader


def create_student(timm: Any, student_key: str, num_classes: int) -> nn.Module:
    timm_name = STUDENT_MODELS[student_key]
    try:
        return timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    except Exception as error:
        available = timm.list_models(f"*{student_key.split('_')[0]}*")[:20]
        raise RuntimeError(
            f"Failed to create timm model {timm_name!r} for {student_key}; "
            f"nearby models={available}"
        ) from error


def create_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def top1_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == targets).sum().item())


def forward_student_features(
    student: nn.Module,
    images: torch.Tensor,
    stages: tuple[int, ...],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Mirror timm's DeiT forward while retaining official OFA stage tokens."""
    selected_blocks = {STAGE_TO_BLOCK[stage]: stage for stage in stages}
    tokens = student.patch_embed(images)
    tokens = student._pos_embed(tokens)
    tokens = student.patch_drop(tokens)
    tokens = student.norm_pre(tokens)
    captured: dict[int, torch.Tensor] = {}
    for block_index, block in enumerate(student.blocks):
        tokens = block(tokens)
        if block_index in selected_blocks:
            captured[selected_blocks[block_index]] = tokens
    tokens = student.norm(tokens)
    logits = student.forward_head(tokens)
    return [captured[stage] for stage in stages], logits


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    projector: OFAProjector,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> tuple[float, float, float, float, float]:
    student.train()
    teacher.eval()
    projector.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_ofa = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad(), autocast_context(amp_enabled):
            teacher_logits = teacher(images)
        with autocast_context(amp_enabled):
            student_features, student_logits = forward_student_features(
                student,
                images,
                args.ofa_stages,
            )
            projected_logits = projector(student_features)
            ce = F.cross_entropy(
                student_logits,
                targets,
                label_smoothing=args.label_smoothing,
            )
        target_mask = F.one_hot(
            targets,
            num_classes=student_logits.shape[-1],
        ).to(dtype=torch.float32)
        final_distillation = ofa_loss(
            student_logits.float(),
            teacher_logits.float(),
            target_mask,
            args.ofa_eps,
            args.ofa_temperature,
        )
        intermediate_distillation = sum(
            ofa_loss(
                head_logits.float(),
                teacher_logits.float(),
                target_mask,
                args.ofa_eps,
                args.ofa_temperature,
            )
            for head_logits in projected_logits
        )
        loss = (
            args.gt_loss_weight * ce.float()
            + args.kd_loss_weight * final_distillation
            + args.ofa_loss_weight * intermediate_distillation
        )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_ce += float(ce.detach()) * batch_size
        total_kd += float(final_distillation.detach()) * batch_size
        total_ofa += float(intermediate_distillation.detach()) * batch_size
        correct += top1_correct(student_logits.detach(), targets)

    denominator = max(1, total)
    return (
        total_loss / denominator,
        total_ce / denominator,
        total_kd / denominator,
        total_ofa / denominator,
        100.0 * correct / denominator,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(amp_enabled):
            logits = model(images)
        correct += top1_correct(logits, targets)
        total += targets.size(0)
    return 100.0 * correct / max(1, total)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    effective_warmup = warmup_epochs if epochs > warmup_epochs else 0

    def lr_multiplier(epoch_index: int) -> float:
        if effective_warmup and epoch_index < effective_warmup:
            return (epoch_index + 1) / effective_warmup
        cosine_epochs = max(1, epochs - effective_warmup)
        progress = (epoch_index - effective_warmup) / cosine_epochs
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier), effective_warmup


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def public_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def checkpoint_payload(
    student: nn.Module,
    projector: OFAProjector,
    epoch: int,
    accuracy: float,
    best_accuracy: float,
    args: argparse.Namespace,
    teacher_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": student.state_dict(),
        "ofa_projector": projector.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "method": "OFA",
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
    estimated_300 = average_epoch * 300
    estimated_planned = average_epoch * args.planned_epochs
    summary = {
        "status": "complete" if latest_epoch == args.student_epochs else "running",
        "method": "OFA",
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
        "gain_over_vanilla_pp": best_accuracy - VANILLA_TOP1[args.dataset][args.student],
        "epoch_times": epoch_times,
        "avg_epoch_seconds": average_epoch,
        "planned_epochs": args.planned_epochs,
        "estimated_planned_seconds": estimated_planned,
        "estimated_planned_human": format_duration(estimated_planned),
        "estimated_300_seconds": estimated_300,
        "estimated_300_human": format_duration(estimated_300),
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
    log("OFFICIAL-BEHAVIOR-BASED OFA / RESNET56 -> DEIT-TI")
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
        log(f"[ENV] gpu_memory_gib={properties.total_memory / (1024**3):.2f}")
    log(f"[ENV] device={device} amp={amp_enabled} seed={args.seed}")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] teacher_root={args.teacher_root.resolve()}")
    log(f"[PATH] run_dir={run_dir.resolve()}")
    log(
        f"[MODE] smoke={args.smoke} timing_run={args.timing_run} "
        f"student_epochs={args.student_epochs} planned_epochs={args.planned_epochs}"
    )
    log(
        f"[PROTOCOL] name={args.protocol_name} optimizer=AdamW "
        f"lr={args.lr} weight_decay={args.weight_decay} "
        f"warmup={args.warmup_epochs} cosine batch={args.batch_size} "
        f"image={args.image_size}"
    )
    log(
        f"[OFA] loss={args.gt_loss_weight}*CE+{args.kd_loss_weight}*final_OFA+"
        f"{args.ofa_loss_weight}*sum(stage_OFA) eps={args.ofa_eps} "
        f"temperature={args.ofa_temperature} stages={args.ofa_stages}"
    )
    log(
        f"[OFFICIAL] repository={OFFICIAL_REPOSITORY} "
        f"commit={OFFICIAL_COMMIT}"
    )
    log(
        "[ADAPTER] teacher=final class logits student_blocks="
        f"{tuple(STAGE_TO_BLOCK[stage] for stage in args.ofa_stages)} "
        "token_grid=14x14 patch_merge=14x14->7x7 "
        "stage_heads=official_transformer_projectors"
    )

    train_loader, test_loader = build_loaders(args, device)
    teacher, teacher_payload, teacher_spec = load_teacher(
        args.dataset,
        device=device,
        checkpoint_root=args.teacher_root,
    )
    student = create_student(timm, args.student, NUM_CLASSES[args.dataset]).to(device)
    projector = OFAProjector(
        stages=args.ofa_stages,
        embed_dim=STUDENT_EMBED_DIM,
        patch_grid=args.feature_grid,
        num_classes=NUM_CLASSES[args.dataset],
    ).to(device)

    with torch.no_grad():
        probe = torch.zeros(2, 3, args.image_size, args.image_size, device=device)
        teacher_logits_probe = teacher(probe)
        student_probe, logits_probe = forward_student_features(
            student,
            probe,
            args.ofa_stages,
        )
        projected_probe = projector(student_probe)
    expected_logits_shape = (2, NUM_CLASSES[args.dataset])
    expected_token_shape = (2, 197, STUDENT_EMBED_DIM)
    if tuple(teacher_logits_probe.shape) != expected_logits_shape:
        raise RuntimeError(f"Unexpected teacher logits: {tuple(teacher_logits_probe.shape)}")
    if [tuple(feature.shape) for feature in student_probe] != [
        expected_token_shape for _ in args.ofa_stages
    ]:
        raise RuntimeError(
            f"Unexpected student tokens: {[tuple(feature.shape) for feature in student_probe]}"
        )
    if tuple(logits_probe.shape) != expected_logits_shape:
        raise RuntimeError(f"Unexpected student logits: {tuple(logits_probe.shape)}")
    if [tuple(logits.shape) for logits in projected_probe] != [
        expected_logits_shape for _ in args.ofa_stages
    ]:
        raise RuntimeError(
            f"Unexpected projected logits: {[tuple(logits.shape) for logits in projected_probe]}"
        )
    log(
        f"[TEACHER] selected={teacher_spec['selected_kind']} epoch={teacher_payload['epoch']} "
        f"top1={float(teacher_payload['accuracy']):.2f}% sha256={teacher_spec['sha256']}"
    )
    log(
        f"[MODEL] teacher_params={count_parameters(teacher):,} "
        f"student={STUDENT_MODELS[args.student]} "
        f"student_params={count_parameters(student):,} "
        f"ofa_trainable_params={count_parameters(projector):,}"
    )
    log(
        f"[FEATURE_CHECK] teacher_logits={expected_logits_shape} "
        f"student_tokens={[expected_token_shape for _ in args.ofa_stages]} "
        f"projected_logits={[expected_logits_shape for _ in args.ofa_stages]}"
    )

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
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
        f"[STUDENT] optimizer=adamw lr={args.lr} weight_decay={args.weight_decay} "
        f"epochs={args.student_epochs} effective_warmup={effective_warmup}"
    )

    best_accuracy = 0.0
    latest_accuracy = 0.0
    epoch_times: list[float] = []
    training_start = time.time()

    for epoch_index in range(args.student_epochs):
        epoch = epoch_index + 1
        epoch_start = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        loss, ce, final_distillation, intermediate_distillation, train_accuracy = train_one_epoch(
            student,
            teacher,
            projector,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            amp_enabled,
        )
        latest_accuracy = evaluate(student, test_loader, device, amp_enabled)
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        previous_best = best_accuracy
        best_accuracy = max(best_accuracy, latest_accuracy)

        payload = checkpoint_payload(
            student,
            projector,
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
            f"[OFA][{epoch:03d}/{args.student_epochs:03d}] loss={loss:.4f} "
            f"ce={ce:.4f} final={final_distillation:.4f} "
            f"intermediate={intermediate_distillation:.4f} "
            f"train_acc={train_accuracy:.2f}% "
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
        f"[FINAL_RESULT] ofa_best_top1={best_accuracy:.2f}% "
        f"vanilla_top1={vanilla:.2f}% gain_over_vanilla={best_accuracy - vanilla:+.2f}pp"
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
    log("[DONE] OFA training completed successfully; resources may be released.")


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] OFA training did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
