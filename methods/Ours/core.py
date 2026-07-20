#!/usr/bin/env python3
"""Train DeiT-Ti with the provided grid-preserving Ours module."""

from __future__ import annotations

import argparse
import json
import math
import platform
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.Ours.ours import Ours
from methods.KD.core import (
    NUM_CLASSES,
    STUDENT_MODELS,
    VANILLA_TOP1,
    autocast_context,
    build_loaders,
    count_parameters,
    create_grad_scaler,
    ensure_timm,
    evaluate,
    format_duration,
    log,
    public_args,
    seed_everything,
    top1_correct,
)
from teacher_checkpoints import DEFAULT_CHECKPOINT_ROOT, load_teacher


SOURCE_SNIPPET_SHA256 = "8649078970b93d750a956994611b65cdec0c24f907d35d86f29d635e8a3b8624"
TEACHER_CHANNELS = (16, 32, 64)
STUDENT_CHANNELS = 192
NUM_STUDENT_BLOCKS = 12


class AdaptiveGuidanceController:
    """Apply the ALG loss-derivative schedule from Eqs. (10)-(19).

    ALG smooths the epoch-level LG loss over 50 epochs, differentiates that
    smoothed loss, smooths the derivative over another 50 epochs, and disables
    guidance when the final derivative is no longer below tau. Epoch one has
    no previous loss; its derivative is defined as zero for initialization and
    is never allowed to stop guidance by itself.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.schedule = args.beta_schedule
        self.beta_on = float(args.beta_on)
        self.threshold = float(args.alg_threshold)
        self.smoothing_window = int(args.alg_smoothing_window)
        self.manual_stop_epoch = args.guidance_stop_epoch
        self.active = True
        self.stop_epoch: int | None = None
        self.guidance_loss_history: list[float] = []
        self.derivative_history: list[float] = []
        self.smoothed_derivative_history: list[float | None] = []
        self.beta_history: list[float] = []

    def beta_for_epoch(self, epoch: int) -> float:
        if self.schedule == "manual_stop":
            assert self.manual_stop_epoch is not None
            active = epoch <= self.manual_stop_epoch
        else:
            active = self.active
        beta = self.beta_on if active else 0.0
        self.beta_history.append(beta)
        return beta

    def _current_derivative(self) -> float:
        epoch = len(self.guidance_loss_history)
        current = self.guidance_loss_history[-1]
        if epoch == 1:
            return 0.0
        if epoch <= self.smoothing_window:
            previous_mean = sum(self.guidance_loss_history[:-1]) / (epoch - 1)
            return (current - previous_mean) / epoch
        return (
            current - self.guidance_loss_history[-1 - self.smoothing_window]
        ) / self.smoothing_window

    def observe(self, epoch: int, guidance_loss: float) -> dict[str, Any]:
        if epoch != len(self.guidance_loss_history) + 1:
            raise ValueError(
                "AdaptiveGuidanceController observations must be consecutive: "
                f"expected={len(self.guidance_loss_history) + 1} got={epoch}"
            )
        self.guidance_loss_history.append(float(guidance_loss))

        if self.schedule == "manual_stop":
            assert self.manual_stop_epoch is not None
            if epoch >= self.manual_stop_epoch and self.stop_epoch is None:
                self.stop_epoch = self.manual_stop_epoch
            self.active = epoch < self.manual_stop_epoch
            self.derivative_history.append(0.0)
            self.smoothed_derivative_history.append(None)
            return self.state_dict()

        derivative = self._current_derivative()
        self.derivative_history.append(derivative)
        window = self.derivative_history[-self.smoothing_window :]
        smoothed_derivative = sum(window) / len(window)
        self.smoothed_derivative_history.append(smoothed_derivative)
        if epoch > 1 and self.active and smoothed_derivative >= self.threshold:
            self.active = False
            self.stop_epoch = epoch
        return self.state_dict()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule,
            "beta_on": self.beta_on,
            "active": self.active,
            "stop_epoch": self.stop_epoch,
            "threshold": self.threshold,
            "smoothing_window": self.smoothing_window,
            "manual_stop_epoch": self.manual_stop_epoch,
            "guidance_loss_history": list(self.guidance_loss_history),
            "derivative_history": list(self.derivative_history),
            "smoothed_derivative_history": list(
                self.smoothed_derivative_history
            ),
            "beta_history": list(self.beta_history),
        }


def install_signal_handlers() -> None:
    def handle_signal(signum: int, frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        log("=" * 72)
        log(f"[FATAL][SIGNAL] Received {signal_name}; external termination requested.")
        if frame is not None:
            traceback.print_stack(frame)
        log("[FATAL] Ours training was interrupted before normal completion.")
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)


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
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--drop-path-rate",
        type=float,
        default=0.0,
        help="DeiT stochastic-depth rate; kept at 0 for the shared KD protocol.",
    )
    parser.add_argument(
        "--fusion-ratio",
        type=float,
        default=0.5,
        help="Lambda in lambda*L_fuse + (1-lambda)*L_align.",
    )
    parser.add_argument(
        "--beta-schedule",
        choices=("alg", "manual_stop"),
        default="alg",
        help=(
            "alg implements ALG Eqs. (10)-(19); manual_stop uses an "
            "explicitly supplied stop epoch."
        ),
    )
    parser.add_argument(
        "--beta-on",
        type=float,
        default=2.5,
        help="ALG beta while feature guidance is active (paper default: 2.5).",
    )
    parser.add_argument(
        "--alg-threshold",
        type=float,
        default=-0.02,
        help="Tau in ALG Eq. (19) (paper default: -0.02).",
    )
    parser.add_argument(
        "--alg-smoothing-window",
        type=int,
        default=50,
        help="Window used in both ALG smoothing stages (paper default: 50).",
    )
    parser.add_argument(
        "--guidance-stop-epoch",
        type=int,
        default=None,
        help="Last guided epoch; required only for --beta-schedule manual_stop.",
    )
    parser.add_argument(
        "--teacher-image-size",
        type=int,
        default=32,
        help=(
            "CNN teacher input size before stage extraction. The supplied/public "
            "DeiT-CIFAR source configuration uses 32."
        ),
    )
    parser.add_argument(
        "--grid-resize-mode",
        choices=("teacher", "larger"),
        default="teacher",
        help=(
            "teacher follows the V3 manuscript by resizing each projected "
            "student stage to the CNN stage grid; larger reproduces the "
            "supplied model snippet's max(student, teacher) grid rule."
        ),
    )
    parser.add_argument(
        "--max-teacher-runtime-gap-pp",
        type=float,
        default=5.0,
        help=(
            "Maximum allowed drop from checkpoint Top-1 when the teacher is "
            "evaluated at --teacher-image-size before a full run."
        ),
    )
    parser.add_argument(
        "--allow-teacher-runtime-gap",
        action="store_true",
        help="Explicitly allow a full run despite failing the teacher input-size audit.",
    )
    parser.add_argument("--feature-grid", type=int, default=14)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--deform-kernel-size", type=int, default=5)
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
        args.run_name = f"ours_{args.dataset}_{args.student}_{suffix}"

    for field in (
        "student_epochs",
        "batch_size",
        "image_size",
        "smoke_train_samples",
        "smoke_test_samples",
        "lr",
        "min_lr",
        "feature_grid",
        "teacher_image_size",
        "num_heads",
        "deform_kernel_size",
        "beta_on",
        "alg_smoothing_window",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if args.alg_threshold >= 0:
        raise ValueError("--alg-threshold must be negative for a decreasing loss")
    if args.max_teacher_runtime_gap_pp < 0:
        raise ValueError("--max-teacher-runtime-gap-pp must be non-negative")
    if args.beta_schedule == "manual_stop":
        if args.guidance_stop_epoch is None or args.guidance_stop_epoch <= 0:
            raise ValueError(
                "--beta-schedule manual_stop requires a positive --guidance-stop-epoch"
            )
    elif args.guidance_stop_epoch is not None:
        raise ValueError(
            "--guidance-stop-epoch is only valid with --beta-schedule manual_stop"
        )
    if not 0.0 <= args.fusion_ratio <= 1.0:
        raise ValueError("--fusion-ratio must be in [0, 1]")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if not 0.0 <= args.drop_path_rate < 1.0:
        raise ValueError("--drop-path-rate must be in [0, 1)")
    if args.min_lr > args.lr:
        raise ValueError("--min-lr must not exceed --lr")
    if args.image_size != 224:
        raise ValueError("The fixed dataset protocols require --image-size 224")
    if args.feature_grid != 14:
        raise ValueError("DeiT-Ti patch features require --feature-grid 14")
    if args.deform_kernel_size % 2 == 0:
        raise ValueError("--deform-kernel-size must be odd")
    if any(channels % args.num_heads for channels in TEACHER_CHANNELS):
        raise ValueError(
            f"--num-heads must divide every teacher channel count {TEACHER_CHANNELS}"
        )


def forward_teacher_features(
    teacher: torch.nn.Module,
    images: torch.Tensor,
    teacher_image_size: int,
) -> list[torch.Tensor]:
    if images.shape[-2:] != (teacher_image_size, teacher_image_size):
        images = F.interpolate(
            images,
            size=(teacher_image_size, teacher_image_size),
            mode="bilinear",
            align_corners=False,
        )
    stem = teacher.stem(images)
    stage1 = teacher.stage1(stem)
    stage2 = teacher.stage2(stage1)
    stage3 = teacher.stage3(stage2)
    return [stage1, stage2, stage3]


@torch.inference_mode()
def evaluate_teacher_at_runtime_size(
    teacher: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
    teacher_image_size: int,
) -> float:
    teacher.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images = F.interpolate(
            images,
            size=(teacher_image_size, teacher_image_size),
            mode="bilinear",
            align_corners=False,
        )
        with autocast_context(amp_enabled):
            logits = teacher(images)
        correct += top1_correct(logits, targets)
        total += targets.size(0)
    return 100.0 * correct / max(1, total)


def create_ours_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    effective_warmup = warmup_epochs if epochs > warmup_epochs else 0
    minimum_ratio = min_lr / base_lr

    def lr_multiplier(epoch_index: int) -> float:
        if effective_warmup and epoch_index < effective_warmup:
            return (epoch_index + 1) / effective_warmup
        cosine_epochs = max(1, epochs - effective_warmup)
        progress = min(
            max((epoch_index - effective_warmup) / cosine_epochs, 0.0), 1.0
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier), effective_warmup


def forward_student_features(
    student: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    final_tokens, intermediate_features = student.forward_intermediates(
        images,
        indices=list(range(NUM_STUDENT_BLOCKS)),
        norm=False,
        output_fmt="NCHW",
    )
    logits = student.forward_head(final_tokens)
    return list(intermediate_features), logits


def create_ours_student(
    timm: Any,
    student_key: str,
    num_classes: int,
    drop_path_rate: float,
) -> torch.nn.Module:
    timm_name = STUDENT_MODELS[student_key]
    try:
        return timm.create_model(
            timm_name,
            pretrained=False,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to create timm model {timm_name!r} with "
            f"drop_path_rate={drop_path_rate}"
        ) from error


def train_one_epoch(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    ours: Ours,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
    beta: float,
) -> tuple[float, float, float, float, float, float]:
    student.train()
    teacher.eval()
    ours.train()
    total_loss = 0.0
    total_ce = 0.0
    total_alignment = 0.0
    total_fusion = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        teacher_features: list[torch.Tensor] | None = None
        if beta > 0.0:
            with torch.no_grad(), autocast_context(amp_enabled):
                teacher_features = forward_teacher_features(
                    teacher,
                    images,
                    args.teacher_image_size,
                )
        with autocast_context(amp_enabled):
            student_features, student_logits = forward_student_features(student, images)
            ce = F.cross_entropy(
                student_logits,
                targets,
                label_smoothing=args.label_smoothing,
            )
            if teacher_features is None:
                alignment_loss = ce.new_zeros(())
                fusion_loss = ce.new_zeros(())
                feature_loss = ce.new_zeros(())
                loss = ce
            else:
                alignment_loss, fusion_loss, _, _, _ = ours(
                    student_features,
                    teacher_features,
                )
                feature_loss = (
                    args.fusion_ratio * fusion_loss
                    + (1.0 - args.fusion_ratio) * alignment_loss
                )
                loss = ce + beta * feature_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_ce += float(ce.detach()) * batch_size
        total_alignment += float(alignment_loss.detach()) * batch_size
        total_fusion += float(fusion_loss.detach()) * batch_size
        correct += top1_correct(student_logits.detach(), targets)

    denominator = max(1, total)
    average_alignment = total_alignment / denominator
    average_fusion = total_fusion / denominator
    average_feature = (
        args.fusion_ratio * average_fusion
        + (1.0 - args.fusion_ratio) * average_alignment
    )
    return (
        total_loss / denominator,
        total_ce / denominator,
        average_alignment,
        average_fusion,
        average_feature,
        100.0 * correct / denominator,
    )


def checkpoint_payload(
    student: torch.nn.Module,
    ours: Ours,
    epoch: int,
    accuracy: float,
    best_accuracy: float,
    args: argparse.Namespace,
    teacher_spec: dict[str, Any],
    controller: AdaptiveGuidanceController,
) -> dict[str, Any]:
    return {
        "model": student.state_dict(),
        "ours": ours.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "method": "Ours",
        "student": args.student,
        "timm_model": STUDENT_MODELS[args.student],
        "dataset": args.dataset,
        "num_classes": NUM_CLASSES[args.dataset],
        "teacher": teacher_spec,
        "source_snippet_sha256": SOURCE_SNIPPET_SHA256,
        "guidance_controller": controller.state_dict(),
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
    aggregation_weights: list[list[float]],
    controller: AdaptiveGuidanceController,
    teacher_runtime_top1: float,
) -> None:
    average_epoch = sum(epoch_times) / max(1, len(epoch_times))
    summary = {
        "status": "complete" if latest_epoch == args.student_epochs else "running",
        "method": "Ours",
        "dataset": args.dataset,
        "student": args.student,
        "timm_model": STUDENT_MODELS[args.student],
        "teacher": teacher_spec,
        "source_snippet_sha256": SOURCE_SNIPPET_SHA256,
        "paper_loss_equation": (
            "CE + beta(e) * (lambda * L_fuse + (1-lambda) * L_align)"
        ),
        "guidance_controller": controller.state_dict(),
        "teacher_runtime_top1": teacher_runtime_top1,
        "teacher_checkpoint_top1": float(teacher_spec["top1"]),
        "teacher_runtime_gap_pp": (
            teacher_runtime_top1 - float(teacher_spec["top1"])
        ),
        "teacher_runtime_audit_passed": (
            teacher_runtime_top1 - float(teacher_spec["top1"])
            >= -args.max_teacher_runtime_gap_pp
        ),
        "student_epochs": args.student_epochs,
        "latest_epoch": latest_epoch,
        "best_top1": best_accuracy,
        "latest_top1": latest_accuracy,
        "vanilla_top1": VANILLA_TOP1[args.dataset][args.student],
        "gain_over_vanilla_pp": (
            best_accuracy - VANILLA_TOP1[args.dataset][args.student]
        ),
        "aggregation_weights": aggregation_weights,
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


def aggregation_weights_list(ours: Ours) -> list[list[float]]:
    return ours.aggregation.normalized_weights().cpu().tolist()


def top_aggregation_weights(ours: Ours) -> str:
    weights = ours.aggregation.normalized_weights().cpu()
    stage_summaries = []
    for stage, stage_weights in enumerate(weights, 1):
        values, indices = torch.topk(stage_weights, k=3)
        pairs = ",".join(
            f"b{int(index)}={float(value):.3f}"
            for value, index in zip(values, indices, strict=True)
        )
        stage_summaries.append(f"stage{stage}[{pairs}]")
    return " ".join(stage_summaries)


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
    log("OURS / RESNET56 -> DEIT-TI")
    log("=" * 72)
    log(
        f"[ENV] python={platform.python_version()} torch={torch.__version__} "
        f"timm={timm.__version__} torchvision={__import__('torchvision').__version__}"
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
        f"[PROTOCOL] name={args.protocol_name} optimizer=AdamW lr={args.lr} "
        f"min_lr={args.min_lr} weight_decay={args.weight_decay} "
        f"warmup={args.warmup_epochs} "
        f"cosine batch={args.batch_size} image={args.image_size} "
        f"label_smoothing={args.label_smoothing} "
        f"drop_path={args.drop_path_rate}"
    )
    log(
        f"[OURS] loss=CE+beta(e)*(lambda*L_fuse+(1-lambda)*L_align) "
        f"beta_on={args.beta_on} lambda={args.fusion_ratio}"
    )
    log(
        f"[OURS] student_blocks=all_12 aggregation=learnable_uniform_init "
        f"teacher_stages=1/2/3 teacher_input={args.teacher_image_size} "
        f"student_grid={args.feature_grid}x{args.feature_grid} "
        f"stage_grid={args.grid_resize_mode} "
        f"projection=1x1 deform_kernel={args.deform_kernel_size} "
        f"qkv_kernel=1 heads={args.num_heads}"
    )
    if args.beta_schedule == "alg":
        log(
            f"[BETA] schedule=alg_exact beta_on={args.beta_on} "
            f"metric=L_align threshold={args.alg_threshold} "
            f"smoothing_window={args.alg_smoothing_window} "
            "active_condition=smoothed_derivative<threshold"
        )
        log(
            "[BETA] ALG Eqs.(10)-(19): smooth the epoch loss over 50 epochs, "
            "differentiate it, smooth that derivative over 50 epochs, and switch "
            "permanently to CE-only when the derivative reaches tau=-0.02."
        )
    else:
        log(
            f"[BETA] schedule=manual_stop beta_on={args.beta_on} "
            f"last_guided_epoch={args.guidance_stop_epoch}"
        )
    log(f"[SOURCE] provided_snippet_sha256={SOURCE_SNIPPET_SHA256}")
    log(
        "[REPRO_STATUS] Paper-confirmed: Eq.(4), lambda=0.5, ALG beta=2.5, "
        "tau=-0.02 and 50-epoch smoothing, all-block aggregation, 1x1 "
        "projection/QKV, teacher-grid bilinear alignment, 5x5 deformable "
        "attention, and frozen teacher."
    )
    log(
        "[REPRO_STATUS] Ours-specific reproduction choices: L_align drives the "
        "ALG controller; epoch-1 derivative is initialized to zero; attention "
        "heads/reduction/loss reduction follow the supplied source."
    )

    train_loader, test_loader = build_loaders(args, device)
    teacher, teacher_payload, teacher_spec = load_teacher(
        args.dataset,
        device=device,
        checkpoint_root=args.teacher_root,
    )
    teacher_runtime_top1 = evaluate_teacher_at_runtime_size(
        teacher,
        test_loader,
        device,
        amp_enabled,
        args.teacher_image_size,
    )
    student = create_ours_student(
        timm,
        args.student,
        NUM_CLASSES[args.dataset],
        args.drop_path_rate,
    ).to(device)
    ours = Ours(
        student_channels=STUDENT_CHANNELS,
        teacher_channels=TEACHER_CHANNELS,
        num_student_blocks=NUM_STUDENT_BLOCKS,
        num_heads=args.num_heads,
        spatial_kernel_size=args.deform_kernel_size,
        grid_resize_mode=args.grid_resize_mode,
    ).to(device)

    with torch.no_grad():
        probe = torch.zeros(2, 3, args.image_size, args.image_size, device=device)
        student_probe, logits_probe = forward_student_features(student, probe)
        teacher_probe = forward_teacher_features(
            teacher, probe, args.teacher_image_size
        )
        if args.grid_resize_mode == "teacher":
            target_spatial_sizes = [feature.shape[-2:] for feature in teacher_probe]
        else:
            target_spatial_sizes = [
                (
                    max(feature.shape[-2], args.feature_grid),
                    max(feature.shape[-1], args.feature_grid),
                )
                for feature in teacher_probe
            ]
        if max(height * width for height, width in target_spatial_sizes) > 4096:
            raise RuntimeError(
                "Ours grid-space attention target is too large for a safe run: "
                f"{target_spatial_sizes}. Use the source-compatible teacher input "
                "size/checkpoint and verify its accuracy with --timing-run."
            )
        (
            alignment_probe,
            fusion_probe,
            aligned_probe,
            fused_probe,
            target_probe,
        ) = ours(
            student_probe,
            teacher_probe,
        )
    expected_teacher_raw = [tuple(feature.shape) for feature in teacher_probe]
    expected_student = [
        (2, STUDENT_CHANNELS, args.feature_grid, args.feature_grid)
    ] * NUM_STUDENT_BLOCKS
    expected_targets = [tuple(feature.shape) for feature in target_probe]
    if [tuple(feature.shape) for feature in student_probe] != expected_student:
        raise RuntimeError(
            f"Unexpected student features: {[tuple(x.shape) for x in student_probe]}"
        )
    if [tuple(feature.shape) for feature in aligned_probe] != expected_targets:
        raise RuntimeError(
            f"Unexpected aligned features: {[tuple(x.shape) for x in aligned_probe]}"
        )
    if [tuple(feature.shape) for feature in fused_probe] != expected_targets:
        raise RuntimeError(
            f"Unexpected fused features: {[tuple(x.shape) for x in fused_probe]}"
        )
    if tuple(logits_probe.shape) != (2, NUM_CLASSES[args.dataset]):
        raise RuntimeError(f"Unexpected logits: {tuple(logits_probe.shape)}")
    if not bool(torch.isfinite(alignment_probe + fusion_probe)):
        raise RuntimeError("Non-finite Ours probe loss")

    log(
        f"[TEACHER] selected={teacher_spec['selected_kind']} "
        f"epoch={teacher_payload['epoch']} "
        f"top1={float(teacher_payload['accuracy']):.2f}% "
        f"sha256={teacher_spec['sha256']}"
    )
    log(
        f"[TEACHER_RUNTIME_AUDIT] checkpoint_top1_at_training_recipe="
        f"{float(teacher_payload['accuracy']):.2f}% "
        f"runtime_top1_at_{args.teacher_image_size}px={teacher_runtime_top1:.2f}% "
        f"gap={teacher_runtime_top1 - float(teacher_payload['accuracy']):+.2f}pp"
    )
    runtime_gap = teacher_runtime_top1 - float(teacher_payload["accuracy"])
    if runtime_gap < -args.max_teacher_runtime_gap_pp:
        log(
            "[TEACHER_RUNTIME_AUDIT][WARN] Runtime low-resolution teacher accuracy "
            f"is more than {args.max_teacher_runtime_gap_pp:.1f}pp below the "
            "checkpoint record."
        )
        if not (args.smoke or args.timing_run or args.allow_teacher_runtime_gap):
            raise RuntimeError(
                "Teacher runtime accuracy audit failed before the full run. "
                "Review --teacher-image-size using a timing run, or pass "
                "--allow-teacher-runtime-gap only after deliberately accepting "
                "the mismatch."
            )
    log(
        f"[MODEL] teacher_params={count_parameters(teacher):,} "
        f"student={STUDENT_MODELS[args.student]} "
        f"student_params={count_parameters(student):,} "
        f"ours_trainable_params={count_parameters(ours):,}"
    )
    log(
        f"[FEATURE_CHECK] teacher_raw={expected_teacher_raw} "
        f"student_blocks={NUM_STUDENT_BLOCKS}x{expected_student[0]} "
        f"stage_targets={expected_targets} "
        f"aligned={expected_targets} fused={expected_targets} "
        f"probe_align={float(alignment_probe):.4f} "
        f"probe_fuse={float(fusion_probe):.4f}"
    )
    log(f"[AGGREGATION_INIT] {top_aggregation_weights(ours)}")

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(ours.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler, effective_warmup = create_ours_scheduler(
        optimizer,
        args.student_epochs,
        args.warmup_epochs,
        args.min_lr,
        args.lr,
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
    controller = AdaptiveGuidanceController(args)
    training_start = time.time()
    for epoch_index in range(args.student_epochs):
        epoch = epoch_index + 1
        epoch_start = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        beta = controller.beta_for_epoch(epoch)
        (
            loss,
            ce,
            alignment_loss,
            fusion_loss,
            feature_loss,
            train_accuracy,
        ) = train_one_epoch(
            student,
            teacher,
            ours,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            amp_enabled,
            beta,
        )
        controller_state = (
            controller.observe(epoch, alignment_loss)
            if beta > 0.0
            else controller.state_dict()
        )
        raw_derivative = (
            controller_state["derivative_history"][-1]
            if beta > 0.0
            else None
        )
        smoothed_derivative = (
            controller_state["smoothed_derivative_history"][-1]
            if beta > 0.0
            else None
        )
        raw_derivative_text = (
            "n/a"
            if raw_derivative is None or args.beta_schedule == "manual_stop"
            else f"{float(raw_derivative):.6f}"
        )
        smoothed_derivative_text = (
            "n/a"
            if smoothed_derivative is None
            else f"{float(smoothed_derivative):.6f}"
        )
        if beta > 0.0 and not controller.active:
            log(
                f"[BETA_TRANSITION] guidance disabled after epoch={epoch} "
                f"L_align={alignment_loss:.6f} "
                f"raw_derivative={raw_derivative_text} "
                f"smoothed_derivative={smoothed_derivative_text} "
                f"threshold={args.alg_threshold}; subsequent epochs are CE-only."
            )
        latest_accuracy = evaluate(student, test_loader, device, amp_enabled)
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        previous_best = best_accuracy
        best_accuracy = max(best_accuracy, latest_accuracy)
        payload = checkpoint_payload(
            student,
            ours,
            epoch,
            latest_accuracy,
            best_accuracy,
            args,
            teacher_spec,
            controller,
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
            aggregation_weights=aggregation_weights_list(ours),
            controller=controller,
            teacher_runtime_top1=teacher_runtime_top1,
        )
        average_epoch = sum(epoch_times) / len(epoch_times)
        suffix = " saved_best" if saved_best else ""
        log(
            f"[OURS][{epoch:03d}/{args.student_epochs:03d}] loss={loss:.4f} "
            f"ce={ce:.4f} align={alignment_loss:.4f} "
            f"fuse={fusion_loss:.4f} feature={feature_loss:.4f} "
            f"beta={beta:.4f} guidance_active_next={controller.active} "
            f"alg_raw_derivative={raw_derivative_text} "
            f"alg_smoothed_derivative={smoothed_derivative_text} "
            f"guidance_stop_epoch={controller.stop_epoch} "
            f"train_acc={train_accuracy:.2f}% val_acc={latest_accuracy:.2f}% "
            f"best={best_accuracy:.2f}% lr={epoch_lr:.6g} "
            f"time={epoch_seconds:.1f}s avg_epoch={average_epoch:.1f}s "
            f"est_planned={format_duration(average_epoch * args.planned_epochs)} "
            f"elapsed={format_duration(elapsed)}{suffix}"
        )
        scheduler.step()

    elapsed = time.time() - training_start
    average_epoch = sum(epoch_times) / len(epoch_times)
    vanilla = VANILLA_TOP1[args.dataset][args.student]
    log("=" * 72)
    log(
        f"[FINAL_RESULT] ours_best_top1={best_accuracy:.2f}% "
        f"vanilla_top1={vanilla:.2f}% "
        f"gain_over_vanilla={best_accuracy - vanilla:+.2f}pp"
    )
    log(
        f"[TIMING] avg_epoch={average_epoch:.1f}s "
        f"planned_epochs={args.planned_epochs} "
        f"estimated_total={format_duration(average_epoch * args.planned_epochs)} "
        f"elapsed={format_duration(elapsed)}"
    )
    log(f"[AGGREGATION_FINAL] {top_aggregation_weights(ours)}")
    log(
        f"[BETA_FINAL] schedule={controller.schedule} "
        f"stop_epoch={controller.stop_epoch} "
        f"guided_epochs={sum(beta > 0.0 for beta in controller.beta_history)} "
        f"ce_only_epochs={sum(beta == 0.0 for beta in controller.beta_history)} "
        "full_history_saved_in=summary.json"
    )
    log(f"[FINAL_RESULT] best_checkpoint={best_checkpoint.resolve()}")
    log(f"[FINAL_RESULT] latest_checkpoint={latest_checkpoint.resolve()}")
    log(f"[FINAL_RESULT] summary={summary_path.resolve()}")
    log("[DONE] Ours training completed successfully; resources may be released.")


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Ours training did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
