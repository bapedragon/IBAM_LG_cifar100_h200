#!/usr/bin/env python3
"""Train DeiT-Ti with the provided grid-preserving Ours module."""

from __future__ import annotations

import argparse
import json
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
    create_scheduler,
    create_student,
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
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=2.5,
        help="Weight applied to the combined alignment/fusion feature loss.",
    )
    parser.add_argument(
        "--fusion-ratio",
        type=float,
        default=0.5,
        help="Lambda in lambda*L_fuse + (1-lambda)*L_align.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Fixed beta multiplier; the supplied source did not include an adaptive controller.",
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
        "feature_grid",
        "num_heads",
        "deform_kernel_size",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if args.distill_weight < 0 or args.beta < 0:
        raise ValueError("--distill-weight and --beta must be non-negative")
    if not 0.0 <= args.fusion_ratio <= 1.0:
        raise ValueError("--fusion-ratio must be in [0, 1]")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
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
    feature_grid: int,
) -> list[torch.Tensor]:
    stem = teacher.stem(images)
    stage1 = teacher.stage1(stem)
    stage2 = teacher.stage2(stage1)
    stage3 = teacher.stage3(stage2)
    return [
        F.adaptive_avg_pool2d(stage1, (feature_grid, feature_grid)),
        F.adaptive_avg_pool2d(stage2, (feature_grid, feature_grid)),
        F.adaptive_avg_pool2d(stage3, (feature_grid, feature_grid)),
    ]


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

        with torch.no_grad(), autocast_context(amp_enabled):
            teacher_features = forward_teacher_features(
                teacher,
                images,
                args.feature_grid,
            )
        with autocast_context(amp_enabled):
            student_features, student_logits = forward_student_features(student, images)
            ce = F.cross_entropy(
                student_logits,
                targets,
                label_smoothing=args.label_smoothing,
            )
            alignment_loss, fusion_loss, _, _ = ours(
                student_features,
                teacher_features,
            )
            feature_loss = (
                args.fusion_ratio * fusion_loss
                + (1.0 - args.fusion_ratio) * alignment_loss
            )
            loss = ce + args.beta * args.distill_weight * feature_loss

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
        f"weight_decay={args.weight_decay} warmup={args.warmup_epochs} "
        f"cosine batch={args.batch_size} image={args.image_size}"
    )
    log(
        f"[OURS] loss=CE+beta*weight*(lambda*L_fuse+(1-lambda)*L_align) "
        f"beta={args.beta} weight={args.distill_weight} "
        f"lambda={args.fusion_ratio}"
    )
    log(
        f"[OURS] student_blocks=all_12 aggregation=learnable_uniform_init "
        f"teacher_stages=1/2/3 grid={args.feature_grid}x{args.feature_grid} "
        f"projection=1x1 deform_kernel={args.deform_kernel_size} "
        f"qkv_kernel=1 heads={args.num_heads}"
    )
    log(
        "[NOTE] The supplied Ours snippet does not include the external adaptive "
        "beta controller; this run records and uses a fixed --beta value."
    )
    log(f"[SOURCE] provided_snippet_sha256={SOURCE_SNIPPET_SHA256}")

    train_loader, test_loader = build_loaders(args, device)
    teacher, teacher_payload, teacher_spec = load_teacher(
        args.dataset,
        device=device,
        checkpoint_root=args.teacher_root,
    )
    student = create_student(timm, args.student, NUM_CLASSES[args.dataset]).to(device)
    ours = Ours(
        student_channels=STUDENT_CHANNELS,
        teacher_channels=TEACHER_CHANNELS,
        num_student_blocks=NUM_STUDENT_BLOCKS,
        num_heads=args.num_heads,
        spatial_kernel_size=args.deform_kernel_size,
    ).to(device)

    with torch.no_grad():
        probe = torch.zeros(2, 3, args.image_size, args.image_size, device=device)
        teacher_probe = forward_teacher_features(teacher, probe, args.feature_grid)
        student_probe, logits_probe = forward_student_features(student, probe)
        alignment_probe, fusion_probe, aligned_probe, fused_probe = ours(
            student_probe,
            teacher_probe,
        )
    expected_teacher = [
        (2, channels, args.feature_grid, args.feature_grid)
        for channels in TEACHER_CHANNELS
    ]
    expected_student = [
        (2, STUDENT_CHANNELS, args.feature_grid, args.feature_grid)
    ] * NUM_STUDENT_BLOCKS
    if [tuple(feature.shape) for feature in teacher_probe] != expected_teacher:
        raise RuntimeError(
            f"Unexpected teacher features: {[tuple(x.shape) for x in teacher_probe]}"
        )
    if [tuple(feature.shape) for feature in student_probe] != expected_student:
        raise RuntimeError(
            f"Unexpected student features: {[tuple(x.shape) for x in student_probe]}"
        )
    if [tuple(feature.shape) for feature in aligned_probe] != expected_teacher:
        raise RuntimeError(
            f"Unexpected aligned features: {[tuple(x.shape) for x in aligned_probe]}"
        )
    if [tuple(feature.shape) for feature in fused_probe] != expected_teacher:
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
        f"[MODEL] teacher_params={count_parameters(teacher):,} "
        f"student={STUDENT_MODELS[args.student]} "
        f"student_params={count_parameters(student):,} "
        f"ours_trainable_params={count_parameters(ours):,}"
    )
    log(
        f"[FEATURE_CHECK] teacher={expected_teacher} "
        f"student_blocks={NUM_STUDENT_BLOCKS}x{expected_student[0]} "
        f"aligned={expected_teacher} fused={expected_teacher} "
        f"probe_align={float(alignment_probe):.4f} "
        f"probe_fuse={float(fusion_probe):.4f}"
    )
    log(f"[AGGREGATION_INIT] {top_aggregation_weights(ours)}")

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(ours.parameters()),
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
        )
        average_epoch = sum(epoch_times) / len(epoch_times)
        suffix = " saved_best" if saved_best else ""
        log(
            f"[OURS][{epoch:03d}/{args.student_epochs:03d}] loss={loss:.4f} "
            f"ce={ce:.4f} align={alignment_loss:.4f} "
            f"fuse={fusion_loss:.4f} feature={feature_loss:.4f} "
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
