# Ours: grid-preserving CNN-to-ViT distillation

This folder runs the supplied Ours module with the repository's frozen
ResNet56 teachers and a DeiT-Ti student. The integration separates settings
confirmed by the working paper/source from reproduction choices that are not
available in the supplied materials. See [`PAPER_AUDIT.md`](PAPER_AUDIT.md)
for the evidence matrix.

## Implemented Ours objective

The executable now follows working-paper Eq. (4) directly:

```text
L_total(e) = CE + beta(e) * [0.5 * L_fuse + 0.5 * L_align]
```

- `L_align`: MSE between the projected/aligned student grid and CNN grid
- `L_fuse`: MSE between the fused grid and CNN grid
- guidance active: `beta(e) = beta_on`
- guidance inactive: `beta(e) = 0`, teacher/feature-module forward passes are
  skipped and training continues with CE only

The previous extra fixed multiplication
`CE + 1.0 * 2.5 * (...)` has been removed. `beta=2.5` is represented once as
`beta(e)`, following the ALG paper cited by V3.

## Feature path matched to the supplied Ours module

- frozen ResNet56 teacher stages 1/2/3
- patch-grid outputs from all 12 DeiT-Ti blocks
- one learned convex 12-block mixture per CNN stage
- stage-specific `1 x 1` channel projection
- bilinear resizing of each projected student stage to its teacher-stage
  resolution, as written in V3
- channel attention and `5 x 5` deformable spatial attention
- four-head convolutional cross-attention with `1 x 1` Q/K/V
- teacher and Ours module discarded at inference

The supplied model snippet instead resizes both tensors to the larger of the
student/teacher grids. Because V3 explicitly says "teacher resolution", the
paper-aligned `teacher` mode is the default. The snippet-compatible behavior
is retained only as the explicit option `--grid-resize-mode larger` and must
not be mixed into the main comparison silently.

## Adaptive beta from ALG

The default controller now implements ALG Eqs. (10)-(19), not a plateau
proxy. It uses `beta=2.5`, `tau=-0.02`, and a 50-epoch window in both
smoothing steps. Guidance remains active while the twice-smoothed loss
derivative is below `tau`; when it reaches `tau`, that epoch is the last
guided epoch and all subsequent epochs use CE only.

V3 does not state which of its two feature losses should be observed by the
ALG controller. This implementation observes `L_align`, because it is the
direct CNN/ViT feature distance corresponding most closely to ALG's `L_LG`.
At epoch 1, where ALG's published expression has no previous loss, the raw
derivative is initialized to zero and cannot stop guidance. These two boundary
decisions are explicitly saved in logs/checkpoints. `manual_stop` remains
available only for controlled diagnostics.

## Dataset-specific base protocols

The draft's single 300-epoch statement is not used for Flowers-102 or
Chaoyang, per the experiment-team correction. The shared optimizer family and
dataset-specific schedules already established for the KD table are retained.

| Dataset | Epochs | Batch | Optimizer | LR / min LR | Weight decay | Warm-up | Schedule |
|---|---:|---:|---|---:|---:|---:|---|
| CIFAR-100 | 300 | 128 | AdamW | `5e-4` / `5e-6` | `0.05` | 20 | Cosine |
| Flowers-102 | 200 | 64 | AdamW | `5e-4` / `5e-6` | `0.05` | 5 | Cosine |
| Chaoyang | 100 | 64 | AdamW | `5e-4` / `5e-6` | `0.05` | 5 | Cosine |

All use 224-pixel student inputs, label smoothing `0.1`, AMP, seed `42`, no
external student pretraining, the established dataset splits, and best Top-1
checkpoint reporting. Augmentation is the repository's common
`RandomResizedCrop(scale=0.8..1.0) + RandomHorizontalFlip` pipeline. These
regularization/checkpoint choices are recorded experiment settings, not
claimed as working-paper Ours specifications.

## Teacher input-size safety audit

The supplied Ours module has a separate teacher input-size setting, and the
public DeiT-CIFAR source configuration uses 32 pixels. Ours therefore defaults
to `--teacher-image-size 32`; this also keeps stage-grid cross-attention
tractable. The teachers currently stored in this repository were trained at
224 pixels, however, so their classifier/features may not remain valid after
the source-required resize.

Before training, the code evaluates the selected teacher at 32 pixels and
prints `[TEACHER_RUNTIME_AUDIT]`. A full run is blocked when the drop exceeds
5 percentage points unless the mismatch is deliberately overridden. If it
fails, do not override it merely to obtain a number: prepare a compatible
32-pixel teacher checkpoint or recover the actual per-dataset Ours config.

## H200 execution

Timing checks (full dataset, two epochs):

```bash
python methods/Ours/cifar100/train.py --timing-run --num-workers 4
python methods/Ours/flowers102/train.py --timing-run --num-workers 4
python methods/Ours/chaoyang/train.py --timing-run --num-workers 4
```

Conditional full runs after the timing log and teacher audit are accepted:

```bash
python methods/Ours/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name ours_cifar100_deit_ti_300ep --output-dir /app/output
python methods/Ours/flowers102/train.py --student-epochs 200 --num-workers 4 --run-name ours_flowers102_deit_ti_200ep --output-dir /app/output
python methods/Ours/chaoyang/train.py --student-epochs 100 --num-workers 4 --run-name ours_chaoyang_deit_ti_100ep --output-dir /app/output
```

For a manual diagnostic stop epoch, use:

```text
--beta-schedule manual_stop --guidance-stop-epoch <LAST_GUIDED_EPOCH>
```

Every epoch prints total/CE/alignment/fusion loss, beta, guidance state,
train/validation/best Top-1, learning rate, epoch time, and projected duration.
Failures end with `[FATAL]`; successful completion ends with `[DONE]`.
