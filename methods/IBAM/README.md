# IBAM (Ours): grid-preserving CNN-to-ViT distillation

This folder integrates the provided Ours source with the repository's fixed
ResNet56 teachers and a timm DeiT-Ti student. The original snippet depends on
the project's pycls configuration and custom model attributes, so the training
adapter in this folder exposes the same feature operations through the models
already used by the H200 experiments.

## Source provenance and integration boundary

- Provided source SHA-256:
  `8649078970b93d750a956994611b65cdec0c24f907d35d86f29d635e8a3b8624`
- Student: timm `deit_tiny_patch16_224`, trained from scratch
- Teacher: the frozen ResNet56 checkpoint selected in
  `checkpoints/teachers/manifest.json`
- Student features: patch-token grids from all 12 DeiT blocks
- Teacher features: ResNet56 stages 1, 2, and 3, pooled to `14 x 14`
- Alignment: one learned convex mixture of the 12 student blocks per teacher
  stage, followed by a stage-specific `1 x 1` channel projection
- Enhancement: channel attention plus deformable spatial attention with a
  `5 x 5` kernel
- Fusion: four-head grid-space cross-attention with convolutional Q/K/V
- Evaluation: student classification head only; IBAM is a training-time module

The original snippet returns an intermediate feature loss but does not contain
the outer training loop or the adaptive guidance controller mentioned in the
working paper. Therefore, this integration does **not** invent an unverified
adaptive schedule. The default executable objective is recorded explicitly:

```text
L_total = CE + beta * 2.5 * (0.5 * L_fuse + 0.5 * L_align)
beta = 1.0 (fixed)
```

`2.5` is the feature-loss weight used by the source-compatible DeiT
configuration. `--distill-weight`, `--fusion-ratio`, and `--beta` are exposed
as command-line options so a confirmed final-paper setting can replace these
values without changing the implementation. Logit KD is not added.

## Fixed dataset protocols

The base student protocol is identical to the one already used for KD, CRD,
ReviewKD, MGD, and OFA within each dataset. Only the IBAM-specific loss differs.

| Dataset | Epochs | Batch | Optimizer | LR | Weight decay | Warm-up | Schedule |
|---|---:|---:|---|---:|---:|---:|---|
| CIFAR-100 | 300 | 128 | AdamW | `5e-4` | `0.05` | 20 | Cosine |
| Flowers-102 | 200 | 64 | AdamW | `5e-4` | `0.05` | 5 | Cosine |
| Chaoyang | 100 | 64 | AdamW | `5e-4` | `0.05` | 5 | Cosine |

All three use 224-pixel inputs, label smoothing `0.1`, AMP, seed `42`, no
external pretraining, the established dataset splits, and best-checkpoint
Top-1 reporting.

## Validation completed before H200 submission

- Python compilation: passed
- DeiT-Ti intermediate extraction: 12 tensors of shape `B x 192 x 14 x 14`
- Teacher extraction: three tensors of shape `B x {16,32,64} x 14 x 14`
- Alignment and fused feature shapes: passed
- CPU forward and backward through deformable attention: passed
- One complete synthetic student optimization step: passed
- Student and IBAM strict checkpoint reload: passed

## Recommended execution order

Run the two-epoch full-data timing check for each dataset before its full run:

```bash
python methods/IBAM/cifar100/train.py --timing-run --num-workers 4
python methods/IBAM/flowers102/train.py --timing-run --num-workers 4
python methods/IBAM/chaoyang/train.py --timing-run --num-workers 4
```

Full runs must use `/app/output` so the H200 runner retains the artifacts:

```bash
python methods/IBAM/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name ibam_cifar100_deit_ti_300ep --output-dir /app/output
python methods/IBAM/flowers102/train.py --student-epochs 200 --num-workers 4 --run-name ibam_flowers102_deit_ti_200ep --output-dir /app/output
python methods/IBAM/chaoyang/train.py --student-epochs 100 --num-workers 4 --run-name ibam_chaoyang_deit_ti_100ep --output-dir /app/output
```

Every epoch prints total, CE, alignment, fusion, and combined feature losses;
train and validation Top-1; best Top-1; learning rate; elapsed time; and the
estimated full-run duration. A failure ends with `[FATAL]` and a traceback; a
successful run ends with `[DONE]`.
