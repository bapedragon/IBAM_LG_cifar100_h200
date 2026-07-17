# Standard logit KD

This folder contains the standard logit knowledge-distillation baseline for the
heterogeneous ResNet56-to-ViT setting. Logit KD does not require teacher and
student feature shapes to match; both models only need the same output classes.

## Folder structure

```text
KD/
  core.py              shared KD training, evaluation, logging, and checkpoint code
  cifar100/            CIFAR-100 wrapper and dataset-specific protocol
  flowers102/          Flowers-102 wrapper and dataset-specific protocol
  chaoyang/            Chaoyang wrapper and dataset-specific protocol
```

The dataset `train.py` files are intentionally thin wrappers. Keeping one
shared `core.py` prevents fixes or hyperparameter changes from drifting between
datasets.

## Dataset-specific base protocol

The prior draft's single 300-epoch protocol is no longer treated as the default
for every dataset. Each dataset uses one documented base student protocol, and
all compared KD methods must reuse that protocol within the dataset.

| Dataset | Epochs | Batch | LR | Weight decay | Warm-up | Resolution |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 300 | 128 | `5e-4` | `0.05` | 20 | 224 |
| Flowers-102 | 200 | 64 | `5e-4` | `0.05` | 5 | 224 |
| Chaoyang | 100 | 64 | `5e-4` | `0.05` | 5 | 224 |

All current protocols use AdamW, cosine decay, PyTorch CUDA AMP, no external
pretraining, and Top-1 evaluation. See each dataset README for its augmentation,
split, provenance, and exact commands.

Teacher checkpoints are loaded from `checkpoints/teachers/manifest.json`,
verified by SHA-256, placed in evaluation mode, and frozen for the entire run.

## KD-specific choices

```text
loss = (1 - alpha) * CE + alpha * T^2 * KL(student/T, teacher/T)
```

- Temperature `T`: `4.0`
- KD weight `alpha`: `0.9`
- Label smoothing: `0.1`
- Seed: `42`

The V2 draft does not specify these KD-specific values. They are explicit
implementation choices and are printed and stored in every `summary.json`.

## Hyperparameter policy

No per-method hyperparameter search is used for the current baseline suite.
Dataset-specific base training settings may differ, but they are fixed before
running the compared methods. Method-specific coefficients are documented
separately and are not changed silently after observing a result.

## Student implementation status

The `timm==1.0.27` path is verified for DeiT-Ti, ConViT, PiT, and PVTv2. CvT,
T2T-7, and T2T-14 require their official model implementations before their
runs because they are not registered in this `timm` release.

The primary Table-2 scope is DeiT-Ti on CIFAR-100, Flowers-102, and Chaoyang.
ConViT-Tiny and PiT-Tiny results are retained as exploratory runs.

## Completed results

| Dataset | Student | Best Top-1 | Vanilla | Gain | Status |
|---|---|---:|---:|---:|---|
| CIFAR-100 | DeiT-Ti | **67.00%** | 65.08% | +1.92pp | Complete |
| Flowers-102 | DeiT-Ti | **45.88%** | 50.06% | -4.18pp | Complete |
| Chaoyang | DeiT-Ti | **80.60%** | 82.00% | -1.40pp | Complete |
| CIFAR-100 | ConViT-Tiny | **73.59%** | 74.87% | -1.28pp | Exploratory |
| CIFAR-100 | PiT-Tiny | **72.22%** | 73.16% | -0.94pp | Exploratory |

Detailed records are stored under `<dataset>/results/deit_ti/`; exploratory
CIFAR-100 students remain under `cifar100/results/<student>/`.
