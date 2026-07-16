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

## Common student protocol

These settings match the LG, ALG, and Ours student protocol in the V2 draft:

- Student epochs: 300
- Optimizer: AdamW
- Initial learning rate: `5e-4`
- Weight decay: `0.05`
- LR schedule: 20-epoch warm-up followed by cosine decay
- Batch size: `128`
- Image resolution: `224 x 224`
- Framework: PyTorch with CUDA AMP
- Student initialization: no pretrained weights
- Evaluation: Top-1 accuracy every epoch; highest Top-1 saved as `student_best.pt`

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
The documented method-specific configuration is fixed and reused across
datasets and students. The paper must describe this as fixed or recommended
method-specific hyperparameters rather than claiming a hyperparameter search.

## Student implementation status

The `timm==1.0.27` path is verified for DeiT-Ti, ConViT, PiT, and PVTv2. CvT,
T2T-7, and T2T-14 require their official model implementations before their
runs because they are not registered in this `timm` release.

Start with the full-data CIFAR-100/DeiT-Ti timing run documented in
`cifar100/README.md`. Do not launch the full 21-run matrix until that log has
been checked.

## Completed results

| Dataset | Student | Best Top-1 | Vanilla | Gain | Status |
|---|---|---:|---:|---:|---|
| CIFAR-100 | DeiT-Ti | **67.00%** | 65.08% | +1.92pp | Complete |
| CIFAR-100 | ConViT-Tiny | **73.59%** | 74.87% | -1.28pp | Complete |
| CIFAR-100 | PiT-Tiny | **72.22%** | 73.16% | -0.94pp | Complete |

Detailed records are stored under `cifar100/results/<student>/`.
