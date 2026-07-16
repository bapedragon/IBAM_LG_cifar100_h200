# KD method experiment organization

Each comparison method gets its own folder under `methods/`. Every method
folder contains one shared implementation plus dataset-specific wrappers and
README files. This keeps method-specific protocols separate without duplicating
the full training loop three times.

```text
methods/
  KD/                  implemented, 1/3 primary Table-2 runs complete
  CRD/                 planned
  ReviewKD/            planned
  MGD/                 planned
  OFA/                 planned
```

## Fixed inputs

Teacher selection is complete. Every KD method must use the files recorded in
`checkpoints/teachers/manifest.json`; a method-specific teacher replacement is
not allowed.

The primary Table-2 matrix now contains three datasets and one ViT student:

- Datasets: CIFAR-100, Flowers-102, Chaoyang
- Student: DeiT-Ti
- Runs per KD method: `3 datasets x 1 student = 3 runs`

Completed ConViT-Tiny and PiT-Tiny runs are retained as exploratory results,
but they are not required for the primary Table-2 matrix.

## Dataset-specific student protocols

The previous draft incorrectly claimed one identical 300-epoch protocol across
all datasets. The revised experiment policy uses a documented base protocol per
dataset. Within one dataset, that protocol must remain identical across KD,
CRD, ReviewKD, MGD, and OFA.

| Dataset | Epochs | Batch | Optimizer | LR | Warm-up | Schedule | Status |
|---|---:|---:|---|---:|---:|---|---|
| CIFAR-100 | 300 | 128 | AdamW | `5e-4` | 20 | Cosine | Existing protocol |
| Flowers-102 | 200 | 64 | AdamW | `5e-4` | 5 | Cosine | Fixed as `flowers102_deit_ti_common_kd_v1` |
| Chaoyang | TBD | TBD | TBD | TBD | TBD | TBD | Review before launch |

Seed `42`, exact augmentation, method loss weights, temperatures, feature
adapters, and best-versus-latest reporting are implementation choices that must
be recorded explicitly in each run summary.

## Baselines

1. Logit KD: first implementation target; directly supports CNN teacher to ViT student.
2. CRD: requires an explicit representation projection rule.
3. ReviewKD: requires a documented CNN-stage to ViT-token mapping.
4. MGD: requires a documented token-grid reshape and feature adapter.
5. OFA: heterogeneous architecture baseline.

Before launching all 21 runs, validate one full-data timing run using
`CIFAR-100 / ResNet56 -> DeiT-Ti / logit KD`.

Current timing command:

```bash
python methods/KD/cifar100/train.py --student deit_ti --timing-run --batch-size 128 --num-workers 4
```

## Required output contract

Every H200 job must write collected artifacts under `/app/output` and print:

- CUDA availability and GPU name
- dataset, teacher checkpoint path/hash, and student model
- epoch, total/CE/KD losses, validation Top-1, best Top-1, and elapsed time
- final selected checkpoint and `summary.json` paths
- a `[FATAL]` traceback on failure or `[DONE]` on successful completion

Recommended result path:

```text
/app/output/kd/<method>/<dataset>/<student>/<run-name>/
```
