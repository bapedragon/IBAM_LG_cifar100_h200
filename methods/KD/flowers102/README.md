# KD on Flowers-102

## Scope

This is the primary Flowers-102 row for Table 2:

- Teacher: ResNet56
- Student: DeiT-Ti
- Method: standard logit KD
- Student initialization: from scratch, without external pretraining

The same Flowers base protocol must be reused for CRD, ReviewKD, MGD, and OFA.
Only the method-specific transfer loss and its documented coefficients may
change.

## Fixed inputs

- Teacher: CIFAR-style ResNet56, frozen
- Teacher checkpoint: `checkpoints/teachers/flowers102/teacher_resnet56_flowers_best.pt`
- Teacher checkpoint epoch/Top-1: 291 / 64.64%
- Paper teacher reference: 66.33%
- Dataset: official Oxford Flowers 102 `train + val` for training and `test` for evaluation
- DeiT-Ti Vanilla reference: 50.06%

The official dataset is downloaded and verified automatically. The common
student and KD-loss protocol is recorded below and in `methods/KD/README.md`.

## Literature check

There is no single universal Flowers-102 KD recipe. The selected protocol is a
documented synthesis of recurring settings from official sources:

- The official CRD benchmark uses dataset-specific schedules and, on
  CIFAR-100, trains for 240 epochs with batch size 64 and temperature 4.
- The official DeiT recipe uses AdamW, learning rate `5e-4`, weight decay
  `0.05`, cosine decay, 5 warm-up epochs, batch size 64, 224-pixel inputs, and
  label smoothing `0.1`.
- A published standard-KD experiment on Flowers-102 uses 200 epochs, batch size
  64, and learning rate `5e-4`.

Primary sources:

- CRD official code:
  https://github.com/HobbitLong/RepDistiller/blob/master/train_student.py
- DeiT official code:
  https://github.com/facebookresearch/deit/blob/main/main.py
- Flowers-102 standard KD details:
  https://proceedings.mlr.press/v139/wang21a/wang21a-supp.pdf

## Selected Flowers protocol

| Setting | Value |
|---|---|
| Protocol name | `flowers102_deit_ti_common_kd_v1` |
| Maximum epochs | `200` |
| Optimizer | AdamW |
| Initial learning rate | `5e-4` |
| Weight decay | `0.05` |
| LR schedule | 5-epoch warm-up, then cosine decay |
| Batch size | `64` |
| Resolution | `224 x 224` |
| Initialization | From scratch (`pretrained=False`) |
| Training split | Official train + val (`2,040` images) |
| Evaluation split | Official test (`6,149` images) |
| Training augmentation | RandomResizedCrop, horizontal flip |
| Evaluation preprocessing | Resize, center crop |
| Label smoothing | `0.1` |
| AMP | Enabled on CUDA |
| Seed | `42` |
| Checkpoint selection | Highest test Top-1 during the fixed run |

External ImageNet pretraining is intentionally excluded so that improvements
measure the fixed teacher's contribution rather than imported large-scale
knowledge. Mixup, CutMix, and Random Erasing are also excluded from this first
generic-KD matrix so every method receives the same simple classification
pipeline without method-dependent soft-target handling.

## KD-specific fixed choices

| Setting | Value |
|---|---|
| Temperature | `4.0` |
| KD weight | `0.9` |
| Classification weight | `0.1` |

These values remain the fixed standard-KD method configuration; they were not
searched on Flowers-102.

## Timing run

```bash
python methods/KD/flowers102/train.py --student deit_ti --timing-run --num-workers 4
```

## Full run

```bash
python methods/KD/flowers102/train.py --student deit_ti --num-workers 4 --run-name kd_flowers102_deit_ti_200ep --output-dir /app/output
```

The wrapper injects the selected Flowers defaults. Explicit CLI arguments can
override them, but any override creates a different protocol and must use a new
protocol name.
