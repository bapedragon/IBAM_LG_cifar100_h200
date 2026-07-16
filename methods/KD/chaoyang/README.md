# KD on Chaoyang

## Scope

This is the primary Chaoyang row for the generic-KD comparison:

- Teacher: ResNet56
- Student: DeiT-Ti
- Method: standard logit KD
- Student initialization: from scratch, without external pretraining

The same Chaoyang base protocol must be reused for CRD, ReviewKD, MGD, and
OFA. Only each method's transfer loss and documented coefficients may change.

## Fixed inputs

- Teacher: CIFAR-style ResNet56, frozen
- Selected checkpoint: `checkpoints/teachers/chaoyang/teacher_resnet56_chaoyang_latest.pt`
- Selected checkpoint epoch/Top-1: 300 / 81.53%
- Paper teacher reference: 77.20%
- Dataset: official Chaoyang train/test split mounted at `/app/data/chaoyang`
- DeiT-Ti Vanilla reference: 82.00%

Chaoyang intentionally uses the epoch-300 `latest` checkpoint rather than the
83.08% `best` checkpoint. The selected teacher is fixed before the generic-KD
comparison and must not change between methods.

## Protocol basis

There is no universal Chaoyang DeiT/KD recipe. The selected protocol is a
documented fixed baseline derived from the dataset's official training code and
the standard DeiT optimization recipe:

- The official HSA-NRL Chaoyang implementation uses the official 4,021/2,139
  train/test split, trains for 80 epochs with batch size 96 and Adam, and uses
  horizontal flipping. It is a noisy-label ResNet method rather than a
  CNN-to-ViT KD recipe, so its settings are evidence for the dataset scale, not
  a configuration copied verbatim.
- The official DeiT recipe uses AdamW, learning rate `5e-4`, weight decay
  `0.05`, cosine decay, 5 warm-up epochs, 224-pixel inputs, batch size 64, and
  label smoothing `0.1`.
- We use 100 epochs as a round, moderately extended schedule over the official
  Chaoyang 80-epoch run while avoiding the previous draft's unsupported rule
  that every dataset must use 300 epochs.

Primary sources:

- Chaoyang dataset and official HSA-NRL code:
  https://github.com/bupt-ai-cz/HSA-NRL
- Official DeiT training code:
  https://github.com/facebookresearch/deit/blob/main/main.py

## Selected Chaoyang protocol

| Setting | Value |
|---|---|
| Protocol name | `chaoyang_deit_ti_common_kd_v1` |
| Maximum epochs | `100` |
| Optimizer | AdamW |
| Initial learning rate | `5e-4` |
| Weight decay | `0.05` |
| LR schedule | 5-epoch warm-up, then cosine decay |
| Batch size | `64` |
| Resolution | `224 x 224` |
| Initialization | From scratch (`pretrained=False`) |
| Training split | Official train (`4,021` images) |
| Evaluation split | Official test (`2,139` images) |
| Training augmentation | RandomResizedCrop, horizontal flip |
| Evaluation preprocessing | Resize, center crop |
| Label smoothing | `0.1` |
| AMP | Enabled on CUDA |
| Seed | `42` |
| Checkpoint selection | Highest test Top-1 during the fixed run |

External ImageNet pretraining, Mixup, CutMix, and Random Erasing are excluded
from this first generic-KD matrix. This keeps the classification pipeline
identical across KD, CRD, ReviewKD, MGD, and OFA.

## KD-specific fixed choices

| Setting | Value |
|---|---|
| Temperature | `4.0` |
| KD weight | `0.9` |
| Classification weight | `0.1` |

These values are fixed implementation choices and are not tuned after observing
the Chaoyang result.

## Teacher-gap reporting

Training and checkpoint selection always use the raw measured Top-1 accuracy.
The project may later report an additional teacher-gap-adjusted number using
the pre-agreed factor:

```text
paper teacher / selected teacher = 77.20 / 81.5334 = 0.94685
```

This factor is reporting-only. It is not applied to logits, losses, gradients,
or checkpoint selection.

## Timing run

```bash
python methods/KD/chaoyang/train.py --student deit_ti --timing-run --num-workers 4
```

## Full run

```bash
python methods/KD/chaoyang/train.py --student deit_ti --num-workers 4 --run-name kd_chaoyang_deit_ti_100ep --output-dir /app/output
```

The wrapper injects the selected Chaoyang defaults. Explicit CLI arguments can
override them, but any override creates a different protocol and must use a new
protocol name.
