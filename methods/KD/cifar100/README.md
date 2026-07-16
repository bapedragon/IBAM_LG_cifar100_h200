# KD on CIFAR-100

## Fixed inputs

- Teacher: CIFAR-style ResNet56, frozen
- Teacher checkpoint: `checkpoints/teachers/cifar100/teacher_resnet56_cifar100_best.pt`
- Teacher checkpoint epoch/Top-1: 297 / 68.68%
- Paper teacher reference: 70.43%
- Dataset: full CIFAR-100 train/test split, downloaded and verified automatically
- DeiT-Ti Vanilla reference: 65.08%

The common student and KD-loss protocol is recorded in `methods/KD/README.md`.

## First timing run

This uses the full dataset for two epochs, reports the average epoch time and
300-epoch estimate, and writes only temporary Pod-local outputs.

```bash
python methods/KD/cifar100/train.py --student deit_ti --timing-run --batch-size 128 --num-workers 4
```

## Full run

Run this only after the timing log is verified:

```bash
python methods/KD/cifar100/train.py --student deit_ti --student-epochs 300 --batch-size 128 --num-workers 4 --output-dir /app/output
```

## Completed results

| Student | Best Top-1 | Best epoch | Vanilla | Gain | Result record |
|---|---:|---:|---:|---:|---|
| DeiT-Ti | **67.00%** | 191 | 65.08% | +1.92pp | `results/deit_ti/` |
| ConViT-Tiny | **73.59%** | 266 | 74.87% | -1.28pp | `results/convit/` |
| PiT-Tiny | **72.22%** | 185 | 73.16% | -0.94pp | `results/pit/` |
