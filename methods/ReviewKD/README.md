# ReviewKD

This folder applies ReviewKD to the fixed ResNet56-to-DeiT-Ti comparison.
The ABF and hierarchical context loss behavior follows the authors' official
implementation.

## Official-code provenance

- Paper: Distilling Knowledge via Knowledge Review, CVPR 2021
- Official repository: https://github.com/dvlab-research/ReviewKD
- Pinned commit: `cede6ea6387ae9b6127de0e561507177bf19c11e`
- Official implementation references:
  - `CIFAR-100/model/reviewkd.py`
  - `CIFAR-100/train.py`
  - `CIFAR-100/script/reviewKD.sh`

## Method settings

The official ResNet56-to-ResNet20 CIFAR-100 command uses a ReviewKD loss weight
of `0.6`. The official training code uses CE, no logit KL by default, and
linearly ramps the feature loss over 20 epochs. We keep those method settings
fixed for all three datasets:

```text
total loss = CE + ramp(epoch, 20) * 0.6 * HCL
```

HCL compares every feature at its full adapted grid and at pooled grids
`4x4`, `2x2`, and `1x1`.

## CNN-to-ViT adapter

ReviewKD was designed for hierarchical CNN features, while DeiT-Ti keeps one
`14x14` patch grid throughout its transformer blocks. The architecture bridge
is therefore explicit:

- Teacher: post-activation ResNet56 `stage1`, `stage2`, and `stage3`
- Teacher spatial adapter: adaptive average pooling of each stage to `14x14`
- Student: DeiT-Ti block indexes `3`, `7`, and `11` (4th, 8th, and 12th block)
- Student spatial adapter: remove the CLS token and reshape patch tokens to
  `14x14`
- Student channels: `192, 192, 192`
- Teacher channels: `16, 32, 64`
- Official ABF direction: deep-to-shallow attention fusion
- ABF hidden channels: `192`

This mapping is the heterogeneous CNN-to-ViT adaptation; the ABF gating and HCL
calculation retain the official method behavior.

## Fixed dataset protocols

| Dataset | Epochs | Batch | Optimizer | LR | Weight decay | LR warm-up |
|---|---:|---:|---|---:|---:|---:|
| CIFAR-100 | 300 | 128 | AdamW | `5e-4` | `0.05` | 20 |
| Flowers-102 | 200 | 64 | AdamW | `5e-4` | `0.05` | 5 |
| Chaoyang | 100 | 64 | AdamW | `5e-4` | `0.05` | 5 |

All use cosine decay, image size `224`, label smoothing `0.1`, seed `42`, AMP,
no external pretraining, and the same fixed teacher used by KD and CRD.

## First run

```bash
python methods/ReviewKD/cifar100/train.py --timing-run --num-workers 4
```

After the timing run:

```bash
python methods/ReviewKD/cifar100/train.py --num-workers 4 --run-name reviewkd_cifar100_deit_ti_300ep --output-dir /app/output
```

## Completed results

| Dataset | Best Top-1 | Best epoch | Latest Top-1 | Vanilla | Gain |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 | **72.84%** | 236 | 72.48% | 65.08% | +7.76pp |

The complete record is stored under `cifar100/results/deit_ti/`.
