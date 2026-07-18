# IBAM teacher checkpoints and KD preparation on H200

This repository contains the reusable ResNet56 teacher-training workflow, the
three fixed teacher checkpoints, and the shared preparation files for the IBAM
KD experiments on the KAU H200 runner.

Repository:

```text
https://github.com/bapedragon/IBAM_teacher_KD_H200.git
```

## Fixed teachers for KD

| Dataset | Selected checkpoint | Epoch | Top-1 | Paper teacher Top-1 | Gap |
|---|---|---:|---:|---:|---:|
| CIFAR-100 | Best | 297 | 68.68% | 70.43% | -1.75pp |
| Flowers-102 | Best | 291 | 64.64% | 66.33% | -1.69pp |
| Chaoyang | Latest | 300 | 81.53% | 77.20% | +4.33pp |

All three files passed SHA-256 verification, strict `state_dict` loading, and a
forward-pass check. Chaoyang intentionally uses the epoch-300 `latest`
checkpoint rather than its 83.08% `best` checkpoint. These selections are now
fixed and must be reused across every student and every compared KD method.

## Repository files

```text
checkpoints/teachers/
  manifest.json
  cifar100/teacher_resnet56_cifar100_best.pt
  flowers102/teacher_resnet56_flowers_best.pt
  chaoyang/teacher_resnet56_chaoyang_latest.pt
teacher_checkpoints.py
methods/
  README.md
  KD/
    README.md
    core.py
    cifar100/{train.py,README.md}
    flowers102/{train.py,README.md}
    chaoyang/{train.py,README.md}
  CRD/
    README.md
    official_crd.py
    core.py
    cifar100/{train.py,README.md}
    flowers102/{train.py,README.md}
    chaoyang/{train.py,README.md}
  ReviewKD/
    README.md
    official_reviewkd.py
    core.py
    cifar100/{train.py,README.md}
    flowers102/{train.py,README.md}
    chaoyang/{train.py,README.md}
  MGD/
    README.md
    official_mgd.py
    core.py
    cifar100/{train.py,README.md}
    flowers102/{train.py,README.md}
    chaoyang/{train.py,README.md}
  OFA/
    README.md
    official_ofa.py
    core.py
    cifar100/{train.py,README.md}
    flowers102/{train.py,README.md}
    chaoyang/{train.py,README.md}
train_teacher_cifar100.py
train_teacher_flowers.py
train_teacher_chaoyang.py
requirements.txt
.gitignore
README.md
```

- `train_teacher_cifar100.py`: downloads/verifies CIFAR-100 and trains the
  CIFAR-100 ResNet56 teacher.
- `train_teacher_flowers.py`: downloads/verifies Oxford Flowers 102 and trains
  the Flowers ResNet56 teacher.
- `train_teacher_chaoyang.py`: validates the mounted official Chaoyang dataset
  and trains the Chaoyang ResNet56 teacher.
- `teacher_checkpoints.py`: verifies hashes and metadata, strictly loads the
  selected weight, freezes the teacher, and exposes it to KD training code.
- `methods/`: separates KD implementations by method and dataset. Each method
  has its own protocol README and uses dataset-specific execution wrappers.
- `methods/KD/core.py`: shared standard-logit-KD training, evaluation, logging,
  and checkpoint implementation.
- `methods/CRD/official_crd.py`: CRD loss and memory bank ported from the
  authors' official RepDistiller repository.
- `methods/CRD/core.py`: ResNet56-to-DeiT-Ti adapter, official CRD sampling,
  training, logging, and checkpoint implementation.
- `methods/ReviewKD/`: official-code-based ReviewKD loss with the documented
  CNN-stage-to-ViT-token adapter and dataset-specific wrappers.
- `methods/MGD/`: official-code-based masked generative loss with the
  documented ResNet-stage-to-DeiT-grid adapter and dataset-specific wrappers.
- `methods/OFA/`: official-behavior-based adaptive target-enhancement loss and
  intermediate logit projectors for heterogeneous ResNet56-to-DeiT-Ti KD.

Legacy LG student training, LG checkpoint evaluation, the downloaded LG weight,
and GitHub-token artifact upload experiments have been removed. H200 artifacts
are collected only through `/app/output`.

Verify all selected teacher files after cloning:

```bash
python teacher_checkpoints.py --dataset all
```

Load one fixed teacher from upcoming KD code:

```python
from teacher_checkpoints import load_teacher

teacher, checkpoint, teacher_spec = load_teacher("chaoyang", device="cuda")
```

## KD methods

The method and dataset protocols are kept under `methods/`. See
[methods/KD/README.md](methods/KD/README.md) for standard logit KD and
[methods/CRD/README.md](methods/CRD/README.md) for CRD. The next comparison
method is documented in
[methods/ReviewKD/README.md](methods/ReviewKD/README.md).
The next official-code-based comparison is documented in
[methods/MGD/README.md](methods/MGD/README.md).
The final heterogeneous comparison is documented in
[methods/OFA/README.md](methods/OFA/README.md).

Current completed results. Table 2 now uses DeiT-Ti only; the other students
below are retained as exploratory runs.

| Method | Dataset | Student | Best Top-1 | Vanilla | Gain |
|---|---|---|---:|---:|---:|
| KD | CIFAR-100 | DeiT-Ti | **67.00%** | 65.08% | +1.92pp |
| KD | Flowers-102 | DeiT-Ti | **45.88%** | 50.06% | -4.18pp |
| KD | Chaoyang | DeiT-Ti | **80.60%** | 82.00% | -1.40pp |
| CRD | CIFAR-100 | DeiT-Ti | **67.40%** | 65.08% | +2.32pp |
| CRD | Flowers-102 | DeiT-Ti | **46.63%** | 50.06% | -3.43pp |
| CRD | Chaoyang | DeiT-Ti | **76.48%** | 82.00% | -5.52pp |
| ReviewKD | CIFAR-100 | DeiT-Ti | **72.84%** | 65.08% | +7.76pp |
| ReviewKD | Flowers-102 | DeiT-Ti | **50.76%** | 50.06% | +0.70pp |
| ReviewKD | Chaoyang | DeiT-Ti | **81.53%** | 82.00% | -0.47pp |
| MGD | CIFAR-100 | DeiT-Ti | **73.71%** | 65.08% | +8.63pp |
| MGD | Flowers-102 | DeiT-Ti | **51.57%** | 50.06% | +1.51pp |
| MGD | Chaoyang | DeiT-Ti | **81.81%** | 82.00% | -0.19pp |
| KD | CIFAR-100 | ConViT-Tiny | **73.59%** | 74.87% | -1.28pp |
| KD | CIFAR-100 | PiT-Tiny | **72.22%** | 73.16% | -0.94pp |

First full-data timing run:

```bash
python methods/KD/cifar100/train.py --student deit_ti --timing-run --batch-size 128 --num-workers 4
```

First 300-epoch run after timing verification:

```bash
python methods/KD/cifar100/train.py --student deit_ti --student-epochs 300 --batch-size 128 --num-workers 4 --output-dir /app/output
```

## Environment

```text
Image: pytorch/pytorch:latest
Language: Python
GPU allocation: 7 (one whole H200 GPU)
```

Install dependencies when running outside the provided H200 image:

```bash
pip install -r requirements.txt
```

`torch` and `torchvision` are already included in the H200 PyTorch image.
Flowers additionally requires `scipy`, and the ViT students require
`timm==1.0.27`.

## Shared teacher protocol

The paper explicitly identifies the datasets, ResNet56 teacher, scratch
training, 224 x 224 input, PyTorch, and Top-1 evaluation. It does not fully
specify the teacher optimization recipe, so all three scripts use the same
scaffold choices:

- Optimizer: SGD
- Initial learning rate: `0.1`
- Momentum: `0.9`
- Weight decay: `5e-4`
- Warm-up: 5 epochs for a 300-epoch run
- LR schedule: cosine decay
- Batch size: `128`
- Image resolution: `224 x 224`
- Seed: `42`
- AMP: enabled on CUDA
- Training artifacts: best checkpoint, latest checkpoint, and `summary.json`
- KD teacher selection: fixed separately in `checkpoints/teachers/manifest.json`

The H200 runner collects only files under `/app/output`. Timing runs intentionally
omit that path so their temporary checkpoints disappear with the Pod.

## CIFAR-100 teacher

The dataset is downloaded automatically under `./data` with verified mirror
fallback.

Timing test:

```bash
python train_teacher_cifar100.py --teacher-epochs 2 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_cifar100_timing_2ep
```

Full run:

```bash
python train_teacher_cifar100.py --teacher-epochs 300 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_cifar100_300ep --output-dir /app/output
```

Collected files:

```text
/app/output/teacher_resnet56_cifar100_300ep/teacher_resnet56_best.pt
/app/output/teacher_resnet56_cifar100_300ep/teacher_resnet56_latest.pt
/app/output/teacher_resnet56_cifar100_300ep/summary.json
```

## Flowers teacher

The official Oxford Flowers 102 images, labels, and splits are downloaded and
verified automatically. Training uses official `train + val` and evaluation
uses the official `test` split.

Timing test:

```bash
python train_teacher_flowers.py --teacher-epochs 2 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_flowers_timing_2ep
```

Full run:

```bash
python train_teacher_flowers.py --teacher-epochs 300 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_flowers_300ep --output-dir /app/output
```

Collected files:

```text
/app/output/teacher_resnet56_flowers_300ep/teacher_resnet56_flowers_best.pt
/app/output/teacher_resnet56_flowers_300ep/teacher_resnet56_flowers_latest.pt
/app/output/teacher_resnet56_flowers_300ep/summary.json
```

## Chaoyang teacher

The official dataset is mounted read-only at:

```text
/app/data/chaoyang/
```

The script searches up to three nested levels for the actual dataset root and
then requires this structure:

```text
train/       4,021 images
test/        2,139 images
train.json
test.json
README.md
```

It verifies the official class counts for normal, serrated, adenocarcinoma, and
adenoma before training. The original dataset must not be committed to this
repository because its license prohibits redistribution.

Timing test:

```bash
python train_teacher_chaoyang.py --data-dir /app/data/chaoyang --teacher-epochs 2 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_chaoyang_timing_2ep
```

Full run:

```bash
python train_teacher_chaoyang.py --data-dir /app/data/chaoyang --teacher-epochs 300 --batch-size 128 --num-workers 4 --run-name teacher_resnet56_chaoyang_300ep --output-dir /app/output
```

Collected files:

```text
/app/output/teacher_resnet56_chaoyang_300ep/teacher_resnet56_chaoyang_best.pt
/app/output/teacher_resnet56_chaoyang_300ep/teacher_resnet56_chaoyang_latest.pt
/app/output/teacher_resnet56_chaoyang_300ep/summary.json
```

## Important log lines

Each script prints the CUDA/GPU environment, resolved data and output paths,
dataset sizes, model parameter count, per-epoch loss/accuracy/time, best Top-1,
estimated 300-epoch time, and final checkpoint paths.

A successful run ends with:

```text
[FINAL_RESULT] teacher_best_top1=...
[TIMING] teacher_avg_epoch=... estimated_300_teacher=...
[FINAL_RESULT] best_checkpoint=...
[DONE] ... completed successfully; resources may be released.
```

Any dataset, runtime, or external termination problem is printed with a
`[FATAL]` marker and traceback before the process exits.
