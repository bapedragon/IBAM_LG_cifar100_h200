# IBAM LG-style CIFAR-100 H200 scaffold

This repository is a first-pass, runnable scaffold for checking the LG-style
training flow used as a baseline in **iBKD: Inductive Bias-Aware Knowledge
Distillation for Vision Transformer under Data Scarcity**.

The experiment trains a CIFAR-style ResNet56 teacher and then a DeiT-Ti student
on CIFAR-100. During student training, the final ResNet feature map is pooled to
the DeiT 14 x 14 patch grid, projected from 64 to 192 channels, and matched to
the DeiT patch tokens with feature MSE:

```text
total loss = classification cross-entropy + feature_weight * feature MSE
```

The Table 1 reference value is **77.38% Top-1** for CIFAR-100 / DeiT-Ti / LG.
This code is **not an exact reproduction claim**: the LG loss weight, matching
layer, augmentation, seed, checkpoint rule, teacher recipe, and exact official
LG commit/config were not specified in the available experiment context.

## Files

- `train_lg_cifar100.py`: dataset download, teacher training, LG-style student
  training, evaluation, logging, and checkpointing
- `requirements.txt`: Python dependencies
- `data/`: automatically created CIFAR-100 download directory
- `outputs/`: automatically created checkpoint and summary directory

## Training choices in this scaffold

- Input resolution: 224 x 224
- Student: `timm` `deit_tiny_patch16_224`, trained from scratch
- Student optimizer: AdamW, learning rate `5e-4`, weight decay `0.05`
- Student schedule: cosine decay with 20 warm-up epochs for a 300-epoch run
- Teacher: CIFAR-style ResNet56, trained from scratch
- Teacher optimizer: SGD with momentum by default (configurable)
- Augmentation: random resized crop and horizontal flip
- Validation: deterministic resize and center crop
- Best checkpoint criterion: validation Top-1 accuracy
- Mixed precision: enabled automatically on CUDA
- Seed: 42 by default

## Install

The H200 image `pytorch/pytorch:latest` already contains PyTorch. Install the
remaining declared dependencies if the runner does not do so automatically:

```bash
pip install -r requirements.txt
```

For the H200 Issue runner, `train_lg_cifar100.py` checks for `timm` at startup
and automatically installs `timm==1.0.27` when it is absent. The Issue command
therefore only needs to invoke the training file.

## Smoke test

The smoke mode uses deterministic subsets (1,024 training and 512 test images
by default), while still exercising download, teacher training, feature
distillation, validation, and checkpoint saving.

```bash
python train_lg_cifar100.py --smoke --teacher-epochs 1 --student-epochs 1 --batch-size 64 --num-workers 2
```

`--smoke` by itself also changes the default 300/300 epochs to 1/1. Explicit
epoch flags take precedence.

For an even smaller diagnostic run:

```bash
python train_lg_cifar100.py --smoke --teacher-epochs 1 --student-epochs 1 --batch-size 16 --num-workers 2 --smoke-train-samples 128 --smoke-test-samples 128
```

## Full experiment

The full mode uses all 50,000 CIFAR-100 training images and all 10,000 test
images.

```bash
python train_lg_cifar100.py --teacher-epochs 300 --student-epochs 300 --batch-size 128 --num-workers 4
```

To reuse a previously trained teacher:

```bash
python train_lg_cifar100.py --teacher-checkpoint ./outputs/lg_cifar100_deit_tiny_full/teacher_resnet56_best.pt --student-epochs 300 --batch-size 128 --num-workers 4
```

Useful optional flags include `--feature-weight`, `--seed`, `--student-lr`,
`--teacher-lr`, `--no-amp`, `--data-dir`, and `--output-dir`. Run
`python train_lg_cifar100.py --help` for the complete list.

## Outputs and Jenkins-friendly logs

Smoke outputs are saved under `./outputs/lg_cifar100_deit_tiny_smoke/`; full
outputs are saved under `./outputs/lg_cifar100_deit_tiny_full/`.

The script prints and flushes the information needed in an Issue/Jenkins
report:

- Python process start, dependency check/install, and core import completion
- CIFAR-100 train/test download start and completion
- Python, PyTorch, torchvision, and timm versions
- CUDA availability, GPU count/name/memory, and AMP state
- batch size, epoch counts, dataset sizes, seed, and output paths
- per-epoch loss, train accuracy, validation accuracy, best accuracy, LR, and time
- classification and feature losses during student training
- final Top-1 accuracy and best checkpoint path
- a `[FATAL]` traceback if execution fails

Each run saves:

- `teacher_resnet56_best.pt`
- `student_deit_tiny_lg_best.pt`
- `summary.json`

## Recommended H200 Issue values

### First smoke run

- **Title:** `[Request]: 박철현 LG CIFAR-100 H200 test run`
- **사용자 ID:** `bapedragon`
- **GitHub 링크:** `https://github.com/bapedragon/IBAM_LG_cifar100_h200.git`
- **실행 명령어:** `python train_lg_cifar100.py --smoke --teacher-epochs 1 --student-epochs 1 --batch-size 64 --num-workers 2`
- **사용 이미지:** `pytorch/pytorch:latest`
- **사용 언어:** `Python`
- **GPU 할당량:** `1`

### Full 300-epoch run (after smoke succeeds)

- **Title:** `[Request]: 박철현 LG CIFAR-100 H200 300-epoch run`
- **사용자 ID:** `bapedragon`
- **GitHub 링크:** `https://github.com/bapedragon/IBAM_LG_cifar100_h200.git`
- **실행 명령어:** `python train_lg_cifar100.py --teacher-epochs 300 --student-epochs 300 --batch-size 128 --num-workers 4`
- **사용 이미지:** `pytorch/pytorch:latest`
- **사용 언어:** `Python`
- **GPU 할당량:** `7`
