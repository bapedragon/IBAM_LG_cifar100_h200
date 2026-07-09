# IBAM LG-style CIFAR-100 H200 full training

This repository runs the LG-style baseline used for the IBAM/iBKD experiment
check on H200:

- Dataset: CIFAR-100
- Teacher: ResNet56, trained from scratch
- Student: DeiT-Ti, trained from scratch
- Method: LG-style feature distillation
- Target paper reference: **77.38% Top-1** for CIFAR-100 / DeiT-Ti / LG

This code is meant to run reliably on the KAU H200 GitHub-Issue runner. It is
not an exact official LG reproduction because the paper draft does not fully
specify every LG implementation detail such as the feature matching layer, loss
weight, exact augmentation recipe, teacher recipe, seed, and checkpoint rule.

## Current status

The smoke test has already been confirmed on H200 with:

- CUDA available
- H200 MIG detected
- CIFAR-100 download through verified mirror fallback
- ResNet56 teacher training
- DeiT-Ti student training
- LG-style feature MSE loss
- final result and checkpoint logs printed

The full run should now be submitted without `--smoke`.

## Files

- `train_lg_cifar100.py`: full training script
- `train_teacher_cifar100.py`: teacher-only ResNet56 training and optional
  GitHub artifact upload script
- `eval_lg_deit_cifar100.py`: evaluates an already-trained LG/pycls DeiT-Tiny
  `.pyth` checkpoint on CIFAR-100
- `requirements.txt`: declared Python dependencies
- `.gitignore`: excludes local datasets, outputs, caches, and macOS metadata

Runtime-created paths:

- `./data`: CIFAR-100 download/extraction directory
- `./outputs/lg_cifar100_deit_tiny_full`: checkpoints and `summary.json`

## Evaluate a provided LG DeiT-Tiny checkpoint

If an LG paper/repository checkpoint is already provided, do not train the
teacher or student again. Evaluation only needs the trained student checkpoint:

```bash
python eval_lg_deit_cifar100.py --checkpoint ./deit-ti_c100_LG.pyth --batch-size 256 --num-workers 4
```

If the checkpoint is hosted at a public direct-download URL:

```bash
python eval_lg_deit_cifar100.py --checkpoint-url "https://..." --batch-size 256 --num-workers 4
```

Teacher weights are not needed for this evaluation step. The teacher is only
needed when training a new distilled student.

The evaluator loads LG/pycls-style DeiT-Tiny checkpoint keys such as:

```text
patch_embed.projection.weight
layers.0.attn.qkv_transform.weight
head.weight
```

and reports:

```text
[CKPT] recorded_test_err=... recorded_top1=...%
[FINAL_RESULT] evaluated_top1=... reference_lg_top1=77.38% gap_to_reference=...
[DONE] Evaluation completed successfully; resources may be released.
```

Note: the downloaded `deit-ti_c100_LG.pyth` checkpoint inspected locally records
`test_err=21.85`, i.e. Top-1 `78.15%`, inside the checkpoint metadata. The
evaluation result should be close to the checkpoint's recorded Top-1 if the
evaluation transform matches the original code.

## Teacher-only training and GitHub upload

Use `train_teacher_cifar100.py` when a reusable ResNet56 teacher checkpoint is
needed before LG/ALG/Ours student training. This script does not import `timm`
and does not train the student.

Teacher smoke/upload test:

```bash
python train_teacher_cifar100.py --smoke --teacher-epochs 2 --batch-size 128 --num-workers 4 --upload-to-github --github-token "YOUR_1DAY_GITHUB_TOKEN" --upload-every-n-epochs 1
```

The token is never printed or saved by the script. If the token is pasted into
an H200 Issue command, use a short-lived fine-grained token scoped only to this
repository, then revoke it after the run.

The smoke run uploads:

```text
teacher_checkpoints/teacher_resnet56_cifar100_smoke/teacher_resnet56_best.pt
teacher_checkpoints/teacher_resnet56_cifar100_smoke/teacher_resnet56_latest.pt
teacher_checkpoints/teacher_resnet56_cifar100_smoke/summary.json
```

Full teacher run:

```bash
python train_teacher_cifar100.py --teacher-epochs 300 --batch-size 128 --num-workers 4 --upload-to-github --github-token "YOUR_1DAY_GITHUB_TOKEN" --upload-every-n-epochs 10
```

The important log lines are:

```text
[TEACHER][001/300] ... time=... avg_epoch=... est_300=...
[FINAL_RESULT] teacher_best_top1=... reference_teacher_top1=70.43% ...
[TIMING] teacher_avg_epoch=... estimated_300_teacher=...
[UPLOAD] remote_best=github://bapedragon/IBAM_LG_cifar100_h200/teacher_checkpoints/...
```

## Full H200 run

The shortest full-run command is:

```bash
python train_lg_cifar100.py
```

This defaults to:

- Teacher epochs: 300
- Student epochs: 300
- Batch size: 128
- DataLoader workers: 0
- Image size: 224
- Student optimizer: AdamW
- Student learning rate: 5e-4
- Student weight decay: 0.05
- Student schedule: 20-epoch warm-up followed by cosine decay
- Feature loss weight: 1.0
- Seed: 42

For the H200 Issue form, the explicit full-run command is:

```bash
python train_lg_cifar100.py --teacher-epochs 300 --student-epochs 300 --batch-size 128 --num-workers 4
```

If the H200 container has enough shared memory, for example `/dev/shm=16GB`,
`--num-workers 4` is preferred for the full run. If a DataLoader shared-memory
error appears, retry with `--num-workers 0`.

## Full-dataset timing run

Before another 300+300 epoch run, use this timing command to measure the real
epoch time on H200:

```bash
python train_lg_cifar100.py --timing-run --batch-size 128 --num-workers 4
```

`--timing-run` uses the full CIFAR-100 train/test splits, not the smoke subset.
When the epoch counts are left at their defaults, it automatically changes the
run to:

- Teacher epochs: 2
- Student epochs: 2
- Batch size: 128
- DataLoader workers: 4

At the end, check these lines:

```text
[TIMING] teacher_avg_epoch=...s student_avg_epoch=...s
[TIMING] estimated_300_teacher_plus_300_student=...
```

If the estimated time is longer than the H200 job time limit, do not retry the
full 300+300 run as a single job. Split the workflow into teacher training and
student training, or reduce validation frequency.

## Smoke test command

Smoke mode is retained only for diagnostics:

```bash
python train_lg_cifar100.py --smoke --teacher-epochs 1 --student-epochs 1 --batch-size 64 --num-workers 0
```

Smoke accuracy is not meaningful. It only checks that the full pipeline runs.

## Training protocol

### Matched to the paper draft

- CIFAR-100
- ResNet56 teacher trained from scratch
- DeiT-Ti student
- Student training for 300 epochs
- AdamW for student optimization
- Initial student learning rate `5e-4`
- Student weight decay `0.05`
- 20-epoch warm-up followed by cosine decay
- Batch size `128`
- Image resolution `224 x 224`
- PyTorch implementation
- Top-1 accuracy reporting

### Set in this repository because the draft does not fully specify it

- Teacher training recipe: 300 epochs, SGD, LR 0.1, momentum 0.9, weight decay
  5e-4, 5-epoch warm-up, cosine decay
- LG feature matching: ResNet56 `stage3` feature to DeiT-Ti patch tokens
- Teacher feature pooling to the DeiT `14 x 14` patch grid
- 1x1 convolution projection from 64 to 192 channels
- Per-token LayerNorm before feature MSE
- Feature loss weight `1.0`
- Final loss: classification cross-entropy + feature MSE
- Label smoothing `0.1`
- RandomResizedCrop and RandomHorizontalFlip for training augmentation
- Seed `42`
- DeiT-Ti pretrained weights are not used
- Best checkpoint is selected by the highest measured CIFAR-100 test Top-1

## Dataset handling

The script downloads CIFAR-100 automatically under `./data`.

Because the official Toronto host can reset connections inside the H200 runner,
the script tries verified sources in this order:

1. A pinned Hugging Face mirror of the original `cifar-100-python.tar.gz`
2. An SJTU mirror
3. The official Toronto URL used by torchvision

Every downloaded archive must match the official CIFAR-100 MD5:

```text
eb9058c3a382ffc7106e4002c42a8d85
```

So using a mirror changes only the download host, not the dataset contents.

## Logs to check after completion

The important lines in the H200 Issue report are:

```text
[TEACHER] ...
[STUDENT] ...
[FINAL_RESULT] student_best_top1=... teacher_best_top1=... reference_lg_top1=77.38%
[FINAL_RESULT] student_gap_to_reference=... teacher_gap_to_reference=...
[DONE] Training completed successfully; resources may be released.
```

For comparison:

- Paper teacher reference on CIFAR-100: **70.43%**
- Paper LG reference for CIFAR-100 / DeiT-Ti: **77.38%**

## Recommended H200 Issue values

- **Title:** `[Request]: 박철현 LG CIFAR-100 H200 300-epoch run`
- **사용자 ID:** `bapedragon`
- **GitHub 링크:** `https://github.com/bapedragon/IBAM_LG_cifar100_h200.git`
- **실행 명령어:** `python train_lg_cifar100.py --teacher-epochs 300 --student-epochs 300 --batch-size 128 --num-workers 4`
- **사용 이미지:** `pytorch/pytorch:latest`
- **사용 언어:** `Python`
- **GPU 할당량:** `7`
- **추가 필요 모듈:** 없음. `timm==1.0.27`이 없으면 스크립트 시작 시 자동 설치함.

### Timing run issue values

- **Title:** `[Request]: 박철현 LG CIFAR-100 H200 timing run`
- **사용자 ID:** `bapedragon`
- **GitHub 링크:** `https://github.com/bapedragon/IBAM_LG_cifar100_h200.git`
- **실행 명령어:** `python train_lg_cifar100.py --timing-run --batch-size 128 --num-workers 4`
- **사용 이미지:** `pytorch/pytorch:latest`
- **사용 언어:** `Python`
- **GPU 할당량:** `7`
- **추가 필요 모듈:** 없음. `timm==1.0.27`이 없으면 스크립트 시작 시 자동 설치함.

### Provided LG checkpoint evaluation issue values

Use this when the trained `deit-ti_c100_LG.pyth` checkpoint is already available
from the repository or from a public direct-download URL.

If the checkpoint is inside the cloned repo:

- **Title:** `[Request]: 박철현 LG DeiT-Tiny CIFAR-100 checkpoint eval`
- **사용자 ID:** `bapedragon`
- **GitHub 링크:** `https://github.com/bapedragon/IBAM_LG_cifar100_h200.git`
- **실행 명령어:** `python eval_lg_deit_cifar100.py --checkpoint ./deit-ti_c100_LG.pyth --batch-size 256 --num-workers 4`
- **사용 이미지:** `pytorch/pytorch:latest`
- **사용 언어:** `Python`
- **GPU 할당량:** `1`
- **추가 필요 모듈:** 없음

If the checkpoint is hosted at a direct URL:

- **Title:** `[Request]: 박철현 LG DeiT-Tiny CIFAR-100 checkpoint eval`
- **사용자 ID:** `bapedragon`
- **GitHub 링크:** `https://github.com/bapedragon/IBAM_LG_cifar100_h200.git`
- **실행 명령어:** `python eval_lg_deit_cifar100.py --checkpoint-url "https://..." --batch-size 256 --num-workers 4`
- **사용 이미지:** `pytorch/pytorch:latest`
- **사용 언어:** `Python`
- **GPU 할당량:** `1`
- **추가 필요 모듈:** 없음

### Teacher smoke/upload issue values

Use this first to confirm that H200 can train the teacher and upload the
checkpoint/summary back to GitHub.

- **Title:** `[Request]: 박철현 ResNet56 CIFAR-100 teacher smoke upload`
- **사용자 ID:** `bapedragon`
- **GitHub 링크:** `https://github.com/bapedragon/IBAM_LG_cifar100_h200.git`
- **실행 명령어:** `python train_teacher_cifar100.py --smoke --teacher-epochs 2 --batch-size 128 --num-workers 4 --upload-to-github --github-token "YOUR_1DAY_GITHUB_TOKEN" --upload-every-n-epochs 1`
- **사용 이미지:** `pytorch/pytorch:latest`
- **사용 언어:** `Python`
- **GPU 할당량:** `7`
- **추가 필요 모듈:** 없음

### Teacher full 300-epoch issue values

Run this only after the smoke/upload path is confirmed. A short-lived token is
recommended. The command uploads intermediate latest/best checkpoints every 10
epochs and uploads final artifacts at the end.

- **Title:** `[Request]: 박철현 ResNet56 CIFAR-100 teacher 300-epoch run`
- **사용자 ID:** `bapedragon`
- **GitHub 링크:** `https://github.com/bapedragon/IBAM_LG_cifar100_h200.git`
- **실행 명령어:** `python train_teacher_cifar100.py --teacher-epochs 300 --batch-size 128 --num-workers 4 --upload-to-github --github-token "YOUR_1DAY_GITHUB_TOKEN" --upload-every-n-epochs 10`
- **사용 이미지:** `pytorch/pytorch:latest`
- **사용 언어:** `Python`
- **GPU 할당량:** `7`
- **추가 필요 모듈:** 없음
