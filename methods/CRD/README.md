# Contrastive Representation Distillation (CRD)

This folder implements CRD for the fixed ResNet56-to-DeiT-Ti comparison.
Method-specific code is ported from the authors' official RepDistiller
repository rather than reimplementing CRD from a secondary source.

## Official-code provenance

- Paper: Contrastive Representation Distillation, ICLR 2020
- Official repository: https://github.com/HobbitLong/RepDistiller
- Pinned official commit: `b84f547c5db6a35318d4671d7d5c4de74c822403`
- Official files used as the implementation basis:
  - `crd/criterion.py`
  - `crd/memory.py`
  - `dataset/cifar100.py`
  - `train_student.py`
- License: BSD 2-Clause, copied to `OFFICIAL_CODE_LICENSE.txt`

`official_crd.py` preserves the official symmetric CRD loss, linear normalized
embeddings, two-sided memory bank, alias sampler, and noise-contrastive loss.
Only device handling was modernized so the code works on current PyTorch and
the H200 runner.

## Official CRD method settings

| Setting | Value |
|---|---:|
| Classification weight (`gamma`) | `1.0` |
| Logit-KD weight (`alpha`) | `0.0` |
| CRD weight (`beta`) | `0.8` |
| Projection dimension | `128` |
| Negative samples (`nce_k`) | `16,384` |
| NCE temperature (`nce_t`) | `0.07` |
| Memory momentum (`nce_m`) | `0.5` |
| Sampling mode | `exact` |

The standalone official CRD command uses `-r 1 -a 0 -b 0.8`. Accordingly,
these experiments use classification CE plus CRD and do not add logit KL.

## CNN-to-ViT adaptation

The official implementation consumes one pooled representation from each
network. The heterogeneous connection is therefore:

- Teacher representation: ResNet56 stage3 global-average-pooled feature,
  dimension `64`
- Student representation: DeiT-Ti CLS pre-logits representation,
  dimension `192`
- Official CRD linear embeddings: `64 -> 128` and `192 -> 128`

No spatial feature grid is matched or discarded beyond the global pooling
already required by CRD. This explicit adapter is the only architecture bridge.

## Fixed dataset protocols

The base student settings remain identical to the completed KD runs.

| Dataset | Epochs | Batch | Optimizer | LR | Weight decay | Warm-up |
|---|---:|---:|---|---:|---:|---:|
| CIFAR-100 | 300 | 128 | AdamW | `5e-4` | `0.05` | 20 |
| Flowers-102 | 200 | 64 | AdamW | `5e-4` | `0.05` | 5 |
| Chaoyang | 100 | 64 | AdamW | `5e-4` | `0.05` | 5 |

All use cosine decay, 224-pixel inputs, label smoothing `0.1`, no external
pretraining, AMP, seed `42`, and the same dataset-specific augmentation and
teacher checkpoint as standard KD.

## First timing run

Start with CIFAR-100:

```bash
python methods/CRD/cifar100/train.py --timing-run --num-workers 4
```

After timing verification:

```bash
python methods/CRD/cifar100/train.py --num-workers 4 --run-name crd_cifar100_deit_ti_300ep --output-dir /app/output
```
