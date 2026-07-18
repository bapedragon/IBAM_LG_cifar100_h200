# One-for-All Knowledge Distillation (OFA-KD)

This folder applies OFA-KD to the fixed ResNet56-to-DeiT-Ti comparison. The
adaptive target-enhancement loss, four DeiT intermediate stages, patch merging,
transformer projector depth, initialization, and loss coefficients follow the
authors' official implementation.

## Official-code provenance

- Paper: One-for-All: Bridge the Gap Between Heterogeneous Architectures in
  Knowledge Distillation, NeurIPS 2023
- Official repository: https://github.com/Hao840/OFAKD
- Pinned commit: `f7bb896cac9879040800bde08a8cc2057a904c52`
- Official sources: `distillers/ofa.py`, `distillers/utils.py`, and
  `custom_forward/vision_transformer.py`
- Official CIFAR ViT protocol: `configs/cifar/vit_mlp.yaml`

The official repository does not contain a license file at the pinned commit.
Therefore, `official_ofa.py` is an independently structured compatibility
implementation of the documented behavior, not a verbatim copy.

Official OFA defaults used here:

```text
total loss = 1.0 * CE + 1.0 * final OFA + 1.0 * sum(intermediate OFA)
epsilon = 1.0
temperature = 1.0
student stages = 1, 2, 3, 4
```

## ResNet56-to-DeiT-Ti connection

OFA was designed for heterogeneous architectures and already projects student
features into class-logit space. The connection is:

- Teacher knowledge: frozen ResNet56 final class logits
- Student stage mapping: DeiT blocks 1, 3, 9, and 11 (zero-based indices)
- Each student feature: one CLS token plus a `14 x 14` patch grid, dimension 192
- Official projection: `14 x 14 -> 7 x 7` patch merging, then 3/2/1/1
  transformer blocks for stages 1/2/3/4, CLS selection, and a class head
- Intermediate and final student logits both use OFA adaptive target enhancement

The projector is trained jointly with the student and is stored in each
checkpoint. It is used only during training; evaluation uses the student model.

## Fixed dataset protocols

| Dataset | Epochs | Batch | Optimizer | LR | Weight decay | LR warm-up |
|---|---:|---:|---|---:|---:|---:|
| CIFAR-100 | 300 | 128 | AdamW | `5e-4` | `0.05` | 20 |
| Flowers-102 | 200 | 64 | AdamW | `5e-4` | `0.05` | 5 |
| Chaoyang | 100 | 64 | AdamW | `5e-4` | `0.05` | 5 |

All use cosine decay, 224-pixel inputs, label smoothing `0.1`, AMP, seed `42`,
no external pretraining, and the same dataset split and fixed teacher used by
KD, CRD, ReviewKD, and MGD. These shared settings are intentionally held fixed
for fair per-dataset comparison; OFA-specific loss and projector settings come
from the official implementation.

## First timing run

```bash
python methods/OFA/cifar100/train.py --timing-run --num-workers 4
```

After timing verification:

```bash
python methods/OFA/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name ofa_cifar100_deit_ti_300ep --output-dir /app/output
```
