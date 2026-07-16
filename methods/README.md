# KD method experiment organization

Each comparison method gets its own folder under `methods/`. Every method
folder contains one shared implementation plus dataset-specific wrappers and
README files. This keeps method-specific protocols separate without duplicating
the full training loop three times.

```text
methods/
  KD/                  implemented, 3/21 runs complete
  CRD/                 planned
  ReviewKD/            planned
  MGD/                 planned
  OFA/                 planned
```

## Fixed inputs

Teacher selection is complete. Every KD method must use the files recorded in
`checkpoints/teachers/manifest.json`; a method-specific teacher replacement is
not allowed.

The target matrix contains three datasets and seven ViT students:

- Datasets: CIFAR-100, Flowers-102, Chaoyang
- Students: DeiT-Ti, ConViT, CvT, PiT, PVTv2, T2T-7, T2T-14
- Runs per KD method: `3 datasets x 7 students = 21 runs`

## Common student protocol

The following settings come from the V2 paper draft and must remain common
across the compared methods:

- 300 epochs
- AdamW
- Initial learning rate `5e-4`
- Weight decay `0.05`
- 20-epoch warm-up followed by cosine decay
- Batch size `128`
- Image resolution `224 x 224`
- PyTorch and AMP on CUDA
- Top-1 accuracy evaluation

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
