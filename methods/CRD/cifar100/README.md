# CRD on CIFAR-100

- Teacher: fixed ResNet56 best checkpoint, 68.68%
- Student: DeiT-Ti from scratch
- Base protocol: 300 epochs, batch 128, AdamW `5e-4`, warm-up 20, cosine
- CRD: official settings recorded in `methods/CRD/README.md`

Timing run:

```bash
python methods/CRD/cifar100/train.py --timing-run --num-workers 4
```

Full run:

```bash
python methods/CRD/cifar100/train.py --num-workers 4 --run-name crd_cifar100_deit_ti_300ep --output-dir /app/output
```

## Completed result

Best Top-1 is **67.40%** at epoch 83; epoch-300 latest Top-1 is 63.74%.
The complete record is stored under `results/deit_ti/`.
