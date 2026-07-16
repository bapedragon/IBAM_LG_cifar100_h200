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
