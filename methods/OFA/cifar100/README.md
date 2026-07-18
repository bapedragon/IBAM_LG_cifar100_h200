# OFA: CIFAR-100 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, Top-1 `68.68%`
- Student: DeiT-Ti from scratch
- Base protocol: 300 epochs, batch 128, AdamW `5e-4`, warm-up 20, cosine
- OFA: stages 1/2/3/4, epsilon `1.0`, temperature `1.0`
- Loss weights: CE `1.0`, final OFA `1.0`, intermediate OFA `1.0`
- Seed: `42`

Timing:

```bash
python methods/OFA/cifar100/train.py --timing-run --num-workers 4
```

Full:

```bash
python methods/OFA/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name ofa_cifar100_deit_ti_300ep --output-dir /app/output
```
