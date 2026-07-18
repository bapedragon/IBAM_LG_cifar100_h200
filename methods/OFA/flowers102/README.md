# OFA: Flowers-102 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, Top-1 `64.64%`
- Student: DeiT-Ti from scratch
- Base protocol: 200 epochs, batch 64, AdamW `5e-4`, warm-up 5, cosine
- OFA: stages 1/2/3/4, epsilon `1.0`, temperature `1.0`
- Loss weights: CE `1.0`, final OFA `1.0`, intermediate OFA `1.0`
- Seed: `42`

Timing:

```bash
python methods/OFA/flowers102/train.py --timing-run --num-workers 4
```

Full:

```bash
python methods/OFA/flowers102/train.py --student-epochs 200 --num-workers 4 --run-name ofa_flowers102_deit_ti_200ep --output-dir /app/output
```
