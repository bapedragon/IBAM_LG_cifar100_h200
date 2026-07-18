# OFA: Chaoyang / DeiT-Ti

- Teacher: fixed ResNet56 latest checkpoint, epoch 300 / Top-1 `81.53%`
- Student: DeiT-Ti from scratch
- Base protocol: 100 epochs, batch 64, AdamW `5e-4`, warm-up 5, cosine
- OFA: stages 1/2/3/4, epsilon `1.0`, temperature `1.0`
- Loss weights: CE `1.0`, final OFA `1.0`, intermediate OFA `1.0`
- Dataset mount: `/app/data/chaoyang`
- Seed: `42`

Timing:

```bash
python methods/OFA/chaoyang/train.py --timing-run --num-workers 4
```

Full:

```bash
python methods/OFA/chaoyang/train.py --student-epochs 100 --num-workers 4 --run-name ofa_chaoyang_deit_ti_100ep --output-dir /app/output
```

Any teacher-gap adjustment is reporting-only and is not applied to training or
checkpoint selection.
