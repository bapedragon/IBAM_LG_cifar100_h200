# CRD on Chaoyang

- Teacher: fixed ResNet56 latest checkpoint, epoch 300 / 81.53%
- Student: DeiT-Ti from scratch
- Base protocol: 100 epochs, batch 64, AdamW `5e-4`, warm-up 5, cosine
- Dataset mount: `/app/data/chaoyang`
- CRD: official settings recorded in `methods/CRD/README.md`

Timing run:

```bash
python methods/CRD/chaoyang/train.py --timing-run --num-workers 4
```

Full run:

```bash
python methods/CRD/chaoyang/train.py --num-workers 4 --run-name crd_chaoyang_deit_ti_100ep --output-dir /app/output
```

Any teacher-gap adjustment is reporting-only and is applied after training to
the raw best Top-1 result.
