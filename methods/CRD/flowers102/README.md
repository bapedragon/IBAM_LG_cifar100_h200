# CRD on Flowers-102

- Teacher: fixed ResNet56 best checkpoint, 64.64%
- Student: DeiT-Ti from scratch
- Base protocol: 200 epochs, batch 64, AdamW `5e-4`, warm-up 5, cosine
- CRD: official settings recorded in `methods/CRD/README.md`

Timing run:

```bash
python methods/CRD/flowers102/train.py --timing-run --num-workers 4
```

Full run:

```bash
python methods/CRD/flowers102/train.py --num-workers 4 --run-name crd_flowers102_deit_ti_200ep --output-dir /app/output
```
