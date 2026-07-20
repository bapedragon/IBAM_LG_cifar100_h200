# Ours: Chaoyang / DeiT-Ti

- Data: official mounted dataset under `/app/data/chaoyang`
- Teacher: fixed ResNet56 latest checkpoint, epoch 300, Top-1 `81.53%`
- Student: DeiT-Ti from scratch
- Base protocol: 100 epochs, batch 64, AdamW `5e-4`, weight decay `0.05`
- Schedule: 5-epoch warm-up followed by cosine decay
- Input and regularization: 224 pixels, label smoothing `0.1`, seed `42`
- Ours loss: `CE + 1.0 * 2.5 * (0.5 * L_fuse + 0.5 * L_align)`

Timing run:

```bash
python methods/Ours/chaoyang/train.py --timing-run --num-workers 4
```

Full run after the timing log is verified:

```bash
python methods/Ours/chaoyang/train.py --student-epochs 100 --num-workers 4 --run-name ours_chaoyang_deit_ti_100ep --output-dir /app/output
```

The working-paper Ours reference is `86.35%` Top-1. Raw measured accuracy must
be retained in the run summary; no teacher-gap correction is applied by code.
