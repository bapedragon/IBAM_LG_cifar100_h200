# IBAM: Flowers-102 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, epoch 291, Top-1 `64.64%`
- Student: DeiT-Ti from scratch
- Split: official `train + val` for training and official `test` for evaluation
- Base protocol: 200 epochs, batch 64, AdamW `5e-4`, weight decay `0.05`
- Schedule: 5-epoch warm-up followed by cosine decay
- Input and regularization: 224 pixels, label smoothing `0.1`, seed `42`
- IBAM loss: `CE + 1.0 * 2.5 * (0.5 * L_fuse + 0.5 * L_align)`

Timing run:

```bash
python methods/IBAM/flowers102/train.py --timing-run --num-workers 4
```

Full run after the timing log is verified:

```bash
python methods/IBAM/flowers102/train.py --student-epochs 200 --num-workers 4 --run-name ibam_flowers102_deit_ti_200ep --output-dir /app/output
```

The working-paper Ours reference is `70.31%` Top-1. The exact external adaptive
beta controller is not part of the provided source snippet, so the executable
uses and records fixed `beta=1.0` unless a confirmed value is supplied.
