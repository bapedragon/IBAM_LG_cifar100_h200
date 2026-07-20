# Ours: Flowers-102 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, epoch 291, recorded Top-1 `64.64%`
- Student: DeiT-Ti from scratch
- Split: official `train + val` for training; official `test` for evaluation
- Base protocol: 200 epochs, batch 64, AdamW `5e-4`, minimum LR `5e-6`,
  weight decay `0.05`, 5-epoch warm-up, cosine decay
- Input/recorded choices: 224 pixels, label smoothing `0.1`, seed `42`
- Loss: `CE + beta(e) * (0.5 * L_fuse + 0.5 * L_align)`
- Adaptive beta: exact ALG equations with `beta=2.5`, `tau=-0.02`, two
  50-epoch smoothing stages; `L_align` is the recorded controller signal
- Working-paper comparison target: `70.31%` Top-1

No supplied Flowers-specific Ours teacher-size config is available. The
source-compatible 32-pixel setting is therefore audited against this
repository's 224-trained teacher before any full run; a drop over 5 points
blocks training.

Timing run:

```bash
python methods/Ours/flowers102/train.py --timing-run --num-workers 4
```

Full run only after the timing log and teacher audit pass:

```bash
python methods/Ours/flowers102/train.py --student-epochs 200 --num-workers 4 --run-name ours_flowers102_deit_ti_200ep --output-dir /app/output
```

The 200-epoch dataset protocol is retained intentionally; the draft's single
300-epoch statement is marked for correction. See
[`../PAPER_AUDIT.md`](../PAPER_AUDIT.md) for confirmed and unresolved settings.
