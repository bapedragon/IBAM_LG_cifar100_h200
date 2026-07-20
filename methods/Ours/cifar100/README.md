# Ours: CIFAR-100 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, epoch 297, recorded Top-1 `68.68%`
- Student: DeiT-Ti from scratch
- Base protocol: 300 epochs, batch 128, AdamW `5e-4`, minimum LR `5e-6`,
  weight decay `0.05`, 20-epoch warm-up, cosine decay
- Input/recorded choices: 224 pixels, label smoothing `0.1`, seed `42`
- Ours: all 12 student blocks, ResNet stages 1/2/3, learned stage mixtures,
  `1x1` projection/QKV, `5x5` deformable attention, four heads
- Loss: `CE + beta(e) * (0.5 * L_fuse + 0.5 * L_align)`
- Adaptive beta: exact ALG equations with `beta=2.5`, `tau=-0.02`, two
  50-epoch smoothing stages; `L_align` is the recorded controller signal
- Working-paper comparison target: `82.42%` Top-1

The public DeiT-CIFAR source config uses a 32-pixel teacher input, so the code
evaluates the current 224-trained checkpoint at 32 before training. A drop over
5 percentage points blocks a full run and indicates that a compatible teacher
checkpoint/config is needed.

Timing run:

```bash
python methods/Ours/cifar100/train.py --timing-run --num-workers 4
```

Full run only after the timing log and teacher audit pass:

```bash
python methods/Ours/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name ours_cifar100_deit_ti_300ep --output-dir /app/output
```

The ALG equations/values are paper-confirmed; selecting `L_align` as the
controller signal and the epoch-1 initialization are documented reproduction
choices. See [`../PAPER_AUDIT.md`](../PAPER_AUDIT.md).
