# Ours: CIFAR-100 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, epoch 297, Top-1 `68.68%`
- Student: DeiT-Ti from scratch
- Base protocol: 300 epochs, batch 128, AdamW `5e-4`, weight decay `0.05`
- Schedule: 20-epoch warm-up followed by cosine decay
- Input and regularization: 224 pixels, label smoothing `0.1`, seed `42`
- Ours: all 12 student blocks, ResNet stages 1/2/3, `14 x 14` grid,
  `5 x 5` deformable spatial attention, four attention heads
- Loss: `CE + 1.0 * 2.5 * (0.5 * L_fuse + 0.5 * L_align)`

Timing run:

```bash
python methods/Ours/cifar100/train.py --timing-run --num-workers 4
```

Full run after the timing log is verified:

```bash
python methods/Ours/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name ours_cifar100_deit_ti_300ep --output-dir /app/output
```

The working-paper Ours reference is `82.42%` Top-1. It is a comparison target,
not a guaranteed output, because the exact external adaptive beta controller
was not included in the provided source snippet.
