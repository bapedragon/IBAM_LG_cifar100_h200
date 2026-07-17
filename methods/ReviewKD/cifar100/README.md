# ReviewKD: CIFAR-100 / DeiT-Ti

- Teacher: fixed ResNet56 best checkpoint, Top-1 `68.68%`
- Student: DeiT-Ti, scratch
- Epochs: `300`
- Batch: `128`
- AdamW: LR `5e-4`, weight decay `0.05`
- LR schedule: 20-epoch warm-up and cosine decay
- ReviewKD: `CE + ramp * 0.6 * HCL`, feature ramp `20` epochs
- Image size: `224`
- Label smoothing: `0.1`
- Seed: `42`

Timing:

```bash
python methods/ReviewKD/cifar100/train.py --timing-run --num-workers 4
```

Full:

```bash
python methods/ReviewKD/cifar100/train.py --num-workers 4 --run-name reviewkd_cifar100_deit_ti_300ep --output-dir /app/output
```

## Completed result

Best Top-1 is **72.84%** at epoch 236; epoch-300 latest Top-1 is 72.48%.
The complete record is stored under `results/deit_ti/`.
