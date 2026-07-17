# Flowers-102 / DeiT-Ti / KD result

## Result

| Item | Value |
|---|---:|
| Method | Standard logit KD |
| Teacher | ResNet56, 64.64% Top-1 |
| Student | DeiT-Ti |
| Epochs | 200 |
| Vanilla Top-1 | 50.06% |
| KD best Top-1 | **45.88%** |
| Gain over Vanilla | **-4.18pp** |
| Best epoch | 50 |
| Latest Top-1 | 44.53% |
| Elapsed time | 35m 48s |

The best checkpoint, not the epoch-200 latest checkpoint, is the selected
student result.

## Fixed configuration

- Protocol: `flowers102_deit_ti_common_kd_v1`
- Temperature: `4.0`
- KD weight: `0.9`
- Classification weight: `0.1`
- Optimizer: AdamW, learning rate `5e-4`, weight decay `0.05`
- Schedule: 5-epoch warm-up and cosine decay
- Batch size: `64`
- Image resolution: `224 x 224`
- Label smoothing: `0.1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_401`
- `artifact_manifest.json`: archive and checkpoint integrity metadata

The checkpoint ZIP is kept outside Git history. Its canonical local filename
is recorded in `artifact_manifest.json`.
