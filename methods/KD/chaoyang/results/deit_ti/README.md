# Chaoyang / DeiT-Ti / KD result

## Result

| Item | Value |
|---|---:|
| Method | Standard logit KD |
| Teacher | ResNet56 latest, 81.53% Top-1 |
| Student | DeiT-Ti |
| Epochs | 100 |
| Vanilla Top-1 | 82.00% |
| KD best Top-1 | **80.60%** |
| Gain over Vanilla | **-1.40pp** |
| Best epoch | 94 |
| Latest Top-1 | 80.22% |
| Elapsed time | 13m 12s |

The raw measured best Top-1 is retained. Any teacher-gap adjustment is a
reporting-only calculation and is not stored as the training result.

## Fixed configuration

- Protocol: `chaoyang_deit_ti_common_kd_v1`
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
- `training.log`: complete H200 log from job `bapedragon_403`
- `artifact_manifest.json`: archive and checkpoint integrity metadata

The checkpoint ZIP is kept outside Git history. Its canonical local filename
is recorded in `artifact_manifest.json`.
