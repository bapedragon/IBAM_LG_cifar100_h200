# CIFAR-100 / DeiT-Ti / CRD result

## Result

| Item | Value |
|---|---:|
| Method | CRD |
| Teacher | ResNet56, 68.68% Top-1 |
| Student | DeiT-Ti |
| Epochs | 300 |
| Vanilla Top-1 | 65.08% |
| CRD best Top-1 | **67.40%** |
| Gain over Vanilla | **+2.32pp** |
| Best epoch | 83 |
| Latest Top-1 | 63.74% |
| Elapsed time | 2h 59m 24s |

The selected result is the best checkpoint at epoch 83.

## Method configuration

- Official repository: `HobbitLong/RepDistiller`
- Pinned commit: `b84f547c5db6a35318d4671d7d5c4de74c822403`
- Loss: `CE + 0.8 * CRD`, without logit KL
- Projection dimension: `128`
- Negatives: `16,384`
- NCE temperature/momentum: `0.07 / 0.5`
- Base protocol: `cifar100_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_405`
- `artifact_manifest.json`: archive and checkpoint integrity metadata
