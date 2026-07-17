# Flowers-102 / DeiT-Ti / CRD result

## Result

| Item | Value |
|---|---:|
| Method | CRD |
| Teacher | ResNet56, 64.64% Top-1 |
| Student | DeiT-Ti |
| Epochs | 200 |
| Vanilla Top-1 | 50.06% |
| CRD best Top-1 | **46.63%** |
| Gain over Vanilla | **-3.43pp** |
| Best epoch | 101 |
| Latest Top-1 | 46.06% |
| Elapsed time | 37m 22s |

## Method configuration

- Official repository: `HobbitLong/RepDistiller`
- Pinned commit: `b84f547c5db6a35318d4671d7d5c4de74c822403`
- Loss: `CE + 0.8 * CRD`, without logit KL
- Projection dimension: `128`
- Negatives: `16,384`
- NCE temperature/momentum: `0.07 / 0.5`
- Base protocol: `flowers102_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_407`
- `artifact_manifest.json`: archive and checkpoint integrity metadata
