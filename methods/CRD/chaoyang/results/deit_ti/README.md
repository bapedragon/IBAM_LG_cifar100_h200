# Chaoyang / DeiT-Ti / CRD result

## Result

| Item | Value |
|---|---:|
| Method | CRD |
| Teacher | ResNet56 latest, 81.53% Top-1 |
| Student | DeiT-Ti |
| Epochs | 100 |
| Vanilla Top-1 | 82.00% |
| CRD best Top-1 | **76.48%** |
| Gain over Vanilla | **-5.52pp** |
| Best epoch | 75 |
| Latest Top-1 | 75.41% |
| Elapsed time | 14m 11s |

The raw measured best Top-1 is retained; no teacher-gap scaling was applied to
training, checkpoint selection, or the stored result.

## Method configuration

- Official repository: `HobbitLong/RepDistiller`
- Pinned commit: `b84f547c5db6a35318d4671d7d5c4de74c822403`
- Loss: `CE + 0.8 * CRD`, without logit KL
- Projection dimension: `128`
- Negatives: `16,384`
- NCE temperature/momentum: `0.07 / 0.5`
- Base protocol: `chaoyang_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_409`
- `artifact_manifest.json`: archive and checkpoint integrity metadata
