# CIFAR-100 / DeiT-Ti / ReviewKD result

## Result

| Item | Value |
|---|---:|
| Method | ReviewKD |
| Teacher | ResNet56, 68.68% Top-1 |
| Student | DeiT-Ti |
| Epochs | 300 |
| Vanilla Top-1 | 65.08% |
| ReviewKD best Top-1 | **72.84%** |
| Gain over Vanilla | **+7.76pp** |
| Best epoch | 236 |
| Latest Top-1 | 72.48% |
| Elapsed time | 2h 44m 03s |

The selected result is the best checkpoint at epoch 236.

## Method configuration

- Official repository: `dvlab-research/ReviewKD`
- Pinned commit: `cede6ea6387ae9b6127de0e561507177bf19c11e`
- Loss: `CE + ramp(epoch, 20) * 0.6 * HCL`
- Student blocks: `3, 7, 11`
- Teacher stages: `stage1, stage2, stage3`
- Base protocol: `cifar100_deit_ti_common_kd_v1`
- Seed: `42`

## Files

- `summary.json`: complete machine-readable configuration and metrics
- `training.log`: complete H200 log from job `bapedragon_411`
- `artifact_manifest.json`: archive and checkpoint integrity metadata
