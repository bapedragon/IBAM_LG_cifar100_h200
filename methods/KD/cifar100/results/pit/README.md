# CIFAR-100 / PiT-Tiny / KD result

## Result

| Item | Value |
|---|---:|
| Method | Standard logit KD |
| Teacher | ResNet56, 68.68% Top-1 |
| Student | PiT-Tiny |
| Epochs | 300 |
| Vanilla Top-1 | 73.16% |
| KD best Top-1 | **72.22%** |
| Gain over Vanilla | **-0.94pp** |
| Best epoch | 185 |
| Latest Top-1 | 71.54% |
| Elapsed time | 2h 37m 31s |

The best checkpoint, not the epoch-300 latest checkpoint, is the selected
student result. The fixed teacher is 4.48pp below the Vanilla PiT result, so
the negative KD gain is retained as a valid fixed-configuration baseline result.

## Fixed KD configuration

- Temperature: `4.0`
- KD weight: `0.9`
- Label smoothing: `0.1`
- Seed: `42`
- Optimizer: AdamW
- Initial learning rate: `5e-4`
- Weight decay: `0.05`
- Warm-up: 20 epochs
- Schedule: cosine decay
- Batch size: `128`
- Image resolution: `224 x 224`

No per-method hyperparameter search was performed.

## Files

- `summary.json`: machine-readable configuration, metrics, timing, and teacher metadata
- `training.log`: complete H200 log from job `bapedragon_397`
- `artifact_manifest.json`: archive and checkpoint hashes

The checkpoint ZIP is kept outside Git history to keep future H200 clones
small. Its canonical local archive name is recorded in `artifact_manifest.json`.
