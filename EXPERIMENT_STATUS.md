# Experiment status

Last updated: 2026-07-20

This document summarizes the completed H200 teacher and DeiT-Ti knowledge-distillation runs. Student scores use the best checkpoint Top-1 accuracy. A timing run is a two-epoch execution check and is not treated as a final result.

## 1. Selected teacher checkpoints

| Dataset | Teacher | Selected checkpoint | Epoch | Top-1 | Paper reference | Gap |
|---|---|---|---:|---:|---:|---:|
| CIFAR-100 | ResNet56 | Best | 297 | **68.68%** | 70.43% | -1.75pp |
| Flowers-102 | ResNet56 | Best | 291 | **64.64%** | 66.33% | -1.69pp |
| Chaoyang | ResNet56 | Latest | 300 | **81.53%** | 77.20% | +4.33pp |

For Chaoyang, the epoch-300 latest checkpoint was deliberately fixed for every KD method because it is closer to the paper reference than the 83.08% best checkpoint.

## 2. Main DeiT-Ti comparison

Values in parentheses are gains over the corresponding Vanilla DeiT-Ti result.

| Method | CIFAR-100 | Flowers-102 | Chaoyang |
|---|---:|---:|---:|
| Vanilla | 65.08% | 50.06% | 82.00% |
| KD | 67.00% (+1.92pp) | 45.88% (-4.18pp) | 80.60% (-1.40pp) |
| CRD | 67.40% (+2.32pp) | 46.63% (-3.43pp) | 76.48% (-5.52pp) |
| ReviewKD | 72.84% (+7.76pp) | 50.76% (+0.70pp) | 81.53% (-0.47pp) |
| **MGD** | **73.71% (+8.63pp)** | **51.57% (+1.51pp)** | **81.81% (-0.19pp)** |
| OFA | 66.18% (+1.10pp) | 44.07% (-5.99pp) | 80.04% (-1.96pp) |

## 3. Completed full-run details

| Method | Dataset | Epochs | Best epoch | Best Top-1 | Latest Top-1 | Status |
|---|---|---:|---:|---:|---:|---|
| KD | CIFAR-100 | 300 | 191 | **67.00%** | 66.14% | Complete |
| KD | Flowers-102 | 200 | 50 | **45.88%** | 44.53% | Complete |
| KD | Chaoyang | 100 | 94 | **80.60%** | 80.22% | Complete |
| CRD | CIFAR-100 | 300 | 83 | **67.40%** | 63.74% | Complete |
| CRD | Flowers-102 | 200 | 101 | **46.63%** | 46.06% | Complete |
| CRD | Chaoyang | 100 | 75 | **76.48%** | 75.41% | Complete |
| ReviewKD | CIFAR-100 | 300 | 236 | **72.84%** | 72.48% | Complete |
| ReviewKD | Flowers-102 | 200 | 158 | **50.76%** | 50.46% | Complete |
| ReviewKD | Chaoyang | 100 | 78 | **81.53%** | 80.27% | Complete |
| MGD | CIFAR-100 | 300 | 250 | **73.71%** | 73.25% | Complete |
| MGD | Flowers-102 | 200 | 101 | **51.57%** | 50.94% | Complete |
| MGD | Chaoyang | 100 | 61 | **81.81%** | 80.04% | Complete |
| OFA | CIFAR-100 | 300 | 227 | **66.18%** | 65.92% | Complete |
| OFA | Flowers-102 | 200 | 190 | **44.07%** | 43.96% | Complete |
| OFA | Chaoyang | 100 | 74 | **80.04%** | 79.43% | Complete |

## 4. Current progress

| Method | CIFAR-100 | Flowers-102 | Chaoyang |
|---|---|---|---|
| KD | Full run complete | Full run complete | Full run complete |
| CRD | Full run complete | Full run complete | Full run complete |
| ReviewKD | Full run complete | Full run complete | Full run complete |
| MGD | Full run complete | Full run complete | Full run complete |
| OFA | Full run complete | Full run complete | Full run complete |
| Ours | Timing pending | Timing pending | Timing pending |

The five generic KD matrices are complete. Ours has passed the V3/ALG
paper/source audit, exact ALG-equation unit tests, teacher-grid and
source-compatible grid tests, local structural forward/backward test, real
DeiT/teacher feature-path test, and strict checkpoint-reload test.
The next step is the CIFAR-100 two-epoch full-data timing run. Its teacher
runtime audit must pass before any full run, followed by Flowers-102 and
Chaoyang.

## 5. Exploratory runs outside Table 2

| Method | Dataset | Student | Epochs | Best Top-1 | Vanilla | Gain |
|---|---|---|---:|---:|---:|---:|
| KD | CIFAR-100 | ConViT-Tiny | 300 | **73.59%** | 74.87% | -1.28pp |
| KD | CIFAR-100 | PiT-Tiny | 300 | **72.22%** | 73.16% | -0.94pp |

## 6. Current interpretation

- MGD is currently the strongest completed generic KD method on all three datasets.
- On CIFAR-100, MGD improves over Vanilla by 8.63 percentage points; on Flowers-102, it improves by 1.51 points.
- On Chaoyang, the selected teacher accuracy (81.53%) is slightly below Vanilla DeiT-Ti (82.00%), so improving the student through KD is intrinsically difficult. MGD is currently closest to Vanilla at -0.19 points.
- All reported values above are raw measured best-checkpoint accuracies. No teacher-gap proportional correction has been applied.

## 7. Newly verified artifact archives

| Method | Dataset | Source job | Archive | Verification |
|---|---|---|---|---|
| ReviewKD | Flowers-102 | `bapedragon_414` | `ReviewKD_Flowers102_DeiT-Ti_200ep_seed42_Top1-50.76.zip` | Best/latest load and finite-tensor checks passed |
| ReviewKD | Chaoyang | `bapedragon_417` | `ReviewKD_Chaoyang_DeiT-Ti_100ep_seed42_Top1-81.53.zip` | Best/latest load and finite-tensor checks passed |
| MGD | CIFAR-100 | `bapedragon_419` | `MGD_CIFAR100_DeiT-Ti_300ep_seed42_Top1-73.71.zip` | Best/latest load and finite-tensor checks passed |
| MGD | Flowers-102 | `bapedragon_421` | `MGD_Flowers102_DeiT-Ti_200ep_seed42_Top1-51.57.zip` | Best/latest load and finite-tensor checks passed |
| MGD | Chaoyang | `bapedragon_423` | `MGD_Chaoyang_DeiT-Ti_100ep_seed42_Top1-81.81.zip` | Best/latest load and finite-tensor checks passed |
| OFA | CIFAR-100 | `bapedragon_425` | `OFA_CIFAR100_DeiT-Ti_300ep_seed42_Top1-66.18.zip` | Best/latest load and finite-tensor checks passed |
| OFA | Flowers-102 | `bapedragon_427` | `OFA_Flowers102_DeiT-Ti_200ep_seed42_Top1-44.07.zip` | Student/projector strict load and finite-tensor checks passed |
| OFA | Chaoyang | `bapedragon_429` | `OFA_Chaoyang_DeiT-Ti_100ep_seed42_Top1-80.04.zip` | Student/projector strict load and finite-tensor checks passed |

The large ZIP and checkpoint files are stored outside Git. Each result folder tracks the summary, complete training log, and SHA-256 integrity manifest.
