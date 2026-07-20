# Ours paper/source audit

This audit separates (1) values stated in V3/ALG, (2) details present only in
the supplied Ours model source, and (3) unavoidable reproduction choices.
The dataset-specific base protocols are intentionally documented elsewhere.

## Confirmed by V3 and the cited ALG paper

| Item | Implemented behavior |
|---|---|
| Total objective | `CE + beta(e) * [lambda*L_fuse + (1-lambda)*L_align]` |
| Loss balance | `lambda=0.5` |
| Alignment/fusion losses | MSE to the frozen CNN feature |
| Student aggregation | features from all `N=12` DeiT blocks with learned convex weights, uniformly initialized |
| Channel/grid alignment | stage-specific `1x1` projection and bilinear resize to the teacher-stage resolution |
| Fusion | channel attention, deformable spatial attention, and grid-preserving convolutional cross-attention |
| Kernels | deformable spatial kernel `5x5`; Q/K/V projections `1x1` |
| Adaptive beta | ALG `beta=2.5`, `tau=-0.02`, 50-epoch loss smoothing/differentiation and 50-epoch derivative smoothing |
| Stop rule | guide while the twice-smoothed derivative is `< tau`; the crossing epoch is the last guided epoch, then CE only |
| Teacher/inference | teacher frozen during training; teacher and Ours modules removed at inference |

The ALG derivative controller follows Eqs. (10)-(19). For epoch `e<=50`, it
uses the current LG loss versus the mean of previous available losses. For
`e>50`, it uses the current loss versus the loss 50 epochs earlier. Those raw
derivatives are averaged over up to 50 epochs before applying `tau`.

## Fixed by the supplied Ours model source, not stated numerically in V3

The supplied file is identified by SHA-256
`8649078970b93d750a956994611b65cdec0c24f907d35d86f29d635e8a3b8624`.

| Item | Source-derived setting |
|---|---|
| CNN stages | ResNet56 stages 1/2/3 |
| Aggregation | one learned 12-block mixture per CNN stage |
| Attention heads | 4 |
| Channel-attention reduction | 16 |
| Cross-attention dropout | 0 |
| Multi-stage loss reduction | sum of per-stage mean-squared errors |

There is one paper/source inconsistency. V3 says to resample to the teacher
resolution, while the supplied source resizes both tensors to the larger grid.
The main experiment defaults to paper-aligned `--grid-resize-mode teacher`.
`--grid-resize-mode larger` is retained only for an explicitly labeled source
compatibility ablation.

## Ours-specific reproduction choices not fixed by either paper

| Choice | Repository decision | Reason |
|---|---|---|
| Signal observed by ALG | epoch-average `L_align` | It is the direct CNN/ViT distance most closely corresponding to ALG's `L_LG`; V3 does not say whether to observe align, fuse, or the combined loss. |
| Epoch-1 derivative | initialize to `0` and forbid stopping at epoch 1 | ALG's published early-epoch expression has no previous value at epoch 1. |
| Flowers/Chaoyang teacher input size | provisional `32` with runtime accuracy audit | ALG confirms 32 for its CNN path, but no Ours per-dataset config/checkpoint was supplied. |
| Teacher mismatch tolerance | block a full run when runtime Top-1 drops by more than 5 pp | Safety gate, not a paper hyperparameter. |

The current teacher checkpoints were trained at 224 pixels. Therefore every
timing run reports `[TEACHER_RUNTIME_AUDIT]`; a full run is blocked on a large
32-pixel accuracy drop unless the mismatch is deliberately overridden. A
failed audit should normally be fixed with a compatible teacher/config rather
than overridden.

## Shared experiment choices (not Ours-specific)

Dataset-specific epochs, batch size/warm-up, common augmentation,
normalization, label smoothing `0.1`, seed `42`, AMP, and best-checkpoint
selection belong to the shared comparison protocol. V3's single 300-epoch
statement is intentionally not applied to Flowers/Chaoyang because the team
has marked it for correction.

## Full-run gate

1. Run the dataset wrapper with `--timing-run`.
2. Confirm dataset/split, feature shapes, finite loss, exact ALG parameters,
   epoch timing, and `[TEACHER_RUNTIME_AUDIT]`.
3. If the teacher audit passes, run the dataset-specific full command.
4. Keep the generated `summary.json`, which records the complete loss,
   derivative, beta, stop-epoch, and aggregation-weight histories.
