# Multi-seed sweep: CAM loss vs. baseline (Stanford Cars)

**Date:** 2026-11-02
**Question:** Is the accuracy gain from the CAM loss real, or seed noise?

## Setup

| | |
|---|---|
| Data | Stanford Cars, 196 classes; SAM pseudo-masks |
| Model | BiT-M R50x1 + Conv-Adapters, `center_cams: true` |
| Loss | `CE + λ · squared_containment`, 3-epoch linear λ warm-up |
| Grid | shots ∈ {1, 2, 4, 8, 16} × λ ∈ {0, 0.5, 3} |
| Seeds | 42, 43, 44 (seed also changes the train/val/test split) |
| Training | 15 epochs, checkpoint selected on `val_cls_loss` after warm-up |
| Augmentation | **off** in all runs (see note below) |
| Total | 45 runs |

Where each run came from:

| Seed | λ=0 | λ=0.5, 3 |
|---|---|---|
| 42 | local, RTX 3060 | local, RTX 3060 |
| 43, 44 | Kaggle T4, kernel v2 | Kaggle T4, kernel v3 |

Within a seed, λ=0 and λ>0 share the same split and the same training subset, so the comparison is paired.

---

## 1. Test accuracy (%), mean ± std over 3 seeds

| shots | λ=0 | λ=0.5 | λ=3 |
|---|---|---|---|
| 1 | 5.2 ± 0.7 | 5.4 ± 0.6 | **7.3 ± 2.4** |
| 2 | 9.7 ± 1.2 | 12.1 ± 0.9 | **14.2 ± 1.3** |
| 4 | 16.8 ± 1.2 | 20.4 ± 2.5 | **22.2 ± 2.5** |
| 8 | 31.6 ± 2.8 | 34.8 ± 2.4 | **35.8 ± 2.6** |
| 16 | 49.3 ± 1.4 | **51.7 ± 1.1** | 51.1 ± 1.5 |

The std here (1–3 pp) is mostly **split noise**: each seed has a different test set. The paired comparison below removes it.

## 2. Paired gain over λ=0 (same seed, same split), test accuracy

| shots | λ=0.5: gain (pp) | p | λ=3: gain (pp) | p |
|---|---|---|---|---|
| 1 | +0.19 ± 0.11 | 0.095 | +2.06 ± 1.74 | 0.18 |
| 2 | +2.37 ± 0.63 | 0.023 | **+4.51 ± 0.33** | 0.002 |
| 4 | +3.61 ± 1.50 | 0.053 | **+5.40 ± 1.47** | 0.024 |
| 8 | +3.15 ± 0.54 | 0.009 | **+4.20 ± 0.19** | 0.001 |
| 16 | **+2.37 ± 0.29** | 0.005 | +1.81 ± 0.16 | 0.002 |

Gain = mean ± std over 3 seeds; p = paired t-test (df = 2). **Every one of the 30 paired comparisons is positive**, and 3 of 3 seeds improve in each row.

Pooled over all shots:

| | mean gain | positive | sign test p | Wilcoxon p |
|---|---|---|---|---|
| λ=0.5 | **+2.34 pp** | 15/15 | 0.0001 | 0.0001 |
| λ=3 | **+3.60 pp** | 15/15 | 0.0001 | 0.0001 |

The paired gains are much tighter than the raw seed-to-seed spread (e.g. 8 shots, λ=3: +4.20 ± 0.19 pp gain vs ±2.6 pp raw std). The effect is consistent; the raw noise comes from the split, not from the CAM loss.

Validation accuracy shows the same pattern: positive in 30/30 paired comparisons, +0.9 to +4.7 pp.

## 3. Localization (test, true-class CAM)

Containment = share of CAM energy outside the mask (lower is better; a flat CAM scores ~0.68).

| shots | λ=0 (seed 42 only) | λ=0.5 (3 seeds) | λ=3 (3 seeds) |
|---|---|---|---|
| 1 | 0.545 | 0.535 ± 0.003 | 0.473 ± 0.045 |
| 2 | 0.511 | 0.428 ± 0.011 | 0.333 ± 0.008 |
| 4 | 0.481 | 0.385 ± 0.004 | 0.289 ± 0.002 |
| 8 | 0.456 | 0.356 ± 0.003 | 0.267 ± 0.004 |
| 16 | 0.438 | 0.325 ± 0.004 | **0.244 ± 0.006** |

- Fully negative CAMs: ~0% in every λ>0 run (the old sign-flip problem is gone).
- Localization is very stable across seeds (std ≤ 0.01 except 1 shot).
- λ=0 localization is only available for seed 42. The Kaggle λ=0 runs had no masks loaded, so their mask-based metrics are empty; their accuracy is valid because λ=0 never uses masks.

---

## 4. Conclusions

1. **The accuracy gain is real, not seed noise.** It's positive in 30/30 paired comparisons, with p = 0.0001 pooled for both λ values.
2. **λ=3 is the better setting below 16 shots** (+4 to +5.4 pp at 2–8 shots). At 16 shots, λ=0.5 and λ=3 are about equal (+1.8 to +2.4 pp).
3. **The relative gain is largest with little data:** +47% relative at 2 shots and +32% at 4 shots for λ=3, vs +4% at 16 shots. That's consistent with the masks acting as a prior.
4. **1 shot is the weak spot:** the gain is small for λ=0.5 (+0.1 to +0.3 pp) and noisy for λ=3 (+0.4 to +3.9 pp depending on seed). With 196 images in total, the model barely trains past the warm-up before overfitting.
5. **Localization improves monotonically with λ** at every shot level.

## 5. Caveats

- **No augmentation.** A bug made `train_augment: true` silently do nothing, both here and in all STATE_001 runs. It's now fixed (`AugmentedSubset.__getitems__`). Whether the gain survives with real augmentation is still untested.
- **Mixed environments:** seed 42 ran locally (RTX 3060, PyTorch 2.6), seeds 43–44 on Kaggle (T4, PyTorch 2.11). Seed 42 has the highest λ=0 accuracy at every shot level, which could be split luck or the environment. Pairing within a seed keeps the gains valid either way.
- **3 seeds** give paired t-tests with only 2 degrees of freedom. The sign and Wilcoxon tests over all 15 pairs are the stronger evidence.
- Several λ>0 runs picked their best checkpoint at epoch 15, the last one, so they might gain a bit more with longer training.

## 6. Bugs found during this experiment (fixed, not yet committed)

| Bug | Effect | Fix |
|---|---|---|
| `AugmentedSubset` lacked `__getitems__` | DataLoader skipped augmentation in every run; PyTorch ≥ 2.10 raises an error | added `__getitems__` |
| Manifest paths parsed with `Path().stem` | On Linux, Windows-style paths never matched, so 0 masks loaded and the CAM loss was silently 0 (Kaggle v2) | `PureWindowsPath(...).stem`, plus a hard error when manifests list no usable mask |

## 7. Next steps

1. Repeat λ ∈ {0, 3} with **augmentation on**, 3 seeds, all in one environment.
2. Try λ ∈ {5, 10} at 2–8 shots, since λ=3 may not be the ceiling.
3. Try longer training (25 epochs) for λ>0.

---

### Data

- Per-run table (45 runs): [results/seed_sweep/all_seeds_runs.csv](results/seed_sweep/all_seeds_runs.csv)
- Paired differences: [results/seed_sweep/paired_diffs.csv](results/seed_sweep/paired_diffs.csv)
- Paired-gain summary: [results/seed_sweep/paired_gain_summary.csv](results/seed_sweep/paired_gain_summary.csv)
- Mean/std table: [results/seed_sweep/mean_std_by_shots_lambda.csv](results/seed_sweep/mean_std_by_shots_lambda.csv)
- Kaggle scripts: [kaggle/](kaggle/) (`build_kernel.py`, `runner_template.py`, `smoke_test_template.py`, `analyze_seeds.py`)
- Kaggle kernel: https://www.kaggle.com/code/soroushakhavan/cam-loss-seed-sweep (v2: λ=0 runs, v3: λ>0 runs)
