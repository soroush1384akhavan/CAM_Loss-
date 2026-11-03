# STATE_002 — Multi-seed sweep, Kaggle pipeline, two data-loading bugs fixed

| | |
|---|---|
| Date | 2026-11-03 |
| Branch | `lab-work-imported` |
| Commit | `7a512d6` — *fix: cross-OS mask manifests and DataLoader augmentation; add seed sweep* |
| Previous state | [STATE_001](STATE_001_run_analysis_fixes.md) (`a1ac006`) |
| Restore this state | `git checkout 7a512d6` |
| Code diff vs previous | `git diff a1ac006 7a512d6` |

---

## 1. Project Goal

Unchanged from STATE_001:

```text
loss = CE(GAP(cam), y) + λ_cam · squared_containment(centered CAM, SAM pseudo-mask)
```

**New in this state:** the main claim now has multi-seed evidence. On Stanford Cars, the CAM loss improves few-shot accuracy in **30 of 30 paired comparisons** (3 seeds × 5 shot levels × 2 λ values).

## 2. Code Components

| Part | File | Status in this state |
|---|---|---|
| Model | [src/model_with_adapter_bit.py](../../src/model_with_adapter_bit.py) | Unchanged (BiT-M R50x1 + adapters, `center_cams`) |
| Losses | [src/losses.py](../../src/losses.py) | Unchanged |
| Training | [src/train.py](../../src/train.py) | Unchanged |
| Data | [src/data.py](../../src/data.py) | **Two bug fixes:** `AugmentedSubset.__getitems__` (augmentation now actually runs through the DataLoader); manifest paths parsed with `PureWindowsPath` (masks load on Linux). **New guard:** raises an error if manifests list zero usable masks. |
| Sweep | [sweep.py](../../sweep.py) | Grid set to what was run: shots {1,2,4,8,16} × λ {0, 0.5, 3}, 15 epochs, 1 repeat per call |
| Kaggle | [kaggle/](../../kaggle/) | **New:** `build_kernel.py` (embeds the code into a kernel), `runner_template.py` (one seed per T4 GPU), `smoke_test_template.py` (CPU check that masks load and the CAM loss is live), `analyze_seeds.py` |
| Results | [results/seed_sweep/](../../results/seed_sweep/) | **New:** all 45 runs, paired differences, mean/std tables |

## 3. Active Config

The committed `configs/config.yaml` is unchanged from STATE_001. The sweep overrides `lambda_cam`, `epochs`, `train_shots_per_class` and `seed` per run.

Settings used by the multi-seed sweep:

| Key | Value |
|---|---|
| data | Stanford Cars, 196 classes, SAM masks (14,393 images with a mask) |
| model | `adapter_bit`, gamma=8, frozen backbone, `center_cams: true` |
| loss | `squared_containment`, λ ∈ {0, 0.5, 3}, 3-epoch linear warm-up |
| training | 15 epochs, lr 3e-4, cosine, batch 8, AMP; checkpoint selected on `val_cls_loss` after warm-up |
| augmentation | **off** (`train_augment: false` on Kaggle, and effectively off locally because of the bug) |
| seeds | 42 (local, RTX 3060, PyTorch 2.6), 43 and 44 (Kaggle T4, PyTorch 2.11) |

## 4. Latest Results

Full report: [SEED_SWEEP_RESULTS.md](../../SEED_SWEEP_RESULTS.md).

**Test accuracy (%), mean ± std over 3 seeds**

| shots | λ=0 | λ=0.5 | λ=3 |
|---|---|---|---|
| 1 | 5.2 ± 0.7 | 5.4 ± 0.6 | **7.3 ± 2.4** |
| 2 | 9.7 ± 1.2 | 12.1 ± 0.9 | **14.2 ± 1.3** |
| 4 | 16.8 ± 1.2 | 20.4 ± 2.5 | **22.2 ± 2.5** |
| 8 | 31.6 ± 2.8 | 34.8 ± 2.4 | **35.8 ± 2.6** |
| 16 | 49.3 ± 1.4 | **51.7 ± 1.1** | 51.1 ± 1.5 |

**Paired gain over λ=0 (same seed and split), pooled over shots**

| | mean gain | positive | sign test p |
|---|---|---|---|
| λ=0.5 | +2.34 pp | 15/15 | 0.0001 |
| λ=3 | +3.60 pp | 15/15 | 0.0001 |

- λ=3 is best at 2–8 shots (+4.2 to +5.4 pp). At 16 shots both λ give about +2 pp.
- Containment (CAM energy outside the mask) at 16 shots: 0.44 (λ=0) → 0.33 (λ=0.5) → 0.24 (λ=3). No λ>0 run has fully negative CAMs.

## 5. Known Issues

| # | Issue | Status |
|---|---|---|
| A1 | **Augmentation never ran** in any STATE_001 run or in the sweep (DataLoader bypassed `AugmentedSubset.__getitem__`) | ✅ Code fixed and tested; ⚠️ results with augmentation not yet measured |
| A2 | **Masks did not load on Linux** (Windows-style manifest paths); Kaggle v2 trained with CAM loss = 0 | ✅ Fixed + guard; confirmed by Kaggle smoke test (14,393 masks found) |
| A3 | λ=0 localization metrics exist only for seed 42 (Kaggle λ=0 runs had no masks) | ℹ️ Accuracy of those runs is valid; λ=0 never uses masks |
| A4 | Seeds split across two environments (local vs Kaggle) | ℹ️ Paired gains stay valid; next sweep should use one environment |
| A5 | Uncommitted user WIP: `main.py` `--config` argument, `configs/config_cars_full.yaml`, `configs/config_pets_full.yaml`, full-shot `config.yaml` | ⚠️ `main.py` change breaks one test in `test_normalization.py` (it patches `main.CONFIG_PATH`) |
| P5 | Old runs trained with ImageNet normalization | Historical, unchanged |
| N1 | Old checkpoints load with centered CAMs under the current config | Unchanged |

## 6. Test Status

| Test | Result |
|---|---|
| `python tests/test_state001_fixes.py` | ✅ 28/28 (6 new: DataLoader augmentation, manifest stems on Windows/POSIX, zero-mask guard) |
| `python tests/test_cam_softmax_loss.py` | ✅ 19/19 |
| `python tests/test_normalization.py` | ✅ 13/13 at this commit; ❌ 1 error with the uncommitted `main.py` (A5) |
| Kaggle CPU smoke test | ✅ masks load on Linux, CAM loss non-zero, gradient reaches the model |

## 7. Diff from STATE_001

### 7.1 Code Changes

`git diff --stat a1ac006 7a512d6`: 14 files, +849 / −6 (excluding `project_states/`).

| File | STATE_001 | STATE_002 |
|---|---|---|
| `src/data.py` | `Path(mask_path).stem`; `AugmentedSubset` overrides only `__getitem__` | `PureWindowsPath(...).stem`; `AugmentedSubset.__getitems__`; error when manifests list no usable mask |
| `sweep.py` | λ {0, 0.3, 0.5, 1, 3}, 20 epochs, 3 repeats | λ {0, 0.5, 3}, 15 epochs, 1 repeat (seeds supplied externally) |
| `tests/test_state001_fixes.py` | 22 checks | 28 checks |
| `kaggle/` | — | Kernel builder, sweep runner, smoke test, analysis |
| `results/seed_sweep/` | — | 45-run results (CSV) |
| docs | — | `SEED_SWEEP_RESULTS.md`; correction note on `COMPARE_LAMBDA0_VS_05_FA.md` |

### 7.2 Behaviour Changes

| Behaviour | STATE_001 | STATE_002 |
|---|---|---|
| `train_augment: true` | Silently no augmentation | Augmentation applied |
| Training on Linux/Kaggle | 0 masks, CAM loss silently 0 | Masks load; error if none found |
| Evidence for the CAM-loss gain | 1 pair of runs (16 shots) | 45 runs, 3 seeds, 5 shot levels |

### 7.3 Compatibility

- All STATE_001 numbers are still valid as "no augmentation" results.
- Runs with `train_augment: true` from this state on are **not** comparable to earlier runs.

## 8. Next Steps

1. Repeat λ ∈ {0, 3} **with augmentation**, 3 seeds, all on Kaggle. This is the decisive check.
2. Add a "mask the background at train time" baseline using the same SAM masks.
3. Add a second and third fine-grained dataset (CUB-200, FGVC-Aircraft).
4. Literature search on attention/CAM supervision and on the class-shared degeneracy.
5. Try λ ∈ {5, 10} at 2–8 shots, and 25 epochs for λ>0.
