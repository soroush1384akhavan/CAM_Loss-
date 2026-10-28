# STATE_001 — Run-Analysis Fixes

| | |
|---|---|
| Date | 2026-10-21 |
| Branch | `lab-work-imported` |
| Commit | `a1ac006` — *fix: CAM sign flip, warm-up-aware selection, signed metrics, augmentation* |
| Previous State | [STATE_000](STATE_000_baseline_snapshot.md) (`5fa3e1d`) |
| Restore | `git checkout a1ac006` |
| Code diff | `git diff 5fa3e1d a1ac006` |

---

## 1. Project Goal

Unchanged from STATE_000:

```text
loss = CE(GAP(cam), y) + λ_cam · cam_loss(cam, SAM pseudo-mask)
```

## 2. Code Components

| Component | File | Status in this state |
|---|---|---|
| Model | [src/model_with_adapter_bit.py](../../src/model_with_adapter_bit.py) | BiT-M R50x1 + Adapter. **New `center_cams` option:** after computing CAM, applies `cam ← cam − mean_k(cam)`. Softmax probabilities are unchanged. |
| Losses | [src/losses.py](../../src/losses.py) | Same 5 losses. `allclass_squared_containment` now takes `topk` (correct class + k strongest competitors) and runs in fp32. |
| Training | [src/train.py](../../src/train.py) | **One monitor** controls both best-checkpoint selection and early stopping. Only post-warmup epochs are eligible and counted toward patience. **Test runs on the best checkpoint.** |
| Data | [src/data.py](../../src/data.py) | **Joint image+mask augmentation** on train only: `RandomResizedCrop(scale=0.7–1.0)` and `hflip`, via `AugmentedSubset`. Val and test unchanged. |
| Test metrics | [src/cam_debug.py](../../src/cam_debug.py) | New metrics: `signed_inside_mean`, `signed_outside_mean`, `abs_peak_negative`, `cam_all_negative`. For all-negative CAMs, `containment` is now `None` (not 0). |
| Normalization | [src/normalization.py](../../src/normalization.py) | Unchanged |

## 3. Active Config (`configs/config.yaml`)

| Key | Value |
|---|---|
| data | Stanford Cars (196 classes) |
| model_type | `adapter_bit`, gamma=8, freeze_backbone=true, **center_cams=true** |
| optimizer | lr=3e-4, wd=1e-4, cosine, epochs=50, batch=8, AMP |
| cam_loss_type | **`squared_containment`** (`cam_allclass_topk: 5` only when using allclass) |
| lambda_cam | 0.5 (start 0, **warm-up 3 epochs, power 1.0**) |
| early stopping | **monitor=`val_cls_loss`**, patience=8 (post-warmup) |
| augmentation | **train_augment=true**, crop scale [0.7, 1.0], hflip |
| runs_dir | **`runs/bitm_r50x1_adapter_g8_16shot_state001`** (separate from previous runs) |

## 4. Latest Results

**No full training run has been done with this state yet.** STATE_000 results are the reference baseline.

The only run is a **2-epoch smoke test** on Cars, verifying pipeline health — not a scientific result:

| Metric (test, after 2 epochs) | Value | Worst in STATE_000 |
|---|---|---|
| test acc | ~0.31 (2 epochs only) | — |
| pointing_hit | 0.83 | 0.08 |
| signed in / out | +11.8 / +3.6 | −37.7 / −8.0 |
| cam_all_negative | 0.000 | 0.70 |

## 5. Known Issues

| # | Issue | Status |
|---|---|---|
| P1 | Sign flip from class-shared component | ✅ Fixed with `center_cams`; needs full-run confirmation |
| P2 | Early stopping on `val_loss` included λ·cam | ✅ Fixed |
| P3 | Three separate criteria for selection, test, and early stop | ✅ Fixed |
| P4 | Fake `containment=0` for all-negative CAMs | ✅ Fixed |
| P5 | Old runs trained with ImageNet normalization | ⚠️ Historical; must be rerun with this state |
| P6 | allclass averaged over absent classes | ✅ Added `cam_allclass_topk`; default loss changed to `squared` |
| P7 | Warm-up too long | ✅ 3 epochs, linear |
| P8 | No augmentation | ✅ Added |
| P9 | `test_cam_softmax_loss.py` failed to run | ✅ Fixed |
| N1 | **New:** STATE_000 checkpoints trained without `center_cams`. Loading with current config shows centered CAMs but logits and accuracy are unchanged. | ℹ️ Set `center_cams: false` to reproduce old CAMs exactly |
| N2 | **New:** Cars baseline with λ=0 and multiple seeds still missing | ⏳ Next task |

## 6. Test Status

| Test | Result |
|---|---|
| `python tests/test_normalization.py` | ✅ 13/13 |
| `python tests/test_cam_softmax_loss.py` | ✅ 19/19 (was ImportError) |
| `python tests/test_state001_fixes.py` | ✅ 22/22 (new) |
| 2-epoch smoke run on Cars (`main.py`) | ✅ Full pipeline ran: training, best-checkpoint selection, cam_debug, report |

## 7. Diff from STATE_000

### 7.1 Code Changes

`git diff --stat 5fa3e1d a1ac006`: 11 files, +419 / −72.

| File | STATE_000 | STATE_001 |
|---|---|---|
| `src/model_with_adapter_bit.py` | `cams = cam_conv(features)` | + `cams -= cams.mean(dim=1)` when `center_cams` is active |
| `src/model_factory.py` | — | passes `center_cams` to BiT |
| `src/losses.py` | allclass over all K classes, input dtype | allclass with optional `topk`, computed in fp32 |
| `src/train.py` | best by `val_acc`, early stop by monitor, patience from epoch 1, test on last weights | one monitor for both; selection and patience only post-warmup; test on best; new `selected_epoch` output |
| `src/cam_debug.py` | 10 metrics, all relu- or min-max-based | 14 metrics; 4 signed metrics added; `containment=None` for all-negative CAMs |
| `src/data.py` | Resize only | + `load(idx, augment)`, `_load_augmented`, `AugmentedSubset` for train |
| `main.py`, `sweep.py` | — | pass `train_augment`, `augment_crop_scale`, `augment_hflip` |
| `tests/` | 1 healthy + 1 broken | 3 healthy |

### 7.2 Config Changes

| Key | STATE_000 | STATE_001 |
|---|---|---|
| `center_cams` | — (false) | **true** |
| `early_stopping_monitor` | `val_loss` | **`val_cls_loss`** |
| `lambda_cam_warmup_epochs` | 10 | **3** |
| `lambda_cam_warmup_power` | 2.0 | **1.0** |
| `cam_loss_type` | `allclass_squared_containment` | **`squared_containment`** |
| `cam_allclass_topk` | — (all classes) | **5** |
| `train_augment` | — (false) | **true** |
| `augment_crop_scale` / `augment_hflip` | — | **[0.7, 1.0] / true** |
| `runs_dir` | `…_cam005` | **`…_state001`** |

### 7.3 Training Behaviour Changes

| Behaviour | STATE_000 | STATE_001 |
|---|---|---|
| When λ>0 runs stop | Always around epoch 11–16, right after warm-up | Only when `val_cls_loss` doesn't improve for 8 epochs **after** warm-up |
| Model tested | Last epoch | Best (by monitor, post-warmup only) |
| Correct-class CAM on object (Cars) | As low as −37.7 | Should be positive; +11.8 in smoke run |
| Containment for all-negative CAM | 0 (fake) | `None` |

### 7.4 Backward Compatibility

- **STATE_000 numbers are not directly comparable to STATE_001.** Normalization, augmentation, warm-up, selection criterion, and CAM centering all changed.
- Old checkpoints load without modification because `state_dict` is unchanged.

## 8. Next Steps

1. λ=0 baseline on Cars and Pets, with 3 seeds.
2. `squared_containment` and `softmax_containment` with λ ∈ {0.1, 0.3, 1.0} and 3 seeds.
3. Log results in **STATE_002** with criteria: `signed_inside_mean > 0`, `cam_all_negative < 5%`, pointing > 0.6, accuracy drop < 1 pp vs baseline.
