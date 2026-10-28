# STATE_000 — Baseline Snapshot (Before Fixes)

| | |
|---|---|
| Date | 2026-10-21 (code corresponds to lab work up to 2026-09-07) |
| Branch | `lab-work-imported` |
| Commit | `5fa3e1d` — *snapshot: state 000 before run-analysis fixes* |
| Previous State | — (first state) |
| Restore | `git checkout 5fa3e1d` |

---

## 1. Project Goal

Train a few-shot classifier (16 samples per class) using the loss:

```text
loss = CE(GAP(cam), y) + λ_cam · cam_loss(cam, SAM pseudo-mask)
```

Goal: the correct-class CAM should activate on the **object itself**, not the background.

## 2. Code Components

| Component | File | Status in this state |
|---|---|---|
| Model | [src/model_with_adapter_bit.py](../../src/model_with_adapter_bit.py) | BiT-M R50x1 frozen + parallel Conv-Adapter on each bottleneck. GroupNorms trainable. 1×1 conv head outputs CAM directly; `logit = GAP(cam)`. **CAM is not centered.** |
| Losses | [src/losses.py](../../src/losses.py) | `containment` (relu-ratio), `squared_containment`, `allclass_squared_containment` (averaged over **all** K classes), `containment_pointing`, `softmax_containment` |
| Training | [src/train.py](../../src/train.py) | Power warm-up for λ. Early stopping on `val_loss` which **includes λ·cam**. Best checkpoint selected by `val_acc`. Test and `cam_debug` run on **last-epoch weights**. |
| Data | [src/data.py](../../src/data.py) | ImageFolder with per-class SAM masks, `random_split`, subsample to 16 samples. **No augmentation.** Normalization read from config (behavior since 2026-09-07). |
| Normalization | [src/normalization.py](../../src/normalization.py) | BiT uses `[0.5, 0.5, 0.5]`, ResNet uses ImageNet stats. Saved in checkpoint. |
| Test metrics | [src/cam_debug.py](../../src/cam_debug.py) | IoU with min-max; `containment` and `mean_pos_*` based on **relu**. No signed metrics. |

## 3. Active Config (`configs/config.yaml`)

| Key | Value |
|---|---|
| data | Stanford Cars (196 classes) |
| model_type | `adapter_bit`, gamma=8, freeze_backbone=true |
| optimizer | lr=3e-4, wd=1e-4, cosine, epochs=50, batch=8, AMP |
| cam_loss_type | `allclass_squared_containment` |
| lambda_cam | 0.5 (start 0, warm-up 10 epochs, power 2.0) |
| early stopping | monitor=`val_loss`, patience=8 |
| normalize | `[0.5, 0.5, 0.5]` |

## 4. Latest Results (`runs/bitm_r50x1_adapter_g8_16shot_cam005`)

| Run | Dataset | Loss | λ | test acc | pointing | avg cam_max |
|---|---|---|---|---|---|---|
| 09-05_20-47 | Pets | — | 0 | **0.895** | 0.78 | 9.3 |
| 09-05_21-19 | Pets | squared | 0.5 | 0.873 | 0.96 | 17.3 |
| 09-06_12-47 | Pets | allclass | 10 | 0.830 | 0.73 | 8.2 |
| 09-07_10-51 ✱ | Cars | softmax | 10 | 0.368 | 0.82 | 0.3 |
| 09-07_11-21 ✱ | Cars | allclass | 10 | 0.413 | 0.35 | 4.8 |
| 09-07_12-24 ✱ | Cars | allclass | 0.5 | 0.483 | **0.08** | **−0.8** |

✱ Only these runs were trained with the correct BiT normalization. No Cars baseline with λ=0 exists in this state.

## 5. Known Issues

Details and evidence in [RUNS_ANALYSIS_FA.md](../../RUNS_ANALYSIS_FA.md).

| # | Issue | Severity |
|---|---|---|
| P1 | **Sign flip:** the class-shared component (costless for CE) goes strongly negative on the object; the squared loss counts this as "energy inside". CAM peaks on background (pointing=0.08). | 🔴 |
| P2 | Early stopping monitors `val_loss` which includes λ·cam. All λ>0 runs stopped right after warm-up. | 🔴 |
| P3 | Three separate criteria: test on last-epoch weights, best by val_acc, early stop by val_loss. | 🟠 |
| P4 | `containment` based on relu returns 0 for all-negative CAMs — a falsely optimal score. | 🟠 |
| P5 | Runs before 09-07_10-51 were trained with ImageNet normalization while config says 0.5. | 🟠 (historical) |
| P6 | `allclass_squared` averages over 195 classes absent from the image. | 🟠 |
| P7 | Warm-up is too long; λ=10 with start=5 effectively starts without warm-up. | 🟡 |
| P8 | No augmentation: train≈1.0 vs val≈0.47 on Cars. | 🟡 |
| P9 | `tests/test_cam_softmax_loss.py` imports old names `CAMMaskLoss` and `CAMSoftmaxMaskLoss` and **fails to run**. | 🟡 |

## 6. Test Status

| Test | Result |
|---|---|
| `python tests/test_normalization.py` | ✅ All 13 tests pass |
| `python tests/test_cam_softmax_loss.py` | ❌ `ImportError: cannot import name 'CAMMaskLoss'` |

## 7. Diff from Previous State

This is the first state.
