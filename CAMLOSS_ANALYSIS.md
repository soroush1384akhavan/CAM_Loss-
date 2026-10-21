# Root-Cause Analysis: Why CAMLoss Does Not Improve Accuracy

**Project:** `D:\CVLab\mainProj` — Stanford Cars (196 classes), BiT-M R50x1 + ConvAdapter, CAM supervision from SAM pseudo-masks
**Analysis date:** 2026-09-05
**Runs compared:** `runs/bitm_r50x1_adapter_g8_16shot_cam005/`

---

## Contents

- [0. Verdict](#0-verdict)
- [1. Direct CAM-vs-baseline comparison](#1-direct-cam-vs-baseline-comparison)
- [2. Root cause #1 — the gradient cannot move any logit](#2-root-cause-1--the-gradient-cannot-move-any-logit)
- [3. Root cause #2 — the softmax temperature confines the gradient to ~3 cells](#3-root-cause-2--the-softmax-temperature-confines-the-gradient-to-3-cells)
- [4. Root cause #3 — the warm-up fires after learning is over](#4-root-cause-3--the-warm-up-fires-after-learning-is-over)
- [5. Root cause #4 — full interpolation and zero augmentation](#5-root-cause-4--full-interpolation-and-zero-augmentation)
- [6. Root cause #5 — architectural limits](#6-root-cause-5--architectural-limits)
- [7. Root cause #6 — a real bug in the other two variants](#7-root-cause-6--a-real-bug-in-the-other-two-variants)
- [8. Evaluation-methodology defects](#8-evaluation-methodology-defects)
- [9. Ruled out](#9-ruled-out)
- [10. What CAMLoss actually achieved](#10-what-camloss-actually-achieved)
- [11. Prioritised recommendations](#11-prioritised-recommendations)
- [12. Summary](#12-summary)
- [Appendix: evidence sources](#appendix-evidence-sources)

---

## 0. Verdict

CAMLoss in this project **is not buggy** — it works, and it optimises exactly what it was defined
to optimise (`pointing_hit` goes from **5.7% to 57.5%**, a 10× gain). That is also precisely why
it does nothing for accuracy:

> **The objective as written lives in the null space of the GAP operator, so to first order it
> has zero effect on any logit.**

Every other problem — warm-up scheduling, softmax temperature, the ~2 pp noise floor — **masks**
this zero effect rather than causing it. The one genuine implementation bug is in
`containment_pointing`, which is homogeneous of degree 1 and is therefore the only variant that
actually damages accuracy.

Causes ranked by importance:

| # | Root cause | Kind | Key evidence |
|---|---|---|---|
| 1 | Gradient is mean-free ⇒ zero effect on logits | Structural (by design) | `Σ ∂L/∂z = −2.7e−08`; grad on the other 195 maps `= 0.000e+00` |
| 2 | `T=1` ⇒ gradient lands on 2.8 of 49 cells | Hyperparameter | `entropy_norm = 0.267`, `peak_prob = 0.649` |
| 3 | Warm-up runs to epoch 10; model saturates at epoch 7 | Scheduling | ratio ≤ 0.063 through epoch 7, `train_acc = 0.999` |
| 4 | 3,136 images, zero augmentation | Data | train 1.000 / val 0.48 |
| 5 | All-negative CAMs + a 7×7 grid with ~6 DOF | Architecture | `cam_all_negative = 0.954` |
| 6 | `containment_pointing` is degree-1 homogeneous | **Bug** | scaling: 0.221 / 0.883 / 3.532 |
| 7 | 2 pp noise floor > the 0.8 pp effect | Methodology | two identical baselines: p = 0.034 |

---

## 1. Direct CAM-vs-baseline comparison

### 1.1 The controlled pair

`diff` on the configs confirms the CAM run and BASE-A differ in **exactly one line**
(`lambda_cam: 0.05` vs `0.0`). The other two baselines are effectively identical as well
(BASE-A → BASE-B differs only by `early_stopping_patience: 12 → 8`; BASE-B → BASE-C only by two
unused `attack_normalize_*` keys).

| Run | λ_cam | Epochs | Best val acc | Test acc (1,618) | IoU (true CAM) | pointing_hit |
|---|---|---|---|---|---|---|
| `2026-08-29_10-58-30` **CAM** | 0.05 | 20 | 0.4852 | **0.4815** | **0.268** | **0.575** |
| `2026-08-29_11-56-35` **BASE-A** (exact twin) | 0 | 20 | 0.4830 | 0.4895 | 0.154 | 0.057 |
| `2026-08-29_15-24-19` **BASE-B** | 0 | 16 | 0.5052 | 0.5068 | 0.154 | 0.059 |
| `2026-08-30_22-50-35` **BASE-C** | 0 | 17 | 0.4948 | 0.5093 | 0.156 | 0.063 |

### 1.2 Exact paired McNemar (identical test split)

```text
CAM    vs BASE-A :  Δ = -0.0080   only-CAM 90 / only-BASE 103   p = 0.388   ← the CAM effect
CAM    vs BASE-B :  Δ = -0.0253                                 p = 0.0067
CAM    vs BASE-C :  Δ = -0.0278                                 p = 0.0031

BASE-A vs BASE-B :  Δ = -0.0173                                 p = 0.060   ← two identical runs!
BASE-A vs BASE-C :  Δ = -0.0198                                 p = 0.034   ← two identical runs!
BASE-B vs BASE-C :  Δ = -0.0025                                 p = 0.83
```

### 1.3 Consequence: the experiment has no statistical power

The last two lines above are the single most important number in this analysis:

> **A paired significance test declares a "significant difference" (p = 0.034) between two runs
> that differ in nothing but nondeterminism.**

So the dominant variance component is **run-to-run**, not sample-to-sample. The noise floor of
this setup is roughly **2 percentage points**, while the effect under investigation is
**0.8 pp**. With one CAM run against three baselines and no seed replication, this experiment
could never have resolved an effect of that size — in either direction.

---

## 2. Root cause #1 — the gradient cannot move any logit

This is the primary cause; everything else is secondary.

### 2.1 Analytical argument

`CAMSoftmaxContainmentLoss` in [`src/losses.py:174`](src/losses.py):

```python
p = F.softmax(target_cam.reshape(b, -1) / self.temperature, dim=1)
containment = (p * outside).sum(dim=1)      # L = Σ_j p_j · o_j
```

Derivative w.r.t. each cell:

```
∂L/∂z_i = (p_i / T) · (o_i − L)
```

Summed over all 49 cells:

```
Σ_i ∂L/∂z_i = (1/T) · (Σ_i p_i o_i − L · Σ_i p_i) = (1/T) · (L − L) = 0
```

And on the other side, [`src/model_with_adapter_bit.py:133`](src/model_with_adapter_bit.py):

```python
def forward(self, x):
    features = self.backbone.forward_features(x)
    cams = self.cam_conv(features)
    return self.gap(cams).flatten(1), cams     # logit_k = mean_j(cam_k,j)
```

So `logit_k` is **exactly the spatial mean** of the CAM, and the CAMLoss gradient is
**mean-free**. The loss is therefore a **pure spatial-redistribution operator**: it preserves the
sum of the map (= the class score) and only changes its shape.

### 2.2 Numerical verification

```text
grad norm on the true-class CAM     : 0.0517
grad norm on the other 195 maps     : 0.000e+00        ← no margin can move
Σ spatial gradient, per sample      : -2.65e-08
                                      -2.83e-08
                                      -3.66e-08
                                      -1.43e-09        ← zero to float precision
```

Two independent consequences, both fatal:

1. **Zero effect on the logit level** (mean-freeness).
2. **Zero effect on the margin**: the loss touches only the ground-truth CAM, and the other 195
   maps receive exactly zero gradient — while CE depends on logit **differences**, not on the
   absolute level of the true-class map.

### 2.3 Corroboration from the project's own diagnostics

`probe_cam_head` recorded an **identical trajectory** in both runs:

| Metric | CAM λ=0.05 | BASE-A λ=0 |
|---|---|---|
| `head_weight_norm` (ep1 → ep20) | 11.305 → **18.877** | 11.305 → **18.931** |
| `head_bias_mean` | −0.004 (flat) | −0.004 (flat) |
| `val_logit_margin` | ≈ −0.56 | ≈ −0.44 |

CAMLoss does not move the head's scale and does not move the margin; it only changes the spatial
pattern. Likewise `val_logit_target` rises from −20.1 to −15.3 (the absolute level lifts) while
`val_logit_margin` stays essentially flat — i.e. the quantity that determines accuracy did not
move.

### 2.4 There is no gradient conflict; there is orthogonality

`grad_probe` in [`src/reports/grad_probe.py`](src/reports/grad_probe.py) backprops the two terms
separately. `cosine_ce_cam` for the `all` group stays between **+0.03 and +0.36** across all 20
epochs of both runs and **never goes meaningfully negative** (only the `adapters` group dips to a
negligible −0.05 … −0.10 a few times).

> **CAMLoss does not fight CE. It is orthogonal to it — and that orthogonality is intrinsic to the
> design, not incidental.** This is why the softmax variant neither helps nor seriously hurts.

---

## 3. Root cause #2 — the softmax temperature confines the gradient to ~3 cells

Because `∂L/∂z_j ∝ p_j`, **the width of `p` determines how many cells actually receive gradient.**

The real CAM scale in these runs is large: `cam_min ≈ −34 … −52`, `cam_max ≈ −0.4 … −3.3`
(std ≈ 8). With `cam_softmax_temperature: 1.0` the softmax is effectively one-hot. From
`cam_metrics` at epoch 20:

| Metric | CAM λ=0.05 | BASE-A | ep1 (both) |
|---|---|---|---|
| `val_softmax_peak_prob` | **0.649** | 0.441 | ~0.10 |
| `val_softmax_entropy_norm` | **0.267** | 0.421 | 0.729 |
| Effective cells = `49^entropy_norm` | **2.8 of 49** | 5.1 | 17 |

So by the end of training only **6% of the map** receives meaningful gradient. CAMLoss is in
practice a **peak-mover**, not a distribution-shaper:

- it relocates the argmax onto the car → `pointing_hit` 0.015 → **0.552** (val)
- but ~46 cells stay untouched → the mean of the map (= the logit) does not change

Moving the argmax of a 49-cell map has no effect on its mean. This is exactly the mechanism the
project's own `README.md` warned about:

> *"low softmax entropy starves other cells of gradient since the gradient scales with `p_j`"*

…but `T` was never tuned and stayed at its default of 1. `T` should be on the scale of the CAM's
own std, i.e. somewhere around **5–10**.

---

## 4. Root cause #3 — the warm-up fires after learning is over

Config: `lambda_cam: 0.05`, `lambda_cam_warmup_epochs: 10`, `lambda_cam_warmup_power: 2.0`
⇒ λ(e) = 0.05 · (e/10)²

From `grad_probe` (CAM run):

| Epoch | λ | train_acc | train CE | ‖g_ce‖ | **λ‖g_cam‖ / ‖g_ce‖** |
|---|---|---|---|---|---|
| 1 | 0.0005 | 0.085 | 5.02 | 27.6 | **1.4e-05** |
| 3 | 0.0045 | 0.741 | — | 8.9 | 3.3e-03 |
| 5 | 0.0125 | **0.974** | 0.18 | 3.58 | 0.023 |
| 7 | 0.0245 | **0.999** | 0.012 | 1.42 | 0.063 |
| 9 | 0.0405 | 0.998 | 0.006 | 0.62 | 0.70 |
| 18 | 0.0500 | 0.999 | 0.0017 | 0.13 | **0.82** |

Three observations:

1. The model reaches 99.9% on the 3,136 training images by **epoch 7**.
2. **Through the entire real learning phase (epochs 1–7) CAMLoss contributes under 6.3% of the
   gradient** — effectively absent. With `power=2.0` the warm-up curve is convex and pushes 90% of
   its weight into the second half.
3. The ratio only grows from epoch 9 — **and not because λ grew, but because the denominator
   collapsed** (‖g_ce‖ from 27.6 to 0.13). That apparent "influence" is an artefact of the CE
   gradient vanishing.

> CAMLoss becomes dominant precisely in the regime where there is nothing left to learn. From epoch
> 7 on it merely selects among parameter settings that **all** achieve train CE ≈ 0.

---

## 5. Root cause #4 — full interpolation and zero augmentation

- `train_shots_per_class: 16` × 196 classes = **3,136 training images**
- **2,716,268** trainable parameters out of 26,216,620 (backbone frozen)
- The transform in [`src/data.py:124`](src/data.py):

```python
self.image_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
])
```

> **There is no augmentation at all** — no flip, no random crop, no colour jitter, no scale jitter.

Result: `train_acc` = 0.974 at epoch 5 and **1.000** at epoch 16, against `val_acc` ≈ 0.48 — a
**~52 pp** generalisation gap.

**Why this neutralises CAMLoss:** once the model has memorised the training set, the decision
function is pinned on the training points. An auxiliary regulariser that only changes **where**
evidence is drawn from — and does so in a mean-free way — is performing a reparametrisation inside
the same flat minimum. There is no mechanism by which that reparametrisation would generalise
better.

The binding constraint in this experiment is **overfitting**, and CAMLoss does not touch it.

---

## 6. Root cause #5 — architectural limits

### 6.1 The all-negative CAM regime

From `test_localization_summary.json` (baseline):

```text
cam_max_mean          = -3.31     ← even the maximum of the true-class map is negative
positive_cell_frac    = 0.0026    ← 99.7% of cells are ≤ 0
val_cam_all_negative  = 0.954     ← in 95% of val samples the whole true-class CAM is negative
val_logit_target      = -20.1
```

Because CE only constrains logit **differences**, the entire logit field can sit at −20 at no cost.
With 196 classes and a GAP head this is the default state. Even in the CAM run,
`cam_all_negative` only drops from 0.954 to 0.601.

Direct consequence — all three losses evaluated numerically on a real all-negative CAM:

| Loss | Value | Grad norm | Status |
|---|---|---|---|
| `containment` (ratio) | **0.000000** | **0.000e+00** | Mathematically disarmed — relu gates everything |
| `containment_pointing` | 5.43 | active | Active but destructive (§7) |
| `softmax_containment` | 0.699 | 4.6e-02 | **The only healthy variant** |

This is why **only** the softmax variant managed to move localisation at all. The project's own
`README.md` had already predicted this failure mode ("an all-negative CAM scores a perfect 0 with
zero gradient").

### 6.2 The 7×7 grid carries only ~6 independent degrees of freedom

From [`erf_suite/erf_probe_summary.json`](erf_suite/erf_probe_summary.json):

```text
mean_sigma_px                 = 45.5     (20% of the 224 px width)
cam_lag1_autocorrelation      = 0.72
effective_independent_samples = 6.06     out of 49 cells
```

And from `mask_audit`: `pure_object_cell_frac = 0.153` → only **7.5 of 49 cells** are fully object
(49% fully background, 35% mixed).

> The CAM map has roughly **6 independent degrees of freedom** in practice. You cannot encode
> useful class-discriminative information for a 196-way fine-grained problem in *where* 6 numbers
> sit.

### 6.3 The frozen backbone

With `freeze_backbone: true`, CAMLoss cannot change the features — only the ConvAdapters
(γ=8, k=3), the GroupNorm affine parameters, and one `Conv2d(2048, 196, 1)`. That is
**2.72 M of 26.2 M parameters (10.4%)**, and the representation the CAM is built on is fixed.

The adapter initialisation itself is sound: the class actually instantiated is `ConvAdapter`
([`src/conv_adapter.py:91`](src/conv_adapter.py)), whose output gate
`se = nn.Parameter(torch.zeros(...))` is zero-init, so the adapter contributes exactly zero at
step 0 and the pretrained residual path is preserved. (`ConvAdapterDesign1`, which has
`adapt_scale=1.0` and no zero-init, is dead code here.)

---

## 7. Root cause #6 — a real bug in the other two variants

This is the only place where a genuine implementation defect exists.

### 7.1 Measured scaling behaviour

I scaled the whole CAM by α and measured each loss:

| Loss | α = 0.25 | α = 1 | α = 4 | Behaviour |
|---|---|---|---|---|
| `softmax_containment` | 0.690 | 0.726 | 0.659 | approximately level-invariant ✔ |
| `containment` (ratio) | 0.7006 | 0.7006 | 0.7006 | **exactly** scale-invariant ✔ |
| `containment_pointing` | **0.221** | **0.883** | **3.532** | **homogeneous of degree 1** ✘ |

### 7.2 Where the bug is

In [`src/losses.py:102`](src/losses.py):

```python
energy_outside = (target_cam ** 2 * outside).sum(dim=(1, 2, 3))   # degree 2
energy_total   = target_cam.abs().sum(dim=(1, 2, 3)) + self.eps   # degree 1
containment    = energy_outside / energy_total                    # ⇒ degree 1
```

A quadratic numerator over a linear denominator gives a ratio that is homogeneous of degree 1.
That loss is minimised by **shrinking the whole CAM toward zero** — and shrinking the true-class
CAM directly shrinks the true-class logit, which directly increases CE.

> **This is a genuine gradient conflict, not orthogonality.**

Side note: the `pointing_penalty` block (using `max_inside`/`max_outside`) in this same class is
**commented out**, so `inside_weight` and `margin` are dead parameters and the class name is
misleading.

### 7.3 Confirmation in the results (ResNet18 runs)

| Loss | λ | Best val acc | Δ vs baseline |
|---|---|---|---|
| baseline | 0 | 0.189 – 0.201 | — |
| `softmax_containment` | 15 | 0.197 – 0.205 | ≈ 0 |
| `containment` (ratio) | 15 | **0.149** | **−4 to −5 pp** |
| `containment_pointing` | 5 | 0.164 – 0.165 | −3 pp |
| `containment_pointing` | 15 | **0.149** | **−4 to −5 pp** |

This separation is the strongest empirical support for the analysis in §2: the mean-free variant
tracks the baseline, while the two variants that do shift the CAM level cost 3–5 pp of accuracy.

---

## 8. Evaluation-methodology defects

### 8.1 Nondeterminism ⇒ a 2 pp noise floor

`set_seed` in [`src/utils.py:8`](src/utils.py) only sets the base seeds:

```python
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

Missing: an explicit `generator` for the DataLoader, and
`torch.use_deterministic_algorithms`. The runs also used `cudnn.benchmark = True`
([`main.py:38`](main.py)) and AMP (`model_info.json`: `device=cuda`, `use_amp=true`). All three
are nondeterminism sources.

Evidence: BASE-A / BASE-B / BASE-C, with the same seed, diverge from **epoch 1**
(`train_acc` = 0.085 / 0.093 / 0.091).

**Hidden coupling:** `run_grad_probe` iterates `train_loader` (which has `shuffle=True`) every
epoch and consumes the global RNG. So any change to the diagnostics configuration changes the
shuffle order of every subsequent epoch — the diagnostics are coupled to the training outcome.

### 8.2 The model being tested is not the model that was selected

In [`src/train.py:594`](src/train.py), `run_cam_debug` is handed the in-memory `model` object —
i.e. the **last-epoch weights**. `best_*.pth` is never reloaded for test evaluation.

The CAM run's best val was epoch 19 (0.4852) but epoch 20 (0.472) was tested. That is a systematic
penalty the pipeline applied to the CAM run.

### 8.3 `val_loss` is not comparable across λ values

In [`src/evaluate.py:37`](src/evaluate.py):

```python
loss = loss_cls + lambda_cam * loss_cam     # ← the early-stopping criterion
```

with `early_stopping_monitor: val_loss`. In the CAM run `cam_loss` falls from 0.94 to 0.20
⇒ up to **0.037** of `val_loss` reduction that has nothing to do with classification but does move
the stopping point. The early-stopping monitor should be `loss_cls` only.

### 8.4 Other issues

- **The headline metric is `max` over 20 epochs of val accuracy** — on the very split used for
  model selection. Optimistic bias plus `max` noise, reported without a CI.
- **Accuracy is a mean of per-batch means** ([`src/train.py:104`](src/train.py)): with
  `batch_size: 8`, the final partial batch is counted at full weight.
- **The splits are not stratified** ([`src/data.py:278`](src/data.py) uses `random_split`): with
  196 classes and 1,618 test samples (~8.3 per class) the per-class distribution is uneven. It is
  shared across runs so it adds no bias, but it does add variance.
- **`_compute_cam_mask_iou`** in [`src/cam_debug.py:21`](src/cam_debug.py) min-max normalises the
  CAM before thresholding, so it manufactures a "hot" region even from an entirely negative map.
  The baseline's IoU of 0.154 is non-zero for that reason and should not be over-trusted; the
  `containment` and `pointing_hit` metrics are more meaningful.

---

## 9. Ruled out

These can be removed from the suspect list, with evidence.

### 9.1 Mask quality ✔ healthy

From [`diagnostics/mask_audit.json`](runs/bitm_r50x1_adapter_g8_16shot_cam005/2026-08-29_10-58-30/diagnostics/mask_audit.json)
— every finding is OK/INFO:

```text
usable mask rate              : 89.3%   (7134 ok / 856 excluded_by_manifest of 7990)
polarity                      : center 0.683  ≫  border 0.074   → mask=1 means object ✔
raw_pixel_max                 : 255            → no 0/1-vs-0/255 scale problem ✔
n_mask_unreadable             : 0
n_mask_empty_after_threshold  : 0
coverage on the 7×7 grid      : 0.312
uniform-CAM reference         : 0.688
perfect-CAM floor             : 0.082
```

### 9.2 The loss really was optimised ✔

`val_softmax_containment` fell from 0.966 to **0.478**, where:

```text
flat (uniform) CAM = 0.696
perfect CAM (floor) = 0.082
```

That is roughly **30% of the way from flat to perfect**. The optimiser did its job; the problem is
in the definition of the objective, not in minimising it. (In the baseline the same metric stays at
0.966 → 0.948, i.e. **worse than flat**.)

### 9.3 Class imbalance ✘ not a factor

Stanford Cars is roughly balanced, and λ acts identically on all classes. The only dataset-side
effect is `shots_per_class: 16`, covered in §5.

### 9.4 Preprocessing mismatch ✘ not a factor

`normalize_mean/std: [0.5, 0.5, 0.5]` is correct for BiT (BiT was pretrained with exactly this
normalisation, not with ImageNet statistics). Masks are resized with NEAREST, binarised at `> 0.5`,
then area-downsampled to the 7×7 grid — which is the right choice for soft coverage.

---

## 10. What CAMLoss actually achieved

This section matters, because the experiment **did not fail** — it just measured something other
than its stated target.

### 10.1 Test results (masked samples)

| Metric | BASE-A | CAM λ=0.05 | Change |
|---|---|---|---|
| `pointing_hit` | 0.057 | **0.575** | **×10.0** |
| `iou_true_label` | 0.154 | **0.268** | **+74%** |
| `mean_pos_inside` | 0.0033 | **0.363** | ×110 |
| `positive_cell_frac` | 0.0026 | 0.0347 | ×13 |
| `cam_max` | −3.31 | −0.36 | +2.95 |
| `val_cam_all_negative` | 0.954 | 0.601 | −37% |
| **accuracy** | 0.4895 | 0.4815 | **−0.8 pp (p = 0.39)** |

### 10.2 The publishable finding

Mask coverage on the CAM grid is **0.312**, so chance level for `pointing_hit` is also 0.312.
The baseline scores **0.057**:

> A model at 49% accuracy places its evidence peak on the car **5.5× less often than chance** —
> i.e. the baseline is strongly **anti-localised**.

CAMLoss fixes this completely (0.057 → 0.575) with no meaningful accuracy cost. That is clean
empirical support for:

> **In a GAP-CAM architecture, faithfulness and accuracy are decoupled.**

This is not a negative result; it is a positive finding about the nature of CAM. If the research
goal is interpretability or localisation, the experiment succeeded. If the goal is accuracy, the
loss design has to change (§11).

---

## 11. Prioritised recommendations

### A) If the goal is accuracy — the loss must change, not λ

| # | Action | Why |
|---|---|---|
| A1 | **Masked-GAP head**: `logit_k = Σ_j m_j·cam_{k,j} / Σ_j m_j`, CE on that | Not mean-free; acts directly on the score and the margin |
| A2 | Auxiliary term `CE(GAP(cam ⊙ mask), y)` alongside the main CE | Forces the model to draw its evidence from inside the mask |
| A3 | Apply the loss to **all** classes, not just the true one (e.g. background penalty on the top-k competitors) | Right now the other 195 maps get zero gradient ⇒ the margin cannot move |
| A4 | Add **augmentation**: `RandomResizedCrop(224, scale=(0.7,1.0))` + `RandomHorizontalFlip` | With `train_acc = 1.000` by epoch 16, this is the only large lever left |
| A5 | `warmup_epochs: 10 → 3`, or `warmup_power: 2.0 → 1.0` | λ must be live during the learning phase (epochs 1–7), not after it |
| A6 | Unfreeze the last backbone stages, or fine-tune the whole network | BiT uses GroupNorm + weight standardisation specifically so that fine-tuning at small batch sizes works; with only 10.4% of parameters trainable there is no headroom to exploit spatial supervision |

### B) If the goal is localisation (where it already succeeded)

| # | Action | Why |
|---|---|---|
| B1 | `cam_softmax_temperature: 1.0 → 5…10`, tuned against `softmax_entropy_norm` (target 0.5–0.7) | Only 2.8 of 49 cells currently receive gradient |
| B2 | Add a raw-units `relu(cam) · outside` term | Drives background to zero **absolutely**, not relatively — exactly what the README proposed |
| B3 | Supervise at higher resolution (e.g. stride 16 ⇒ a 14×14 grid) | The 7×7 grid has only ~6 independent DOF |
| B4 | Remove or fix `containment_pointing` (make the denominator degree 2 as well) | It is degree-1 homogeneous and crushes the logit |

### C) Protocol fixes — without these no result is defensible

| # | Action | Why |
|---|---|---|
| C1 | **At least 3 seeds per arm**, report mean ± std | The noise floor is 2 pp and the effect under study is 0.8 pp |
| C2 | `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark = False`, explicit DataLoader `generator` | Same-seed runs diverge from epoch 1 |
| C3 | Evaluate `best_*.pth` on test, not the last-epoch weights | The CAM run was tested at epoch 20 (0.472) instead of epoch 19 (0.485) |
| C4 | Make the early-stopping `val_loss` be `loss_cls` only | It currently includes `λ·cam_loss` and is not comparable across λ |
| C5 | Report test accuracy with a CI and paired McNemar, not `max` over val | The current metric is on the model-selection split and is optimistically biased |
| C6 | Stratify the splits | ~8.3 samples per class under a random split |
| C7 | Give `run_grad_probe` its own loader with a fixed seed | Diagnostics are consuming the training RNG |

---

## 12. Summary

CAMLoss does not improve accuracy for these reasons, in order of importance:

1. **Its gradient is mean-free** (`Σ ∂L/∂z = 0`, confirmed analytically and numerically), and since
   `logit = GAP(cam)`, the term has **no** first-order effect on any logit. It also touches only
   the true-class map, so the margin does not move either.
2. **`T=1` reduced it to a single-cell peak-mover** (2.8 effective cells of 49) — which explains
   why `pointing_hit` went up 10× while the logits stayed put.
3. **A quadratic warm-up out to epoch 10, while the model hit `train_acc = 0.999` at epoch 7**;
   CAMLoss contributed under 6% of the gradient throughout the entire learning phase.
4. **The binding constraint is overfitting** (3,136 images, zero augmentation, train 1.000 /
   val 0.48), and CAMLoss does not address it.
5. **The 2 pp noise floor of the protocol** could never have resolved a 0.8 pp effect — two
   completely identical baselines are declared "different" at p = 0.034.

And the **only real bug** is in `containment_pointing`, which is homogeneous of degree 1, is
minimised by shrinking the CAM, and is therefore the one variant that genuinely costs 3–5 pp of
accuracy.

The positive result should not be overlooked: the baseline is **anti-localised**
(`pointing_hit = 0.057` against a chance level of 0.312) and CAMLoss brings it to 0.575 at no
meaningful accuracy cost — clean evidence that faithfulness and accuracy are decoupled in GAP-CAM.

---

## Appendix: evidence sources

| Topic | Path |
|---|---|
| The three loss definitions | [`src/losses.py`](src/losses.py) — `CAM_LOSS_REGISTRY` |
| `logits = GAP(cams)` | [`src/model_with_adapter_bit.py:133`](src/model_with_adapter_bit.py) |
| Loss composition and λ warm-up | [`src/train.py`](src/train.py) — `train_one_epoch`, `_lambda_cam_for_epoch` |
| `val_loss` includes λ·cam_loss | [`src/evaluate.py:37`](src/evaluate.py) |
| Absence of augmentation | [`src/data.py:124`](src/data.py) |
| Incomplete seeding | [`src/utils.py:8`](src/utils.py) |
| CE-vs-CAM gradient comparison | [`src/reports/grad_probe.py`](src/reports/grad_probe.py) |
| In-training CAM metrics | [`src/reports/cam_stats.py`](src/reports/cam_stats.py) |
| Test dump and overlays | [`src/cam_debug.py`](src/cam_debug.py) |
| Mask health | `runs/.../diagnostics/mask_audit.json` |
| Test localisation metrics | `runs/.../cam_debug/test_localization_summary.json` |
| Per-epoch history | `runs/.../metrics.csv`, `runs/.../diagnostics/*.jsonl` |
| ERF probe | [`erf_suite/erf_probe_summary.json`](erf_suite/erf_probe_summary.json) |
| One-shot sweep (uninformative) | `sweep_results/results.csv` — all at chance (0.006) |













