# STATE_001 — اصلاح مشکلات تحلیل اجراها

| | |
|---|---|
| تاریخ ثبت | 2026-10-21 |
| Branch | `lab-work-imported` |
| Commit | `a1ac006` — *fix: CAM sign flip, warm-up-aware selection, signed metrics, augmentation* |
| State قبلی | [STATE_000](STATE_000_baseline_snapshot.md) (`5fa3e1d`) |
| برگشت به این state | `git checkout a1ac006` |
| مقایسهٔ کد با state قبلی | `git diff 5fa3e1d a1ac006` |

---

## ۱. هدف پروژه

بدون تغییر نسبت به STATE_000:

```text
loss = CE(GAP(cam), y) + λ_cam · cam_loss(cam, SAM pseudo-mask)
```

## ۲. اجزای کد

| بخش | فایل | وضعیت در این state |
|---|---|---|
| مدل | [src/model_with_adapter_bit.py](../src/model_with_adapter_bit.py) | BiT-M R50x1 + Adapter. **گزینهٔ جدید `center_cams`:** `cam ← cam − mean_k(cam)`. احتمال‌های softmax دقیقاً همان قبلی‌اند. |
| lossها | [src/losses.py](../src/losses.py) | همان ۵ loss قبلی. `allclass_squared_containment` حالا `topk` می‌گیرد (کلاس درست به‌همراه k رقیب قوی) و در fp32 محاسبه می‌شود. |
| آموزش | [src/train.py](../src/train.py) | **یک monitor** هم best checkpoint را انتخاب می‌کند و هم early stopping را. فقط epochهای بعد از warm-up قابل انتخاب‌اند و در patience شمرده می‌شوند. **تست روی best checkpoint** اجرا می‌شود. |
| داده | [src/data.py](../src/data.py) | **augmentation مشترک تصویر و ماسک** فقط روی train: `RandomResizedCrop(scale=0.7–1.0)` و `hflip`، از طریق `AugmentedSubset`. val و test بدون تغییر. |
| متریک تست | [src/cam_debug.py](../src/cam_debug.py) | متریک‌های جدید: `signed_inside_mean`، `signed_outside_mean`، `abs_peak_negative` و `cam_all_negative`. برای CAM تمام‌منفی، `containment` حالا تعریف‌نشده (`None`) است، نه ۰. |
| نرمال‌سازی | [src/normalization.py](../src/normalization.py) | بدون تغییر |

## ۳. Config فعال (`configs/config.yaml`)

| کلید | مقدار |
|---|---|
| data | Stanford Cars (196 کلاس) |
| model_type | `adapter_bit`، gamma=8، freeze_backbone=true، **center_cams=true** |
| optimizer | lr=3e-4، wd=1e-4، cosine، epochs=50، batch=8، AMP |
| cam_loss_type | **`squared_containment`** (`cam_allclass_topk: 5` فقط در صورت استفاده از allclass) |
| lambda_cam | 0.5 (start 0، **warm-up 3 epoch، power 1.0**) |
| early stopping | **monitor=`val_cls_loss`**، patience=8 (بعد از warm-up) |
| augmentation | **train_augment=true**، crop scale [0.7, 1.0]، hflip |
| runs_dir | **`runs/bitm_r50x1_adapter_g8_16shot_state001`** (جدا از اجراهای قبلی) |

## ۴. آخرین نتایج

**هنوز اجرای آموزشی کاملی با این state انجام نشده است.** نتایج STATE_000 با کد قدیمی به دست آمده‌اند و مرجع مقایسه‌اند.

تنها اجرای موجود یک **smoke test دو epochی** روی Cars است. این اجرا فقط سالم بودن مسیر کامل را نشان می‌دهد و نتیجهٔ علمی نیست:

| متریک (test، بعد از ۲ epoch) | مقدار | بدترین مقدار در STATE_000 |
|---|---|---|
| test acc | ~0.31 (فقط ۲ epoch) | — |
| pointing_hit | 0.83 | 0.08 |
| signed in / out | +11.8 / +3.6 | −37.7 / −8.0 |
| cam_all_negative | 0.000 | 0.70 |

## ۵. مشکلات شناخته‌شده

| # | مشکل | وضعیت |
|---|---|---|
| P1 | Sign flip ناشی از مؤلفهٔ مشترک بین کلاس‌ها | ✅ با `center_cams` اصلاح شد؛ باید در اجرای کامل تأیید شود |
| P2 | early stopping روی `val_loss` شامل λ·cam | ✅ اصلاح شد |
| P3 | سه معیار مختلف برای انتخاب و تست | ✅ اصلاح شد |
| P4 | `containment` جعلی برای CAM تمام‌منفی | ✅ اصلاح شد |
| P5 | اجراهای قدیمی با نرمال‌سازی ImageNet | ⚠️ تاریخی. باید با این state دوباره اجرا شوند |
| P6 | allclass روی کلاس‌های غایب | ✅ گزینهٔ `cam_allclass_topk` اضافه شد؛ loss پیش‌فرض به `squared` تغییر کرد |
| P7 | warm-up طولانی | ✅ ۳ epoch و خطی |
| P8 | augmentation وجود نداشت | ✅ اضافه شد |
| P9 | `test_cam_softmax_loss.py` اجرا نمی‌شد | ✅ اصلاح شد |
| N1 | **جدید:** checkpointهای STATE_000 بدون `center_cams` آموزش دیده‌اند. اگر با config فعلی بارگذاری شوند، CAMها مرکزشده نمایش داده می‌شوند ولی logitها و دقت همان می‌ماند. | ℹ️ برای بازتولید دقیق CAMهای قدیمی `center_cams: false` بگذارید |
| N2 | **جدید:** Cars baseline با λ=0 و چند seed هنوز وجود ندارد | ⏳ کار بعدی |

## ۶. وضعیت تست‌ها

| تست | نتیجه |
|---|---|
| `python tests/test_normalization.py` | ✅ ۱۳ از ۱۳ |
| `python tests/test_cam_softmax_loss.py` | ✅ ۱۹ از ۱۹ (قبلاً ImportError می‌داد) |
| `python tests/test_state001_fixes.py` | ✅ ۲۲ از ۲۲ (جدید) |
| smoke run دو epochی روی Cars (`main.py`) | ✅ مسیر کامل اجرا شد: آموزش، انتخاب best، cam_debug و report |

## ۷. تفاوت با STATE_000

### ۷.۱ تغییرات کد

`git diff --stat 5fa3e1d a1ac006`: ۱۱ فایل، ‎+419 / −72.

| فایل | STATE_000 | STATE_001 |
|---|---|---|
| `src/model_with_adapter_bit.py` | `cams = cam_conv(features)` | به‌اضافهٔ `cams -= cams.mean(dim=1)` در صورت فعال بودن `center_cams` |
| `src/model_factory.py` | — | پاس دادن `center_cams` به BiT |
| `src/losses.py` | allclass روی همهٔ K کلاس، با dtype ورودی | allclass با `topk` اختیاری و محاسبه در fp32 |
| `src/train.py` | best بر اساس `val_acc`، early stop بر اساس monitor، patience از epoch 1، تست روی وزن آخر | یک monitor برای هر دو، انتخاب و patience فقط بعد از warm-up، تست روی best، و خروجی جدید `selected_epoch` |
| `src/cam_debug.py` | ۱۰ متریک، همه مبتنی بر relu یا min-max | ۱۴ متریک؛ ۴ متریک علامت‌دار اضافه شد و `containment=None` برای CAM تمام‌منفی |
| `src/data.py` | فقط Resize | به‌اضافهٔ `load(idx, augment)`، `_load_augmented` و `AugmentedSubset` برای train |
| `main.py`, `sweep.py` | — | پاس دادن `train_augment`، `augment_crop_scale` و `augment_hflip` |
| `tests/` | ۱ تست سالم و ۱ تست خراب | ۳ تست سالم |

### ۷.۲ تغییرات config

| کلید | STATE_000 | STATE_001 |
|---|---|---|
| `center_cams` | — (false) | **true** |
| `early_stopping_monitor` | `val_loss` | **`val_cls_loss`** |
| `lambda_cam_warmup_epochs` | 10 | **3** |
| `lambda_cam_warmup_power` | 2.0 | **1.0** |
| `cam_loss_type` | `allclass_squared_containment` | **`squared_containment`** |
| `cam_allclass_topk` | — (همهٔ کلاس‌ها) | **5** |
| `train_augment` | — (false) | **true** |
| `augment_crop_scale` / `augment_hflip` | — | **[0.7, 1.0] / true** |
| `runs_dir` | `…_cam005` | **`…_state001`** |

### ۷.۳ تغییر رفتار در آموزش

| رفتار | STATE_000 | STATE_001 |
|---|---|---|
| زمان توقف اجراهای λ>0 | همیشه حدود epoch 11 تا 16، درست بعد از warm-up | تنها وقتی که `val_cls_loss` در ۸ epoch **بعد از** warm-up بهتر نشود |
| مدلی که تست می‌شود | آخرین epoch | best (با monitor، فقط از epochهای بعد از warm-up) |
| CAM کلاس درست روی شیء (Cars) | تا ‎−37.7 | باید مثبت باشد؛ در smoke run برابر +11.8 است |
| containment برای CAM تمام‌منفی | 0 (جعلی) | `None` |

### ۷.۴ سازگاری با اجراهای قبلی

- **عدد اجراهای STATE_000 با STATE_001 مستقیماً قابل مقایسه نیست.** نرمال‌سازی، augmentation، warm-up، معیار انتخاب و مرکزسازی همه عوض شده‌اند.
- checkpointهای قدیمی بدون تغییر بارگذاری می‌شوند، چون state_dict تغییری نکرده است.

## ۸. قدم بعدی

1. Baseline با λ=0 روی Cars و Pets، با ۳ seed.
2. `squared_containment` و `softmax_containment` با λ ∈ {0.1, 0.3, 1.0} و ۳ seed.
3. ثبت نتایج در **STATE_002**، با این معیارها: `signed_inside_mean > 0`، `cam_all_negative < 5%`، pointing بیشتر از 0.6 و افت دقت کمتر از ۱ واحد درصد نسبت به baseline.
