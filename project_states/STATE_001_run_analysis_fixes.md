# STATE_001 — رفع مشکلات تحلیل اجراها

| | |
|---|---|
| تاریخ ثبت | ۱۴۰۵-۰۷-۳۰ |
| Branch | `lab-work-imported` |
| Commit | `a1ac006` — *fix: CAM sign flip, warm-up-aware selection, signed metrics, augmentation* |
| State قبلی | [STATE_000](STATE_000_baseline_snapshot.md) (`5fa3e1d`) |
| بازگشت به این state | `git checkout a1ac006` |
| مقایسه کد با state قبلی | `git diff 5fa3e1d a1ac006` |

---

## ۱. هدف پروژه

بدون تغییر نسبت به STATE_000:

```text
loss = CE(GAP(cam), y) + λ_cam · cam_loss(cam, SAM pseudo-mask)
```

## ۲. اجزای کد

| بخش | فایل | وضعیت در این state |
|---|---|---|
| مدل | [src/model_with_adapter_bit.py](../src/model_with_adapter_bit.py) | BiT-M R50x1 + Adapter. **گزینه جدید `center_cams`:** پس از محاسبه CAM، `cam ← cam − mean_k(cam)` اعمال می‌شود. احتمال‌های softmax دست‌نخورده می‌مانند. |
| تابع هزینه | [src/losses.py](../src/losses.py) | همان ۵ loss قبلی. `allclass_squared_containment` حالا `topk` می‌گیرد (کلاس درست + k رقیب قوی) و در fp32 محاسبه می‌شود. |
| آموزش | [src/train.py](../src/train.py) | **یک monitor** هم best checkpoint را انتخاب می‌کند هم early stopping را کنترل می‌کند. فقط epoch‌های بعد از warm-up قابل انتخاب‌اند و در patience حساب می‌شوند. **تست روی best checkpoint** اجرا می‌شود. |
| داده | [src/data.py](../src/data.py) | **augmentation مشترک تصویر و ماسک** فقط روی train: `RandomResizedCrop(scale=0.7–1.0)` و `hflip`، از طریق `AugmentedSubset`. val و test بدون تغییر. |
| متریک‌های تست | [src/cam_debug.py](../src/cam_debug.py) | متریک‌های جدید: `signed_inside_mean`، `signed_outside_mean`، `abs_peak_negative`، `cam_all_negative`. برای CAM تمام‌منفی، `containment` حالا `None` است نه ۰. |
| نرمال‌سازی | [src/normalization.py](../src/normalization.py) | بدون تغییر |

## ۳. تنظیمات فعال (`configs/config.yaml`)

| کلید | مقدار |
|---|---|
| data | Stanford Cars (196 کلاس) |
| model_type | `adapter_bit`، gamma=8، freeze_backbone=true، **center_cams=true** |
| optimizer | lr=3e-4، wd=1e-4، cosine، epochs=50، batch=8، AMP |
| cam_loss_type | **`squared_containment`** (گزینه `cam_allclass_topk: 5` فقط در صورت استفاده از allclass) |
| lambda_cam | 0.5 (شروع از 0، **warm-up 3 epoch، توان 1.0**) |
| early stopping | **monitor=`val_cls_loss`**، patience=8 (بعد از warm-up) |
| augmentation | **train_augment=true**، crop scale [0.7, 1.0]، hflip |
| runs_dir | **`runs/bitm_r50x1_adapter_g8_16shot_state001`** (جدا از اجراهای قبلی) |

## ۴. آخرین نتایج

**هنوز اجرای آموزشی کاملی با این state انجام نشده است.** نتایج STATE_000 مبنای مقایسه‌اند.

تنها اجرای موجود یک **smoke test دو epochی** روی Cars است که فقط سلامت pipeline را تأیید می‌کند، نه یک نتیجه علمی:

| متریک (test، بعد از ۲ epoch) | مقدار | بدترین مقدار در STATE_000 |
|---|---|---|
| test acc | ~0.31 (فقط ۲ epoch) | — |
| pointing_hit | 0.83 | 0.08 |
| signed in / out | +11.8 / +3.6 | −37.7 / −8.0 |
| cam_all_negative | 0.000 | 0.70 |

## ۵. مشکلات شناخته‌شده

| # | مشکل | وضعیت |
|---|---|---|
| P1 | Sign flip ناشی از مؤلفه مشترک بین کلاس‌ها | ✅ با `center_cams` رفع شد؛ باید در اجرای کامل تأیید شود |
| P2 | early stopping روی `val_loss` که شامل λ·cam بود | ✅ رفع شد |
| P3 | سه معیار جداگانه برای انتخاب، تست و توقف | ✅ رفع شد |
| P4 | `containment=0` جعلی برای CAM تمام‌منفی | ✅ رفع شد |
| P5 | اجراهای قدیمی با نرمال‌سازی ImageNet | ⚠️ تاریخی؛ باید با این state دوباره اجرا شوند |
| P6 | allclass روی کلاس‌های غایب در تصویر | ✅ گزینه `cam_allclass_topk` اضافه شد؛ loss پیش‌فرض به `squared` تغییر کرد |
| P7 | warm-up بیش از حد طولانی | ✅ 3 epoch، خطی |
| P8 | نبود augmentation | ✅ اضافه شد |
| P9 | خرابی `test_cam_softmax_loss.py` | ✅ رفع شد |
| N1 | **جدید:** checkpoint‌های STATE_000 بدون `center_cams` آموزش دیده‌اند. بارگذاری با config فعلی، CAM‌ها را مرکزشده نشان می‌دهد ولی logit‌ها و دقت تغییری نمی‌کنند. | ℹ️ برای بازتولید دقیق CAM‌های قدیمی، `center_cams: false` بگذارید |
| N2 | **جدید:** baseline Cars با λ=0 و چند seed هنوز موجود نیست | ⏳ کار بعدی |

## ۶. وضعیت تست‌ها

| تست | نتیجه |
|---|---|
| `python tests/test_normalization.py` | ✅ 13 از 13 |
| `python tests/test_cam_softmax_loss.py` | ✅ 19 از 19 (قبلاً ImportError می‌داد) |
| `python tests/test_state001_fixes.py` | ✅ 22 از 22 (تست جدید) |
| smoke run دو epochی روی Cars (`main.py`) | ✅ pipeline کامل اجرا شد: آموزش، انتخاب best، cam_debug، report |

## ۷. تفاوت با STATE_000

### ۷.۱ تغییرات کد

`git diff --stat 5fa3e1d a1ac006`: 11 فایل، ‎+419 / −72.

| فایل | STATE_000 | STATE_001 |
|---|---|---|
| `src/model_with_adapter_bit.py` | `cams = cam_conv(features)` | + `cams -= cams.mean(dim=1)` در صورت فعال بودن `center_cams` |
| `src/model_factory.py` | — | `center_cams` را به BiT پاس می‌دهد |
| `src/losses.py` | allclass روی همه K کلاس، با dtype ورودی | allclass با `topk` اختیاری، محاسبه در fp32 |
| `src/train.py` | best بر اساس `val_acc`، early stop بر اساس monitor، patience از epoch 1، تست روی وزن آخر | یک monitor برای هر دو؛ انتخاب و patience فقط بعد از warm-up؛ تست روی best؛ خروجی جدید `selected_epoch` |
| `src/cam_debug.py` | 10 متریک، همه مبتنی بر relu یا min-max | 14 متریک؛ 4 متریک علامت‌دار اضافه شد؛ `containment=None` برای CAM تمام‌منفی |
| `src/data.py` | فقط Resize | + `load(idx, augment)`، `_load_augmented` و `AugmentedSubset` برای train |
| `main.py`، `sweep.py` | — | `train_augment`، `augment_crop_scale` و `augment_hflip` پاس داده می‌شوند |
| `tests/` | 1 تست سالم + 1 تست خراب | 3 تست سالم |

### ۷.۲ تغییرات تنظیمات

| کلید | STATE_000 | STATE_001 |
|---|---|---|
| `center_cams` | — (false) | **true** |
| `early_stopping_monitor` | `val_loss` | **`val_cls_loss`** |
| `lambda_cam_warmup_epochs` | 10 | **3** |
| `lambda_cam_warmup_power` | 2.0 | **1.0** |
| `cam_loss_type` | `allclass_squared_containment` | **`squared_containment`** |
| `cam_allclass_topk` | — (همه کلاس‌ها) | **5** |
| `train_augment` | — (false) | **true** |
| `augment_crop_scale` / `augment_hflip` | — | **[0.7, 1.0] / true** |
| `runs_dir` | `…_cam005` | **`…_state001`** |

### ۷.۳ تغییر رفتار در آموزش

| رفتار | STATE_000 | STATE_001 |
|---|---|---|
| زمان توقف اجراهای λ>0 | همیشه حدود epoch 11 تا 16، درست بعد از warm-up | فقط وقتی `val_cls_loss` در ۸ epoch **بعد از** warm-up بهتر نشود |
| مدلی که تست می‌شود | آخرین epoch | best (با monitor، فقط از epochهای بعد از warm-up) |
| CAM کلاس درست روی شیء (Cars) | تا ‎−37.7 | باید مثبت باشد؛ در smoke run برابر +11.8 |
| containment برای CAM تمام‌منفی | 0 (جعلی) | `None` |

### ۷.۴ سازگاری با اجراهای قبلی

- **نتایج STATE_000 با STATE_001 مستقیماً قابل مقایسه نیستند.** نرمال‌سازی، augmentation، warm-up، معیار انتخاح و مرکزسازی همه عوض شده‌اند.
- checkpoint‌های قدیمی بدون هیچ تغییری بارگذاری می‌شوند چون `state_dict` تغییر نکرده است.

## ۸. قدم بعدی

1. Baseline با λ=0 روی Cars و Pets، با ۳ seed.
2. `squared_containment` و `softmax_containment` با λ ∈ {0.1, 0.3, 1.0} و ۳ seed.
3. ثبت نتایج در **STATE_002** با این معیارها: `signed_inside_mean > 0`، `cam_all_negative < 5%`، pointing بیشتر از 0.6 و افت دقت کمتر از ۱ واحد درصد نسبت به baseline.
