# STATE_002 — sweep چند-seed، پایپ‌لاین Kaggle و رفع دو باگ بارگذاری داده

| | |
|---|---|
| تاریخ ثبت | ۱۴۰۵-۰۸-۱۲ |
| Branch | `lab-work-imported` |
| Commit | `7a512d6` — *fix: cross-OS mask manifests and DataLoader augmentation; add seed sweep* |
| State قبلی | [STATE_001](STATE_001_run_analysis_fixes.md) (`a1ac006`) |
| بازگشت به این state | `git checkout 7a512d6` |
| مقایسه کد با state قبلی | `git diff a1ac006 7a512d6` |

---

## ۱. هدف پروژه

بدون تغییر نسبت به STATE_001:

```text
loss = CE(GAP(cam), y) + λ_cam · squared_containment(CAM مرکزشده، ماسک SAM)
```

**چیز تازه در این state:** ادعای اصلی حالا با چند seed پشتیبانی می‌شود. روی Stanford Cars، CAM loss دقت few-shot را در **۳۰ مقایسهٔ زوجی از ۳۰ مقایسه** بالا برده است (۳ seed × ۵ سطح shots × ۲ مقدار λ).

## ۲. اجزای کد

| بخش | فایل | وضعیت در این state |
|---|---|---|
| مدل | [src/model_with_adapter_bit.py](../src/model_with_adapter_bit.py) | بدون تغییر (BiT-M R50x1 + adapter، `center_cams`) |
| lossها | [src/losses.py](../src/losses.py) | بدون تغییر |
| آموزش | [src/train.py](../src/train.py) | بدون تغییر |
| داده | [src/data.py](../src/data.py) | **رفع دو باگ:** `AugmentedSubset.__getitems__` (augmentation حالا واقعاً از طریق DataLoader اعمال می‌شود)؛ پارس مسیرهای manifest با `PureWindowsPath` (ماسک‌ها روی لینوکس بارگذاری می‌شوند). **محافظ جدید:** اگر manifestها هیچ ماسک قابل‌استفاده‌ای نداشته باشند، خطا می‌دهد. |
| sweep | [sweep.py](../sweep.py) | شبکهٔ آزمایش همان چیزی است که اجرا شد: shots ‏{1,2,4,8,16} × λ ‏{0, 0.5, 3}، ۱۵ epoch، یک تکرار در هر اجرا |
| Kaggle | [kaggle/](../kaggle/) | **جدید:** `build_kernel.py` (کد را داخل kernel جاسازی می‌کند)، `runner_template.py` (هر seed روی یک GPU از T4)، `smoke_test_template.py` (بررسی روی CPU که ماسک‌ها بارگذاری می‌شوند و CAM loss فعال است)، `analyze_seeds.py` |
| نتایج | [results/seed_sweep/](../results/seed_sweep/) | **جدید:** هر ۴۵ ران، تفاوت‌های زوجی، جدول میانگین و انحراف معیار |

## ۳. تنظیمات فعال

`configs/config.yaml` کامیت‌شده نسبت به STATE_001 تغییری نکرده است. sweep برای هر ران `lambda_cam`، `epochs`، `train_shots_per_class` و `seed` را خودش تعیین می‌کند.

تنظیمات sweep چند-seed:

| کلید | مقدار |
|---|---|
| داده | Stanford Cars، ۱۹۶ کلاس، ماسک SAM (۱۴٬۳۹۳ تصویر ماسک دارند) |
| مدل | `adapter_bit`، gamma=8، backbone منجمد، `center_cams: true` |
| loss | `squared_containment`، λ ∈ {0, 0.5, 3}، warm-up خطی ۳ epoch |
| آموزش | ۱۵ epoch، lr ‏3e-4، cosine، batch 8، AMP؛ انتخاب checkpoint با `val_cls_loss` بعد از warm-up |
| augmentation | **خاموش** (روی Kaggle با `train_augment: false`، و روی سیستم محلی به‌خاطر باگ عملاً خاموش) |
| seedها | 42 (محلی، RTX 3060، PyTorch 2.6)، 43 و 44 (Kaggle T4، PyTorch 2.11) |

## ۴. آخرین نتایج

گزارش کامل: [SEED_SWEEP_RESULTS.md](../SEED_SWEEP_RESULTS.md)

**دقت test (٪)، میانگین ± انحراف معیار روی ۳ seed**

| shots | λ=0 | λ=0.5 | λ=3 |
|---|---|---|---|
| 1 | 5.2 ± 0.7 | 5.4 ± 0.6 | **7.3 ± 2.4** |
| 2 | 9.7 ± 1.2 | 12.1 ± 0.9 | **14.2 ± 1.3** |
| 4 | 16.8 ± 1.2 | 20.4 ± 2.5 | **22.2 ± 2.5** |
| 8 | 31.6 ± 2.8 | 34.8 ± 2.4 | **35.8 ± 2.6** |
| 16 | 49.3 ± 1.4 | **51.7 ± 1.1** | 51.1 ± 1.5 |

**بهبود زوجی نسبت به λ=0 (همان seed و همان split)، روی همهٔ shotها**

| | میانگین بهبود | مثبت | p آزمون علامت |
|---|---|---|---|
| λ=0.5 | ‎+2.34 واحد درصد | ۱۵ از ۱۵ | 0.0001 |
| λ=3 | ‎+3.60 واحد درصد | ۱۵ از ۱۵ | 0.0001 |

- λ=3 در ۲ تا ۸ shot بهترین است (‎+4.2 تا ‎+5.4 واحد). در ۱۶ shot هر دو λ حدود ‎+2 واحد می‌دهند.
- containment (سهم انرژی CAM بیرون ماسک) در ۱۶ shot: ‏0.44 (λ=0) → 0.33 (λ=0.5) → 0.24 (λ=3). هیچ ران λ>0 CAM تمام‌منفی ندارد.

## ۵. مشکلات شناخته‌شده

| # | مشکل | وضعیت |
|---|---|---|
| A1 | **augmentation در هیچ‌کدام از ران‌های STATE_001 و sweep اجرا نشد** (DataLoader از `AugmentedSubset.__getitem__` رد می‌شد) | ✅ کد اصلاح و تست شد؛ ⚠️ نتیجهٔ آموزش با augmentation هنوز اندازه‌گیری نشده |
| A2 | **ماسک‌ها روی لینوکس بارگذاری نمی‌شدند** (مسیرهای ویندوزی در manifest)؛ Kaggle v2 با CAM loss = 0 آموزش دید | ✅ اصلاح شد + محافظ؛ smoke test روی Kaggle تأیید کرد (۱۴٬۳۹۳ ماسک پیدا شد) |
| A3 | متریک‌های محلی‌سازی λ=0 فقط برای seed 42 وجود دارد (ران‌های λ=0 روی Kaggle ماسک نداشتند) | ℹ️ دقت آن ران‌ها معتبر است، چون λ=0 از ماسک استفاده نمی‌کند |
| A4 | seedها در دو محیط مختلف اجرا شده‌اند (محلی و Kaggle) | ℹ️ بهبودهای زوجی معتبر می‌مانند؛ sweep بعدی باید در یک محیط باشد |
| A5 | کار در جریان کاربر که کامیت نشده: آرگومان `--config` در `main.py`، `configs/config_cars_full.yaml`، `configs/config_pets_full.yaml`، و `config.yaml` با همهٔ داده | ⚠️ تغییر `main.py` یک تست در `test_normalization.py` را خراب می‌کند (تست `main.CONFIG_PATH` را patch می‌کند) |
| P5 | ران‌های قدیمی با نرمال‌سازی ImageNet | تاریخی، بدون تغییر |
| N1 | checkpointهای قدیمی با config فعلی CAM مرکزشده نشان می‌دهند | بدون تغییر |

## ۶. وضعیت تست‌ها

| تست | نتیجه |
|---|---|
| `python tests/test_state001_fixes.py` | ✅ ۲۸ از ۲۸ (۶ تست جدید: augmentation از مسیر DataLoader، stem مسیرهای ویندوزی و POSIX، محافظ ماسک صفر) |
| `python tests/test_cam_softmax_loss.py` | ✅ ۱۹ از ۱۹ |
| `python tests/test_normalization.py` | ✅ ۱۳ از ۱۳ در این کامیت؛ ❌ یک خطا با `main.py` کامیت‌نشده (A5) |
| smoke test روی CPU در Kaggle | ✅ ماسک‌ها روی لینوکس بارگذاری می‌شوند، CAM loss غیرصفر است و گرادیان به مدل می‌رسد |

## ۷. تفاوت با STATE_001

### ۷.۱ تغییرات کد

`git diff --stat a1ac006 7a512d6`: ‏۱۴ فایل، ‎+849 / −6 (بدون `project_states/`).

| فایل | STATE_001 | STATE_002 |
|---|---|---|
| `src/data.py` | `Path(mask_path).stem`؛ `AugmentedSubset` فقط `__getitem__` را override می‌کند | `PureWindowsPath(...).stem`؛ `AugmentedSubset.__getitems__`؛ خطا وقتی manifestها ماسک قابل‌استفاده ندارند |
| `sweep.py` | λ ‏{0, 0.3, 0.5, 1, 3}، ۲۰ epoch، ۳ تکرار | λ ‏{0, 0.5, 3}، ۱۵ epoch، ۱ تکرار (seedها از بیرون داده می‌شوند) |
| `tests/test_state001_fixes.py` | ۲۲ بررسی | ۲۸ بررسی |
| `kaggle/` | — | سازندهٔ kernel، اجراکنندهٔ sweep، smoke test، تحلیل |
| `results/seed_sweep/` | — | نتایج ۴۵ ران (CSV) |
| مستندات | — | `SEED_SWEEP_RESULTS.md`؛ اصلاحیه روی `COMPARE_LAMBDA0_VS_05_FA.md` |

### ۷.۲ تغییر رفتار

| رفتار | STATE_001 | STATE_002 |
|---|---|---|
| `train_augment: true` | بی‌صدا بدون augmentation | augmentation اعمال می‌شود |
| آموزش روی لینوکس/Kaggle | صفر ماسک، CAM loss بی‌صدا صفر | ماسک‌ها بارگذاری می‌شوند؛ اگر نباشند خطا می‌دهد |
| شاهد برای بهبود CAM loss | یک جفت ران (۱۶ shot) | ۴۵ ران، ۳ seed، ۵ سطح shots |

### ۷.۳ سازگاری

- همهٔ اعداد STATE_001 به‌عنوان نتایج «بدون augmentation» معتبر می‌مانند.
- ران‌هایی که از این state به بعد با `train_augment: true` اجرا شوند با ران‌های قبلی **قابل مقایسه نیستند**.

## ۸. قدم بعدی

1. تکرار λ ∈ {0, 3} **با augmentation**، ۳ seed، همه روی Kaggle. این بررسی تعیین‌کننده است.
2. baseline «حذف پس‌زمینه هنگام آموزش» با همان ماسک‌های SAM.
3. اضافه کردن دیتاست ریزدانهٔ دوم و سوم (CUB-200، FGVC-Aircraft).
4. جست‌وجوی منابع دربارهٔ نظارت روی attention/CAM و دربارهٔ تباهیدگی مؤلفهٔ مشترک بین کلاس‌ها.
5. امتحان λ ∈ {5, 10} در ۲ تا ۸ shot، و ۲۵ epoch برای λ>0.
