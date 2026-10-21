# STATE_000 — snapshot قبل از اصلاحات

| | |
|---|---|
| تاریخ ثبت | 2026-10-21 (کد مربوط به کار آزمایشگاه تا 2026-09-07) |
| Branch | `lab-work-imported` |
| Commit | `5fa3e1d` — *snapshot: state 000 before run-analysis fixes* |
| State قبلی | — (اولین state) |
| برگشت به این state | `git checkout 5fa3e1d` |

---

## ۱. هدف پروژه

آموزش یک طبقه‌بند few-shot (۱۶ نمونه برای هر کلاس) با این تابع هزینه:

```text
loss = CE(GAP(cam), y) + λ_cam · cam_loss(cam, SAM pseudo-mask)
```

هدف این است که CAM کلاس درست روی **خود شیء** فعال شود، نه روی پس‌زمینه.

## ۲. اجزای کد

| بخش | فایل | وضعیت در این state |
|---|---|---|
| مدل | [src/model_with_adapter_bit.py](../src/model_with_adapter_bit.py) | BiT-M R50x1 منجمد، Conv-Adapter موازی با هر bottleneck، GroupNormها قابل آموزش، و هد `1x1 conv` که مستقیماً CAM می‌دهد و `logit = GAP(cam)` است. **CAM مرکزسازی نمی‌شود.** |
| lossها | [src/losses.py](../src/losses.py) | `containment` (relu-ratio)، `squared_containment`، `allclass_squared_containment` (میانگین روی **همهٔ** K کلاس)، `containment_pointing` و `softmax_containment` |
| آموزش | [src/train.py](../src/train.py) | warm-up توانی برای λ. early stopping روی `val_loss`، که **شامل λ·cam** است. best checkpoint بر اساس `val_acc` انتخاب می‌شود. تست/`cam_debug` روی **وزن آخرین epoch** اجرا می‌شود. |
| داده | [src/data.py](../src/data.py) | ImageFolder به‌همراه ماسک SAM برای هر کلاس، `random_split` و subsample به ۱۶ نمونه. **هیچ augmentationی وجود ندارد.** نرمال‌سازی از config خوانده می‌شود (این رفتار از 2026-09-07 شروع شد). |
| نرمال‌سازی | [src/normalization.py](../src/normalization.py) | BiT با `[0.5]*3` و ResNet با آمار ImageNet. در checkpoint ذخیره می‌شود. |
| متریک تست | [src/cam_debug.py](../src/cam_debug.py) | IoU با min-max، و `containment` و `mean_pos_*` بر اساس **relu**. متریک علامت‌دار ندارد. |

## ۳. Config فعال (`configs/config.yaml`)

| کلید | مقدار |
|---|---|
| data | Stanford Cars (196 کلاس) |
| model_type | `adapter_bit`، gamma=8، freeze_backbone=true |
| optimizer | lr=3e-4، wd=1e-4، cosine، epochs=50، batch=8، AMP |
| cam_loss_type | `allclass_squared_containment` |
| lambda_cam | 0.5 (start 0، warm-up 10 epoch، power 2.0) |
| early stopping | monitor=`val_loss`، patience=8 |
| normalize | `[0.5, 0.5, 0.5]` |

## ۴. آخرین نتایج (`runs/bitm_r50x1_adapter_g8_16shot_cam005`)

| اجرا | داده | loss | λ | test acc | pointing | cam_max میانگین |
|---|---|---|---|---|---|---|
| 09-05_20-47 | Pets | — | 0 | **0.895** | 0.78 | 9.3 |
| 09-05_21-19 | Pets | squared | 0.5 | 0.873 | 0.96 | 17.3 |
| 09-06_12-47 | Pets | allclass | 10 | 0.830 | 0.73 | 8.2 |
| 09-07_10-51 ✱ | Cars | softmax | 10 | 0.368 | 0.82 | 0.3 |
| 09-07_11-21 ✱ | Cars | allclass | 10 | 0.413 | 0.35 | 4.8 |
| 09-07_12-24 ✱ | Cars | allclass | 0.5 | 0.483 | **0.08** | **−0.8** |

✱ فقط این اجراها با نرمال‌سازی درست BiT آموزش دیده‌اند. Cars baseline با λ=0 در این state وجود ندارد.

## ۵. مشکلات شناخته‌شده

جزئیات و شواهد در [RUNS_ANALYSIS_FA.md](../RUNS_ANALYSIS_FA.md) آمده است.

| # | مشکل | شدت |
|---|---|---|
| P1 | **Sign flip:** مؤلفهٔ مشترک بین کلاس‌ها (رایگان برای CE) روی شیء شدیداً منفی می‌شود و loss مربعی آن را «انرژی داخل» حساب می‌کند. نتیجه این است که CAM روی پس‌زمینه پیک می‌زند (pointing 0.08). | 🔴 |
| P2 | early stopping روی `val_loss` شامل λ·cam است. همهٔ اجراهای λ>0 درست بعد از warm-up متوقف شده‌اند. | 🔴 |
| P3 | تست روی وزن آخر اجرا می‌شود، best بر اساس val_acc انتخاب می‌شود و early stop بر اساس val_loss. یعنی سه معیار مختلف داریم. | 🟠 |
| P4 | `containment` بر اساس relu برای CAM تمام‌منفی برابر 0 می‌شود، یعنی بهترین نمرهٔ جعلی. | 🟠 |
| P5 | اجراهای قبل از 09-07_10-51 با نرمال‌سازی ImageNet آموزش دیده‌اند، ولی config آن‌ها 0.5 می‌گوید. | 🟠 (تاریخی) |
| P6 | `allclass_squared` روی ۱۹۵ کلاسی که در تصویر نیستند میانگین می‌گیرد. | 🟠 |
| P7 | warm-up خیلی طولانی است و λ=10 با start=5 عملاً warm-up ندارد. | 🟡 |
| P8 | augmentation وجود ندارد: train≈1.0 در برابر val≈0.47 روی Cars. | 🟡 |
| P9 | `tests/test_cam_softmax_loss.py` نام‌های قدیمی `CAMMaskLoss` و `CAMSoftmaxMaskLoss` را import می‌کند و **اجرا نمی‌شود**. | 🟡 |

## ۶. وضعیت تست‌ها

| تست | نتیجه |
|---|---|
| `python tests/test_normalization.py` | ✅ همهٔ ۱۳ تست پاس می‌شوند |
| `python tests/test_cam_softmax_loss.py` | ❌ `ImportError: cannot import name 'CAMMaskLoss'` |

## ۷. تفاوت با state قبلی

اولین state است.
