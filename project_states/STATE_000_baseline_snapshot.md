# STATE_000 — اسنپ‌شات اولیه (قبل از اصلاحات)

| | |
|---|---|
| تاریخ ثبت | ۱۴۰۵-۰۷-۳۰ (کد مربوط به کار آزمایشگاه تا ۱۴۰۳-۰۶-۱۶) |
| Branch | `lab-work-imported` |
| Commit | `5fa3e1d` — *snapshot: state 000 before run-analysis fixes* |
| State قبلی | — (اولین state) |
| بازگشت به این state | `git checkout 5fa3e1d` |

---

## ۱. هدف پروژه

آموزش یک طبقه‌بند few-shot (۱۶ نمونه برای هر کلاس) با تابع هزینه زیر:

```text
loss = CE(GAP(cam), y) + λ_cam · cam_loss(cam, SAM pseudo-mask)
```

هدف: CAM کلاس درست باید روی **خود شیء** فعال شود، نه روی پس‌زمینه.

## ۲. اجزای کد

| بخش | فایل | وضعیت در این state |
|---|---|---|
| مدل | [src/model_with_adapter_bit.py](../src/model_with_adapter_bit.py) | BiT-M R50x1 منجمد + Conv-Adapter موازی روی هر bottleneck. GroupNorm‌ها قابل آموزش‌اند. هد `1×1 conv` مستقیماً CAM می‌دهد و `logit = GAP(cam)`. **CAM مرکزسازی نمی‌شود.** |
| تابع هزینه | [src/losses.py](../src/losses.py) | `containment` (نسبت relu)، `squared_containment`، `allclass_squared_containment` (میانگین روی **همه** K کلاس)، `containment_pointing`، `softmax_containment` |
| آموزش | [src/train.py](../src/train.py) | warm-up توانی برای λ. early stopping روی `val_loss` که **شامل λ·cam هم هست**. best checkpoint بر اساس `val_acc` انتخاب می‌شود. تست و `cam_debug` روی **وزن‌های آخرین epoch** اجرا می‌شوند. |
| داده | [src/data.py](../src/data.py) | ImageFolder با ماسک SAM، `random_split` و subsample به ۱۶ نمونه. **هیچ augmentation‌ای ندارد.** نرمال‌سازی از config خوانده می‌شود (از ۱۴۰۳-۰۶-۱۶ به بعد). |
| نرمال‌سازی | [src/normalization.py](../src/normalization.py) | BiT با `[0.5, 0.5, 0.5]` و ResNet با آمار ImageNet. در checkpoint ذخیره می‌شود. |
| متریک‌های تست | [src/cam_debug.py](../src/cam_debug.py) | IoU با min-max؛ `containment` و `mean_pos_*` مبتنی بر **relu**. متریک علامت‌دار ندارد. |

## ۳. تنظیمات فعال (`configs/config.yaml`)

| کلید | مقدار |
|---|---|
| data | Stanford Cars (196 کلاس) |
| model_type | `adapter_bit`، gamma=8، freeze_backbone=true |
| optimizer | lr=3e-4، wd=1e-4، cosine، epochs=50، batch=8، AMP |
| cam_loss_type | `allclass_squared_containment` |
| lambda_cam | 0.5 (شروع از 0، warm-up 10 epoch، توان 2.0) |
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

✱ فقط این اجراها با نرمال‌سازی صحیح BiT آموزش دیده‌اند. در این state هیچ baseline‌ای با λ=0 روی Cars وجود ندارد.

## ۵. مشکلات شناخته‌شده

جزئیات و شواهد در [RUNS_ANALYSIS_FA.md](../RUNS_ANALYSIS_FA.md) آمده است.

| # | مشکل | شدت |
|---|---|---|
| P1 | **Sign flip:** مؤلفه مشترک بین کلاس‌ها (که برای CE رایگان است) روی شیء به‌شدت منفی می‌شود. loss مربعی این را به عنوان «انرژی داخل» حساب می‌کند؛ نتیجه: CAM روی پس‌زمینه پیک می‌زند (pointing=0.08). | 🔴 |
| P2 | early stopping روی `val_loss` کار می‌کند که شامل λ·cam است. همه اجراهای λ>0 درست بعد از warm-up متوقف شده‌اند. | 🔴 |
| P3 | سه معیار جداگانه: تست روی وزن آخر، best بر اساس val_acc، و early stop بر اساس val_loss. | 🟠 |
| P4 | برای CAM تمام‌منفی، `containment` بر اساس relu برابر 0 می‌شود — یعنی بهترین نمره، اما جعلی. | 🟠 |
| P5 | اجراهای قبل از 09-07_10-51 با نرمال‌سازی ImageNet آموزش دیده‌اند، ولی config آن‌ها 0.5 نشان می‌دهد. | 🟠 (تاریخی) |
| P6 | `allclass_squared` روی ۱۹۵ کلاسی که در تصویر حضور ندارند هم میانگین می‌گیرد. | 🟠 |
| P7 | warm-up بیش از حد طولانی است؛ λ=10 با start=5 عملاً بدون warm-up شروع می‌شود. | 🟡 |
| P8 | augmentation ندارد: train≈1.0 در برابر val≈0.47 روی Cars. | 🟡 |
| P9 | `tests/test_cam_softmax_loss.py` نام‌های قدیمی `CAMMaskLoss` و `CAMSoftmaxMaskLoss` را import می‌کند و **اجرا نمی‌شود**. | 🟡 |

## ۶. وضعیت تست‌ها

| تست | نتیجه |
|---|---|
| `python tests/test_normalization.py` | ✅ همه ۱۳ تست پاس |
| `python tests/test_cam_softmax_loss.py` | ❌ `ImportError: cannot import name 'CAMMaskLoss'` |

## ۷. تفاوت با state قبلی

اولین state است.
