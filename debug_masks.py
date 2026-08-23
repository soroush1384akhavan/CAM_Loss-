# debug_masks.py
import yaml
from pathlib import Path
from torchvision import datasets
from src.mask_config_utils import build_mask_config

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_dir = config["data_dir"]
mask_root = config["mask_root"]

img_dataset = datasets.ImageFolder(root=data_dir)
image_classes = set(img_dataset.classes)

mask_dirs, mask_manifests = build_mask_config(mask_root)
mask_classes = set(mask_dirs.keys())

print("=== class numbers ===")
print("image classes:", len(image_classes))
print("mask classes :", len(mask_classes))

print("\n=== masksroot which is not on data ===")
missing = image_classes - mask_classes
print(missing if missing else "none")

print("\n-----------------")
print("image sample:", sorted(image_classes)[:3])
print("mask  sample:", sorted(mask_classes)[:3])

common = sorted(image_classes & mask_classes)
if common:
    c = common[0]
    md = Path(mask_dirs[c])
    mf = Path(mask_manifests[c])
    print(f"\n=== {c} ===")
    print("mask_dir:", md, "-> exists:", md.exists())
    if md.exists():
        pngs = list(md.glob("*.png"))
        print("nums : ", len(pngs))
        if pngs:
            print("samples :", pngs[0].name)
    print("manifest:", mf, "-> exists:", mf.exists())
else:
    print("\nهیچ کلاس مشترکی بین image_classes و mask_classes نیست!")
    
# 4) اسم فایل‌های واقعی تصویر برای همون کلاس نمونه
img_class_dir = Path(data_dir) / c
img_files = list(img_class_dir.glob("*"))
print(f"\n=== نمونه فایل‌های تصویر تو '{c}' ===")
print("تعداد:", len(img_files))
for p in img_files[:5]:
    print(" -", p.name, "  stem:", p.stem)

mask_files = list(md.glob("*.png"))
print(f"\n=== نمونه فایل‌های ماسک تو '{c}' ===")
print("تعداد:", len(mask_files))
for p in mask_files[:5]:
    print(" -", p.name, "  stem:", p.stem)

# 5) آیا هیچ overlap بین stem‌ها هست؟
img_stems = {p.stem for p in img_files}
mask_stems = {p.stem for p in mask_files}
print("\n=== overlap بین stem تصویر و ماسک ===")
print("common:", len(img_stems & mask_stems))

# 6) محتوای چند خط اول manifest
print(f"\n=== چند خط اول manifest ===")
with open(mf, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i > 5:
            break
        print(line.strip())