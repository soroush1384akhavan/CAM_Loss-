"""CPU smoke test on Kaggle: do masks load on Linux, and does the CAM loss reach the model?"""
import base64
import glob
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

CODE_B64 = "__CODE_B64__"
CODE = Path("/tmp/camcode")
CODE.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall(CODE)
os.chdir(CODE)
sys.path.insert(0, str(CODE))

import timm  # noqa: E402

if timm.__version__ != "1.0.29":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.29"], check=False)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from src.data import _stem_from_any_path, get_dataloaders  # noqa: E402
from src.losses import build_cam_loss  # noqa: E402
from src.mask_config_utils import build_mask_config  # noqa: E402
from src.model_factory import build_model  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}   {detail}", flush=True)


def find_dir(suffix):
    for pattern in ("/kaggle/input/*/" + suffix, "/kaggle/input/*/*/" + suffix, "/kaggle/input/*/*/*/" + suffix):
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    raise FileNotFoundError(suffix)


print("python", sys.version.split()[0], "torch", torch.__version__, "os", os.name, flush=True)
data_dir = find_dir("cars_dataset/cars_dataset")
mask_root = find_dir("sam_pseudo_masks_cars/sam_pseudo_masks_cars")

# 1. stem parsing on Linux
check("Windows path stem on Linux", _stem_from_any_path(r"D:\a\b\masks\test_01537.png") == "test_01537")

# 2. same loaders as the sweep (seed 43, 1 shot, no augmentation)
cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
mask_dirs, mask_manifests = build_mask_config(mask_root)
train_loader, val_loader, test_loader, classes = get_dataloaders(
    data_dir=data_dir, image_size=224, batch_size=8, val_ratio=cfg["val_ratio"], seed=43, num_workers=0,
    normalize_mean=cfg["normalize_mean"], normalize_std=cfg["normalize_std"],
    mask_dirs=mask_dirs, mask_manifests=mask_manifests, test_ratio=cfg["test_ratio"],
    train_shots_per_class=1, train_fraction=None, train_augment=False,
)
base_ds = train_loader.dataset.dataset
valid = sum(len(v) for v in base_ds.valid_mask_stems_by_class.values() if v)
check("manifests list the expected number of masks", valid == 14393, f"valid stems={valid}")

seen = with_mask = 0
batches = []
for i, b in enumerate(train_loader):
    seen += len(b["label"])
    with_mask += int(b["has_mask"].sum())
    batches.append(b)
    if i == 5:
        break
check("training batches carry masks", with_mask > 0.6 * seen, f"{with_mask}/{seen} samples have a mask")

# 3. CAM loss is non-zero and reaches the model
cfg.update(bit_weights_path=hf_hub_download("timm/resnetv2_50x1_bit.goog_in21k", "model.safetensors",
                                            local_dir="/tmp/bitw"))
model = build_model(cfg, len(classes))
model.train()
cam_loss_fn = build_cam_loss(cfg)
b = next(bb for bb in batches if bb["has_mask"].sum() > 0)
logits, cams = model(b["image"])
cam_loss = cam_loss_fn(cams=cams, mask=b["mask"], labels=b["label"], has_mask=b["has_mask"])
ce = nn.CrossEntropyLoss()(logits, b["label"])
check("CAM loss is non-zero", cam_loss.item() > 1e-4, f"cam_loss={cam_loss.item():.4f} ce={ce.item():.4f}")
model.zero_grad()
cam_loss.backward()
g = model.cam_conv.weight.grad
check("CAM loss sends gradient into the model", g is not None and g.abs().sum().item() > 0,
      f"|grad cam_conv|={0 if g is None else g.abs().sum().item():.3e}")
check("CAMs are centered (center_cams)", cams.mean(dim=1).abs().max().item() < 1e-3)

print("\nSMOKE TEST PASSED" if all(results) else "\nSMOKE TEST FAILED", flush=True)
sys.exit(0 if all(results) else 1)
