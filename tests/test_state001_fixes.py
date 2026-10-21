"""
Checks for the STATE_001 fixes (see project_states/STATE_001_run_analysis_fixes.md):
CAM centering, top-k all-class loss, signed test metrics, checkpoint selection /
early stopping after the lambda warm-up, and joint image+mask augmentation.

Run from anywhere:  python tests/test_state001_fixes.py
Exits non-zero if any check fails.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cam_debug import _localization_metrics  # noqa: E402
from src.data import ImageFolderWithClassMasks, AugmentedSubset  # noqa: E402
from src.losses import CAMAllClassSquaredContainmentLoss, CAMSquaredContainmentLoss, build_cam_loss  # noqa: E402
from src.model_with_adapter_bit import BiTMR50x1CAMWithAdapter  # noqa: E402
from src.train import _monitor_value, run_training  # noqa: E402

torch.manual_seed(0)
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}   {detail}")
    if not cond:
        ok = False


B, K, H, W = 4, 6, 7, 7
mask = torch.zeros(B, 1, H, W)
mask[:, :, 2:5, 2:5] = 1.0
has_mask = torch.ones(B)
labels = torch.tensor([0, 1, 2, 3])

# 1. The exploit behind the sign flip: a blob shared by every class, strongly
#    negative on the object. CE cannot see it, and the squared losses reward it.
base = torch.randn(B, K, H, W)
blob = torch.zeros(B, 1, H, W)
blob[:, :, 2:5, 2:5] = -60.0
exploited = base + blob
legacy = CAMAllClassSquaredContainmentLoss(topk=None)
squared = CAMSquaredContainmentLoss()
check("shared negative blob does not change CE",
      torch.allclose(F.cross_entropy(base.mean((2, 3)), labels),
                     F.cross_entropy(exploited.mean((2, 3)), labels), atol=1e-4))
check("shared negative blob lowers the uncentered squared loss",
      squared(exploited, mask, labels, has_mask) < 0.5 * squared(base, mask, labels, has_mask))
centered = exploited - exploited.mean(dim=1, keepdim=True)
check("centering removes the blob exactly",
      torch.allclose(centered, base - base.mean(dim=1, keepdim=True), atol=1e-4))

# 2. center_cams in the real model: probabilities unchanged, class-mean is zero.
model = BiTMR50x1CAMWithAdapter(num_classes=K, pretrained=False, gamma=8).eval()
x = torch.randn(2, 3, 64, 64)
with torch.no_grad():
    logits_raw, cams_raw = model(x)
    model.center_cams = True
    logits_c, cams_c = model(x)
check("center_cams keeps softmax probabilities",
      torch.allclose(logits_raw.softmax(1), logits_c.softmax(1), atol=1e-5))
check("center_cams makes the per-cell class mean zero",
      cams_c.mean(dim=1).abs().max().item() < 1e-4, f"max={cams_c.mean(dim=1).abs().max().item():.2e}")

# 3. top-k all-class loss.
cams = torch.randn(B, K, H, W)
check("topk=None reproduces the legacy all-class loss",
      torch.allclose(CAMAllClassSquaredContainmentLoss()(cams, mask, labels, has_mask),
                     legacy(cams, mask, labels, has_mask)))
check("topk >= K-1 equals all classes",
      torch.allclose(CAMAllClassSquaredContainmentLoss(topk=K)(cams, mask, labels, has_mask),
                     legacy(cams, mask, labels, has_mask)))
target_only = squared(cams, mask, labels, has_mask)
check("topk=0 equals the true-class squared loss",
      torch.allclose(CAMAllClassSquaredContainmentLoss(topk=0)(cams, mask, labels, has_mask), target_only))
check("factory passes cam_allclass_topk",
      build_cam_loss({"cam_loss_type": "allclass_squared_containment", "cam_allclass_topk": 3}).topk == 3)

# 4. Signed test metrics: an all-negative CAM must not score a perfect containment.
full_mask = torch.zeros(1, 224, 224)
full_mask[:, 64:160, 64:160] = 1.0
negative_cam = torch.full((H, W), -5.0)
negative_cam[2:5, 2:5] = -40.0
m = _localization_metrics(negative_cam, full_mask)
check("all-negative CAM: containment undefined, not 0", m["containment"] is None)
check("all-negative CAM is flagged", m["cam_all_negative"] == 1.0 and m["abs_peak_negative"] == 1.0)
check("signed means expose the negative object",
      m["signed_inside_mean"] < m["signed_outside_mean"],
      f"in={m['signed_inside_mean']:.1f} out={m['signed_outside_mean']:.1f}")
good = torch.zeros(H, W)
good[2:5, 2:5] = 5.0
g = _localization_metrics(good, full_mask)
check("positive object CAM keeps a real containment", g["containment"] is not None and g["containment"] < 0.2)

# 5. Monitor lookup.
check("val_cls_loss -> val cls_loss", _monitor_value("val_cls_loss", {"cls_loss": 1}, {"cls_loss": 2}) == 2)
check("train_acc -> train acc", _monitor_value("train_acc", {"acc": 0.3}, {"acc": 0.9}) == 0.3)
check("unknown monitor -> None", _monitor_value("foo", {}, {}) is None)


# 6. Selection and early stopping start only after the warm-up. With lr=0 the
#    monitor never moves, so the first eligible epoch is selected and training
#    stops exactly `patience` epochs later.
class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.cam_conv = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        cams = self.cam_conv(x)
        return cams.mean((2, 3)), cams


class _Fake(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return {"image": torch.randn(3, 7, 7), "label": torch.tensor(i % 3),
                "mask": torch.ones(1, 7, 7), "has_mask": torch.tensor(1.0), "image_path": f"{i}.png"}


tiny = _Tiny()
loader = DataLoader(_Fake(), batch_size=4)
warmup, patience = 3, 2
with tempfile.TemporaryDirectory() as tmp:
    result = run_training(
        model=tiny, train_loader=loader, val_loader=loader, classes=["a", "b", "c"],
        ce_loss_fn=nn.CrossEntropyLoss(), cam_loss_fn=CAMSquaredContainmentLoss(),
        optimizer=torch.optim.SGD(tiny.parameters(), lr=0.0), scheduler=None, scaler=None,
        device="cpu", lambda_cam=1.0, use_amp=False, epochs=20,
        config={"diagnostics": False, "cam_debug": False, "early_stopping": True,
                "early_stopping_monitor": "val_cls_loss", "early_stopping_patience": patience,
                "lambda_cam_warmup_epochs": warmup, "model_type": "plain"},
        runs_dir=tmp,
    )
check("selected checkpoint is the first post-warm-up epoch", result["selected_epoch"] == warmup,
      f"selected={result['selected_epoch']}")
check("early stopping counts patience only after the warm-up",
      result["stopped_early"] and result["last_epoch_ran"] == warmup + patience,
      f"stopped at {result['last_epoch_ran']}")

# 7. Joint augmentation: the mask must follow the image through crop and flip.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "data" / "cls").mkdir(parents=True)
    (root / "masks").mkdir()
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:, :60] = (255, 0, 0)           # red object on the left
    Image.fromarray(img).save(root / "data" / "cls" / "a.png")
    msk = np.zeros((120, 160), dtype=np.uint8)
    msk[:, :60] = 255
    Image.fromarray(msk).save(root / "masks" / "a.png")

    ds = ImageFolderWithClassMasks(str(root / "data"), image_size=64,
                                   mask_dirs={"cls": str(root / "masks")},
                                   normalize_mean=[0.5] * 3, normalize_std=[0.5] * 3,
                                   augment_crop_scale=(0.3, 1.0))
    aug = AugmentedSubset(ds, [0])
    agree = []
    flipped = 0
    for _ in range(30):
        s = aug[0]
        red = (s["image"][0] > 0.5) & (s["image"][1] < -0.5)
        agree.append((red.float() == s["mask"][0]).float().mean().item())
        flipped += int(s["mask"][0, :, -1].mean() > 0.5)
    plain = ds[0]
    check("augmented mask matches the augmented image", min(agree) > 0.97, f"min agreement={min(agree):.3f}")
    check("horizontal flip happens", 0 < flipped < 30, f"{flipped}/30 flipped")
    check("non-augmented path is unchanged in shape", plain["mask"].shape == (1, 64, 64)
          and plain["has_mask"].item() == 1.0)
    check("AugmentedSubset still exposes the base dataset", aug.dataset is ds and aug.indices == [0])

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
raise SystemExit(0 if ok else 1)
