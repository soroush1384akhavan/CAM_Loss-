import torch
from pathlib import Path
import yaml

from src.adversarial import (
    denormalize_images,
    normalize_images,
    unpack_batch,
    get_logits,
    evaluate_adversarial_robustness,
)
from src.data import get_dataloaders
from src.utils import get_device
from src.model_factory import build_model
from src.normalization import configure_normalization, validate_checkpoint_normalization


CONFIG_PATH = Path("configs/config.yaml")
config_path = Path(CONFIG_PATH)

with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

mean, std = configure_normalization(config)

device = get_device()


train_loader, val_loader, test_loader, classes = get_dataloaders(
    data_dir=config["data_dir"],
    image_size=config.get("image_size", 224),
    batch_size=config.get("batch_size", 32),
    val_ratio=config.get("val_ratio", 0.2),
    seed=config.get("seed", 42),
    num_workers=config.get("num_workers", 0),
    normalize_mean=mean,
    normalize_std=std,
    mask_dirs=config.get("mask_dirs", {}),
    mask_manifests=config.get("mask_manifests", {}),
    test_ratio=config.get("test_ratio", 0.1),
    train_shots_per_class=config.get("train_shots_per_class", None),
    train_fraction=config.get("train_fraction", None),
)

num_classes = len(classes)

model = build_model(config, num_classes).to(device)


CHECKPOINT_PATH = "/media/vision/ShareStorage-2/soroush/CVLab/runs/2026-08-21_16-28-47/last_resnet18_cam.pth"

if CHECKPOINT_PATH is None:
    runs_dir = Path(config.get("runs_dir", "runs"))
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")

    latest_run = run_dirs[-1]
    print(f"Using latest run: {latest_run}")

    candidate = latest_run / "best_resnet18_cam.pth"
    if not candidate.exists():
        candidate = latest_run / "last_resnet18_cam.pth"
    if not candidate.exists():
        raise FileNotFoundError(
            f"No checkpoint (best/last) found in {latest_run}"
        )

    CHECKPOINT_PATH = str(candidate)

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
validate_checkpoint_normalization(checkpoint, mean, std)


def _extract_state_dict(checkpoint):

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            print(f"  (state_dict found under key: '{key}')")
            return checkpoint[key]

    looks_like_state_dict = all(
        isinstance(v, torch.Tensor) for v in list(checkpoint.values())[:5]
    )
    if looks_like_state_dict:
        print("  (checkpoint itself looks like a raw state_dict)")
        return checkpoint

    raise KeyError(
        f"Could not find model weights in checkpoint. "
        f"Top-level keys: {list(checkpoint.keys())}"
    )


state_dict = _extract_state_dict(checkpoint)
missing, unexpected = model.load_state_dict(state_dict, strict=False)

if missing:
    print(f"  WARNING - missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
if unexpected:
    print(f"  WARNING - unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
if not missing and not unexpected:
    print("  Checkpoint loaded cleanly (no missing/unexpected keys).")

model.eval()

if isinstance(checkpoint, dict) and "val_acc" in checkpoint:
    print(f"  Checkpoint val_acc (from training): {checkpoint['val_acc']:.4f}")


batch = next(iter(test_loader))
images, labels = unpack_batch(batch)
images = images.to(device)
labels = labels.to(device)

print("\n--- sanity check ---")
print("images raw stats:", images.min().item(), images.max().item(), images.mean().item())

pixels = denormalize_images(images, mean, std)
print("denormalized (should be ~[0,1]):", pixels.min().item(), pixels.max().item())

pixels_clamped = pixels.clamp(0, 1)
print("after clamp:", pixels_clamped.min().item(), pixels_clamped.max().item())

clip_ratio = (pixels != pixels_clamped).float().mean().item()
print("fraction of pixels clipped:", clip_ratio)

renormalized = normalize_images(pixels_clamped, mean, std)
with torch.no_grad():
    logits = get_logits(model(renormalized))
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
print("accuracy after round-trip (no attack):", acc)

with torch.no_grad():
    logits2 = get_logits(model(images))
    preds2 = logits2.argmax(dim=1)
    acc2 = (preds2 == labels).float().mean().item()
print("accuracy direct (no round-trip):", acc2)



print("\n--- full adversarial evaluation ---")
results = evaluate_adversarial_robustness(
    model=model,
    loader=test_loader,
    device=device,
    mean=mean,
    std=std,
    epsilons=[1 / 255, 2 / 255, 4 / 255, 8 / 255],
    pgd_steps=10,
)
