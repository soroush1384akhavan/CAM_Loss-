import csv
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from src.data import get_dataloaders
from src.model import ResNet18CAM

import numpy as np
from PIL import Image, ImageDraw

from src.utils import load_checkpoint, get_device
from src.model_factory import build_model
from src.reports.cam_viz import signed_cam_overlay
from src.normalization import configure_normalization, normalization_for_loader



def _compute_cam_mask_iou(cam: torch.Tensor, mask: torch.Tensor, cam_threshold: float = 0.5):
    if mask.dim() == 3:
        mask = mask[0]

    # Min-max normalize CAM to [0, 1].
    cam_norm = cam - cam.min()
    cam_norm = cam_norm / (cam_norm.max() + 1e-6)

    # Resize CAM to mask resolution.
    cam_resized = F.interpolate(
        cam_norm.unsqueeze(0).unsqueeze(0),
        size=mask.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    cam_bin = (cam_resized >= cam_threshold).float()
    # print(mask)
    # print(cam_bin)
    mask_bin = (mask >= 0.5).float()

    intersection = (cam_bin * mask_bin).sum().item()
    union = ((cam_bin + mask_bin) >= 1).float().sum().item()

    if union < 1e-6:
        return 1.0

    return intersection / union


def _localization_metrics(cam: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """
    Test-split localisation numbers computed exactly the way the loss sees them:
    on the CAM grid, on relu(cam), against the area-downsampled mask.

    Reported alongside the IoU overlays because IoU is computed on a min-max
    normalised CAM, which always produces a hot region regardless of the CAM's
    absolute values. These numbers do not.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)

    cam = cam.unsqueeze(0).unsqueeze(0).float()
    mask_cam = F.interpolate(mask.float(), size=cam.shape[-2:], mode="area")

    inside = mask_cam
    outside = 1.0 - mask_cam
    positive = F.relu(cam)

    energy_in = float((positive * inside).sum())
    energy_out = float((positive * outside).sum())
    area_in = float(inside.sum())
    area_out = float(outside.sum())
    n_cells = float(cam.numel())

    peak_index = int(cam.flatten().argmax())
    peak_coverage = float(inside.flatten()[peak_index])
    abs_peak_index = int(cam.abs().flatten().argmax())
    all_negative = bool((cam <= 0).all())

    return {
        # relu containment is 0/eps = 0 -- a fake perfect score -- for a CAM
        # with no positive cell, so it is left undefined there.
        "containment": None if all_negative else energy_out / (energy_in + energy_out + eps),
        "uniform_containment": area_out / n_cells,
        "floor_containment": float(outside.min()),
        "mean_pos_inside": energy_in / (area_in + eps),
        "mean_pos_outside": energy_out / (area_out + eps),
        "mask_coverage_cam_grid": float(inside.mean()),
        "positive_cell_frac": float((cam > 0).float().mean()),
        "pointing_hit": 1.0 if peak_coverage >= 0.5 else 0.0,
        "cam_min": float(cam.min()),
        "cam_max": float(cam.max()),
        "signed_inside_mean": float((cam * inside).sum()) / (area_in + eps),
        "signed_outside_mean": float((cam * outside).sum()) / (area_out + eps),
        "abs_peak_negative": 1.0 if float(cam.flatten()[abs_peak_index]) < 0 else 0.0,
        "cam_all_negative": 1.0 if all_negative else 0.0,
    }


def _with_header(tile: np.ndarray, title: str) -> np.ndarray:
    header = Image.new("RGB", (tile.shape[1], 22), (18, 18, 18))
    ImageDraw.Draw(header).text((4, 4), title, fill=(235, 235, 235))
    return np.concatenate([np.asarray(header), tile], axis=0)


def _upsample_cam(cam: torch.Tensor, size: int) -> np.ndarray:
    return F.interpolate(
        cam.unsqueeze(0).unsqueeze(0).float(),
        size=(size, size), mode="bilinear", align_corners=False,
    )[0, 0].cpu().numpy()


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    m = mask > 0.5
    eroded = m.copy()
    eroded[1:] &= m[:-1]
    eroded[:-1] &= m[1:]
    eroded[:, 1:] &= m[:, :-1]
    eroded[:, :-1] &= m[:, 1:]
    return m & ~eroded


def run_cam_debug(
    model,
    test_loader,
    classes,
    device,
    run_dir,
    top_k: int = 20,
    cam_iou_threshold: float = 0.5,
    normalize_mean=None,
    normalize_std=None,
    overlay_gamma: float = 0.6,
    overlay_alpha_gain: float = 1.5,
):

    model.eval()

    ce_none = torch.nn.CrossEntropyLoss(reduction="none")

    normalize_mean, normalize_std = normalization_for_loader(test_loader, normalize_mean, normalize_std)
    mean_t = torch.tensor(normalize_mean).view(1, 3, 1, 1)
    std_t = torch.tensor(normalize_std).view(1, 3, 1, 1)

    records = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            has_mask = batch["has_mask"].to(device, non_blocking=True)
            paths = batch["image_path"]

            logits, cams = model(x)
            losses = ce_none(logits, y)
            preds = logits.argmax(dim=1)

            for i in range(x.size(0)):
                sample_has_mask = bool(has_mask[i].item())
                cam_i = cams[i, preds[i]].detach().cpu()
                # The loss supervises the TRUE-class CAM, so score that one too:
                # on a misclassified sample the two are different maps.
                cam_true = cams[i, y[i]].detach().cpu()
                mask_i = masks[i].detach().cpu()

                iou = (
                    _compute_cam_mask_iou(cam_i, mask_i, cam_threshold=cam_iou_threshold)
                    if sample_has_mask else None
                )
                iou_true = (
                    _compute_cam_mask_iou(cam_true, mask_i, cam_threshold=cam_iou_threshold)
                    if sample_has_mask else None
                )
                metrics = (
                    _localization_metrics(cam_true, mask_i)
                    if sample_has_mask else {}
                )

                records.append({
                    "loss": losses[i].item(),
                    "iou": iou,
                    "iou_true_label": iou_true,
                    "has_mask": sample_has_mask,
                    "image": x[i].detach().cpu(),
                    "true_label": y[i].item(),
                    "pred_label": preds[i].item(),
                    "cam": cam_i,
                    "cam_true": cam_true,
                    "mask": mask_i,
                    "image_path": paths[i],
                    "metrics": metrics,
                })

    iou_records = [r for r in records if r["has_mask"] and r["iou"] is not None]

    iou_records.sort(key=lambda r: r["iou"])
    worst = iou_records[:top_k]
    best = list(reversed(iou_records[-top_k:]))

    cam_dir = Path(run_dir) / "cam_debug"
    worst_dir = cam_dir / "worst_iou"
    best_dir = cam_dir / "best_iou"
    worst_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    def _save_group(group, out_dir, prefix):
        summary_rows = []
        for rank, r in enumerate(group, start=1):
            img = r["image"].unsqueeze(0) * std_t + mean_t
            img = img.clamp(0, 1).squeeze(0)
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")

            true_name = classes[r["true_label"]]
            pred_name = classes[r["pred_label"]]

            mask_np = r["mask"].numpy()[0]
            boundary = _mask_boundary(mask_np)
            pred_full = _upsample_cam(r["cam"].float(), img_np.shape[0])
            true_full = _upsample_cam(r["cam_true"].float(), img_np.shape[0])
            scale = max(float(np.abs(pred_full).max()), float(np.abs(true_full).max()), 1e-6)
            max_alpha = min(0.5, max(0.2, 0.28 * overlay_alpha_gain))
            pred_heat, pred_overlay, _ = signed_cam_overlay(
                img_np, pred_full, boundary, max_alpha=max_alpha, gamma=overlay_gamma, scale=scale
            )
            true_heat, true_overlay, _ = signed_cam_overlay(
                img_np, true_full, boundary, max_alpha=max_alpha, gamma=overlay_gamma, scale=scale
            )
            mask_tile = np.repeat(
                (mask_np > 0.5).astype(np.uint8)[:, :, None] * 255, 3, axis=2
            )
            tiles = [
                _with_header(img_np, "input"),
                _with_header(mask_tile, "mask (white = object)"),
                _with_header(pred_heat, "pred CAM [+red / 0-white / -blue]"),
                _with_header(pred_overlay, "pred overlay"),
                _with_header(true_heat, "true CAM [+red / 0-white / -blue]"),
                _with_header(true_overlay, "true overlay"),
            ]
            composite = np.concatenate(tiles, axis=1)

            fname = (
                f"{prefix}_{rank:03d}_true-{true_name}_pred-{pred_name}_"
                f"iou-{r['iou']:.3f}_loss-{r['loss']:.3f}.png"
            )

            Image.fromarray(composite).save(out_dir / fname)

            summary_rows.append([
                rank, r["image_path"], true_name, pred_name, r["iou"], r["loss"]
            ])

        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "image_path", "true_label", "pred_label", "iou", "loss"])
            writer.writerows(summary_rows)

    _save_group(worst, worst_dir, "worst")
    _save_group(best, best_dir, "best")

    _save_full_test_report(records, iou_records, classes, cam_dir, cam_iou_threshold)

    print(f"Saved {len(worst)} worst-IoU CAM overlays to {worst_dir}")
    print(f"Saved {len(best)} best-IoU CAM overlays to {best_dir}")
    if iou_records:
        print(
            f"IoU stats over {len(iou_records)} masked samples: "
            f"min={iou_records[0]['iou']:.3f} max={iou_records[-1]['iou']:.3f} "
            f"mean={statistics.mean(r['iou'] for r in iou_records):.3f}"
        )
    else:
        print(
            "IoU stats: no test sample carried a mask, so there is nothing to "
            "score. See diagnostics/mask_audit.md."
        )


_METRIC_COLUMNS = (
    "containment",
    "uniform_containment",
    "floor_containment",
    "mean_pos_inside",
    "mean_pos_outside",
    "mask_coverage_cam_grid",
    "positive_cell_frac",
    "pointing_hit",
    "cam_min",
    "cam_max",
    "signed_inside_mean",
    "signed_outside_mean",
    "abs_peak_negative",
    "cam_all_negative",
)


def _save_full_test_report(records, iou_records, classes, cam_dir, cam_iou_threshold):
    """
    Per-sample dump of the whole test split, plus its aggregate.

    The best/worst-K overlays show the tails; this is the distribution. Without
    it there is no way to tell a run where most samples are fine and twenty are
    terrible from one where every sample is mediocre.
    """
    cam_dir = Path(cam_dir)
    cam_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "image_path", "true_label", "pred_label", "correct",
        "loss", "has_mask", "iou_pred_label", "iou_true_label",
    ] + list(_METRIC_COLUMNS)

    with open(cam_dir / "all_test_samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in records:
            metrics = r.get("metrics") or {}
            writer.writerow([
                r["image_path"],
                classes[r["true_label"]],
                classes[r["pred_label"]],
                int(r["true_label"] == r["pred_label"]),
                r["loss"],
                int(r["has_mask"]),
                r["iou"],
                r.get("iou_true_label"),
            ] + [metrics.get(name) for name in _METRIC_COLUMNS])

    summary = {
        "n_test_samples": len(records),
        "n_with_mask": len(iou_records),
        "accuracy": (
            sum(1 for r in records if r["true_label"] == r["pred_label"]) / len(records)
            if records else None
        ),
        "iou_threshold": cam_iou_threshold,
    }

    if iou_records:
        summary["iou_pred_label_mean"] = statistics.mean(r["iou"] for r in iou_records)
        true_ious = [r["iou_true_label"] for r in iou_records if r.get("iou_true_label") is not None]
        if true_ious:
            summary["iou_true_label_mean"] = statistics.mean(true_ious)

        for name in _METRIC_COLUMNS:
            values = [
                r["metrics"][name] for r in iou_records
                if r.get("metrics") and r["metrics"].get(name) is not None
            ]
            if values:
                summary[f"{name}_mean"] = statistics.mean(values)
                summary[f"{name}_median"] = statistics.median(values)

    with open(cam_dir / "test_localization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if iou_records:
        print(
            "Test localisation | "
            f"containment={summary.get('containment_mean', float('nan')):.4f} "
            f"(flat={summary.get('uniform_containment_mean', float('nan')):.4f} "
            f"floor={summary.get('floor_containment_mean', float('nan')):.4f}) "
            f"pointing={summary.get('pointing_hit_mean', float('nan')):.4f} "
            f"bg_activation={summary.get('mean_pos_outside_mean', float('nan')):.4f} "
            f"signed in/out={summary.get('signed_inside_mean_mean', float('nan')):+.3f}/"
            f"{summary.get('signed_outside_mean_mean', float('nan')):+.3f} "
            f"all_negative={summary.get('cam_all_negative_mean', float('nan')):.3f}"
        )
    print(f"Saved per-sample test dump to {cam_dir / 'all_test_samples.csv'}")


def main(
    model, test_loader, classes, device, run_dir,
    normalize_mean=None, normalize_std=None,
):
    run_cam_debug(
        model=model,
        test_loader=test_loader,
        classes=classes,
        device=device,
        run_dir=run_dir,
        top_k=5,
        cam_iou_threshold=0.5,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )


if __name__ == "__main__":

    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    normalize_mean, normalize_std = configure_normalization(config)

    _, _, test_loader, classes = get_dataloaders(
            data_dir=config["data_dir"],
            image_size=config.get("image_size", 224),
            batch_size=config.get("batch_size", 8),
            val_ratio=config.get("val_ratio", 0.2),
            seed=config.get("seed", 42),
            num_workers=config.get("num_workers", 0),
            mask_dirs=config.get("mask_dirs", {}),
            mask_manifests=config.get("mask_manifests", None),
            test_ratio=config.get("test_ratio", 0.1),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
    )
    num_classes = len(classes)
    device = get_device()
    
    
    model = build_model(config, num_classes).to(device)
    model = load_checkpoint(model, device=device, normalization=(normalize_mean, normalize_std))


    main(
        model=model, test_loader=test_loader, classes=classes, device=device, run_dir="runs",
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

## for run : python -m src.cam_debug
