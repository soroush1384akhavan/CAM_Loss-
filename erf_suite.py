
"""
Unified ERF / CAM analysis suite for the current project structure.

Expected project API
--------------------
- src.data.get_dataloaders(...)
- src.model_factory.build_model(config, num_classes)
- model(x) -> (logits, cams), where cams is [B, C, Hc, Wc]
- test batches are dicts with at least:
    batch["image"], batch["label"]
  and optionally:
    batch["image_path"], batch["adv_image"]

What this script can do
-----------------------
1) probe:
   Measure dataset-averaged effective receptive field (ERF) of every pooled CAM
   cell for one model or a teacher/student pair. Reports centroid, sigma,
   teacher-student offset, overlap, independent-sample estimate, and CAM
   lag-1 autocorrelation.

2) pages:
   Make a PDF with image + CAM(s) + probabilities + ERF maps for every common
   CAM cell.

3) delta:
   If adversarial inputs are available (batch["adv_image"] or --adv-npy),
   measure perturbation spread, reach under each ERF, student/teacher reach
   ratio, and dead-cell rate.

4) all:
   Run all available analyses. If no adversarial input is supplied, delta is
   skipped with a clear message.

Examples
--------
# One model, using the project's default load_checkpoint behavior:
python -m src.erf_suite --mode probe

# Teacher/student checkpoints, same config:
python -m src.erf_suite --mode all \
    --teacher-ckpt runs/teacher/best.pt \
    --student-ckpt runs/student/best.pt

# Separate model configs:
python -m src.erf_suite --mode all \
    --config configs/config.yaml \
    --teacher-config configs/teacher.yaml \
    --student-config configs/student.yaml \
    --teacher-ckpt runs/teacher/best.pt \
    --student-ckpt runs/student/best.pt

# Delta from aligned .npy adversarial inputs:
python -m src.erf_suite --mode delta \
    --teacher-ckpt runs/teacher/best.pt \
    --student-ckpt runs/student/best.pt \
    --adv-npy runs/adversarial.npy \
    --adv-space input

Notes
-----
- --adv-npy must be aligned with test-loader order. Supported layouts:
    [N,C,H,W] or [N,H,W,C].
- --adv-space input: adversarial array is in the same normalized tensor space
  fed to the model.
- --adv-space pixel: adversarial array is in [0,1] pixel space.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from src.data import get_dataloaders
from src.model_factory import build_model
from src.utils import get_device, load_checkpoint


# ---------------------------------------------------------------------------
# Config / model helpers
# ---------------------------------------------------------------------------

def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(d: dict, path: str, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def infer_norm(config: dict):
    mean = (
        config.get("normalize_mean")
        or deep_get(config, "data.normalize_mean")
        or (0.485, 0.456, 0.406)
    )
    std = (
        config.get("normalize_std")
        or deep_get(config, "data.normalize_std")
        or (0.229, 0.224, 0.225)
    )
    return tuple(float(x) for x in mean), tuple(float(x) for x in std)


def build_test_loader(config: dict):
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
    )
    return test_loader, classes


def _state_dict_candidates(obj):
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "student", "teacher", "net"):
            if key in obj and isinstance(obj[key], dict):
                yield obj[key]
        yield obj
    else:
        yield obj


def _clean_state_dict(state: dict, role: str):
    prefixes = ["module.", "model."]
    if role == "student":
        prefixes += ["student."]
    elif role == "teacher":
        prefixes += ["teacher."]

    out = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        out[k] = v
    return out


def load_explicit_checkpoint(model, path: str, device, role: str):
    ck = torch.load(path, map_location=device)
    model_keys = set(model.state_dict().keys())

    best = None
    best_score = -1
    for cand in _state_dict_candidates(ck):
        if not isinstance(cand, dict):
            continue
        clean = _clean_state_dict(cand, role)
        score = len(model_keys.intersection(clean.keys()))
        if score > best_score:
            best = clean
            best_score = score

    if best is None:
        raise RuntimeError(f"Could not find a state_dict in checkpoint: {path}")

    missing, unexpected = model.load_state_dict(best, strict=False)
    matched = len(model_keys) - len(missing)
    if matched == 0:
        raise RuntimeError(
            f"Checkpoint {path} did not match the built {role} model at all."
        )

    print(
        f"{role:8s}: loaded {path} | matched={matched}/{len(model_keys)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    return model


def build_one_model(
    config: dict,
    num_classes: int,
    device,
    ckpt: Optional[str],
    role: str,
    allow_default_loader: bool = False,
):
    model = build_model(config, num_classes).to(device)

    if ckpt:
        model = load_explicit_checkpoint(model, ckpt, device, role)
    elif allow_default_loader:
        # This mirrors the user's existing cam_debug.py behavior.
        try:
            model = load_checkpoint(model, device=device)
            print(f"{role:8s}: loaded via src.utils.load_checkpoint default behavior")
        except TypeError:
            model = load_checkpoint(model, device=str(device))
            print(f"{role:8s}: loaded via src.utils.load_checkpoint default behavior")
    else:
        print(f"{role:8s}: no checkpoint supplied; using built model weights")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_logits_cams(output):
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        logits, cams = output[0], output[1]
    elif isinstance(output, dict):
        logits = output.get("logits")
        cams = output.get("cams", output.get("cam"))
    else:
        raise TypeError(
            "Expected model(x) -> (logits, cams) or dict with logits/cams."
        )

    if logits is None or cams is None:
        raise ValueError("Could not extract logits/cams from model output.")
    if cams.dim() != 4:
        raise ValueError(f"Expected cams [B,C,H,W], got shape {tuple(cams.shape)}")
    return logits, cams


# ---------------------------------------------------------------------------
# Batch / image helpers
# ---------------------------------------------------------------------------

def get_batch_paths(batch, batch_size: int, start_index: int):
    paths = batch.get("image_path")
    if paths is None:
        return [f"sample_{start_index+i:06d}" for i in range(batch_size)]
    return [str(x) for x in paths]


def denormalize(x: torch.Tensor, mean, std):
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x * std_t + mean_t).clamp(0, 1)


def tensor_to_rgb(x: torch.Tensor, mean, std):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x01 = denormalize(x, mean, std)[0].detach().cpu()
    return x01.permute(1, 2, 0).numpy()


def collect_test_subset(test_loader, max_images: int, device):
    xs, ys, paths, advs = [], [], [], []
    have_adv = True
    seen = 0

    for batch in test_loader:
        x = batch["image"]
        y = batch["label"]
        b = x.size(0)
        take = min(b, max_images - seen)
        if take <= 0:
            break

        xs.append(x[:take].cpu())
        ys.append(y[:take].cpu())
        paths.extend(get_batch_paths(batch, b, seen)[:take])

        if "adv_image" in batch:
            advs.append(batch["adv_image"][:take].cpu())
        else:
            have_adv = False

        seen += take
        if seen >= max_images:
            break

    if not xs:
        raise RuntimeError("Test loader yielded no images.")

    x = torch.cat(xs, 0)
    y = torch.cat(ys, 0)
    adv = torch.cat(advs, 0) if have_adv and advs else None
    return x, y, paths, adv


def load_adv_npy(path: str, n: int, expected_chw: Tuple[int, int, int]):
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if "adv" in obj:
            arr = obj["adv"]
        elif "images" in obj:
            arr = obj["images"]
        else:
            first = obj.files[0]
            arr = obj[first]
    else:
        arr = obj

    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"--adv-npy must be 4-D; got {arr.shape}")

    C, H, W = expected_chw
    if arr.shape[1:] == (C, H, W):
        pass
    elif arr.shape[1:] == (H, W, C):
        arr = arr.transpose(0, 3, 1, 2)
    else:
        raise ValueError(
            f"Adversarial shape {arr.shape} does not match expected "
            f"[N,{C},{H},{W}] or [N,{H},{W},{C}]"
        )

    if len(arr) < n:
        raise ValueError(f"--adv-npy has {len(arr)} samples but {n} are required.")
    return torch.from_numpy(np.ascontiguousarray(arr[:n])).float()


# ---------------------------------------------------------------------------
# ERF core
# ---------------------------------------------------------------------------

def choose_reference_classes(
    reference_model,
    x: torch.Tensor,
    y: torch.Tensor,
    target: str,
):
    with torch.no_grad():
        logits, _ = extract_logits_cams(reference_model(x))
        top2 = logits.topk(min(2, logits.shape[1]), dim=1).indices

    if target == "true":
        return y, None
    if target == "top1":
        return top2[:, 0], None
    if target == "runnerup":
        if top2.shape[1] < 2:
            return top2[:, 0], None
        return top2[:, 1], None
    if target == "margin":
        if top2.shape[1] < 2:
            return top2[:, 0], None
        return top2[:, 1], top2[:, 0]
    raise ValueError(target)


def infer_common_res(models: Dict[str, torch.nn.Module], x: torch.Tensor, user_res: int):
    if user_res > 0:
        return user_res

    sizes = []
    with torch.no_grad():
        for m in models.values():
            _, cams = extract_logits_cams(m(x[:1]))
            sizes.append((cams.shape[-2], cams.shape[-1]))

    if any(h != w for h, w in sizes):
        r = min(min(h, w) for h, w in sizes)
    else:
        r = min(h for h, _ in sizes)

    print(f"common CAM grid: {r}x{r} (native grids: {sizes})")
    return int(r)


def erf_all_cells(
    model,
    x: torch.Tensor,
    pos_cls: torch.Tensor,
    res: int,
    neg_cls: Optional[torch.Tensor] = None,
):
    """
    Returns [res, res, B, H, W]:
      | d pooled_CAM[target, i, j] / d input |
    summed across input channels.
    """
    B, _, H, W = x.shape
    out = np.zeros((res, res, B, H, W), dtype=np.float32)
    idx = torch.arange(B, device=x.device)

    for i in range(res):
        for j in range(res):
            xin = x.detach().clone().requires_grad_(True)
            _, cams = extract_logits_cams(model(xin))
            pooled = F.adaptive_avg_pool2d(cams, (res, res))
            score = pooled[idx, pos_cls, i, j]
            if neg_cls is not None:
                score = score - pooled[idx, neg_cls, i, j]
            g, = torch.autograd.grad(score.sum(), xin, retain_graph=False)
            out[i, j] = g.abs().sum(1).detach().cpu().numpy()

    return out


def norm_map(m: np.ndarray):
    s = float(m.sum())
    if s <= 0:
        return np.full_like(m, 1.0 / m.size, dtype=np.float64)
    return m / s


def erf_stats(m: np.ndarray):
    p = norm_map(m.astype(np.float64))
    H, W = p.shape
    ys, xs = np.mgrid[0:H, 0:W]
    cx = float((p * xs).sum())
    cy = float((p * ys).sum())
    var = float((p * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum()) / 2.0
    sigma = math.sqrt(max(var, 0.0))
    return cx, cy, sigma


def gaussian_overlap(c1, s1, c2, s2):
    d2 = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2
    v = s1 * s1 + s2 * s2
    if v <= 1e-12:
        return 1.0 if d2 <= 1e-12 else 0.0
    return float(math.exp(-d2 / (2 * v)) * (2 * s1 * s2 / v))


def cam_autocorr(model, x_cpu: torch.Tensor, device, batch_size: int):
    num = 0.0
    den = 0.0

    for st in range(0, len(x_cpu), batch_size):
        x = x_cpu[st:st+batch_size].to(device)
        with torch.no_grad():
            logits, cams = extract_logits_cams(model(x))
            idx = torch.arange(len(x), device=device)
            c = logits.argmax(1)
            a = cams[idx, c].float()
            a = a - a.mean((1, 2), keepdim=True)

            if a.shape[-1] > 1:
                num += float((a[:, :, :-1] * a[:, :, 1:]).sum())
                den += float((a[:, :, :-1] ** 2).sum())
            if a.shape[-2] > 1:
                num += float((a[:, :-1, :] * a[:, 1:, :]).sum())
                den += float((a[:, :-1, :] ** 2).sum())

    return num / max(den, 1e-12)


def measure_dataset_erf(
    models: Dict[str, torch.nn.Module],
    reference_name: str,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    device,
    res: int,
    batch_size: int,
    target: str,
):
    H, W = x_cpu.shape[-2:]
    acc = {
        name: np.zeros((res, res, H, W), dtype=np.float64)
        for name in models
    }
    seen = 0

    for st in range(0, len(x_cpu), batch_size):
        xb = x_cpu[st:st+batch_size].to(device)
        yb = y_cpu[st:st+batch_size].to(device)

        pos, neg = choose_reference_classes(
            models[reference_name], xb, yb, target
        )

        for name, model in models.items():
            e = erf_all_cells(model, xb, pos, res, neg)
            acc[name] += e.sum(axis=2)

        seen += len(xb)
        print(f"ERF averaging: {seen}/{len(x_cpu)}", end="\r")

    print()
    return {name: v / max(seen, 1) for name, v in acc.items()}


def build_probe_summary(
    erfs: Dict[str, np.ndarray],
    models: Dict[str, torch.nn.Module],
    x_cpu: torch.Tensor,
    device,
    batch_size: int,
):
    names = list(erfs.keys())
    res = erfs[names[0]].shape[0]
    H, W = erfs[names[0]].shape[-2:]

    summary = {
        "image_height": H,
        "image_width": W,
        "cam_grid": res,
        "models": {},
    }

    per_cell_rows = []
    stats_by_model = {}

    for name in names:
        s = [
            erf_stats(erfs[name][i, j])
            for i in range(res)
            for j in range(res)
        ]
        stats_by_model[name] = s
        sigma_mean = float(np.mean([v[2] for v in s]))
        indep = (H * W) / max(4.0 * sigma_mean * sigma_mean, 1e-12)
        ac = cam_autocorr(models[name], x_cpu, device, batch_size)

        summary["models"][name] = {
            "mean_sigma_px": sigma_mean,
            "mean_sigma_fraction_of_sqrt_area": sigma_mean / math.sqrt(H * W),
            "effective_independent_samples": indep,
            "cam_lag1_autocorrelation": ac,
        }

    if len(names) == 2:
        a_name, b_name = names[0], names[1]
        offsets, overlaps = [], []

        for k in range(res * res):
            a = stats_by_model[a_name][k]
            b = stats_by_model[b_name][k]
            offset = math.hypot(a[0] - b[0], a[1] - b[1])
            ov = gaussian_overlap(a[:2], a[2], b[:2], b[2])
            offsets.append(offset)
            overlaps.append(ov)
            per_cell_rows.append({
                "cell_row": k // res,
                "cell_col": k % res,
                f"{a_name}_cx": a[0],
                f"{a_name}_cy": a[1],
                f"{a_name}_sigma": a[2],
                f"{b_name}_cx": b[0],
                f"{b_name}_cy": b[1],
                f"{b_name}_sigma": b[2],
                "center_offset_px": offset,
                "gaussian_overlap": ov,
            })

        summary["comparison"] = {
            "model_a": a_name,
            "model_b": b_name,
            "mean_center_offset_px": float(np.mean(offsets)),
            "mean_gaussian_overlap": float(np.mean(overlaps)),
        }

    return summary, per_cell_rows


def save_probe_outputs(out_dir: Path, erfs, summary, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "erf_maps.npz", **erfs)

    with open(out_dir / "erf_probe_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if rows:
        with open(out_dir / "erf_per_cell.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n=== ERF PROBE ===")
    for name, v in summary["models"].items():
        print(
            f"{name:8s} | sigma={v['mean_sigma_px']:.2f}px "
            f"| independent≈{v['effective_independent_samples']:.2f} "
            f"| CAM autocorr={v['cam_lag1_autocorrelation']:.3f}"
        )
    if "comparison" in summary:
        c = summary["comparison"]
        print(
            f"comparison | mean offset={c['mean_center_offset_px']:.2f}px "
            f"| mean overlap={c['mean_gaussian_overlap']:.3f}"
        )
    print(f"saved: {out_dir / 'erf_probe_summary.json'}")
    print(f"saved: {out_dir / 'erf_maps.npz'}")


# ---------------------------------------------------------------------------
# PDF pages
# ---------------------------------------------------------------------------

def class_name(classes, idx):
    idx = int(idx)
    return classes[idx] if 0 <= idx < len(classes) else str(idx)


def overlay_cam(ax, img, cam, title):
    ax.imshow(img)
    cam = np.asarray(cam, dtype=np.float32)
    cmin, cmax = float(cam.min()), float(cam.max())
    if abs(cmax - cmin) < 1e-12:
        norm = np.zeros_like(cam)
    else:
        norm = (cam - cmin) / (cmax - cmin)
    ax.imshow(
        norm,
        cmap="jet",
        alpha=0.5,
        interpolation="bilinear",
        extent=(-0.5, img.shape[1]-0.5, img.shape[0]-0.5, -0.5),
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def overlay_erf(ax, img, erf, vmax=None):
    ax.imshow(img)
    e = np.asarray(erf, dtype=np.float32)
    vmax = float(e.max()) if vmax is None else float(vmax)
    n = np.clip(e / max(vmax, 1e-12), 0, 1)
    rgba = plt.cm.inferno(n)
    rgba[..., 3] = 0.10 + 0.85 * n
    ax.imshow(rgba, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])


def get_cam_and_probs(model, x, res):
    with torch.no_grad():
        logits, cams = extract_logits_cams(model(x))
        probs = torch.softmax(logits, dim=1)
        cams = F.adaptive_avg_pool2d(cams, (res, res))
    return logits, probs, cams


def make_erf_pages(
    pdf_path: Path,
    models: Dict[str, torch.nn.Module],
    reference_name: str,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    classes,
    mean,
    std,
    device,
    res: int,
    avg_erfs: Dict[str, np.ndarray],
    n_pages: int,
    target: str,
    page_erf_mode: str,
):
    names = list(models.keys())
    H, W = x_cpu.shape[-2:]
    n_pages = min(n_pages, len(x_cpu))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for b in range(n_pages):
            x = x_cpu[b:b+1].to(device)
            y = y_cpu[b:b+1].to(device)
            img = tensor_to_rgb(x, mean, std)

            ref_pos, ref_neg = choose_reference_classes(
                models[reference_name], x, y, target
            )

            out = {}
            for name, model in models.items():
                logits, probs, cams = get_cam_and_probs(model, x, res)
                out[name] = {
                    "logits": logits[0].detach().cpu(),
                    "probs": probs[0].detach().cpu(),
                    "cams": cams[0].detach().cpu(),
                    "pred": int(logits[0].argmax()),
                }

            if page_erf_mode == "per-image":
                page_erfs = {
                    name: erf_all_cells(model, x, ref_pos, res, ref_neg)[:, :, 0]
                    for name, model in models.items()
                }
            else:
                page_erfs = avg_erfs

            if len(names) == 1:
                fig = plt.figure(figsize=(max(10, res * 1.3), 4 + res * 1.25))
                gs = fig.add_gridspec(
                    res + 1, res + 2,
                    height_ratios=[1.5] + [1] * res,
                    hspace=0.20, wspace=0.12
                )
            else:
                fig = plt.figure(figsize=(max(14, res * 2.4), 4 + res * 1.25))
                gs = fig.add_gridspec(
                    res + 1, 2 * res + 3,
                    height_ratios=[1.5] + [1] * res,
                    hspace=0.20, wspace=0.12
                )

            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(img)
            ax.set_title(f"input\ntrue: {class_name(classes, y.item())}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            # CAM panels
            for mi, name in enumerate(names):
                pred = out[name]["pred"]
                cam = out[name]["cams"][pred].numpy()
                ax = fig.add_subplot(gs[0, 1 + mi])
                overlay_cam(
                    ax,
                    img,
                    cam,
                    f"{name} CAM\npred={class_name(classes, pred)} "
                    f"p={float(out[name]['probs'][pred]):.3f}",
                )

            # top-k probabilities from reference classes
            ax = fig.add_subplot(gs[0, 1 + len(names):])
            ref_probs = out[reference_name]["probs"]
            topk = torch.topk(ref_probs, k=min(5, len(ref_probs))).indices.tolist()
            yy = np.arange(len(topk))
            width = 0.8 / len(names)
            for mi, name in enumerate(names):
                vals = [float(out[name]["probs"][k]) for k in topk]
                ax.barh(yy + (mi - (len(names)-1)/2)*width, vals, height=width*0.9, label=name)
            ax.set_yticks(yy)
            ax.set_yticklabels([class_name(classes, k)[:18] for k in topk], fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_title(f"top-5 by {reference_name}", fontsize=9)
            ax.legend(fontsize=7)

            # ERF grids
            cell_h = H / res
            cell_w = W / res
            if len(names) == 1:
                starts = {names[0]: 0}
            else:
                starts = {names[0]: 0, names[1]: res + 1}

            vmax_by_model = {
                name: float(page_erfs[name].max()) for name in names
            }

            for name in names:
                start_col = starts[name]
                for i in range(res):
                    for j in range(res):
                        ax = fig.add_subplot(gs[i+1, start_col+j])
                        overlay_erf(
                            ax,
                            img,
                            page_erfs[name][i, j],
                            vmax=vmax_by_model[name],
                        )
                        ax.add_patch(Rectangle(
                            (j * cell_w - 0.5, i * cell_h - 0.5),
                            cell_w,
                            cell_h,
                            fill=False,
                            linestyle="--",
                            linewidth=0.8,
                        ))
                        cx, cy, _ = erf_stats(page_erfs[name][i, j])
                        ax.plot(cx, cy, "o", ms=3.5, mfc="white", mec="black", mew=0.6)
                        if i == 0:
                            ax.set_title(f"{name} c{j}", fontsize=7)
                        if j == 0:
                            ax.set_ylabel(f"r{i}", fontsize=7)

                        if len(names) == 2 and name == names[1]:
                            a = erf_stats(page_erfs[names[0]][i, j])
                            d = math.hypot(cx - a[0], cy - a[1])
                            ax.set_xlabel(f"{d:.1f}px", fontsize=6)

            fig.suptitle(
                f"sample {b} | ERF target={target} | page ERF={page_erf_mode}",
                fontsize=12,
            )
            pdf.savefig(fig, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"PDF pages: {b+1}/{n_pages}", end="\r")

    print()
    print(f"saved: {pdf_path}")


# ---------------------------------------------------------------------------
# Delta / ERF overlap
# ---------------------------------------------------------------------------

def delta_energy_map(
    x_input: torch.Tensor,
    adv: torch.Tensor,
    mean,
    std,
    adv_space: str,
):
    """
    Returns spatial perturbation energy [N,H,W].
    """
    if adv_space == "input":
        nat = x_input.float()
        a = adv.float()
    elif adv_space == "pixel":
        mean_t = torch.tensor(mean).view(1, -1, 1, 1)
        std_t = torch.tensor(std).view(1, -1, 1, 1)
        nat = (x_input.float() * std_t + mean_t)
        a = adv.float()
    else:
        raise ValueError(adv_space)

    return ((a - nat) ** 2).sum(1)


def run_delta_overlap(
    erfs: Dict[str, np.ndarray],
    x_cpu: torch.Tensor,
    adv_cpu: torch.Tensor,
    mean,
    std,
    adv_space: str,
    out_dir: Path,
):
    names = list(erfs.keys())
    res = erfs[names[0]].shape[0]
    H, W = x_cpu.shape[-2:]
    P = H * W

    E = {
        name: np.stack([
            [norm_map(erfs[name][i, j]) for j in range(res)]
            for i in range(res)
        ])
        for name in names
    }

    d2 = delta_energy_map(x_cpu, adv_cpu, mean, std, adv_space).numpy()

    spread = []
    reaches = {name: [] for name in names}

    for k in range(len(d2)):
        e = norm_map(d2[k].astype(np.float64))
        spread.append(1.0 / (P * float((e ** 2).sum())))

        for name in names:
            reaches[name].append([
                float((E[name][i, j] * e).sum() * P)
                for i in range(res)
                for j in range(res)
            ])

    spread = np.asarray(spread, dtype=np.float64)
    reaches = {k: np.asarray(v, dtype=np.float64) for k, v in reaches.items()}

    summary = {
        "n_images": len(d2),
        "spread_mean": float(spread.mean()),
        "effective_pixel_fraction_mean": float(spread.mean()),
        "reach_mean": {name: float(v.mean()) for name, v in reaches.items()},
    }

    per_cell_rows = []

    if len(names) == 2:
        teacher = names[0]
        student = names[1]
        ratio = reaches[student] / np.maximum(reaches[teacher], 1e-9)
        dead = reaches[student] < 0.5

        summary.update({
            "reference_model": teacher,
            "comparison_model": student,
            "student_teacher_reach_ratio_mean": float(ratio.mean()),
            "dead_cell_fraction_reach_lt_0_5": float(dead.mean()),
        })

        # ERF centre offsets
        offsets = []
        for i in range(res):
            for j in range(res):
                a = erf_stats(erfs[teacher][i, j])
                b = erf_stats(erfs[student][i, j])
                offsets.append(math.hypot(a[0]-b[0], a[1]-b[1]))
        offsets = np.asarray(offsets)

        ratio_by_cell = ratio.mean(0)
        if len(offsets) > 1 and np.std(offsets) > 1e-12 and np.std(ratio_by_cell) > 1e-12:
            corr = float(np.corrcoef(offsets, ratio_by_cell)[0, 1])
        else:
            corr = None
        summary["corr_center_offset_vs_reach_ratio"] = corr

        for k in range(res * res):
            per_cell_rows.append({
                "cell_row": k // res,
                "cell_col": k % res,
                "center_offset_px": float(offsets[k]),
                f"{teacher}_reach_mean": float(reaches[teacher][:, k].mean()),
                f"{student}_reach_mean": float(reaches[student][:, k].mean()),
                "student_teacher_reach_ratio_mean": float(ratio[:, k].mean()),
                "dead_fraction": float(dead[:, k].mean()),
            })

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "delta_erf_overlap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if per_cell_rows:
        with open(out_dir / "delta_erf_per_cell.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_cell_rows[0].keys()))
            w.writeheader()
            w.writerows(per_cell_rows)

    print("\n=== DELTA / ERF OVERLAP ===")
    print(f"spread mean: {summary['spread_mean']:.3f}")
    for name, val in summary["reach_mean"].items():
        print(f"reach under {name:8s}: {val:.3f}")

    if len(names) == 2:
        print(
            "student/teacher reach ratio: "
            f"{summary['student_teacher_reach_ratio_mean']:.3f}"
        )
        print(
            "dead cells (student reach < 0.5): "
            f"{summary['dead_cell_fraction_reach_lt_0_5']:.1%}"
        )
        if summary["corr_center_offset_vs_reach_ratio"] is not None:
            print(
                "corr(offset, reach ratio): "
                f"{summary['corr_center_offset_vs_reach_ratio']:+.3f}"
            )

    print(f"saved: {out_dir / 'delta_erf_overlap_summary.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument(
        "--mode",
        choices=["all", "probe", "pages", "delta"],
        default="all",
    )
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--teacher-config", default="")
    ap.add_argument("--student-config", default="")
    ap.add_argument("--teacher-ckpt", default="")
    ap.add_argument("--student-ckpt", default="")
    ap.add_argument(
        "--single",
        action="store_true",
        help="Analyze only one model. Uses --student-ckpt if given, otherwise "
             "the project's default load_checkpoint behavior.",
    )
    ap.add_argument(
        "--target",
        choices=["true", "top1", "runnerup", "margin"],
        default="top1",
        help="ERF target class. With two models, class ids are always chosen "
             "from the first/reference model so both models are compared on "
             "the same target.",
    )
    ap.add_argument(
        "--res",
        type=int,
        default=0,
        help="Common CAM grid. 0 = infer a shared square resolution.",
    )
    ap.add_argument("--erf-images", type=int, default=64)
    ap.add_argument("--erf-batch", type=int, default=8)
    ap.add_argument("--page-images", type=int, default=6)
    ap.add_argument(
        "--page-erf",
        choices=["avg", "per-image"],
        default="avg",
        help="Use dataset-averaged or per-image ERF maps in PDF pages.",
    )
    ap.add_argument("--adv-npy", default="")
    ap.add_argument(
        "--adv-space",
        choices=["input", "pixel"],
        default="input",
    )
    ap.add_argument("--out", default="runs/erf_suite")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = read_yaml(args.config)
    teacher_cfg = read_yaml(args.teacher_config) if args.teacher_config else base_cfg
    student_cfg = read_yaml(args.student_config) if args.student_config else base_cfg

    test_loader, classes = build_test_loader(base_cfg)
    num_classes = len(classes)
    device = torch.device(get_device())
    mean, std = infer_norm(base_cfg)

    # Collect a stable subset once; all analyses use the same images/order.
    need_n = max(args.erf_images, args.page_images)
    x_cpu, y_cpu, paths, adv_from_loader = collect_test_subset(
        test_loader, need_n, device
    )

    # Build model(s).
    models = {}

    if args.single:
        ckpt = args.student_ckpt or args.teacher_ckpt
        models["model"] = build_one_model(
            student_cfg,
            num_classes,
            device,
            ckpt if ckpt else None,
            "model",
            allow_default_loader=(not ckpt),
        )
        reference_name = "model"
    else:
        # If both checkpoints are absent, fall back to one default-loaded model.
        if not args.teacher_ckpt and not args.student_ckpt:
            models["model"] = build_one_model(
                base_cfg,
                num_classes,
                device,
                None,
                "model",
                allow_default_loader=True,
            )
            reference_name = "model"
            print(
                "NOTE: no teacher/student checkpoints supplied, so running as a "
                "single-model ERF analysis."
            )
        else:
            if args.teacher_ckpt:
                models["teacher"] = build_one_model(
                    teacher_cfg,
                    num_classes,
                    device,
                    args.teacher_ckpt,
                    "teacher",
                )
            if args.student_ckpt:
                models["student"] = build_one_model(
                    student_cfg,
                    num_classes,
                    device,
                    args.student_ckpt,
                    "student",
                )

            reference_name = "teacher" if "teacher" in models else list(models.keys())[0]

    # Infer common CAM grid from a real sample.
    res = infer_common_res(
        models,
        x_cpu[:1].to(device),
        args.res,
    )

    # ERF average is required by all three analyses.
    n_erf = min(args.erf_images, len(x_cpu))
    erfs = measure_dataset_erf(
        models=models,
        reference_name=reference_name,
        x_cpu=x_cpu[:n_erf],
        y_cpu=y_cpu[:n_erf],
        device=device,
        res=res,
        batch_size=args.erf_batch,
        target=args.target,
    )

    summary, rows = build_probe_summary(
        erfs,
        models,
        x_cpu[:n_erf],
        device,
        args.erf_batch,
    )

    if args.mode in ("all", "probe"):
        save_probe_outputs(out_dir, erfs, summary, rows)

    if args.mode in ("all", "pages"):
        make_erf_pages(
            pdf_path=out_dir / "erf_pages.pdf",
            models=models,
            reference_name=reference_name,
            x_cpu=x_cpu,
            y_cpu=y_cpu,
            classes=classes,
            mean=mean,
            std=std,
            device=device,
            res=res,
            avg_erfs=erfs,
            n_pages=args.page_images,
            target=args.target,
            page_erf_mode=args.page_erf,
        )

    if args.mode in ("all", "delta"):
        adv_cpu = None
        n_delta = min(args.erf_images, len(x_cpu))

        if args.adv_npy:
            C, H, W = x_cpu.shape[1:]
            adv_cpu = load_adv_npy(args.adv_npy, n_delta, (C, H, W))
        elif adv_from_loader is not None:
            adv_cpu = adv_from_loader[:n_delta]

        if adv_cpu is None:
            msg = (
                "DELTA skipped: no adversarial input found. Supply --adv-npy "
                "or return batch['adv_image'] from the test loader."
            )
            if args.mode == "delta":
                raise SystemExit(msg)
            print(msg)
        else:
            run_delta_overlap(
                erfs=erfs,
                x_cpu=x_cpu[:n_delta],
                adv_cpu=adv_cpu[:n_delta],
                mean=mean,
                std=std,
                adv_space=args.adv_space,
                out_dir=out_dir,
            )

    print("\nDone.")
    print(f"output directory: {out_dir}")


if __name__ == "__main__":
    main()
