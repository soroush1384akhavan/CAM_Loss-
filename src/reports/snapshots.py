"""
Per-epoch CAM snapshots on a fixed set of validation images.

The end-of-run `cam_debug` overlays show where the CAM landed; they cannot show
how it got there. These snapshots pin the same images across every epoch, so
the run produces a flip-book: mask drifting away from the object, the CAM
collapsing to a single cell, the background quietly staying positive.

Each row is:

    image | mask | predicted CAM | predicted overlay | true CAM | true overlay

The diverging CAM maps keep the sign visible: red is positive, white is zero,
and blue is negative. ReLU-based containment losses ignore negative cells,
while spatial-softmax containment remains active on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from src.reports.cam_viz import signed_cam_overlay
from src.normalization import (
    IMAGENET_MEAN, IMAGENET_STD, denormalize_images, normalization_for_loader,
)

_CAPTION_HEIGHT = 46
_HEADER_HEIGHT = 22


def _get_font():
    try:
        from PIL import ImageFont
        try:
            return ImageFont.load_default(size=14)
        except TypeError:  # Pillow < 10.1
            return ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return None


def _denormalize(image: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    img = denormalize_images(image.detach().cpu(), mean, std)
    img = img.clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _upsample(cam: torch.Tensor, size: int) -> np.ndarray:
    resized = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0).float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].cpu().numpy()


def _mask_boundary(binary: np.ndarray) -> np.ndarray:
    m = binary > 0.5
    eroded = m.copy()
    eroded[1:, :] &= m[:-1, :]
    eroded[:-1, :] &= m[1:, :]
    eroded[:, 1:] &= m[:, :-1]
    eroded[:, :-1] &= m[:, 1:]
    return m & ~eroded


def _with_header(tile: np.ndarray, title: str, font) -> np.ndarray:
    header = Image.new("RGB", (tile.shape[1], _HEADER_HEIGHT), (18, 18, 18))
    ImageDraw.Draw(header).text((4, 4), title, fill=(235, 235, 235), font=font)
    return np.concatenate([np.asarray(header), tile], axis=0)


class CAMSnapshotter:
    """Holds a fixed sample of images and re-renders their CAMs every epoch."""

    def __init__(
        self,
        loader,
        device: str,
        classes,
        run_dir,
        num_samples: int = 8,
        seed: int = 42,
        max_scan_batches: int = 50,
        overlay_gamma: float = 0.6,
        overlay_alpha_gain: float = 1.5,
        normalize_mean=None,
        normalize_std=None,
        cam_loss_type="containment",
    ):
        self.device = device
        self.classes = list(classes)
        self.out_dir = Path(run_dir) / "diagnostics" / "snapshots"
        self.font = _get_font()
        self.samples: List[dict] = []

        self.overlay_gamma = overlay_gamma
        self.overlay_alpha_gain = overlay_alpha_gain
        self.normalize_mean, self.normalize_std = normalization_for_loader(
            loader, normalize_mean, normalize_std,
        )
        self.cam_loss_type = str(cam_loss_type).lower()

        if loader is None or num_samples <= 0:
            return

        pool: List[dict] = []
        fallback: List[dict] = []

        for batch_index, batch in enumerate(loader):
            if batch_index >= max_scan_batches or len(pool) >= num_samples * 4:
                break
            for i in range(batch["image"].shape[0]):
                entry = {
                    "image": batch["image"][i].clone(),
                    "mask": batch["mask"][i].clone(),
                    "label": int(batch["label"][i].item()),
                    "path": str(batch["image_path"][i]),
                }
                if float(batch["has_mask"][i].item()) > 0.5:
                    pool.append(entry)
                elif len(fallback) < num_samples:
                    fallback.append(entry)

        if not pool:
            print(
                "[reports] snapshots: no masked validation samples found; "
                "falling back to unmasked images (mask panels will be empty)."
            )
            pool = fallback

        if not pool:
            return

        # Spread the picks over the scanned pool instead of taking a
        # contiguous prefix, which would be one or two batches only.
        step = max(1, len(pool) // num_samples)
        self.samples = pool[::step][:num_samples]

    def available(self) -> bool:
        return bool(self.samples)

    @torch.no_grad()
    def capture(self, model, epoch: int) -> Optional[str]:
        if not self.samples:
            return None

        was_training = model.training
        model.eval()

        try:
            images = torch.stack([s["image"] for s in self.samples]).to(self.device)
            labels = torch.tensor(
                [s["label"] for s in self.samples], device=self.device, dtype=torch.long
            )
            logits, cams = model(images)
            cams = cams.detach().float()
            target_cams = cams[torch.arange(len(self.samples), device=cams.device), labels]
            pred_labels = logits.argmax(dim=1)
            pred_cams = cams[torch.arange(len(self.samples), device=cams.device), pred_labels]
        finally:
            model.train(was_training)

        rows = [
            self._render_row(
                sample, target_cams[i].cpu(), pred_cams[i].cpu(), int(pred_labels[i].item())
            )
            for i, sample in enumerate(self.samples)
        ]

        width = max(row.shape[1] for row in rows)
        padded = [
            np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0)), constant_values=18)
            for row in rows
        ]
        grid = np.concatenate(padded, axis=0)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"epoch_{epoch:03d}.png"
        Image.fromarray(grid).save(path)
        return str(path)

    def _render_row(
        self, sample: dict, true_cam: torch.Tensor, pred_cam: torch.Tensor, pred_label: int
    ) -> np.ndarray:
        image_np = _denormalize(sample["image"], self.normalize_mean, self.normalize_std)
        size = image_np.shape[0]

        mask_np = sample["mask"][0].cpu().numpy()
        binary = (mask_np > 0.5).astype(np.float32)

        boundary = _mask_boundary(binary)
        true_full = _upsample(true_cam, size)
        pred_full = _upsample(pred_cam, size)
        scale = max(float(np.abs(true_full).max()), float(np.abs(pred_full).max()), 1e-6)
        max_alpha = min(0.5, max(0.2, 0.28 * self.overlay_alpha_gain))
        true_heat, true_overlay, _ = signed_cam_overlay(
            image_np, true_full, boundary, max_alpha=max_alpha, gamma=self.overlay_gamma, scale=scale
        )
        pred_heat, pred_overlay, _ = signed_cam_overlay(
            image_np, pred_full, boundary, max_alpha=max_alpha, gamma=self.overlay_gamma, scale=scale
        )

        mask_tile = np.repeat((binary * 255).astype(np.uint8)[:, :, None], 3, axis=2)

        tiles = [
            _with_header(image_np, "input", self.font),
            _with_header(mask_tile, "mask (white = object)", self.font),
            _with_header(pred_heat, "pred CAM [+red / 0-white / -blue]", self.font),
            _with_header(pred_overlay, "pred overlay", self.font),
            _with_header(true_heat, "true CAM [+red / 0-white / -blue]", self.font),
            _with_header(true_overlay, "true overlay", self.font),
        ]
        row = np.concatenate(tiles, axis=1)
        return np.concatenate([
            row, self._caption(sample, true_cam, pred_cam, pred_label, binary, row.shape[1])
        ], axis=0)

    def _caption(
        self,
        sample: dict,
        true_cam: torch.Tensor,
        pred_cam: torch.Tensor,
        pred_label: int,
        binary: np.ndarray,
        width: int,
    ) -> np.ndarray:
        hc, wc = true_cam.shape
        mask_cam = F.interpolate(
            torch.from_numpy(binary).unsqueeze(0).unsqueeze(0).float(),
            size=(hc, wc),
            mode="area",
        )[0, 0]

        inside = mask_cam
        outside = 1.0 - mask_cam
        positive = F.relu(true_cam)

        energy_out = float((positive * outside).sum())
        energy_in = float((positive * inside).sum())
        containment = energy_out / (energy_in + energy_out + 1e-6)
        uniform = float(outside.mean())

        mean_pos_out = energy_out / (float(outside.sum()) + 1e-6)
        mean_pos_in = energy_in / (float(inside.sum()) + 1e-6)

        true_name = self.classes[sample["label"]] if sample["label"] < len(self.classes) else "?"
        pred_name = self.classes[pred_label] if pred_label < len(self.classes) else "?"

        collapsed = float(true_cam.max()) <= 0.0
        warning = ""
        if collapsed:
            warning = "  <<< CAM ENTIRELY NEGATIVE: ReLU metric is blind"
            if self.cam_loss_type == "softmax_containment":
                warning += "; softmax loss is still active"

        line_1 = (
            f"{Path(sample['path']).name}  |  true={true_name}  pred={pred_name}  |  "
            f"relu_containment={containment:.3f} (flat baseline={uniform:.3f}){warning}"
        )
        line_2 = (
            f"true range=[{float(true_cam.min()):+.3f}, {float(true_cam.max()):+.3f}]  "
            f"pred range=[{float(pred_cam.min()):+.3f}, {float(pred_cam.max()):+.3f}]  |  "
            f"mean positive: object={mean_pos_in:.4f} background={mean_pos_out:.4f}  |  "
            f"mask coverage={float(inside.mean()):.3f}"
        )

        strip = Image.new("RGB", (width, _CAPTION_HEIGHT), (18, 18, 18))
        draw = ImageDraw.Draw(strip)
        draw.text((6, 5), line_1, fill=(245, 245, 245), font=self.font)
        draw.text((6, 25), line_2, fill=(190, 190, 190), font=self.font)
        return np.asarray(strip)
