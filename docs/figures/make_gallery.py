"""Image galleries and training curves for the presentation, from the local seed-42 sweep."""
import csv
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {0.0: "#2a78d6", 0.5: "#eb6834", 3.0: "#1baf7a"}  # same slots as fig1/fig2
MARKER = {0.0: "o", 0.5: "s", 3.0: "D"}
TILE, HEADER = 224, 21
SNAP_ROWS = {"Land Rover (bad mask)": 0, "Porsche Panamera": 1, "BMW 3 Series Wagon": 2, "GMC Canyon (bad mask)": 3,
             "Dodge Durango": 4, "Dodge Charger": 5, "Aston Martin V8": 6, "Hyundai Veloster": 7}
INPUT, MASK, PRED_CAM, TRUE_CAM = 0, 1, 2, 4
plt.rcParams.update({"font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def run_dir(shots, lam):
    return Path(sorted(glob.glob(str(REPO / f"sweep_results/runs/shots{shots}_lambda{lam}/rep0/2026-10-23_*")))[-1])


def snapshot(shots, lam, epoch=None):
    snaps = sorted((run_dir(shots, lam) / "diagnostics/snapshots").glob("epoch_*.png"))
    path = snaps[-1] if epoch is None else run_dir(shots, lam) / f"diagnostics/snapshots/epoch_{epoch:03d}.png"
    return Image.open(path).convert("RGB")


def tile(img, row, col, row_h=292):
    top = row * row_h + HEADER
    return img.crop((col * TILE, top, (col + 1) * TILE, top + TILE))


def grid(tiles, col_titles, row_titles, path, title, tile_size=1.75):
    rows, cols = len(tiles), len(tiles[0])
    fig, axes = plt.subplots(rows, cols, figsize=(tile_size * cols + 0.5, tile_size * rows + 0.8), squeeze=False)
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.imshow(tiles[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9.5, color=INK)
        axes[r, 0].set_ylabel(row_titles[r], fontsize=8.5, color=INK2)
    fig.suptitle(title, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / path, dpi=150)
    plt.close(fig)
    print("saved", path)


# Figure 4: same images, lambda 0 / 0.5 / 3 at 16 shots
names = ["Porsche Panamera", "BMW 3 Series Wagon", "Dodge Charger", "Hyundai Veloster"]
imgs = {lam: snapshot(16, lam) for lam in (0.0, 0.5, 3.0)}
tiles = [[tile(imgs[0.0], SNAP_ROWS[n], INPUT), tile(imgs[0.0], SNAP_ROWS[n], MASK)]
         + [tile(imgs[lam], SNAP_ROWS[n], TRUE_CAM) for lam in (0.0, 0.5, 3.0)] for n in names]
grid(tiles, ["Input", "SAM mask", "λ = 0", "λ = 0.5", "λ = 3"], names, "fig4_lambda_gallery_16shot.png",
     "True-class CAM at 16 shots, seed 42, last epoch. Fixed validation images.")

# Figure 5: same comparison with very little data
imgs2 = {lam: snapshot(2, lam) for lam in (0.0, 3.0)}
tiles = [[tile(imgs2[0.0], SNAP_ROWS[n], INPUT), tile(imgs2[0.0], SNAP_ROWS[n], MASK),
          tile(imgs2[0.0], SNAP_ROWS[n], TRUE_CAM), tile(imgs2[3.0], SNAP_ROWS[n], TRUE_CAM)] for n in names]
grid(tiles, ["Input", "SAM mask", "λ = 0", "λ = 3"], names, "fig5_lambda_gallery_2shot.png",
     "True-class CAM at 2 shots (392 training images), seed 42, last epoch.")

# Figure 6: how the CAM forms during training
epochs = [1, 3, 5, 10, 15]
epoch_imgs = {e: snapshot(16, 3.0, e) for e in epochs}
tiles = [[tile(epoch_imgs[1], SNAP_ROWS[n], INPUT)] + [tile(epoch_imgs[e], SNAP_ROWS[n], TRUE_CAM) for e in epochs]
         for n in ("BMW 3 Series Wagon", "Dodge Charger")]
grid(tiles, ["Input"] + [f"Epoch {e}" for e in epochs], ["BMW 3 Series Wagon", "Dodge Charger"],
     "fig6_cam_over_epochs.png", "True-class CAM during training, 16 shots, λ = 3, seed 42.", tile_size=1.5)


# Overlays written by cam_debug: 6 tiles per row, header above each.
def debug_tiles(path, cols):
    img = Image.open(path).convert("RGB")
    top = img.height - TILE
    return [img.crop((c * TILE, top, (c + 1) * TILE, top + TILE)) for c in cols]


def debug_files(kind, shots=16, lam=3.0):
    return sorted((run_dir(shots, lam) / f"cam_debug/{kind}").glob(f"{kind.split('_')[0]}_*.png"))


def labels(path):
    m = re.search(r"true-(.+)_pred-(.+)_iou-([\d.]+)_loss", path.name)
    short = lambda s: s.replace("_", " ")
    return short(m.group(1)), short(m.group(2)), float(m.group(3))


# Figure 7: lowest-IoU test images are mostly SAM failures, not CAM failures
worst = debug_files("worst_iou")[:4]
tiles = [debug_tiles(p, [INPUT, MASK, TRUE_CAM]) for p in worst]
grid(tiles, ["Input", "SAM mask", "True-class CAM"], [labels(p)[0][:26] for p in worst], "fig7_sam_mask_failures.png",
     "The four lowest-IoU test images (16 shots, λ = 3, seed 42).", tile_size=1.9)

# Figure 8: well-localized but misclassified images
SHORT = {"Land Rover Range Rover SUV 2012": "Range Rover", "Jeep Patriot SUV 2012": "Jeep Patriot",
         "Aston Martin V8 Vantage Convertible 2012": "Aston Martin V8",
         "Rolls-Royce Phantom Drophead Coupe Convertible 2012": "Rolls-Royce Phantom",
         "Jeep Grand Cherokee SUV 2012": "Jeep Grand Cherokee"}
best = [p for p in debug_files("best_iou") if labels(p)[0] != labels(p)[1]][:3]
tiles = [debug_tiles(p, [INPUT, MASK, PRED_CAM, TRUE_CAM]) for p in best]
grid(tiles, ["Input", "SAM mask", "Predicted-class CAM", "True-class CAM"],
     [f"True: {SHORT.get(labels(p)[0], labels(p)[0][:22])}\nPred: {SHORT.get(labels(p)[1], labels(p)[1][:22])}" for p in best],
     "fig8_right_place_wrong_class.png",
     "Right place, wrong class: top-IoU images the model misclassified (16 shots, λ = 3, seed 42).", tile_size=1.9)


# Figure 9: validation CE curves
def column(shots, lam, name):
    rows = list(csv.DictReader(open(run_dir(shots, lam) / "training_log.csv", encoding="utf-8")))
    return [int(r["epoch"]) for r in rows], [float(r[name]) for r in rows]


fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
for ax, shots in zip(axes, (4, 16)):
    for lam in (0.0, 0.5, 3.0):
        x, y = column(shots, lam, "val_cls_loss")
        ax.plot(x, y, color=COLOR[lam], linewidth=2, label=f"λ = {lam:g}" + (" (baseline)" if lam == 0 else ""))
        best_i = y.index(min(y))
        ax.plot(x[best_i], y[best_i], marker=MARKER[lam], color=COLOR[lam], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_title(f"{shots} shots", loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("Epoch")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
axes[0].set_ylabel("Validation cross-entropy")
axes[0].legend(frameon=False, loc="upper right")
fig.suptitle("Lower validation CE with the CAM loss (marker = lowest point), seed 42", fontsize=10.5, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig9_val_ce_curves.png", dpi=170)
plt.close(fig)
print("saved fig9_val_ce_curves.png")

# Numbers quoted in the presentation.
print("\nmask-failure stats, 16 shots, seed 42 (healthy = mask covers at least 5% of the CAM grid)")
for lam in (0.0, 0.5, 3.0):
    rows = [r for r in csv.DictReader(open(run_dir(16, lam) / "cam_debug/all_test_samples.csv", encoding="utf-8"))
            if r["has_mask"] == "1"]
    tiny = [r for r in rows if float(r["mask_coverage_cam_grid"]) < 0.05]
    good = [r for r in rows if float(r["mask_coverage_cam_grid"]) >= 0.05]
    zero = [r for r in rows if float(r["iou_true_label"]) == 0]
    mean = lambda rs, k: sum(float(r[k]) for r in rs if r[k] not in ("", "None")) / max(1, sum(r[k] not in ("", "None") for r in rs))
    print(f"lambda={lam}: tiny {len(tiny)}/{len(rows)}; iou=0 {len(zero)} of which tiny {sum(float(r['mask_coverage_cam_grid']) < 0.05 for r in zero)}; "
          f"pointing tiny {mean(tiny, 'pointing_hit'):.3f} healthy {mean(good, 'pointing_hit'):.3f}; "
          f"containment tiny {mean(tiny, 'containment'):.3f} healthy {mean(good, 'containment'):.3f}; "
          f"acc tiny {mean(tiny, 'correct'):.3f} healthy {mean(good, 'correct'):.3f}; "
          f"pointing correct {mean([r for r in rows if r['correct'] == '1'], 'pointing_hit'):.3f} wrong {mean([r for r in rows if r['correct'] == '0'], 'pointing_hit'):.3f}")

print("\nepoch of lowest validation CE, seed 42 (epochs run)")
for shots in (1, 2, 4, 8, 16):
    parts = []
    for lam in (0.0, 0.5, 3.0):
        x, y = column(shots, lam, "val_cls_loss")
        parts.append(f"lambda={lam}: {x[y.index(min(y))]} ({len(x)})")
    print(f"shots={shots}: " + ", ".join(parts))
