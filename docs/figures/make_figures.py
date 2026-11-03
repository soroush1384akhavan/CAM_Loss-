"""Figures for the professor presentation, from results/seed_sweep/*.csv."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
runs = pd.read_csv(REPO / "results/seed_sweep/all_seeds_runs.csv")
diffs = pd.read_csv(REPO / "results/seed_sweep/paired_diffs.csv")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {0.0: "#2a78d6", 0.5: "#eb6834", 3.0: "#1baf7a"}  # validated categorical slots 1-3
MARKER = {0.0: "o", 0.5: "s", 3.0: "D"}
SHOTS = [1, 2, 4, 8, 16]

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# Figure 1: test accuracy vs shots, mean +- std over seeds
fig, ax = plt.subplots(figsize=(7.5, 4.6))
agg = runs.groupby(["lambda_cam", "shots"]).test_acc.agg(["mean", "std"]).reset_index()
for lam, g in agg.groupby("lambda_cam"):
    g = g.set_index("shots").loc[SHOTS]
    ax.errorbar(SHOTS, g["mean"] * 100, yerr=g["std"] * 100, color=COLOR[lam], marker=MARKER[lam],
                markersize=7, linewidth=2, capsize=3, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"λ = {lam:g}" + (" (baseline)" if lam == 0 else ""))
ax.set_xscale("log", base=2)
ax.set_xticks(SHOTS, [str(s) for s in SHOTS])
ax.set_xlim(0.8, 20)
ax.set_xlabel("Training images per class (shots)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Stanford Cars: test accuracy vs. shots (mean ± std, 3 seeds)", loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper left")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "fig1_accuracy_vs_shots.png", dpi=180)
plt.close(fig)

# Figure 2: paired gain over lambda=0 (same seed and split), bars = mean, dots = seeds
fig, ax = plt.subplots(figsize=(7.5, 4.4))
width, offsets = 0.36, {0.5: -0.19, 3.0: 0.19}
for lam in (0.5, 3.0):
    g = diffs[diffs.lambda_cam == lam]
    means = g.groupby("shots").d_test.mean().loc[SHOTS] * 100
    xs = [i + offsets[lam] for i in range(len(SHOTS))]
    ax.bar(xs, means, width=width - 0.04, color=COLOR[lam], label=f"λ = {lam:g}", zorder=2)
    for i, s in enumerate(SHOTS):
        vals = g[g.shots == s].d_test * 100
        ax.scatter([xs[i]] * len(vals), vals, s=22, color=INK, zorder=3, linewidths=0)
    tops = g.groupby("shots").d_test.max().loc[SHOTS] * 100
    for x, m, top in zip(xs, means, tops):
        ax.text(x, max(m, top) + 0.3, f"+{m:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
ax.axhline(0, color=INK2, linewidth=1)
ax.set_xticks(range(len(SHOTS)), [str(s) for s in SHOTS])
ax.set_xlabel("Training images per class (shots)")
ax.set_ylabel("Test-accuracy gain over λ=0 (pp)")
ax.set_title("Paired gain from the CAM loss — 30/30 comparisons positive (dots = seeds)", loc="left", fontsize=12)
ax.set_ylim(0, 7.8)
ax.legend(frameon=False, loc="upper right")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "fig2_paired_gain.png", dpi=180)
plt.close(fig)
print("saved figures to", OUT)

# Figure 3: true-class CAM before (sign flip) vs after (class-centering), same images.
# Crops tiles out of the training snapshots: 8 rows x 6 tiles of 224 px, 21 px header per row.
from PIL import Image  # noqa: E402

RUNS = REPO / "runs"
BEFORE = RUNS / "bitm_r50x1_adapter_g8_16shot_cam005/2026-09-07_12-24-37/diagnostics/snapshots/epoch_016.png"
AFTER = RUNS / "bitm_r50x1_adapter_g8_16shot_state001/2026-10-21_15-57-04/diagnostics/snapshots/epoch_020.png"
if BEFORE.exists() and AFTER.exists():
    before, after = Image.open(BEFORE).convert("RGB"), Image.open(AFTER).convert("RGB")
    row_h, header, tile = before.height // 8, 21, 224

    def crop(img, row, col):
        top = row * row_h + header
        return img.crop((col * tile, top, (col + 1) * tile, top + tile))

    rows = [(2, "BMW 3 Series Wagon"), (1, "Porsche Panamera")]
    cols = ["Input", "SAM mask", "Before: no centering\n(CAM negative on car)", "After: centered CAM\n(CAM positive on car)"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(9, 4.9))
    for r, (row, name) in enumerate(rows):
        tiles = [crop(after, row, 0), crop(after, row, 1), crop(before, row, 4), crop(after, row, 4)]
        for c, t in enumerate(tiles):
            ax = axes[r, c]
            ax.imshow(t)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
            if r == 0:
                ax.set_title(cols[c], fontsize=10, color=INK)
        axes[r, 0].set_ylabel(name, fontsize=10, color=INK2)
    fig.suptitle("True-class CAM on the same images (red = positive, blue = negative), both λ=0.5",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_cam_before_after.png", dpi=170)
    plt.close(fig)
    print("saved fig3")
