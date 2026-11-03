"""Combine the local seed-42 sweep with Kaggle seeds and compute mean/std/paired stats."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:\CVLab\mainProj")
OUT = Path(sys.argv[1])  # folder with the Kaggle summary_seed_runs.csv; results are written here too

# Local seed 42: rebuild the same columns from its run folders.
rows = []
for r in csv.DictReader(open(ROOT / "sweep_results/results_raw.csv", encoding="utf-8")):
    run_dir = ROOT / r["run_dir"]
    s = json.load(open(run_dir / "cam_debug/test_localization_summary.json"))
    log = pd.read_csv(run_dir / "training_log.csv")
    rows.append({
        "seed": int(r["seed"]), "shots": int(r["shots"]), "lambda_cam": float(r["lambda_cam"]),
        "best_val_acc": float(r["best_val_acc"]), "test_acc": s["accuracy"],
        "epochs_ran": len(log), "min_val_cls_loss": log.val_cls_loss.min(),
        "containment_mean": s.get("containment_mean"), "pointing_hit_mean": s["pointing_hit_mean"],
        "iou_true_label_mean": s["iou_true_label_mean"], "signed_inside_mean": s["signed_inside_mean_mean"],
        "signed_outside_mean": s["signed_outside_mean_mean"], "cam_all_negative": s["cam_all_negative_mean"],
        "source": "local",
    })
local = pd.DataFrame(rows)

kaggle = pd.read_csv(OUT / "summary_seed_runs.csv")
kaggle["source"] = "kaggle"
cols = [c for c in local.columns if c in kaggle.columns]
df = pd.concat([local[cols], kaggle[cols]], ignore_index=True)
df["shots"] = df.shots.astype(int)
df["lambda_cam"] = df.lambda_cam.astype(float)
df.to_csv(OUT / "all_seeds_runs.csv", index=False)

seeds = sorted(df.seed.unique())
print("seeds:", seeds, "runs:", len(df))
print(df.groupby("seed").size().to_dict())

# 1. mean / std per (shots, lambda)
metrics = ["test_acc", "best_val_acc", "containment_mean", "pointing_hit_mean", "signed_inside_mean",
           "signed_outside_mean", "min_val_cls_loss"]
agg = df.groupby(["shots", "lambda_cam"])[metrics].agg(["mean", "std", "count"])
agg.to_csv(OUT / "mean_std_by_shots_lambda.csv")

pd.set_option("display.width", 250)
t = df.pivot_table(index="shots", columns="lambda_cam", values="test_acc", aggfunc=["mean", "std"]) * 100
print("\nTEST ACC (%) mean / std over seeds")
print(t.round(2))

# 2. paired differences vs lambda=0 within each seed (same split, same subset)
base = df[df.lambda_cam == 0.0].set_index(["seed", "shots"])
diffs = []
for lam in sorted(df.lambda_cam.unique()):
    if lam == 0.0:
        continue
    cur = df[df.lambda_cam == lam].set_index(["seed", "shots"])
    for key in cur.index.intersection(base.index):
        diffs.append({"seed": key[0], "shots": key[1], "lambda_cam": lam,
                      "d_test": cur.loc[key, "test_acc"] - base.loc[key, "test_acc"],
                      "d_val": cur.loc[key, "best_val_acc"] - base.loc[key, "best_val_acc"],
                      "d_containment": cur.loc[key, "containment_mean"] - base.loc[key, "containment_mean"]})
d = pd.DataFrame(diffs)
d.to_csv(OUT / "paired_diffs.csv", index=False)

print("\nPAIRED TEST-ACC GAIN vs lambda=0 (pp): mean, std, n, positive, paired t p-value")
out = []
for (lam, shots), g in d.groupby(["lambda_cam", "shots"]):
    x = g.d_test.values * 100
    p = stats.ttest_1samp(x, 0).pvalue if len(x) > 1 else np.nan
    out.append({"lambda": lam, "shots": shots, "mean_pp": x.mean(), "std_pp": x.std(ddof=1) if len(x) > 1 else np.nan,
                "n": len(x), "positive": int((x > 0).sum()), "t_p": p})
summary = pd.DataFrame(out)
print(summary.round(3).to_string(index=False))
summary.to_csv(OUT / "paired_gain_summary.csv", index=False)

print("\nPOOLED over shots (per lambda): mean gain, positive/total, sign-test p, Wilcoxon p")
for lam, g in d.groupby("lambda_cam"):
    x = g.d_test.values * 100
    k, n = int((x > 0).sum()), len(x)
    sign_p = stats.binomtest(k, n, 0.5).pvalue
    w_p = stats.wilcoxon(x).pvalue if n >= 6 else np.nan
    print(f"lambda={lam}: mean {x.mean():+.2f} pp, {k}/{n} positive, sign p={sign_p:.4f}, wilcoxon p={w_p:.4f}")

# run-to-run noise at lambda=0 (std across seeds), for scale
print("\nSEED NOISE: std of test acc across seeds (pp)")
print((df.groupby(["shots", "lambda_cam"]).test_acc.std().unstack() * 100).round(2))
