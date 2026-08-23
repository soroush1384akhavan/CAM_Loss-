import csv
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data import get_dataloaders
from src.losses import build_cam_loss
from src.model_factory import build_model
from src.train import run_training
from src.utils import get_device, set_seed
from src.mask_config_utils import build_mask_config

# ---------------- sweep configs  ----------------
CONFIG_PATH = Path("configs/config.yaml")
OUTPUT_DIR = Path("sweep_results")
SHOTS = [1, 2, 4, 8, 16]
LAMBDAS = [0.0, 0.3, 0.5, 1.0, 3.0]
SWEEP_EPOCHS = 20
REPEATS = 3          # <-- تعداد تکرار هر ترکیب (shots, lambda)
# -------------------------------------------------


def load_base_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single(base_config: dict, shots: int, lam: float, repeat: int):
    config = dict(base_config)
    config["train_shots_per_class"] = shots
    config["train_fraction"] = None
    config["lambda_cam"] = lam
    config["epochs"] = SWEEP_EPOCHS

    # هر تکرار سیدِ متفاوت می‌گیره تا رن‌ها واقعا مستقل باشن
    base_seed = int(base_config.get("seed", 42))
    seed = base_seed + repeat
    config["seed"] = seed

    # هر رن پوشه‌ی مخصوص به خودش رو داره: shots{shots}_lambda{lam}/rep{repeat}
    config["runs_dir"] = str(
        OUTPUT_DIR / "runs" / f"shots{shots}_lambda{lam}" / f"rep{repeat}"
    )

    set_seed(seed)
    device = get_device()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    mask_root = config.get("mask_root")
    if mask_root:
        mask_dirs, mask_manifests = build_mask_config(mask_root)
        config["mask_dirs"] = mask_dirs
        config["mask_manifests"] = mask_manifests
    else:
        config["mask_dirs"] = config.get("mask_dirs", {})
        config["mask_manifests"] = config.get("mask_manifests", {})

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=config["data_dir"],
        image_size=config.get("image_size", 224),
        batch_size=config.get("batch_size", 32),
        val_ratio=config.get("val_ratio", 0.2),
        seed=seed,
        num_workers=config.get("num_workers", 0),
        mask_dirs=config.get("mask_dirs", {}),
        mask_manifests=config.get("mask_manifests", {}),
        test_ratio=config.get("test_ratio", 0.1),
        train_shots_per_class=config.get("train_shots_per_class"),
        train_fraction=config.get("train_fraction"),
    )

    num_classes = len(classes)

    model = build_model(config, num_classes).to(device)

    ce_loss_fn = nn.CrossEntropyLoss()
    cam_loss_fn = build_cam_loss(config)

    lambda_cam = float(config["lambda_cam"])
    use_amp = bool(config.get("use_amp", True))
    epochs = int(config["epochs"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    scheduler = None
    if str(config.get("scheduler", "none")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(config.get("eta_min", 0.0)),
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and use_amp))

    result = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        classes=classes,
        ce_loss_fn=ce_loss_fn,
        cam_loss_fn=cam_loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        lambda_cam=lambda_cam,
        use_amp=use_amp,
        epochs=epochs,
        config=config,
        runs_dir=config["runs_dir"],
        test_loader=test_loader,
    )

    return {
        "shots": shots,
        "lambda_cam": lam,
        "repeat": repeat,
        "seed": seed,
        "best_val_acc": result["best_val_acc"],
        "last_train_acc": result["history"]["train_acc"][-1],
        "last_val_acc": result["history"]["val_acc"][-1],
        "run_dir": str(result["run_dir"]),
    }


def aggregate_repeats(rows):
    """
    برای یک ترکیب ثابت (shots, lambda) روی چند تکرار میانگین و انحراف معیار می‌گیره.
    """
    best_val_accs = [r["best_val_acc"] for r in rows]
    last_train_accs = [r["last_train_acc"] for r in rows]
    last_val_accs = [r["last_val_acc"] for r in rows]

    return {
        "shots": rows[0]["shots"],
        "lambda_cam": rows[0]["lambda_cam"],
        "n_repeats": len(rows),
        "best_val_acc_mean": statistics.mean(best_val_accs),
        "best_val_acc_std": statistics.pstdev(best_val_accs) if len(best_val_accs) > 1 else 0.0,
        "last_train_acc_mean": statistics.mean(last_train_accs),
        "last_val_acc_mean": statistics.mean(last_val_accs),
        "last_val_acc_std": statistics.pstdev(last_val_accs) if len(last_val_accs) > 1 else 0.0,
        "run_dirs": ";".join(r["run_dir"] for r in rows),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = load_base_config()

    raw_results = []       # هر ران به تنهایی (shots, lambda, repeat)
    avg_results = []       # میانگین روی repeats برای هر (shots, lambda)

    raw_csv_path = OUTPUT_DIR / "results_raw.csv"
    avg_csv_path = OUTPUT_DIR / "results_avg.csv"

    raw_fieldnames = [
        "shots", "lambda_cam", "repeat", "seed",
        "best_val_acc", "last_train_acc", "last_val_acc", "run_dir",
    ]
    avg_fieldnames = [
        "shots", "lambda_cam", "n_repeats",
        "best_val_acc_mean", "best_val_acc_std",
        "last_train_acc_mean",
        "last_val_acc_mean", "last_val_acc_std",
        "run_dirs",
    ]

    with open(raw_csv_path, "w", newline="", encoding="utf-8") as raw_f:
        raw_writer = csv.DictWriter(raw_f, fieldnames=raw_fieldnames)
        raw_writer.writeheader()

        for shots in SHOTS:
            for lam in LAMBDAS:
                combo_rows = []

                for repeat in range(REPEATS):
                    print(f"\n{'='*60}")
                    print(f"RUNNING: shots_per_class={shots} | lambda_cam={lam} | repeat={repeat+1}/{REPEATS}")
                    print(f"{'='*60}\n")

                    row = run_single(base_config, shots, lam, repeat)
                    raw_results.append(row)
                    combo_rows.append(row)

                    raw_writer.writerow(row)
                    raw_f.flush()

                    print(
                        f"\n>>> DONE shots={shots} lambda={lam} repeat={repeat+1}/{REPEATS} "
                        f"-> best_val_acc={row['best_val_acc']:.4f}\n"
                    )

                # میانگین‌گیری روی repeats برای این ترکیب shots/lambda
                avg_row = aggregate_repeats(combo_rows)
                avg_results.append(avg_row)

                print(
                    f"\n>>> AVG over {REPEATS} repeats | shots={shots} lambda={lam} "
                    f"-> best_val_acc_mean={avg_row['best_val_acc_mean']:.4f} "
                    f"(+/- {avg_row['best_val_acc_std']:.4f})\n"
                )

    with open(avg_csv_path, "w", newline="", encoding="utf-8") as avg_f:
        avg_writer = csv.DictWriter(avg_f, fieldnames=avg_fieldnames)
        avg_writer.writeheader()
        for row in avg_results:
            avg_writer.writerow(row)

    print(f"\nRaw (per-repeat) results saved to: {raw_csv_path}")
    print(f"Averaged results saved to: {avg_csv_path}")

    plot_results(avg_results)


def plot_results(avg_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for lam in LAMBDAS:
        xs = []
        ys = []
        yerrs = []
        for shots in SHOTS:
            match = [
                r for r in avg_results
                if r["shots"] == shots and r["lambda_cam"] == lam
            ]
            if match:
                xs.append(shots)
                ys.append(match[0]["best_val_acc_mean"] * 100)
                yerrs.append(match[0]["best_val_acc_std"] * 100)
        ax.errorbar(xs, ys, yerr=yerrs, marker="o", capsize=3, label=f"lambda_cam={lam}")

    ax.set_xlabel("Number of labeled training examples per class (shots)")
    ax.set_ylabel("Best Val Accuracy (%) - mean over repeats")
    ax.set_title(f"Accuracy vs Shots, per lambda_cam (avg of {REPEATS} repeats)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SHOTS)
    ax.set_xticklabels(SHOTS)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "sweep_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()