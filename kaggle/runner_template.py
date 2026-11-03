"""
Kaggle runner: CAM-loss shots x lambda sweep for extra seeds, one seed per GPU.
The project code is embedded below (zip, base64) so this kernel is a frozen
snapshot of the code that produced the local seed-42 sweep.
"""
import base64
import csv
import glob
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

SEEDS = __SEEDS__
LAMBDAS = __LAMBDAS__
CODE_B64 = "__CODE_B64__"

WORK = Path("/kaggle/working")
CODE = Path("/tmp/camcode")
LOGS = WORK / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=False)


# 1. code
CODE.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall(CODE)
print("code files:", sum(1 for _ in CODE.rglob("*.py")), flush=True)

# 2. environment
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
import torch  # noqa: E402

try:
    import timm
    timm_version = timm.__version__
except ImportError:
    timm_version = None
if timm_version != "1.0.29":
    sh(f"{sys.executable} -m pip install -q timm==1.0.29")
sh(f"{sys.executable} -c \"import torch,timm,torchvision;print(torch.__version__,timm.__version__,torchvision.__version__)\"")

# 3. data
def find_dir(suffix):
    for pattern in ("/kaggle/input/*/" + suffix, "/kaggle/input/*/*/" + suffix, "/kaggle/input/*/*/*/" + suffix):
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    raise FileNotFoundError(suffix)


data_dir = find_dir("cars_dataset/cars_dataset")
mask_root = find_dir("sam_pseudo_masks_cars/sam_pseudo_masks_cars")
print("data_dir:", data_dir, "classes:", len(os.listdir(data_dir)), flush=True)
print("mask_root:", mask_root, "classes:", len(os.listdir(mask_root)), flush=True)

# 4. BiT-M weights (same checkpoint the local runs use)
from huggingface_hub import hf_hub_download  # noqa: E402

weights = hf_hub_download("timm/resnetv2_50x1_bit.goog_in21k", "model.safetensors", local_dir="/tmp/bitw")
print("weights:", weights, os.path.getsize(weights), flush=True)

# 5. one config + one process per seed
import yaml  # noqa: E402

base = yaml.safe_load(open(CODE / "configs/config.yaml", encoding="utf-8"))
runner = CODE / "run_one.py"
runner.write_text(
    "import os, sys, pathlib\n"
    "os.chdir('/tmp/camcode'); sys.path.insert(0, '/tmp/camcode')\n"
    "import sweep\n"
    "sweep.CONFIG_PATH = pathlib.Path(sys.argv[1])\n"
    "sweep.OUTPUT_DIR = pathlib.Path(sys.argv[2])\n"
    "sweep.REPEATS = 1\n"
    "sweep.LAMBDAS = __LAMBDAS__\n"
    "sweep.main()\n"
)

procs = []
for gpu, seed in enumerate(SEEDS):
    cfg = dict(base)
    cfg.update(data_dir=data_dir, mask_root=mask_root, bit_weights_path=weights, seed=seed, num_workers=2,
               train_augment=False)  # match the local seed-42 sweep, which ran without augmentation
    cfg_path = CODE / f"configs/kaggle_seed{seed}.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"), sort_keys=False)
    out_dir = WORK / f"sweep_seed{seed}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu % max(1, torch.cuda.device_count())), TQDM_DISABLE="1",
               PYTHONUNBUFFERED="1")
    log = open(LOGS / f"seed{seed}.log", "w", encoding="utf-8")
    procs.append((seed, subprocess.Popen([sys.executable, str(runner), str(cfg_path), str(out_dir)],
                                         stdout=log, stderr=subprocess.STDOUT, env=env), log))
    print(f"started seed {seed} on GPU {gpu}", flush=True)

start = time.time()
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(600)
    for seed, p, _ in procs:
        done = sum(1 for _ in open(WORK / f"sweep_seed{seed}/results_raw.csv")) - 1 if (WORK / f"sweep_seed{seed}/results_raw.csv").exists() else 0
        print(f"[{(time.time() - start) / 3600:.2f}h] seed {seed}: {done}/{5 * len(LAMBDAS)} runs done, running={p.poll() is None}", flush=True)

for seed, p, log in procs:
    log.close()
    print(f"seed {seed} exit code {p.returncode}", flush=True)

# 6. compact summary: test accuracy + localisation per run
rows = []
for seed in SEEDS:
    raw = WORK / f"sweep_seed{seed}/results_raw.csv"
    if not raw.exists():
        continue
    for r in csv.DictReader(open(raw, encoding="utf-8")):
        run_dir = Path(r["run_dir"])
        summary_path = run_dir / "cam_debug/test_localization_summary.json"
        s = json.load(open(summary_path)) if summary_path.exists() else {}
        log_rows = list(csv.DictReader(open(run_dir / "training_log.csv")))
        best = min(log_rows, key=lambda x: float(x["val_cls_loss"]))
        rows.append({
            "seed": seed, "shots": r["shots"], "lambda_cam": r["lambda_cam"],
            "best_val_acc": r["best_val_acc"], "test_acc": s.get("accuracy"),
            "epochs_ran": len(log_rows), "min_val_cls_loss_epoch": best["epoch"],
            "min_val_cls_loss": best["val_cls_loss"],
            "containment_mean": s.get("containment_mean"), "containment_median": s.get("containment_median"),
            "pointing_hit_mean": s.get("pointing_hit_mean"), "iou_true_label_mean": s.get("iou_true_label_mean"),
            "signed_inside_mean": s.get("signed_inside_mean_mean"), "signed_outside_mean": s.get("signed_outside_mean_mean"),
            "cam_all_negative": s.get("cam_all_negative_mean"), "run_dir": str(run_dir),
        })
if rows:
    with open(WORK / "summary_seed_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"summary rows: {len(rows)}", flush=True)

# 7. keep the output small: last-epoch checkpoints are not needed
freed = 0
for p in WORK.rglob("last_*.pth"):
    freed += p.stat().st_size
    p.unlink()
print(f"removed last_*.pth: {freed / 1e9:.2f} GB", flush=True)

if any(p.returncode != 0 for _, p, _ in procs):
    sys.exit(1)
