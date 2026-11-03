"""
Build a self-contained Kaggle kernel from the current code.

The project code (src/, sweep.py, configs/config.yaml) is zipped and embedded
in the kernel script, so each pushed kernel version is a frozen snapshot.

    python kaggle/build_kernel.py --seeds 43 44 --lambdas 0.5 3.0
    python kaggle/build_kernel.py --smoke
    kaggle kernels push -p kaggle/kernel --accelerator NvidiaTeslaT4   # sweep
    kaggle kernels push -p kaggle/kernel                               # smoke test (CPU)
"""
import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATASET = "soroushakhavan/sam-pseudo-masks-cars-and-dataset"


def code_bundle() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (REPO / "src").rglob("*.py"):
            if "__pycache__" not in p.parts:
                z.write(p, p.relative_to(REPO).as_posix())
        z.write(REPO / "sweep.py", "sweep.py")
        z.write(REPO / "configs/config.yaml", "configs/config.yaml")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 3.0])
    parser.add_argument("--smoke", action="store_true", help="CPU check that masks load and the CAM loss is live")
    args = parser.parse_args()

    out = HERE / "kernel"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.py"):
        old.unlink()

    if args.smoke:
        code = (HERE / "smoke_test_template.py").read_text(encoding="utf-8")
        name, gpu = "cam-loss-smoke-test", False
    else:
        code = (HERE / "runner_template.py").read_text(encoding="utf-8")
        code = code.replace("__SEEDS__", repr(args.seeds)).replace("__LAMBDAS__", repr(args.lambdas))
        name, gpu = "cam-loss-seed-sweep", True
    code = code.replace("__CODE_B64__", code_bundle())
    compile(code, name, "exec")

    file_name = name.replace("-", "_") + ".py"
    (out / file_name).write_text(code, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"soroushakhavan/{name}", "title": name, "code_file": file_name,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": [],
        **({"machine_shape": "NvidiaTeslaT4"} if gpu else {}),
    }, indent=2))
    print(f"built {out / file_name} ({len(code)} bytes)")


if __name__ == "__main__":
    main()
