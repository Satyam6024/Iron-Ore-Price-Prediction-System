from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import shutil
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"[workflow] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    python_dir = str(Path(sys.executable).parent)
    env["PATH"] = f"{python_dir}{os.pathsep}{env.get('PATH', '')}"
    subprocess.run(cmd, check=True, env=env)


def ensure_python_deps() -> None:
    required = ["numpy", "pandas", "sklearn", "yaml", "joblib", "mlflow"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing Python dependencies in the current interpreter "
            f"({sys.executable}). Missing: {missing}. "
            "Install with: python -m pip install -r requirements.txt"
        )


def resolve_dvc_command() -> list[str]:
    if shutil.which("dvc") is not None:
        return ["dvc"]
    if importlib.util.find_spec("dvc") is not None:
        return [sys.executable, "-m", "dvc"]
    raise RuntimeError("dvc command/module not found. Install dependencies first: python -m pip install -r requirements.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end workflow: features -> experiments -> params update -> DVC training/registry/deploy"
        )
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip experiment selection and params update. Directly run DVC pipeline.",
    )
    parser.add_argument(
        "--skip-dvc-init",
        action="store_true",
        help="Do not auto-run 'dvc init --no-scm' if .dvc is missing.",
    )
    args = parser.parse_args()

    root = Path(".")
    dvc_dir = root / ".dvc"

    ensure_python_deps()
    dvc_cmd = resolve_dvc_command()

    if not dvc_dir.exists() and not args.skip_dvc_init:
        run([*dvc_cmd, "init", "--no-scm"])

    if not args.skip_experiments:
        # Ensure selected-feature artifacts exist before experiments.
        run([*dvc_cmd, "repro", "feature_selection"])
        run([sys.executable, "dvc_pipeline/src/run_experiments.py"])
        run([sys.executable, "dvc_pipeline/src/update_params_from_experiments.py"])

    # Run remaining pipeline (or entire pipeline if previous stages changed).
    run([*dvc_cmd, "repro"])

    print("[workflow] Completed.")


if __name__ == "__main__":
    main()
