from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from common import ensure_parent, load_params, write_json


def to_python(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python(v) for v in value]
    return value


def main() -> None:
    params_path = Path("params.yaml")
    params = load_params()

    exp_cfg = params["experiments"]
    workflow_cfg = params["workflow"]
    best_result_path = Path(exp_cfg["best_result_path"])
    update_report_path = Path(workflow_cfg["params_update_report_path"])

    if not best_result_path.exists():
        raise FileNotFoundError(f"Best experiment file not found: {best_result_path}")

    with best_result_path.open("r", encoding="utf-8") as f:
        best = json.load(f)

    selected_model = str(best["selected_model_name"])
    selected_params = to_python(best["selected_best_params"])

    old_model = params["training"].get("model_name")
    old_fixed = params["training"].get("fixed_params", {})

    params["training"]["model_name"] = selected_model
    params["training"]["fixed_params"] = selected_params

    with params_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    report = {
        "params_path": str(params_path),
        "old_model_name": old_model,
        "new_model_name": selected_model,
        "old_fixed_params": old_fixed,
        "new_fixed_params": selected_params,
        "source_best_result_path": str(best_result_path),
    }
    ensure_parent(update_report_path)
    write_json(report, update_report_path)

    print(f"[params_update] Updated training.model_name -> {selected_model}")
    print("[params_update] Updated training.fixed_params from best experiment params")
    print(f"[params_update] Report: {update_report_path}")


if __name__ == "__main__":
    main()
