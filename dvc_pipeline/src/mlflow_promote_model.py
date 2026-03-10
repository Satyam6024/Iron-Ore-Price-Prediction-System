from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from common import ensure_parent, load_params, write_json


def parse_metrics(training_metrics: Dict[str, Any]) -> Dict[str, float]:
    test_metrics = training_metrics.get("test_metrics", {})
    return {
        "test_rmse": float(test_metrics.get("rmse", np.inf)),
        "test_mae": float(test_metrics.get("mae", np.inf)),
        "test_mape": float(test_metrics.get("mape", np.inf)),
        "test_r2": float(test_metrics.get("r2", -np.inf)),
    }


def check_testing_gate(metrics: Dict[str, float], gate_cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    max_rmse = float(gate_cfg["max_test_rmse"])
    max_mape = float(gate_cfg["max_test_mape"])
    min_r2 = float(gate_cfg["min_test_r2"])

    conditions = {
        "rmse_ok": metrics["test_rmse"] <= max_rmse,
        "mape_ok": metrics["test_mape"] <= max_mape,
        "r2_ok": metrics["test_r2"] >= min_r2,
    }
    passed = all(conditions.values())
    details = {
        "gate": "testing",
        "thresholds": {
            "max_test_rmse": max_rmse,
            "max_test_mape": max_mape,
            "min_test_r2": min_r2,
        },
        "metrics": metrics,
        "conditions": conditions,
        "passed": passed,
    }
    return passed, details


def check_staging_gate(
    predictions: pd.DataFrame, baseline_metrics: Dict[str, float], gate_cfg: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    y_true = predictions["actual"].astype(float).to_numpy()
    y_pred = predictions["predicted"].astype(float).to_numpy()
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - (ss_res / ss_tot if ss_tot != 0 else 0.0))

    baseline_rmse = float(baseline_metrics["test_rmse"])
    if baseline_rmse <= 1e-8:
        rmse_drift_pct = 0.0 if rmse <= 1e-8 else np.inf
    else:
        rmse_drift_pct = float(((rmse - baseline_rmse) / baseline_rmse) * 100.0)

    max_rmse_drift_pct = float(gate_cfg["max_rmse_drift_pct"])
    max_mape = float(gate_cfg["max_mape"])
    min_r2 = float(gate_cfg["min_r2"])

    conditions = {
        "rmse_drift_ok": rmse_drift_pct <= max_rmse_drift_pct,
        "mape_ok": mape <= max_mape,
        "r2_ok": r2 >= min_r2,
    }
    passed = all(conditions.values())

    details = {
        "gate": "staging",
        "thresholds": {
            "max_rmse_drift_pct": max_rmse_drift_pct,
            "max_mape": max_mape,
            "min_r2": min_r2,
        },
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "rmse_drift_pct_vs_testing_baseline": rmse_drift_pct,
        },
        "conditions": conditions,
        "passed": passed,
    }
    return passed, details


def maybe_set_alias(client, model_name: str, alias: str, version: str) -> None:
    if hasattr(client, "set_registered_model_alias"):
        client.set_registered_model_alias(name=model_name, alias=alias, version=version)


def maybe_delete_alias(client, model_name: str, alias: str) -> None:
    if hasattr(client, "delete_registered_model_alias"):
        try:
            client.delete_registered_model_alias(name=model_name, alias=alias)
        except Exception:
            pass


def maybe_transition_stage(client, model_name: str, version: str, stage: str) -> None:
    if hasattr(client, "transition_model_version_stage"):
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=False,
            )
        except Exception:
            # Stage transitions are optional; aliases are primary mechanism.
            pass


def main() -> None:
    params = load_params()
    train_cfg = params["training"]
    mlflow_cfg = params["mlflow"]
    registry_cfg = params["registry"]

    registration_info_path = Path(registry_cfg["registration_output_path"])
    promotion_output_path = Path(registry_cfg["promotion_output_path"])
    metrics_path = Path(train_cfg["metrics_path"])
    predictions_path = Path(train_cfg["predictions_path"])

    testing_alias = str(registry_cfg.get("testing_alias", "testing"))
    staging_alias = str(registry_cfg.get("staging_alias", "staging"))
    production_alias = str(registry_cfg.get("production_alias", "production"))
    gates_cfg = registry_cfg["gates"]

    if not registration_info_path.exists():
        raise FileNotFoundError(f"Registration report not found: {registration_info_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Training metrics file not found: {metrics_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with registration_info_path.open("r", encoding="utf-8") as f:
        reg = json.load(f)
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics_json = json.load(f)

    preds = pd.read_csv(predictions_path)
    if not {"actual", "predicted"}.issubset(preds.columns):
        raise ValueError("Predictions file must contain columns: actual, predicted")

    model_name = str(reg["model_name"])
    version = str(reg["version"])
    run_id = str(reg["run_id"])

    # Imported lazily so non-MLflow stages can still run without mlflow installed.
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mlflow is required for registry stages. Install with: pip install mlflow"
        ) from exc

    tracking_uri = str(mlflow_cfg["tracking_uri"])
    registry_uri = str(mlflow_cfg.get("registry_uri", tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    base_metrics = parse_metrics(metrics_json)

    # Always ensure candidate is in testing alias first.
    maybe_set_alias(client, model_name, testing_alias, version)
    client.set_model_version_tag(model_name, version, "lifecycle_stage", "testing")

    testing_pass, testing_detail = check_testing_gate(base_metrics, gates_cfg["testing"])
    if not testing_pass:
        client.set_model_version_tag(model_name, version, "lifecycle_status", "failed_testing_gate")
        maybe_delete_alias(client, model_name, staging_alias)
        report = {
            "model_name": model_name,
            "version": version,
            "run_id": run_id,
            "final_state": "testing",
            "reason": "failed_testing_gate",
            "testing_gate": testing_detail,
            "staging_gate": None,
            "aliases": {
                "testing": testing_alias,
                "staging": staging_alias,
                "production": production_alias,
            },
        }
        ensure_parent(promotion_output_path)
        write_json(report, promotion_output_path)
        print(f"[mlflow_promote] v{version} failed testing gate. Kept in testing.")
        print(f"[mlflow_promote] Report: {promotion_output_path}")
        return

    # Move to staging.
    maybe_set_alias(client, model_name, staging_alias, version)
    maybe_transition_stage(client, model_name, version, "Staging")
    client.set_model_version_tag(model_name, version, "lifecycle_stage", "staging")
    client.set_model_version_tag(model_name, version, "lifecycle_status", "passed_testing_gate")

    staging_pass, staging_detail = check_staging_gate(
        predictions=preds, baseline_metrics=base_metrics, gate_cfg=gates_cfg["staging"]
    )
    if not staging_pass:
        # Roll back to testing as requested.
        maybe_set_alias(client, model_name, testing_alias, version)
        maybe_delete_alias(client, model_name, staging_alias)
        client.set_model_version_tag(model_name, version, "lifecycle_stage", "testing")
        client.set_model_version_tag(model_name, version, "lifecycle_status", "failed_staging_gate")
        report = {
            "model_name": model_name,
            "version": version,
            "run_id": run_id,
            "final_state": "testing",
            "reason": "failed_staging_gate_rolled_back_to_testing",
            "testing_gate": testing_detail,
            "staging_gate": staging_detail,
            "aliases": {
                "testing": testing_alias,
                "staging": staging_alias,
                "production": production_alias,
            },
        }
        ensure_parent(promotion_output_path)
        write_json(report, promotion_output_path)
        print(f"[mlflow_promote] v{version} failed staging gate. Rolled back to testing.")
        print(f"[mlflow_promote] Report: {promotion_output_path}")
        return

    # Promote to production.
    maybe_set_alias(client, model_name, production_alias, version)
    maybe_transition_stage(client, model_name, version, "Production")
    client.set_model_version_tag(model_name, version, "lifecycle_stage", "production")
    client.set_model_version_tag(model_name, version, "lifecycle_status", "promoted_to_production")

    report = {
        "model_name": model_name,
        "version": version,
        "run_id": run_id,
        "final_state": "production",
        "reason": "passed_testing_and_staging",
        "testing_gate": testing_detail,
        "staging_gate": staging_detail,
        "aliases": {
            "testing": testing_alias,
            "staging": staging_alias,
            "production": production_alias,
        },
        "production_model_uri": f"models:/{model_name}@{production_alias}",
    }
    ensure_parent(promotion_output_path)
    write_json(report, promotion_output_path)

    print(f"[mlflow_promote] v{version} promoted to production.")
    print(f"[mlflow_promote] Report: {promotion_output_path}")


if __name__ == "__main__":
    main()
