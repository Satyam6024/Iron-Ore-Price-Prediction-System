from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib

from common import ensure_parent, load_params, write_json


def flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in metrics.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=next_key))
        elif isinstance(value, (int, float)):
            flat[next_key] = float(value)
    return flat


def wait_for_registration_ready(client, model_name: str, version: str, timeout_sec: int) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_state = None
    while time.time() < deadline:
        mv = client.get_model_version(name=model_name, version=version)
        status = str(getattr(mv, "status", "UNKNOWN"))
        status_upper = status.upper()
        last_state = {
            "status": status,
            "stage": str(getattr(mv, "current_stage", "None")),
        }
        if "READY" in status_upper:
            return last_state
        if "FAILED" in status_upper:
            raise RuntimeError(f"Model registration failed with status={status}")
        time.sleep(2)

    raise TimeoutError(
        f"Timed out waiting for model registration readiness. Last state={last_state}"
    )


def main() -> None:
    params = load_params()
    train_cfg = params["training"]
    mlflow_cfg = params["mlflow"]
    registry_cfg = params["registry"]

    model_path = Path(train_cfg["output_model_path"])
    metrics_path = Path(train_cfg["metrics_path"])
    predictions_path = Path(train_cfg["predictions_path"])

    output_path = Path(registry_cfg["registration_output_path"])
    model_name = str(registry_cfg["model_name"])
    testing_alias = str(registry_cfg.get("testing_alias", "testing"))
    timeout_sec = int(registry_cfg.get("wait_timeout_sec", 120))

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Training metrics not found: {metrics_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Test predictions not found: {predictions_path}")

    # Imported lazily so non-MLflow stages can still run without mlflow installed.
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mlflow is required for registry stages. Install with: pip install mlflow"
        ) from exc

    tracking_uri = str(mlflow_cfg["tracking_uri"])
    registry_uri = str(mlflow_cfg.get("registry_uri", tracking_uri))
    experiment_name = str(mlflow_cfg["experiment_name"])

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics_json = json.load(f)

    model = joblib.load(model_path)

    with mlflow.start_run(run_name=f"{model_name}_registration") as run:
        run_id = run.info.run_id

        # Log model + artifacts + flattened metrics.
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        mlflow.log_artifact(str(predictions_path), artifact_path="reports")
        mlflow.log_param("registry_model_name", model_name)
        mlflow.log_param("source_model_path", str(model_path))

        flat_metrics = flatten_metrics(metrics_json)
        if flat_metrics:
            mlflow.log_metrics(flat_metrics)

        mlflow.sklearn.log_model(model, artifact_path="model")
        model_uri = f"runs:/{run_id}/model"

    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(registered.version)
    state = wait_for_registration_ready(client, model_name=model_name, version=version, timeout_sec=timeout_sec)

    # Set lifecycle entry point to testing.
    if hasattr(client, "set_registered_model_alias"):
        client.set_registered_model_alias(name=model_name, alias=testing_alias, version=version)
    client.set_model_version_tag(name=model_name, version=version, key="lifecycle_stage", value="testing")
    client.set_model_version_tag(name=model_name, version=version, key="lifecycle_status", value="registered")
    client.set_model_version_tag(name=model_name, version=version, key="source_run_id", value=run_id)

    report = {
        "model_name": model_name,
        "version": version,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
        "testing_alias": testing_alias,
        "model_uri": f"models:/{model_name}/{version}",
        "status": state["status"],
        "current_stage": state["stage"],
    }
    ensure_parent(output_path)
    write_json(report, output_path)

    print(f"[mlflow_register] Model name: {model_name}")
    print(f"[mlflow_register] Registered version: {version}")
    print(f"[mlflow_register] Run ID: {run_id}")
    print(f"[mlflow_register] Testing alias -> v{version}")
    print(f"[mlflow_register] Report: {output_path}")


if __name__ == "__main__":
    main()
