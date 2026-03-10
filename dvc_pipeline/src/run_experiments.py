from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from common import ensure_parent, load_params, write_json
from train_model import build_model_and_space, evaluate


def to_python(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python(v) for v in value]
    return value


def should_minimize(metric_name: str) -> bool:
    metric_name = metric_name.lower()
    return any(token in metric_name for token in ["rmse", "mae", "mape", "loss", "error"])


def main() -> None:
    params = load_params()
    train_cfg = params["training"]
    exp_cfg = params["experiments"]

    train_path = Path(train_cfg["train_data_path"])
    test_path = Path(train_cfg["test_data_path"])
    feature_columns_path = Path(train_cfg["feature_columns_path"])
    target_col = train_cfg["target_column"]

    results_path = Path(exp_cfg["results_path"])
    best_result_path = Path(exp_cfg["best_result_path"])
    model_candidates: List[str] = [str(x) for x in exp_cfg["models"]]
    n_iter = int(exp_cfg["n_iter_per_model"])
    cv_splits = int(exp_cfg["cv_splits"])
    n_jobs = int(exp_cfg["n_jobs"])
    random_state = int(exp_cfg["random_state"])
    optimization_metric = str(exp_cfg["optimization_metric"])
    log_to_mlflow = bool(exp_cfg.get("log_to_mlflow", False))

    if not train_path.exists() or not test_path.exists() or not feature_columns_path.exists():
        raise FileNotFoundError("Training/test/features artifacts are missing. Run feature stage first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    with feature_columns_path.open("r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    X_train = train_df[feature_columns]
    y_train = train_df[target_col]
    X_test = test_df[feature_columns]
    y_test = test_df[target_col]

    mlflow = None
    if log_to_mlflow:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
            tracking_uri = str(params["mlflow"]["tracking_uri"])
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(str(exp_cfg.get("mlflow_experiment_name", "IronOre_Experiments")))
        except Exception:
            mlflow = None

    cv = TimeSeriesSplit(n_splits=cv_splits)
    rows: List[Dict[str, Any]] = []

    for model_name in model_candidates:
        estimator, param_dist = build_model_and_space(model_name=model_name, random_state=random_state)
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1,
        )

        if mlflow is None:
            search.fit(X_train, y_train)
        else:
            with mlflow.start_run(run_name=f"exp_{model_name}"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("n_iter", n_iter)
                search.fit(X_train, y_train)
                mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
                mlflow.log_metric("best_cv_rmse", float(-search.best_score_))

        best_model = search.best_estimator_
        train_metrics = evaluate(y_train, best_model.predict(X_train))
        test_metrics = evaluate(y_test, best_model.predict(X_test))

        row = {
            "model_name": model_name,
            "best_cv_rmse": float(-search.best_score_),
            "best_params": to_python(search.best_params_),
            "train_rmse": float(train_metrics["rmse"]),
            "train_mae": float(train_metrics["mae"]),
            "train_mape": float(train_metrics["mape"]),
            "train_r2": float(train_metrics["r2"]),
            "test_rmse": float(test_metrics["rmse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_mape": float(test_metrics["mape"]),
            "test_r2": float(test_metrics["r2"]),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No experiment results were produced.")

    minimize = should_minimize(optimization_metric)
    sorted_rows = sorted(rows, key=lambda x: x[optimization_metric], reverse=not minimize)
    best = sorted_rows[0]

    ensure_parent(results_path)
    results_df = pd.DataFrame(
        [
            {
                **{k: v for k, v in row.items() if k != "best_params"},
                "best_params": json.dumps(row["best_params"]),
            }
            for row in sorted_rows
        ]
    )
    results_df.to_csv(results_path, index=False)

    best_report = {
        "optimization_metric": optimization_metric,
        "minimize_metric": minimize,
        "selected_model_name": best["model_name"],
        "selected_model_score": best[optimization_metric],
        "selected_best_params": best["best_params"],
        "all_results_path": str(results_path),
    }
    ensure_parent(best_result_path)
    write_json(best_report, best_result_path)

    print(f"[experiments] Results saved: {results_path}")
    print(f"[experiments] Best config saved: {best_result_path}")
    print(
        f"[experiments] Selected model: {best['model_name']} "
        f"({optimization_metric}={best[optimization_metric]:.6f})"
    )


if __name__ == "__main__":
    main()
