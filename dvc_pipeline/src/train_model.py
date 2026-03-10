from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import ensure_parent, load_params, write_json


def build_model_and_space(model_name: str, random_state: int) -> Tuple[Pipeline, Dict[str, object]]:
    if model_name == "ridge":
        estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        )
        param_dist = {
            "model__alpha": np.logspace(-4, 3, 200),
        }
        return estimator, param_dist

    if model_name == "random_forest":
        estimator = Pipeline(
            steps=[
                ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1)),
            ]
        )
        param_dist = {
            "model__n_estimators": [200, 300, 500, 800],
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_split": [2, 4, 8, 12],
            "model__min_samples_leaf": [1, 2, 4, 6],
            "model__max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
        }
        return estimator, param_dist

    if model_name == "gradient_boosting":
        estimator = Pipeline(
            steps=[
                ("model", GradientBoostingRegressor(random_state=random_state)),
            ]
        )
        param_dist = {
            "model__n_estimators": [150, 250, 350, 500],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4, 5],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__min_samples_split": [2, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
        }
        return estimator, param_dist

    if model_name == "extra_trees":
        estimator = Pipeline(
            steps=[
                ("model", ExtraTreesRegressor(random_state=random_state, n_jobs=-1)),
            ]
        )
        param_dist = {
            "model__n_estimators": [200, 300, 500, 800],
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_split": [2, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
        }
        return estimator, param_dist

    raise ValueError(
        "Unsupported model_name. Use one of: ridge, random_forest, gradient_boosting, extra_trees"
    )


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true_np) < 1e-8, 1e-8, y_true_np)

    rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))
    mae = float(mean_absolute_error(y_true_np, y_pred_np))
    mape = float(np.mean(np.abs((y_true_np - y_pred_np) / denom)))
    r2 = float(r2_score(y_true_np, y_pred_np))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def main() -> None:
    params = load_params()
    cfg = params["training"]

    train_path = Path(cfg["train_data_path"])
    test_path = Path(cfg["test_data_path"])
    full_path = Path(cfg["full_data_path"])
    feature_columns_path = Path(cfg["feature_columns_path"])

    target_col = cfg["target_column"]
    date_col = cfg["date_column"]
    model_name = str(cfg["model_name"])
    random_state = int(cfg["random_state"])
    cv_splits = int(cfg["cv_splits"])
    n_iter = int(cfg["n_iter"])
    n_jobs = int(cfg["n_jobs"])
    fixed_params = cfg.get("fixed_params", {}) or {}

    output_model_path = Path(cfg["output_model_path"])
    metrics_path = Path(cfg["metrics_path"])
    predictions_path = Path(cfg["predictions_path"])

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.read_csv(full_path)

    with feature_columns_path.open("r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    X_train = train_df[feature_columns]
    y_train = train_df[target_col]
    X_test = test_df[feature_columns]
    y_test = test_df[target_col]
    X_full = full_df[feature_columns]
    y_full = full_df[target_col]

    selection_mode = "random_search"
    best_cv_rmse = None

    if fixed_params:
        selection_mode = "fixed_params"
        estimator, _ = build_model_and_space(model_name=model_name, random_state=random_state)
        estimator.set_params(**fixed_params)
        estimator.fit(X_train, y_train)
        best_model = estimator
        best_params = fixed_params
    else:
        estimator, param_dist = build_model_and_space(model_name=model_name, random_state=random_state)
        cv = TimeSeriesSplit(n_splits=cv_splits)
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

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_cv_rmse = float(-search.best_score_)
        best_params = {k: str(v) for k, v in search.best_params_.items()}

    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    train_metrics = evaluate(y_train, train_pred)
    test_metrics = evaluate(y_test, test_pred)

    # Retrain on full dataset for final output
    final_model = best_model.fit(X_full, y_full)

    ensure_parent(output_model_path)
    joblib.dump(final_model, output_model_path)

    pred_df = pd.DataFrame(
        {
            date_col: test_df[date_col],
            "actual": y_test,
            "predicted": test_pred,
        }
    )
    ensure_parent(predictions_path)
    pred_df.to_csv(predictions_path, index=False)

    metrics = {
        "model_name": model_name,
        "selection_mode": selection_mode,
        "best_cv_rmse": best_cv_rmse,
        "best_params": best_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_count": int(len(feature_columns)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "full_rows": int(len(full_df)),
        "output_model_path": str(output_model_path),
    }
    write_json(metrics, metrics_path)

    print(f"[training] Model: {model_name}")
    if best_cv_rmse is not None:
        print(f"[training] Best CV RMSE: {best_cv_rmse:.6f}")
    else:
        print("[training] Best CV RMSE: skipped (fixed_params mode)")
    print(f"[training] Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"[training] Saved model: {output_model_path}")
    print(f"[training] Saved metrics: {metrics_path}")
    print(f"[training] Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
