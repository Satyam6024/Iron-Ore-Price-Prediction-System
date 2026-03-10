from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression

from common import ensure_parent, load_params, write_json


def run_random_forest_importance(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_estimators: int,
) -> List[str]:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    scores = model.feature_importances_
    ranked = sorted(zip(X.columns.tolist(), scores.tolist()), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked]


def run_f_regression(X: pd.DataFrame, y: pd.Series) -> List[str]:
    scores, _ = f_regression(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    ranked = sorted(zip(X.columns.tolist(), scores.tolist()), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked]


def main() -> None:
    params = load_params()
    cfg = params["feature_selection"]

    train_input_path = Path(cfg["train_input_path"])
    all_feature_columns_path = Path(cfg["all_feature_columns_path"])
    output_selected_path = Path(cfg["selected_columns_output_path"])
    report_path = Path(cfg["report_path"])

    target_col = str(cfg["target_column"])
    method = str(cfg["method"])
    top_k = int(cfg["top_k"])
    random_state = int(cfg.get("random_state", 42))
    n_estimators = int(cfg.get("n_estimators", 400))
    mandatory_features = list(cfg.get("mandatory_features", []))

    if not train_input_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_input_path}")
    if not all_feature_columns_path.exists():
        raise FileNotFoundError(f"Feature columns file not found: {all_feature_columns_path}")

    train_df = pd.read_csv(train_input_path)
    with all_feature_columns_path.open("r", encoding="utf-8") as f:
        all_features = json.load(f)

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train dataset")

    valid_features = [c for c in all_features if c in train_df.columns]
    if not valid_features:
        raise ValueError("No valid feature columns found in training dataset.")

    X = train_df[valid_features]
    y = train_df[target_col]

    if method == "none":
        ranked_features = valid_features
    elif method == "random_forest_importance":
        ranked_features = run_random_forest_importance(
            X=X, y=y, random_state=random_state, n_estimators=n_estimators
        )
    elif method == "f_regression":
        ranked_features = run_f_regression(X=X, y=y)
    else:
        raise ValueError(
            "Unsupported feature_selection.method. "
            "Use one of: none, random_forest_importance, f_regression"
        )

    top_k = max(1, min(top_k, len(ranked_features)))
    selected = ranked_features[:top_k]

    # Force-include mandatory features while preserving order.
    for feat in mandatory_features:
        if feat in valid_features and feat not in selected:
            selected.append(feat)

    ensure_parent(output_selected_path)
    with output_selected_path.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    report: Dict[str, Any] = {
        "method": method,
        "top_k_requested": int(cfg["top_k"]),
        "selected_count": len(selected),
        "selected_columns_output_path": str(output_selected_path),
        "target_column": target_col,
        "mandatory_features": mandatory_features,
        "top_20_ranked_features": ranked_features[:20],
    }
    write_json(report, report_path)

    print(f"[feature_selection] Method: {method}")
    print(f"[feature_selection] Selected {len(selected)} features")
    print(f"[feature_selection] Output: {output_selected_path}")
    print(f"[feature_selection] Report: {report_path}")


if __name__ == "__main__":
    main()
