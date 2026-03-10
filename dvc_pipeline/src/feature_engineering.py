from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import add_lag_roll_features, add_time_features, ensure_parent, load_params, write_json


def main() -> None:
    params = load_params()
    cfg = params["feature_engineering"]
    training_cfg = params["training"]

    input_path = Path(cfg["input_path"])
    output_path = Path(cfg["output_path"])
    train_output_path = Path(cfg["train_output_path"])
    test_output_path = Path(cfg["test_output_path"])
    feature_columns_path = Path(cfg["feature_columns_path"])
    report_path = Path(cfg["report_path"])

    target_col = cfg["target_column"]
    date_col = cfg["date_column"]
    lags = cfg["lags"]
    rolling_windows = cfg["rolling_windows"]
    test_size = int(training_cfg["test_size"])

    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)

    df = add_time_features(df, date_col=date_col)
    df = add_lag_roll_features(
        df=df,
        target_col=target_col,
        lags=lags,
        rolling_windows=rolling_windows,
    )
    df = df.dropna().reset_index(drop=True)

    if len(df) <= test_size:
        raise ValueError(
            f"Not enough rows after feature engineering. rows={len(df)}, test_size={test_size}"
        )

    feature_columns = [col for col in df.columns if col not in {date_col, target_col}]
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()

    # Save outputs
    ensure_parent(output_path)
    ensure_parent(train_output_path)
    ensure_parent(test_output_path)
    ensure_parent(feature_columns_path)

    df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")
    train_df[date_col] = train_df[date_col].dt.strftime("%Y-%m-%d")
    test_df[date_col] = test_df[date_col].dt.strftime("%Y-%m-%d")

    df.to_csv(output_path, index=False)
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    with feature_columns_path.open("w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    report = {
        "rows_after_feature_engineering": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "target_column": target_col,
        "date_column": date_col,
        "lags": [int(x) for x in lags],
        "rolling_windows": [int(x) for x in rolling_windows],
    }
    write_json(report, report_path)

    print(f"[feature_engineering] Input: {input_path}")
    print(f"[feature_engineering] Output full: {output_path} | rows={len(df)}")
    print(f"[feature_engineering] Output train: {train_output_path} | rows={len(train_df)}")
    print(f"[feature_engineering] Output test: {test_output_path} | rows={len(test_df)}")
    print(f"[feature_engineering] Feature cols: {feature_columns_path} | count={len(feature_columns)}")


if __name__ == "__main__":
    main()
