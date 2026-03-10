from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import yaml


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if not isinstance(params, dict):
        raise ValueError("params.yaml content must be a dictionary")
    return params


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(data: Dict[str, Any], output_path: str | Path) -> None:
    ensure_parent(output_path)
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def convert_volume(value: Any) -> float:
    if pd.isna(value) or value == "":
        return 0.0

    if isinstance(value, (float, int, np.number)):
        return float(value)

    text = str(value).replace(",", "").strip().upper()
    multiplier = 1.0

    if text.endswith("K"):
        multiplier = 1_000.0
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith("B"):
        multiplier = 1_000_000_000.0
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def add_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    frame = df.copy()
    frame["day_of_week"] = frame[date_col].dt.dayofweek
    frame["day_of_month"] = frame[date_col].dt.day
    frame["week_of_year"] = frame[date_col].dt.isocalendar().week.astype(int)
    frame["month"] = frame[date_col].dt.month
    frame["quarter"] = frame[date_col].dt.quarter
    frame["year"] = frame[date_col].dt.year
    frame["is_month_start"] = frame[date_col].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame[date_col].dt.is_month_end.astype(int)
    return frame


def add_lag_roll_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> pd.DataFrame:
    frame = df.copy()

    for lag in lags:
        frame[f"{target_col}_lag_{lag}"] = frame[target_col].shift(int(lag))

    for window in rolling_windows:
        w = int(window)
        shifted = frame[target_col].shift(1)
        frame[f"{target_col}_roll_mean_{w}"] = shifted.rolling(window=w).mean()
        frame[f"{target_col}_roll_std_{w}"] = shifted.rolling(window=w).std()
        frame[f"{target_col}_roll_min_{w}"] = shifted.rolling(window=w).min()
        frame[f"{target_col}_roll_max_{w}"] = shifted.rolling(window=w).max()

    frame["price_return_1"] = frame[target_col].pct_change(1)
    frame["price_return_7"] = frame[target_col].pct_change(7)
    frame["ema_14"] = frame[target_col].ewm(span=14, adjust=False).mean().shift(1)
    frame["volatility_14"] = frame["price_return_1"].rolling(window=14).std().shift(1)

    return frame
