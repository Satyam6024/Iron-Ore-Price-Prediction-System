from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import convert_volume, ensure_parent, load_params, write_json


def main() -> None:
    params = load_params()
    cfg = params["data_cleaning"]

    input_path = Path(cfg["input_path"])
    output_path = Path(cfg["output_path"])
    report_path = Path(cfg["report_path"])
    keep_weekdays_only = bool(cfg.get("keep_weekdays_only", True))

    if not input_path.exists():
        raise FileNotFoundError(f"Ingestion output not found: {input_path}")

    df = pd.read_csv(input_path)
    original_rows = len(df)

    df.columns = [col.strip() for col in df.columns]
    expected_columns = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Price", "Open", "High", "Low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Change %"] = (
        df["Change %"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Change %"] = pd.to_numeric(df["Change %"], errors="coerce")
    df["Vol."] = df["Vol."].apply(convert_volume)

    df = df.dropna(subset=["Date", "Price"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    if keep_weekdays_only:
        df = df[df["Date"].dt.dayofweek < 5]

    numeric_cols = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    df["Vol."] = df["Vol."].fillna(0.0)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    cleaned_rows = len(df)
    date_min = df["Date"].min()
    date_max = df["Date"].max()

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    ensure_parent(output_path)
    df[expected_columns].to_csv(output_path, index=False)

    report = {
        "original_rows": int(original_rows),
        "cleaned_rows": int(cleaned_rows),
        "dropped_rows": int(original_rows - cleaned_rows),
        "date_min": str(date_min.date()) if pd.notna(date_min) else None,
        "date_max": str(date_max.date()) if pd.notna(date_max) else None,
        "keep_weekdays_only": keep_weekdays_only,
        "output_path": str(output_path),
    }
    write_json(report, report_path)

    print(f"[data_cleaning] Input: {input_path} | rows={original_rows}")
    print(f"[data_cleaning] Output: {output_path} | rows={cleaned_rows}")
    print(f"[data_cleaning] Report: {report_path}")


if __name__ == "__main__":
    main()
