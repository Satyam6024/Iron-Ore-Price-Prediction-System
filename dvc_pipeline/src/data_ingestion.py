from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import ensure_parent, load_params, write_json


def main() -> None:
    params = load_params()
    cfg = params["data_ingestion"]

    source_path = Path(cfg["raw_data_path"])
    output_path = Path(cfg["output_path"])
    report_path = Path(cfg["report_path"])

    if not source_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {source_path}")

    df = pd.read_csv(source_path)
    ensure_parent(output_path)
    df.to_csv(output_path, index=False)

    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "source_path": str(source_path),
        "output_path": str(output_path),
    }
    write_json(report, report_path)

    print(f"[data_ingestion] Source: {source_path}")
    print(f"[data_ingestion] Output: {output_path} | shape={df.shape}")
    print(f"[data_ingestion] Report: {report_path}")


if __name__ == "__main__":
    main()
