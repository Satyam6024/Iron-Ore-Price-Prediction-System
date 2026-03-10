from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[1]


def resolve_path(env_name: str, default_relative: str) -> Path:
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        candidate = Path(env_value)
        if candidate.is_absolute():
            return candidate
        return BASE_DIR / candidate
    return BASE_DIR / default_relative


def resolve_first_existing(default_relatives: List[str]) -> Path:
    for rel in default_relatives:
        candidate = BASE_DIR / rel
        if candidate.exists():
            return candidate
    return BASE_DIR / default_relatives[0]


MODEL_PATH = resolve_path(
    "MODEL_PATH",
    str(
        resolve_first_existing(
            [
                "dvc_pipeline/deploy/model.joblib",
                "dvc_pipeline/models/final_model.joblib",
            ]
        ).relative_to(BASE_DIR)
    ),
)
FEATURE_COLUMNS_PATH = resolve_path(
    "FEATURE_COLUMNS_PATH",
    str(
        resolve_first_existing(
            [
                "dvc_pipeline/deploy/feature_columns.json",
                "dvc_pipeline/data/features/selected_feature_columns.json",
                "dvc_pipeline/data/features/feature_columns.json",
            ]
        ).relative_to(BASE_DIR)
    ),
)
MODEL_METADATA_PATH = resolve_path(
    "MODEL_METADATA_PATH",
    "dvc_pipeline/deploy/deployment_manifest.json",
)


app = FastAPI(
    title="Iron Ore Model API",
    description="Prediction API for the trained iron ore price model.",
    version="1.0.0",
)

cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
allow_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
if not allow_origins:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
FEATURE_COLUMNS: List[str] = []
MODEL_METADATA: Dict[str, Any] = {}


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="Feature map for one prediction row."
    )


class BatchPredictRequest(BaseModel):
    rows: List[Dict[str, float]] = Field(
        ..., description="List of feature maps for batch predictions."
    )


class SimplePredictRequest(BaseModel):
    date: str = Field(..., description="Prediction date in YYYY-MM-DD format.")
    open: float = Field(..., description="Open price for the prediction day.")
    high: float = Field(..., description="High price for the prediction day.")
    low: float = Field(..., description="Low price for the prediction day.")
    volume: float = Field(..., description="Volume value for the prediction day.")
    change_pct: float = Field(..., description="Daily change percent (without % sign).")
    price_history: List[float] = Field(
        ...,
        description=(
            "Historical closing prices in chronological order (oldest -> latest), "
            "up to the day before prediction date."
        ),
    )


def load_artifacts() -> None:
    global MODEL
    global FEATURE_COLUMNS
    global MODEL_METADATA

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not FEATURE_COLUMNS_PATH.exists():
        raise RuntimeError(f"Feature columns file not found: {FEATURE_COLUMNS_PATH}")

    MODEL = joblib.load(MODEL_PATH)

    with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as f:
        FEATURE_COLUMNS = json.load(f)

    if not isinstance(FEATURE_COLUMNS, list) or not FEATURE_COLUMNS:
        raise RuntimeError("Feature columns file is invalid or empty.")

    if MODEL_METADATA_PATH.exists():
        with MODEL_METADATA_PATH.open("r", encoding="utf-8") as f:
            MODEL_METADATA = json.load(f)
    else:
        MODEL_METADATA = {}


def ensure_artifacts_loaded() -> None:
    if MODEL is not None and FEATURE_COLUMNS:
        return
    try:
        load_artifacts()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model artifacts: {exc}",
        ) from exc


def validate_feature_map(raw_features: Dict[str, Any]) -> Dict[str, float]:
    missing = [name for name in FEATURE_COLUMNS if name not in raw_features]
    extra = [name for name in raw_features if name not in FEATURE_COLUMNS]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required feature(s).",
                "missing": missing,
            },
        )

    if extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unexpected feature(s) supplied.",
                "extra": extra,
            },
        )

    normalized: Dict[str, float] = {}
    for name in FEATURE_COLUMNS:
        value = raw_features[name]
        try:
            normalized[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be numeric. Got: {value!r}",
            ) from exc

    return normalized


def _extract_required_lags_and_windows() -> tuple[List[int], List[int]]:
    lags: set[int] = set()
    windows: set[int] = set()

    lag_pattern = re.compile(r"^Price_lag_(\d+)$")
    roll_pattern = re.compile(r"^Price_roll_(?:mean|std|min|max)_(\d+)$")

    for feature in FEATURE_COLUMNS:
        lag_match = lag_pattern.match(feature)
        if lag_match:
            lags.add(int(lag_match.group(1)))

        roll_match = roll_pattern.match(feature)
        if roll_match:
            windows.add(int(roll_match.group(1)))

    # Supporting features that depend on specific lookback points.
    if "price_return_1" in FEATURE_COLUMNS:
        lags.update({1, 2})
    if "price_return_7" in FEATURE_COLUMNS:
        lags.update({1, 8})
    if "volatility_14" in FEATURE_COLUMNS:
        windows.add(14)

    return sorted(lags), sorted(windows)


def minimum_history_required() -> int:
    lags, windows = _extract_required_lags_and_windows()
    needed = 1
    if lags:
        needed = max(needed, max(lags))
    if windows:
        needed = max(needed, max(windows))

    if "price_return_1" in FEATURE_COLUMNS:
        needed = max(needed, 2)
    if "price_return_7" in FEATURE_COLUMNS:
        needed = max(needed, 8)
    if "ema_14" in FEATURE_COLUMNS:
        needed = max(needed, 14)
    if "volatility_14" in FEATURE_COLUMNS:
        needed = max(needed, 15)
    return needed


def build_features_from_simple(payload: SimplePredictRequest) -> Dict[str, float]:
    dt = pd.to_datetime(payload.date, errors="coerce")
    if pd.isna(dt):
        raise HTTPException(
            status_code=422,
            detail="Invalid date format. Use YYYY-MM-DD.",
        )

    history = [float(x) for x in payload.price_history]
    if not history:
        raise HTTPException(status_code=422, detail="price_history cannot be empty.")

    min_required = minimum_history_required()
    if len(history) < min_required:
        raise HTTPException(
            status_code=422,
            detail=f"price_history requires at least {min_required} values for current selected features.",
        )

    hist_series = pd.Series(history, dtype=float)
    if not pd.notna(hist_series).all():
        raise HTTPException(status_code=422, detail="price_history contains invalid values.")

    derived: Dict[str, float] = {
        "Open": float(payload.open),
        "High": float(payload.high),
        "Low": float(payload.low),
        "Vol.": float(payload.volume),
        "Change %": float(payload.change_pct),
        "day_of_week": float(dt.dayofweek),
        "day_of_month": float(dt.day),
        "week_of_year": float(dt.isocalendar().week),
        "month": float(dt.month),
        "quarter": float(((dt.month - 1) // 3) + 1),
        "year": float(dt.year),
        "is_month_start": float(1 if dt.is_month_start else 0),
        "is_month_end": float(1 if dt.is_month_end else 0),
    }

    lags, windows = _extract_required_lags_and_windows()
    for lag in lags:
        if len(history) >= lag:
            derived[f"Price_lag_{lag}"] = float(history[-lag])

    for w in windows:
        if len(history) >= w:
            win = hist_series.iloc[-w:]
            derived[f"Price_roll_mean_{w}"] = float(win.mean())
            derived[f"Price_roll_std_{w}"] = float(win.std())
            derived[f"Price_roll_min_{w}"] = float(win.min())
            derived[f"Price_roll_max_{w}"] = float(win.max())

    if len(history) >= 2:
        prev = history[-2]
        denom = prev if abs(prev) > 1e-8 else 1e-8
        derived["price_return_1"] = float((history[-1] - prev) / denom)

    if len(history) >= 8:
        past = history[-8]
        denom = past if abs(past) > 1e-8 else 1e-8
        derived["price_return_7"] = float((history[-1] - past) / denom)

    if len(history) >= 14:
        derived["ema_14"] = float(hist_series.ewm(span=14, adjust=False).mean().iloc[-1])

    if len(history) >= 15:
        vol = hist_series.pct_change().rolling(window=14).std().iloc[-1]
        derived["volatility_14"] = float(0.0 if pd.isna(vol) else vol)

    missing = [name for name in FEATURE_COLUMNS if name not in derived]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Could not derive required feature(s) from simple payload.",
                "missing": missing,
            },
        )

    return {name: float(derived[name]) for name in FEATURE_COLUMNS}


@app.on_event("startup")
def startup_event() -> None:
    load_artifacts()


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Iron Ore Model API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    if MODEL is None:
        try:
            load_artifacts()
        except Exception:
            pass
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "feature_count": len(FEATURE_COLUMNS),
        "model_path": str(MODEL_PATH),
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
    }


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    ensure_artifacts_loaded()
    return {
        "feature_columns": FEATURE_COLUMNS,
        "model_metadata": MODEL_METADATA,
    }


@app.get("/metadata/simple")
def metadata_simple() -> Dict[str, Any]:
    ensure_artifacts_loaded()
    return {
        "input_fields": ["date", "open", "high", "low", "volume", "change_pct", "price_history"],
        "price_history_order": "oldest_to_latest",
        "minimum_price_history_length": minimum_history_required(),
        "model_feature_count": len(FEATURE_COLUMNS),
        "model_feature_columns": FEATURE_COLUMNS,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> Dict[str, float]:
    ensure_artifacts_loaded()

    features = validate_feature_map(payload.features)
    frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    prediction = float(MODEL.predict(frame)[0])
    return {"prediction": prediction}


@app.post("/predict/simple")
def predict_simple(payload: SimplePredictRequest) -> Dict[str, float]:
    ensure_artifacts_loaded()
    features = build_features_from_simple(payload)
    frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prediction = float(MODEL.predict(frame)[0])
    return {"prediction": prediction}


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictRequest) -> Dict[str, Any]:
    ensure_artifacts_loaded()
    if not payload.rows:
        raise HTTPException(status_code=422, detail="rows cannot be empty.")

    normalized_rows: List[Dict[str, float]] = [
        validate_feature_map(row) for row in payload.rows
    ]
    frame = pd.DataFrame(normalized_rows, columns=FEATURE_COLUMNS)
    predictions = [float(x) for x in MODEL.predict(frame).tolist()]

    return {
        "count": len(predictions),
        "predictions": predictions,
    }
