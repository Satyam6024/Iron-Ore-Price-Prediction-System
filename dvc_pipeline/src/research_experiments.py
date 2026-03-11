from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from common import ensure_parent, load_params, write_json
from research_sequence_models import fit_predict_arima_lstm


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / denom))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def lag_frame(series: np.ndarray, lags: Iterable[int]) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(series, dtype=float)})
    frame["t"] = np.arange(len(frame))
    for lag in lags:
        frame[f"lag_{int(lag)}"] = frame["y"].shift(int(lag))
    return frame.dropna().reset_index(drop=True)


def lag_split(
    train_values: np.ndarray,
    test_values: np.ndarray,
    lags: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_t = len(train_values)
    full = np.concatenate([train_values, test_values])
    frame = lag_frame(full, lags=lags)
    train_part = frame[frame["t"] < split_t].copy()
    test_part = frame[frame["t"] >= split_t].copy()

    x_cols = [col for col in frame.columns if col.startswith("lag_")]
    x_train = train_part[x_cols].to_numpy(dtype=float)
    y_train = train_part["y"].to_numpy(dtype=float)
    x_test = test_part[x_cols].to_numpy(dtype=float)
    return x_train, y_train, x_test


def fit_predict_arima(train_values: np.ndarray, test_values: np.ndarray, order: Tuple[int, int, int]) -> np.ndarray:
    try:
        result = ARIMA(train_values, order=order).fit()
        preds: List[float] = []
        for actual in np.asarray(test_values, dtype=float):
            next_pred = float(np.asarray(result.forecast(steps=1), dtype=float)[0])
            preds.append(next_pred)
            try:
                result = result.append([float(actual)], refit=False)
            except Exception:
                history = np.concatenate([np.asarray(train_values, dtype=float), np.asarray(test_values[: len(preds)], dtype=float)])
                result = ARIMA(history, order=order).fit()
        return np.asarray(preds, dtype=float)
    except Exception:
        # Fallback keeps pipeline resilient even when ARIMA fails for a split.
        return np.repeat(float(train_values[-1]), len(test_values))


def fit_predict_holt_winters(train_values: np.ndarray, test_values: np.ndarray) -> np.ndarray:
    history = list(np.asarray(train_values, dtype=float))
    preds: List[float] = []
    for actual in np.asarray(test_values, dtype=float):
        try:
            fitted = ExponentialSmoothing(
                history,
                trend="add",
                damped_trend=True,
                initialization_method="estimated",
            ).fit(optimized=True)
            next_pred = float(np.asarray(fitted.forecast(1), dtype=float)[0])
        except Exception:
            next_pred = float(history[-1])
        preds.append(next_pred)
        history.append(float(actual))
    return np.asarray(preds, dtype=float)


def fit_predict_svr(
    train_values: np.ndarray,
    test_values: np.ndarray,
    lags: Iterable[int],
) -> np.ndarray:
    x_train, y_train, x_test = lag_split(train_values, test_values, lags=lags)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale")),
        ]
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def fit_predict_random_forest(
    train_values: np.ndarray,
    test_values: np.ndarray,
    lags: Iterable[int],
    random_state: int,
) -> np.ndarray:
    x_train, y_train, x_test = lag_split(train_values, test_values, lags=lags)
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def fit_predict_gradient_boosting(
    train_values: np.ndarray,
    test_values: np.ndarray,
    lags: Iterable[int],
    random_state: int,
) -> np.ndarray:
    x_train, y_train, x_test = lag_split(train_values, test_values, lags=lags)
    model = GradientBoostingRegressor(
        random_state=random_state,
        n_estimators=350,
        learning_rate=0.03,
        max_depth=4,
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def compute_hybrid_weights(
    train_values: np.ndarray,
    lags: Iterable[int],
    arima_order: Tuple[int, int, int],
    random_state: int,
) -> Dict[str, float]:
    val_size = max(40, min(200, len(train_values) // 5))
    core = train_values[:-val_size]
    val = train_values[-val_size:]

    arima_val = fit_predict_arima(core, val, order=arima_order)
    gb_val = fit_predict_gradient_boosting(core, val, lags=lags, random_state=random_state)

    arima_rmse = float(np.sqrt(mean_squared_error(val, arima_val)))
    gb_rmse = float(np.sqrt(mean_squared_error(val, gb_val)))
    arima_inv = 1.0 / max(arima_rmse, 1e-8)
    gb_inv = 1.0 / max(gb_rmse, 1e-8)
    total = arima_inv + gb_inv
    return {
        "arima": float(arima_inv / total),
        "gradient_boosting": float(gb_inv / total),
        "validation_arima_rmse": arima_rmse,
        "validation_gradient_boosting_rmse": gb_rmse,
    }


def volatility_regime_mask(
    train_values: np.ndarray,
    test_values: np.ndarray,
    window: int,
    quantile: float,
) -> Tuple[np.ndarray, float]:
    full = pd.Series(np.concatenate([train_values, test_values]), dtype=float)
    vol = full.pct_change().rolling(window=window).std()
    train_vol = vol.iloc[: len(train_values)]
    train_vol_non_null = train_vol.dropna()
    if train_vol_non_null.empty:
        threshold = 0.0
    else:
        threshold = float(train_vol_non_null.quantile(quantile))
    test_vol = vol.iloc[len(train_values) :].to_numpy(dtype=float)
    high_mask = np.where(np.isnan(test_vol), False, test_vol >= threshold)
    return high_mask, threshold


def ewma_variance_rmse(train_values: np.ndarray, test_values: np.ndarray, lam: float = 0.94) -> float:
    train_ret = pd.Series(train_values, dtype=float).pct_change().dropna().to_numpy(dtype=float)
    if len(train_ret) == 0:
        return float("nan")

    sigma2 = float(np.var(train_ret))
    prev_r = float(train_ret[-1])
    test_bridge = np.concatenate([[float(train_values[-1])], np.asarray(test_values, dtype=float)])
    test_ret = pd.Series(test_bridge, dtype=float).pct_change().dropna().to_numpy(dtype=float)
    preds: List[float] = []
    for r in test_ret:
        sigma2 = lam * sigma2 + (1.0 - lam) * (prev_r**2)
        preds.append(float(sigma2))
        prev_r = float(r)
    if not preds:
        return float("nan")
    actual_var = test_ret**2
    return float(np.sqrt(np.mean((actual_var - np.asarray(preds, dtype=float)) ** 2)))


def regime_rmse(y_true: np.ndarray, y_pred: np.ndarray, high_mask: np.ndarray) -> Tuple[float, float]:
    high_idx = np.asarray(high_mask, dtype=bool)
    low_idx = ~high_idx
    if high_idx.any():
        high = float(np.sqrt(mean_squared_error(y_true[high_idx], y_pred[high_idx])))
    else:
        high = float("nan")
    if low_idx.any():
        low = float(np.sqrt(mean_squared_error(y_true[low_idx], y_pred[low_idx])))
    else:
        low = float("nan")
    return high, low


def rolling_rmse_stats(y_true: np.ndarray, y_pred: np.ndarray, window: int) -> Tuple[float, float]:
    err = pd.Series(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    roll = np.sqrt(err.pow(2).rolling(window=window, min_periods=5).mean())
    if roll.dropna().empty:
        fallback = float(np.sqrt(np.mean(np.square(err))))
        return fallback, fallback
    return float(roll.mean(skipna=True)), float(roll.max(skipna=True))


def main() -> None:
    params = load_params()
    cfg = params["research_experiments"]

    input_path = Path(cfg["input_path"])
    date_col = str(cfg["date_column"])
    target_col = str(cfg["target_column"])
    test_size = int(cfg["test_size"])
    lags = [int(x) for x in cfg["lag_features"]]
    arima_order = tuple(int(x) for x in cfg["arima_order"])
    rolling_window = int(cfg["rolling_window"])
    volatility_window = int(cfg["volatility_window"])
    high_quantile = float(cfg["high_volatility_quantile"])
    procurement_volume = float(cfg["procurement_volume_tons"])
    random_state = int(params["training"]["random_state"])
    arima_lstm_cfg = cfg.get("arima_lstm", {})

    results_path = Path(cfg["results_path"])
    summary_path = Path(cfg["summary_path"])
    predictions_path = Path(cfg["predictions_path"])
    arima_lstm_history_path = Path(arima_lstm_cfg["history_path"])

    if not input_path.exists():
        raise FileNotFoundError(f"Research experiments input not found: {input_path}")

    frame = pd.read_csv(input_path)
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)
    if len(frame) <= test_size + max(lags):
        raise ValueError(
            f"Not enough data for research experiments: rows={len(frame)}, "
            f"required>{test_size + max(lags)}"
        )

    values = frame[target_col].to_numpy(dtype=float)
    dates = frame[date_col].dt.strftime("%Y-%m-%d").to_numpy()
    train_values = values[:-test_size]
    test_values = values[-test_size:]
    test_dates = dates[-test_size:]

    predictions: Dict[str, np.ndarray] = {}
    predictions["arima"] = fit_predict_arima(train_values, test_values, order=arima_order)
    predictions["holt_winters"] = fit_predict_holt_winters(train_values, test_values)
    predictions["svr_rbf"] = fit_predict_svr(train_values, test_values, lags=lags)
    predictions["random_forest"] = fit_predict_random_forest(
        train_values, test_values, lags=lags, random_state=random_state
    )
    predictions["gradient_boosting"] = fit_predict_gradient_boosting(
        train_values, test_values, lags=lags, random_state=random_state
    )

    hybrid_weights = compute_hybrid_weights(
        train_values=train_values,
        lags=lags,
        arima_order=arima_order,
        random_state=random_state,
    )
    predictions["hybrid_arima_gradient_boosting"] = (
        hybrid_weights["arima"] * predictions["arima"]
        + hybrid_weights["gradient_boosting"] * predictions["gradient_boosting"]
    )
    predictions["hybrid_arima_lstm"], arima_lstm_details = fit_predict_arima_lstm(
        train_values=train_values,
        test_values=test_values,
        arima_order=arima_order,
        sequence_length=int(arima_lstm_cfg["sequence_length"]),
        hidden_size=int(arima_lstm_cfg["hidden_size"]),
        num_layers=int(arima_lstm_cfg["num_layers"]),
        dropout=float(arima_lstm_cfg["dropout"]),
        learning_rate=float(arima_lstm_cfg["learning_rate"]),
        batch_size=int(arima_lstm_cfg["batch_size"]),
        epochs=int(arima_lstm_cfg["epochs"]),
        validation_ratio=float(arima_lstm_cfg["validation_ratio"]),
        patience=int(arima_lstm_cfg["patience"]),
        random_state=random_state,
    )

    high_vol_mask, vol_threshold = volatility_regime_mask(
        train_values=train_values,
        test_values=test_values,
        window=volatility_window,
        quantile=high_quantile,
    )
    ewma_var_rmse = ewma_variance_rmse(train_values=train_values, test_values=test_values)

    rows: List[Dict[str, Any]] = []
    for model_name, pred in predictions.items():
        base = evaluate(test_values, pred)
        rmse_high, rmse_low = regime_rmse(test_values, pred, high_mask=high_vol_mask)
        roll_mean, roll_max = rolling_rmse_stats(
            test_values,
            pred,
            window=min(max(rolling_window, 5), len(test_values)),
        )
        procurement_cost = float(np.abs(test_values - pred).sum() * procurement_volume)

        rows.append(
            {
                "model_name": model_name,
                "rmse": base["rmse"],
                "mae": base["mae"],
                "mape": base["mape"],
                "r2": base["r2"],
                "rmse_high_volatility": rmse_high,
                "rmse_low_volatility": rmse_low,
                "rolling_rmse_mean": roll_mean,
                "rolling_rmse_max": roll_max,
                "procurement_abs_error_cost": procurement_cost,
            }
        )

    results_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    best_row = results_df.iloc[0].to_dict()

    pred_frame = pd.DataFrame({"Date": test_dates, "actual": test_values})
    for model_name, pred in predictions.items():
        pred_frame[f"pred_{model_name}"] = pred

    ensure_parent(results_path)
    ensure_parent(predictions_path)
    ensure_parent(arima_lstm_history_path)
    results_df.to_csv(results_path, index=False)
    pred_frame.to_csv(predictions_path, index=False)
    write_json(arima_lstm_details, arima_lstm_history_path)

    summary = {
        "input_path": str(input_path),
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_values)),
        "rows_test": int(len(test_values)),
        "lags": lags,
        "arima_order": list(arima_order),
        "best_model_by_rmse": best_row.get("model_name"),
        "best_rmse": float(best_row.get("rmse")),
        "best_procurement_cost": float(best_row.get("procurement_abs_error_cost")),
        "hybrid_weights": hybrid_weights,
        "volatility": {
            "window": volatility_window,
            "high_volatility_quantile": high_quantile,
            "high_volatility_threshold": vol_threshold,
            "ewma_variance_rmse": ewma_var_rmse,
        },
        "arima_lstm": {
            "history_path": str(arima_lstm_history_path),
            "sequence_length": int(arima_lstm_cfg["sequence_length"]),
            "hidden_size": int(arima_lstm_cfg["hidden_size"]),
            "num_layers": int(arima_lstm_cfg["num_layers"]),
            "dropout": float(arima_lstm_cfg["dropout"]),
            "epochs_ran": int(arima_lstm_details["epochs_ran"]),
            "best_epoch": int(arima_lstm_details["best_epoch"]),
            "best_val_loss": float(arima_lstm_details["best_val_loss"]),
            "device": arima_lstm_details["device"],
        },
        "results_path": str(results_path),
        "predictions_path": str(predictions_path),
    }
    write_json(summary, summary_path)

    print(f"[research_experiments] Input: {input_path}")
    print(f"[research_experiments] Results: {results_path}")
    print(f"[research_experiments] Predictions: {predictions_path}")
    print(f"[research_experiments] ARIMA+LSTM history: {arima_lstm_history_path}")
    print(f"[research_experiments] Summary: {summary_path}")
    print(
        f"[research_experiments] Best model: {summary['best_model_by_rmse']} "
        f"(rmse={summary['best_rmse']:.6f})"
    )


if __name__ == "__main__":
    main()
