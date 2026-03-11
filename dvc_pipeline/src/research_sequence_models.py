from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.arima.model import ARIMA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pred = self.head(out[:, -1, :])
        return pred.squeeze(-1)


def _build_sequences(series: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    x_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    for idx in range(sequence_length, len(series)):
        x_rows.append(series[idx - sequence_length : idx])
        y_rows.append(float(series[idx]))
    if not x_rows:
        raise ValueError(
            f"Not enough residual history for LSTM sequences: len={len(series)}, sequence_length={sequence_length}"
        )
    x_arr = np.asarray(x_rows, dtype=np.float32).reshape(-1, sequence_length, 1)
    y_arr = np.asarray(y_rows, dtype=np.float32)
    return x_arr, y_arr


def _scale_series(series: np.ndarray) -> Tuple[np.ndarray, float, float]:
    series = np.asarray(series, dtype=np.float32)
    mean = float(series.mean())
    std = float(series.std())
    if std < 1e-8:
        std = 1.0
    scaled = (series - mean) / std
    return scaled.astype(np.float32), mean, std


def train_lstm_on_residuals(
    residuals: np.ndarray,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    validation_ratio: float,
    patience: int,
    random_state: int,
) -> Dict[str, Any]:
    set_seed(random_state)

    residual_series = pd.Series(np.asarray(residuals, dtype=float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled, mean, std = _scale_series(residual_series.to_numpy(dtype=np.float32))
    x_all, y_all = _build_sequences(scaled, sequence_length=sequence_length)

    if len(x_all) < 32:
        raise ValueError(f"Too few training sequences for ARIMA+LSTM: {len(x_all)}")

    val_count = max(1, int(len(x_all) * validation_ratio))
    if len(x_all) - val_count < 8:
        val_count = max(1, len(x_all) // 5)

    x_train = x_all[:-val_count]
    y_train = y_all[:-val_count]
    x_val = x_all[-val_count:]
    y_val = y_all[-val_count:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualLSTM(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "device": str(device),
        "residual_mean": mean,
        "residual_std": std,
        "sequence_length": sequence_length,
        "history": history,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
    }


def forecast_residuals(
    model_artifact: Dict[str, Any],
    residual_history: np.ndarray,
    horizon: int,
) -> np.ndarray:
    model: ResidualLSTM = model_artifact["model"]
    device = torch.device(model_artifact["device"])
    mean = float(model_artifact["residual_mean"])
    std = float(model_artifact["residual_std"])
    sequence_length = int(model_artifact["sequence_length"])

    history = pd.Series(np.asarray(residual_history, dtype=float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled = ((history.to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
    if len(scaled) < sequence_length:
        raise ValueError(
            f"Residual history shorter than sequence length: len={len(scaled)}, sequence_length={sequence_length}"
        )
    window = scaled[-sequence_length:].copy()

    preds: List[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            x_tensor = torch.from_numpy(window.reshape(1, sequence_length, 1)).to(device)
            next_scaled = float(model(x_tensor).cpu().item())
            next_value = next_scaled * std + mean
            preds.append(next_value)
            window = np.concatenate([window[1:], np.asarray([next_scaled], dtype=np.float32)])

    return np.asarray(preds, dtype=float)


def fit_predict_arima_lstm(
    train_values: np.ndarray,
    test_values: np.ndarray,
    arima_order: Tuple[int, int, int],
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    validation_ratio: float,
    patience: int,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    fitted = ARIMA(np.asarray(train_values, dtype=float), order=arima_order).fit()
    train_residuals = np.asarray(fitted.resid, dtype=float)
    train_residuals = pd.Series(train_residuals).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)

    artifact = train_lstm_on_residuals(
        residuals=train_residuals,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        validation_ratio=validation_ratio,
        patience=patience,
        random_state=random_state,
    )
    result = fitted
    residual_history = list(train_residuals.astype(float))
    hybrid_preds: List[float] = []
    arima_preds: List[float] = []
    residual_preds: List[float] = []

    for actual in np.asarray(test_values, dtype=float):
        arima_pred = float(np.asarray(result.forecast(steps=1), dtype=float)[0])
        residual_pred = float(forecast_residuals(artifact, residual_history=np.asarray(residual_history), horizon=1)[0])
        hybrid_preds.append(arima_pred + residual_pred)
        arima_preds.append(arima_pred)
        residual_preds.append(residual_pred)

        actual_residual = float(actual - arima_pred)
        residual_history.append(actual_residual)
        try:
            result = result.append([float(actual)], refit=False)
        except Exception:
            history_values = np.concatenate(
                [np.asarray(train_values, dtype=float), np.asarray(test_values[: len(hybrid_preds)], dtype=float)]
            )
            result = ARIMA(history_values, order=arima_order).fit()

    history = artifact["history"]
    final_train_loss = float(history[-1]["train_loss"]) if history else float("nan")
    final_val_loss = float(history[-1]["val_loss"]) if history else float("nan")

    details = {
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_ran": len(history),
        "best_epoch": int(artifact["best_epoch"]),
        "best_val_loss": float(artifact["best_val_loss"]),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "device": artifact["device"],
        "rolling_arima_mean_abs_residual_pred": float(np.mean(np.abs(np.asarray(residual_preds, dtype=float)))),
        "rolling_arima_mean_abs_pred": float(np.mean(np.abs(np.asarray(arima_preds, dtype=float)))),
        "history": history,
    }
    return np.asarray(hybrid_preds, dtype=float), details
