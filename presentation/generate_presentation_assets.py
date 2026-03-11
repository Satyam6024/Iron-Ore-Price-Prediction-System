from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "presentation" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11


def save(fig: plt.Figure, name: str) -> None:
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[presentation] saved {path}")


def load_params() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8"))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_market_dynamics(clean_df: pd.DataFrame) -> None:
    data = clean_df.copy()
    data["Price_30dma"] = data["Price"].rolling(30).mean()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.lineplot(data=data, x="Date", y="Price", ax=ax, label="Daily price", linewidth=1.5)
    sns.lineplot(data=data, x="Date", y="Price_30dma", ax=ax, label="30-day moving average", linewidth=2.2)
    ax.set_title("Iron Ore Market Dynamics")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    save(fig, "01_market_dynamics.png")


def plot_volatility(clean_df: pd.DataFrame) -> None:
    data = clean_df.copy()
    data["Return_1d"] = data["Price"].pct_change()
    data["Volatility_14d"] = data["Return_1d"].rolling(14).std()
    threshold = data["Volatility_14d"].dropna().quantile(0.75)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    sns.lineplot(data=data, x="Date", y="Return_1d", ax=axes[0], color="#2b6cb0", linewidth=1.1)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Daily Returns")
    axes[0].set_ylabel("Return")

    sns.lineplot(data=data, x="Date", y="Volatility_14d", ax=axes[1], color="#c05621", linewidth=1.5)
    axes[1].axhline(threshold, color="#7b341e", linestyle="--", linewidth=1.0, label="75th percentile")
    axes[1].set_title("Conditional Volatility Proxy")
    axes[1].set_ylabel("14-day rolling std")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    save(fig, "02_volatility_regime.png")


def plot_seasonality(clean_df: pd.DataFrame) -> None:
    data = clean_df.copy()
    data["Month"] = pd.Categorical(
        data["Date"].dt.strftime("%b"),
        categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        ordered=True,
    )
    data["Weekday"] = pd.Categorical(
        data["Date"].dt.strftime("%a"),
        categories=["Mon", "Tue", "Wed", "Thu", "Fri"],
        ordered=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.boxplot(data=data, x="Month", y="Price", ax=axes[0], color="#90cdf4")
    axes[0].set_title("Monthly Seasonality")
    axes[0].tick_params(axis="x", rotation=45)

    weekday_avg = data.groupby("Weekday", observed=False)["Price"].mean().reset_index()
    sns.barplot(data=weekday_avg, x="Weekday", y="Price", ax=axes[1], color="#f6ad55")
    axes[1].set_title("Average Price by Weekday")
    save(fig, "03_seasonality.png")


def plot_correlation(clean_df: pd.DataFrame) -> None:
    cols = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    corr = clean_df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", square=True, ax=ax)
    ax.set_title("Core Feature Correlation")
    save(fig, "04_correlation_heatmap.png")


def plot_production_comparison(prod_df: pd.DataFrame) -> None:
    ordered = prod_df.sort_values("test_rmse").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.barplot(
        data=ordered,
        x="test_rmse",
        y="model_name",
        hue="model_name",
        dodge=False,
        legend=False,
        ax=axes[0],
        palette="crest",
    )
    axes[0].set_title("Production Model Comparison")
    axes[0].set_xlabel("Test RMSE")
    axes[0].set_ylabel("")

    long_df = ordered.melt(
        id_vars="model_name",
        value_vars=["train_rmse", "test_rmse"],
        var_name="split",
        value_name="rmse",
    )
    sns.barplot(data=long_df, x="model_name", y="rmse", hue="split", ax=axes[1], palette="Set2")
    axes[1].set_title("Generalization Check")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_xlabel("")
    save(fig, "05_production_model_comparison.png")


def plot_research_benchmark(research_df: pd.DataFrame) -> None:
    ordered = research_df.sort_values("rmse").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    sns.barplot(
        data=ordered,
        x="rmse",
        y="model_name",
        hue="model_name",
        dodge=False,
        legend=False,
        ax=axes[0],
        palette="mako",
    )
    axes[0].set_title("Research Benchmark: RMSE")
    axes[0].set_xlabel("RMSE")
    axes[0].set_ylabel("")

    melt_df = ordered.melt(
        id_vars="model_name",
        value_vars=["rmse_high_volatility", "rmse_low_volatility"],
        var_name="regime",
        value_name="regime_rmse",
    )
    sns.barplot(data=melt_df, x="model_name", y="regime_rmse", hue="regime", ax=axes[1], palette="flare")
    axes[1].set_title("Regime-Sensitive Error")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_xlabel("")
    save(fig, "06_research_benchmark.png")


def plot_prediction_traces(pred_df: pd.DataFrame) -> None:
    view = pred_df.copy()
    view["Date"] = pd.to_datetime(view["Date"])
    view = view.tail(80)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.lineplot(data=view, x="Date", y="actual", ax=ax, label="Actual", linewidth=2.2, color="black")
    sns.lineplot(data=view, x="Date", y="pred_arima", ax=ax, label="ARIMA", linewidth=1.5)
    sns.lineplot(data=view, x="Date", y="pred_svr_rbf", ax=ax, label="SVR-RBF", linewidth=1.6)
    sns.lineplot(
        data=view,
        x="Date",
        y="pred_hybrid_arima_lstm",
        ax=ax,
        label="Hybrid ARIMA + LSTM",
        linewidth=1.8,
    )
    ax.set_title("Test Window Prediction Trace")
    ax.set_ylabel("Price")
    save(fig, "07_research_prediction_traces.png")


def plot_sequential_decomposition() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")

    boxes = [
        (0.03, 0.35, 0.16, 0.3, "#bee3f8", "Raw price series"),
        (0.24, 0.35, 0.16, 0.3, "#fbd38d", "ARIMA\nlinear signal"),
        (0.45, 0.35, 0.16, 0.3, "#fed7d7", "Residuals\nnonlinear part"),
        (0.66, 0.35, 0.16, 0.3, "#c6f6d5", "LSTM\nsequence learner"),
        (0.84, 0.35, 0.13, 0.3, "#e9d8fd", "Final\nforecast"),
    ]

    for x, y, w, h, color, label in boxes:
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=color,
            edgecolor="#2d3748",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    arrows = [(0.19, 0.5, 0.24, 0.5), (0.40, 0.5, 0.45, 0.5), (0.61, 0.5, 0.66, 0.5), (0.82, 0.5, 0.84, 0.5)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.8})

    ax.set_title("Sequential Decomposition Philosophy (Paper Concept)")
    save(fig, "08_sequential_decomposition.png")


def plot_mlop_flow() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.axis("off")

    labels = [
        "Raw Data",
        "Cleaning + EDA",
        "Feature Eng.",
        "Experiments",
        "Train",
        "MLflow Registry",
        "Deploy API",
        "Frontend",
    ]
    colors = ["#bee3f8", "#c6f6d5", "#fefcbf", "#fbd38d", "#fed7d7", "#e9d8fd", "#90cdf4", "#fbb6ce"]
    xs = np.linspace(0.03, 0.87, len(labels))

    for x, label, color in zip(xs, labels, colors):
        rect = FancyBboxPatch(
            (x, 0.35),
            0.1,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=color,
            edgecolor="#2d3748",
            linewidth=1.1,
        )
        ax.add_patch(rect)
        ax.text(x + 0.05, 0.49, label, ha="center", va="center", fontsize=10)

    for idx in range(len(labels) - 1):
        ax.annotate(
            "",
            xy=(xs[idx + 1], 0.49),
            xytext=(xs[idx] + 0.1, 0.49),
            arrowprops={"arrowstyle": "->", "lw": 1.6},
        )

    ax.text(0.47, 0.16, "DVC orchestrates stages, MLflow tracks and governs model lifecycle", ha="center", fontsize=11)
    ax.set_title("End-to-End MLOps Workflow")
    save(fig, "09_mlop_flow.png")


def plot_production_fit(test_pred_df: pd.DataFrame) -> None:
    view = test_pred_df.copy()
    view["Date"] = pd.to_datetime(view["Date"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.lineplot(data=view.tail(100), x="Date", y="actual", ax=axes[0], label="Actual", linewidth=2.2, color="black")
    sns.lineplot(data=view.tail(100), x="Date", y="predicted", ax=axes[0], label="Predicted", linewidth=1.8)
    axes[0].set_title("Production Model Fit on Test Window")
    axes[0].set_ylabel("Price")

    residuals = view["actual"] - view["predicted"]
    sns.histplot(residuals, bins=25, kde=True, ax=axes[1], color="#4c51bf")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Actual - Predicted")
    save(fig, "10_production_fit.png")


def plot_svr_formula() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    ax.axis("off")
    formula = r"$f(x) = \sum_{i=1}^{n} (\alpha_i - \alpha_i^*) K(x_i, x) + b$"
    ax.text(0.5, 0.62, formula, ha="center", va="center", fontsize=22)
    ax.text(
        0.5,
        0.22,
        "SVR uses a kernel-induced feature space and a regularized epsilon-insensitive loss.",
        ha="center",
        va="center",
        fontsize=10.5,
    )
    ax.set_title("Support Vector Regression Forecast Function")
    save(fig, "11_svr_formula.png")


def plot_ml_approaches_overview() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.axis("off")

    panels = [
        {
            "xy": (0.03, 0.14),
            "wh": (0.28, 0.72),
            "face": "#dbeafe",
            "title": "Support Vector Regression",
            "body": [
                r"$f(x)=\sum_{i=1}^{n}(\alpha_i-\alpha_i^*)K(x_i,x)+b$",
                "Kernel-space nonlinear predictor",
                "Convex optimization with regularized",
                "epsilon-insensitive loss",
                "Needs careful lag and rolling features",
            ],
        },
        {
            "xy": (0.36, 0.14),
            "wh": (0.28, 0.72),
            "face": "#fde68a",
            "title": "Tree-Based Ensembles",
            "body": [
                "Random Forest: bootstrap + feature subsampling",
                "Gradient Boosting: sequential tree refinement",
                "Variance reduction through decorrelated",
                "or stage-wise combined learners",
            ],
        },
        {
            "xy": (0.69, 0.14),
            "wh": (0.28, 0.72),
            "face": "#fecaca",
            "title": "Core Limitation",
            "body": [
                "Models partition feature space from",
                "contemporaneous predictors",
                "No native ordered temporal memory",
                "Time dependence must be injected via",
                "lags, rolling stats, or exogenous inputs",
            ],
        },
    ]

    for panel in panels:
        x, y = panel["xy"]
        w, h = panel["wh"]
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            facecolor=panel["face"],
            edgecolor="#1f2937",
            linewidth=1.4,
        )
        ax.add_patch(rect)
        ax.text(x + 0.02, y + h - 0.08, panel["title"], ha="left", va="top", fontsize=14, weight="bold")

        current_y = y + h - 0.18
        for idx, line in enumerate(panel["body"]):
            if idx == 0 and line.startswith("$"):
                ax.text(x + 0.02, current_y, line, ha="left", va="top", fontsize=16)
                current_y -= 0.12
            else:
                ax.text(x + 0.025, current_y, f"- {line}", ha="left", va="top", fontsize=10.5)
                current_y -= 0.09

    ax.annotate("", xy=(0.36, 0.5), xytext=(0.31, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.annotate("", xy=(0.69, 0.5), xytext=(0.64, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.8})

    ax.text(
        0.5,
        0.93,
        "Machine Learning Approaches for Iron Ore Forecasting",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        0.5,
        0.06,
        "Nonlinear interactions can be learned, but temporal structure still depends on explicit feature engineering.",
        ha="center",
        va="center",
        fontsize=11,
    )
    save(fig, "12_ml_approaches_overview.png")


def plot_ml_model_comparison(research_df: pd.DataFrame) -> None:
    compare = research_df[
        research_df["model_name"].isin(["svr_rbf", "random_forest", "gradient_boosting", "hybrid_arima_lstm"])
    ].copy()
    label_map = {
        "svr_rbf": "SVR",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "hybrid_arima_lstm": "Hybrid ARIMA + LSTM",
    }
    compare["display_name"] = compare["model_name"].map(label_map)
    compare = compare.sort_values("rmse").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))

    sns.barplot(
        data=compare,
        x="rmse",
        y="display_name",
        hue="display_name",
        dodge=False,
        legend=False,
        palette=["#2563eb", "#d97706", "#059669", "#7c3aed"],
        ax=axes[0],
    )
    axes[0].set_title("Overall Forecast Error")
    axes[0].set_xlabel("RMSE")
    axes[0].set_ylabel("")

    for idx, row in compare.iterrows():
        axes[0].text(row["rmse"] + 0.12, idx, f'{row["rmse"]:.2f}', va="center", fontsize=10)

    regime = compare.melt(
        id_vars="display_name",
        value_vars=["rmse_high_volatility", "rmse_low_volatility"],
        var_name="regime",
        value_name="regime_rmse",
    )
    regime["regime"] = regime["regime"].map(
        {
            "rmse_high_volatility": "High volatility",
            "rmse_low_volatility": "Low volatility",
        }
    )
    sns.barplot(
        data=regime,
        x="display_name",
        y="regime_rmse",
        hue="regime",
        palette=["#dc2626", "#0ea5e9"],
        ax=axes[1],
    )
    axes[1].set_title("Regime-Sensitive Error Comparison")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=18)

    fig.suptitle("SVR, Tree Ensembles, and Hybrid ARIMA + LSTM", fontsize=17, weight="bold", y=1.02)
    save(fig, "13_ml_model_comparison.png")


def plot_arima_lstm_history(history_payload: dict) -> None:
    history_df = pd.DataFrame(history_payload.get("history", []))
    if history_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 4.6))
    sns.lineplot(data=history_df, x="epoch", y="train_loss", ax=ax, label="Train loss", linewidth=1.8)
    sns.lineplot(data=history_df, x="epoch", y="val_loss", ax=ax, label="Validation loss", linewidth=1.8)
    best_epoch = int(history_payload.get("best_epoch", 0))
    if best_epoch > 0:
        ax.axvline(best_epoch, color="#7c3aed", linestyle="--", linewidth=1.0, label="Best epoch")
    ax.set_title("ARIMA + LSTM Residual Training Curve")
    ax.set_ylabel("MSE loss")
    ax.legend()
    save(fig, "14_arima_lstm_history.png")


def main() -> None:
    params = load_params()
    clean_df = pd.read_csv(PROJECT_ROOT / params["data_cleaning"]["output_path"])
    clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce")
    clean_df = clean_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    prod_df = pd.read_csv(PROJECT_ROOT / params["experiments"]["results_path"])
    research_df = pd.read_csv(PROJECT_ROOT / params["research_experiments"]["results_path"])
    pred_df = pd.read_csv(PROJECT_ROOT / params["research_experiments"]["predictions_path"])
    test_pred_df = pd.read_csv(PROJECT_ROOT / params["training"]["predictions_path"])
    research_summary = load_json(PROJECT_ROOT / params["research_experiments"]["summary_path"])
    arima_lstm_history = load_json(PROJECT_ROOT / params["research_experiments"]["arima_lstm"]["history_path"])

    plot_market_dynamics(clean_df)
    plot_volatility(clean_df)
    plot_seasonality(clean_df)
    plot_correlation(clean_df)
    plot_production_comparison(prod_df)
    plot_research_benchmark(research_df)
    plot_prediction_traces(pred_df)
    plot_sequential_decomposition()
    plot_mlop_flow()
    plot_production_fit(test_pred_df)
    plot_svr_formula()
    plot_ml_approaches_overview()
    plot_ml_model_comparison(research_df)
    plot_arima_lstm_history(arima_lstm_history)


if __name__ == "__main__":
    main()
