# Notebook Walkthrough

These notebooks are a stage-by-stage walkthrough of the active project pipeline.
They are now aligned with the research paper themes:

- hybrid linear + nonlinear forecasting
- volatility-aware evaluation
- regime-sensitive analysis
- rolling robustness diagnostics

Recommended order:

1. `00_project_overview.ipynb`
2. `01_data_ingestion.ipynb`
3. `02_data_cleaning.ipynb`
4. `02b_eda.ipynb`
5. `03_feature_engineering.ipynb`
6. `04_feature_selection.ipynb`
7. `05_experiments.ipynb`
8. `06_training.ipynb`
9. `07_mlflow_register.ipynb`
10. `08_mlflow_promote.ipynb`
11. `09_deploy_production.ipynb`

`05_experiments.ipynb` runs both:

- original production experiment selection
- paper-aligned research experiments (`research_experiments.py`)
- ARIMA + LSTM residual training history and benchmark visuals

`02b_eda.ipynb` and `05_experiments.ipynb` also save presentation-ready figures into `presentation/figures`.

Start Jupyter from the project root:

```powershell
cd C:\aditi
.\.venv\Scripts\Activate.ps1
jupyter lab
```

If `jupyter` is not installed in the environment yet:

```powershell
uv pip install jupyterlab
```
