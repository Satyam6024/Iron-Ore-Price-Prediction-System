# DVC Pipeline

This pipeline is parameter-driven through `params.yaml` and has eight stages:

1. `data_ingestion`
2. `data_cleaning`
3. `feature_engineering`
4. `feature_selection`
5. `training`
6. `mlflow_register`
7. `mlflow_promote`
8. `deploy_production`

Final model output:

- `dvc_pipeline/models/final_model.joblib`
- Deployment bundle (only when promotion reaches production):
  - `dvc_pipeline/deploy/model.joblib`
  - `dvc_pipeline/deploy/feature_columns.json`
  - `dvc_pipeline/deploy/deployment_manifest.json`

## Setup

```powershell
cd C:\aditi
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv pip install dvc pyyaml
```

## Initialize DVC (first time)

```powershell
cd C:\aditi
dvc init
```

## Run pipeline

```powershell
dvc repro
```

## Full automated workflow (experiments -> params update -> DVC)

Run:

```powershell
python workflows/run_full_workflow.py
```

This will:

1. Build feature data
2. Run experiments across multiple algorithms
3. Update `params.yaml` with best model + hyperparameters
4. Execute full DVC pipeline (train, register, promote, deploy)

Lifecycle behavior:

- After `training`, model is registered in MLflow and aliased as `testing`.
- `mlflow_promote` applies:
  - Testing gate thresholds
  - Staging gate thresholds
- If staging fails, the model is rolled back to `testing`.
- If staging passes, model is promoted to `production`.
- `deploy_production` runs only if promotion result is `production`; otherwise it fails intentionally and blocks deployment.

## Change parameters

Edit `params.yaml` and rerun:

```powershell
dvc repro
```

DVC will only rerun stages affected by parameter/data/script changes.

## Useful outputs

- Ingestion report: `dvc_pipeline/reports/ingestion_report.json`
- Cleaning report: `dvc_pipeline/reports/cleaning_report.json`
- Feature report: `dvc_pipeline/reports/feature_report.json`
- Feature selection report: `dvc_pipeline/reports/feature_selection_report.json`
- Training metrics: `dvc_pipeline/reports/training_metrics.json`
- Test predictions: `dvc_pipeline/reports/test_predictions.csv`
- Registry registration report: `dvc_pipeline/reports/registry_registration.json`
- Registry promotion report: `dvc_pipeline/reports/registry_promotion.json`

## MLflow UI

```powershell
mlflow ui --backend-store-uri file:./dvc_pipeline/mlruns --port 5000
```

Open: `http://127.0.0.1:5000`
