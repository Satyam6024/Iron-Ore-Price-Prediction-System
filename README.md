# Iron Ore MLOps Project - Run Guide

This guide is for someone who receives this project and wants to run it on their own local system.

## 1) Prerequisites

Install these first:

- Python 3.11 or newer
- `uv` (for virtual environment + package install)
- (Optional) Docker Desktop, if running with containers

## 2) Open Project Folder

Unzip/copy the project and open a terminal in the project root (the folder containing `params.yaml` and `dvc.yaml`).

## 3) Create and Activate Virtual Environment

Windows PowerShell:

```powershell
cd <project-folder>
uv venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4) Install Dependencies

```powershell
uv pip install -r requirements.txt
```

## 5) Run Full Workflow (Data -> Features -> Train -> Registry -> Deploy)

```powershell
python workflows/run_full_workflow.py
```

What this does:

- Builds DVC stages (`data_ingestion`, `data_cleaning`, `feature_engineering`, `feature_selection`, `training`, MLflow register/promotion, deploy)
- Produces deployable artifacts in `dvc_pipeline/deploy/`

If you want to skip experiments and only run/re-run pipeline stages:

```powershell
python workflows/run_full_workflow.py --skip-experiments
```

## 6) Start Backend API

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API docs (Swagger UI):

- http://127.0.0.1:8000/docs

## 7) Test in Swagger UI

In `/docs`:

1. Run `GET /health` (should show model loaded)
2. Run `GET /metadata/simple` (shows minimum history length)
3. Run `POST /predict/simple` with sample body:

```json
{
  "date": "2023-12-29",
  "open": 136.37,
  "high": 136.37,
  "low": 136.37,
  "volume": 0,
  "change_pct": 0.15,
  "price_history": [129.25,129.52,129.85,130.02,129.74,130.12,130.38,130.01,129.88,130.44,131.1,131.45,131.9,132.35,132.8,133.1,133.6,134.0,134.25,134.61,134.9,135.12,135.45,135.0,135.31,135.75,136.07,136.16,136.37,136.37]
}
```

## 8) Use Frontend

Option A (quick): open `frontend/index.html` directly in browser.

- Set API base URL to `http://localhost:8000`
- Click `Load Simple Schema`
- Fill values and click `Predict`

Option B (Docker frontend): see section 10.

## 9) Optional: Open the Notebook Walkthrough

The project also includes stage-by-stage walkthrough notebooks in `notebooks/`.

Recommended order:

1. `notebooks/00_project_overview.ipynb`
2. `notebooks/01_data_ingestion.ipynb`
3. `notebooks/02_data_cleaning.ipynb`
4. `notebooks/03_feature_engineering.ipynb`
5. `notebooks/04_feature_selection.ipynb`
6. `notebooks/05_experiments.ipynb`
7. `notebooks/06_training.ipynb`
8. `notebooks/07_mlflow_register.ipynb`
9. `notebooks/08_mlflow_promote.ipynb`
10. `notebooks/09_deploy_production.ipynb`

To launch Jupyter:

```powershell
jupyter lab
```

If needed:

```powershell
uv pip install jupyterlab
```

## 10) Optional: Start MLflow UI

```powershell
mlflow ui --backend-store-uri file:./dvc_pipeline/mlruns --port 5000
```

Open:

- http://127.0.0.1:5000

## 11) Optional: Run with Docker Compose

Run workflow container once:

```powershell
docker compose run --rm --profile workflow workflow
```

Run serving stack:

```powershell
docker compose up --build backend frontend mlflow
```

URLs:

- Backend API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Frontend: http://localhost:3000
- MLflow: http://localhost:5000

## 12) Common Troubleshooting

- `ModuleNotFoundError`: ensure `.venv` is activated and run `uv pip install -r requirements.txt` again.
- DVC not found: run from activated env (workflow script can also use `python -m dvc` fallback).
- Model not loading in API: run full workflow first so `dvc_pipeline/deploy/model.joblib` exists.
- 422 error in `/predict/simple`: check `date` format (`YYYY-MM-DD`) and ensure `price_history` has at least the minimum length from `GET /metadata/simple`.
