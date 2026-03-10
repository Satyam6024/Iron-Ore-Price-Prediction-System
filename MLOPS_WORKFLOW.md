# End-to-End MLOps Workflow

This project now supports the full flow you asked for:

1. Run experiments
2. Auto-select best hyperparameters
3. Apply feature selection (top features)
4. Update `params.yaml`
5. Trigger DVC pipeline
6. Train model
7. Register model in MLflow
8. Gate through Testing -> Staging -> Production (with rollback)
9. Deploy production-ready model bundle
10. Serve predictions via backend API to frontend

## Architecture

- Experiments script: `dvc_pipeline/src/run_experiments.py`
- Params update script: `dvc_pipeline/src/update_params_from_experiments.py`
- Orchestrator: `workflows/run_full_workflow.py`
- DVC pipeline: `dvc.yaml`
- Feature selection stage: `dvc_pipeline/src/feature_selection.py`
- Registry register stage: `dvc_pipeline/src/mlflow_register_model.py`
- Registry promotion stage: `dvc_pipeline/src/mlflow_promote_model.py`
- Deployment stage: `dvc_pipeline/src/deploy_production_model.py`
- Backend API: `api/main.py`
- Frontend: `frontend/index.html`

User-facing inference endpoint:

- `POST /predict/simple` (small input form + price history, no manual 37-feature entry)

## Local run

```powershell
cd C:\aditi
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
python workflows/run_full_workflow.py
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Frontend:

- Open `frontend/index.html` in browser or run via Docker frontend service below.

MLflow UI:

```powershell
mlflow ui --backend-store-uri file:./dvc_pipeline/mlruns --port 5000
```

## Docker run

### 1) Run full workflow (experiments + DVC + registry + deployment)

```powershell
docker compose run --rm --profile workflow workflow
```

### 2) Start model-serving stack

```powershell
docker compose up --build backend frontend mlflow
```

Endpoints:

- Backend API: `http://localhost:8000`
- Backend docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:3000`
- MLflow UI: `http://localhost:5000`

## Promotion logic

- Model always starts in `testing` alias.
- Fails testing gate -> stays in `testing`.
- Passes testing gate -> moves to `staging`.
- Fails staging gate -> rolled back to `testing`.
- Passes staging gate -> promoted to `production`.
- Deployment stage runs only when final state is `production`.
