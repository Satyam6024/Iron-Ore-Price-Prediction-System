# Iron Ore Model API

This API serves predictions from the deployed model bundle (preferred) or fallback model paths.

Default loading order:

1. `C:\aditi\dvc_pipeline\deploy\model.joblib`
2. `C:\aditi\dvc_pipeline\models\final_model.joblib`

## Setup

```powershell
cd C:\aditi
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r api\requirements_api.txt
```

## Run

```powershell
cd C:\aditi
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open docs: `http://127.0.0.1:8000/docs`

## Endpoints

- `GET /health`
- `GET /metadata` (model feature columns)
- `GET /metadata/simple` (simple input schema + min history length)
- `POST /predict` (advanced: full feature map)
- `POST /predict/simple` (recommended for frontend users)
- `POST /predict/batch`

## Simple prediction payload

```json
{
  "date": "2023-12-29",
  "open": 136.37,
  "high": 136.37,
  "low": 136.37,
  "volume": 0,
  "change_pct": 0.15,
  "price_history": [129.25, 129.52, 129.85, 130.02, 129.74, 130.12, 130.38, 130.01, 129.88, 130.44, 131.1, 131.45, 131.9, 132.35, 132.8, 133.1, 133.6, 134.0, 134.25, 134.61, 134.9, 135.12, 135.45, 135.0, 135.31, 135.75, 136.07, 136.16, 136.37, 136.37]
}
```

The backend derives engineered features internally, so users do not enter all model features.

## Optional env overrides

- `MODEL_PATH`
- `FEATURE_COLUMNS_PATH`
- `MODEL_METADATA_PATH`
- `CORS_ALLOW_ORIGINS`
