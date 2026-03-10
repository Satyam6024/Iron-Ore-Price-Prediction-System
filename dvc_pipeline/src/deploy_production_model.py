from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from common import ensure_parent, load_params, write_json


def main() -> None:
    params = load_params()
    train_cfg = params["training"]
    deploy_cfg = params["deployment"]
    registry_cfg = params["registry"]

    model_path = Path(train_cfg["output_model_path"])
    feature_columns_path = Path(train_cfg["feature_columns_path"])
    promotion_report_path = Path(registry_cfg["promotion_output_path"])

    output_dir = Path(deploy_cfg["output_dir"])
    deploy_model_path = Path(deploy_cfg["deployed_model_path"])
    deploy_features_path = Path(deploy_cfg["deployed_feature_columns_path"])
    manifest_path = Path(deploy_cfg["manifest_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Final model artifact not found: {model_path}")
    if not feature_columns_path.exists():
        raise FileNotFoundError(f"Feature columns file not found: {feature_columns_path}")
    if not promotion_report_path.exists():
        raise FileNotFoundError(f"Promotion report not found: {promotion_report_path}")

    with promotion_report_path.open("r", encoding="utf-8") as f:
        promotion = json.load(f)

    if str(promotion.get("final_state")) != "production":
        raise RuntimeError(
            "Deployment blocked because model is not in production state. "
            f"Current state: {promotion.get('final_state')}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(deploy_model_path)
    ensure_parent(deploy_features_path)

    shutil.copy2(model_path, deploy_model_path)
    shutil.copy2(feature_columns_path, deploy_features_path)

    manifest = {
        "deployed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": promotion["model_name"],
        "model_version": promotion["version"],
        "source_model_path": str(model_path),
        "deployed_model_path": str(deploy_model_path),
        "source_feature_columns_path": str(feature_columns_path),
        "deployed_feature_columns_path": str(deploy_features_path),
        "production_model_uri": promotion.get("production_model_uri"),
        "promotion_reason": promotion.get("reason"),
    }
    write_json(manifest, manifest_path)

    print(f"[deploy] Deployment bundle created in: {output_dir}")
    print(f"[deploy] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
