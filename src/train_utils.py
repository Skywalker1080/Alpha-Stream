import os
import mlflow
from mlflow.tracking import MlflowClient
from logger.logger import get_logger

logger = get_logger()

def _safe_promote_to_production(model_name: str, version: int):
    """Promote model version to Production (safe for DagsHub)."""
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"✅ Promoted {model_name} v{version} → Production")
    except Exception as e:
        logger.warning(f"⚠️ Registry not supported: {e}")


def _get_output_paths(base_dir: str, ticker: str, model_type: str):
    os.makedirs(base_dir, exist_ok=True)
    prefix = f"{ticker}_{model_type}"
    return (
        os.path.join(base_dir, f"{prefix}_model.pt"),
        os.path.join(base_dir, f"{prefix}_scaler.pkl"),
    )