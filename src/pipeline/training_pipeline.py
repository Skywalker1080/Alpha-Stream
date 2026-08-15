import os
import sys
import mlflow
import joblib
from typing import Dict
from src.config.pipeline_config import Config
from src.data.data_ingestion import fetch_ohlcv
from sklearn.preprocessing import StandardScaler
from src.model.model_defination import PrismModel
from src.model.evaluation import evaluate_model_temp
from src.model.provisioning import scaler_path as artifact_path
from src.exception.exceptions import PrismException
from logger.logger import get_logger

logger = get_logger()

def train_parent() -> Dict:
    """
    'Train' parent model on BTC-USD.
    
    With TimesFM, this is a zero-shot evaluation pipeline:
    1. Fetch OHLCV data
    2. Load pre-trained TimesFM model
    3. Evaluate on the dataset
    4. Save scaler + log metrics to MLflow
    
    No actual weight training occurs — TimesFM is a foundation model.
    """
    config = Config()
    parent_ticker = config.parent_ticker
    start = config.start

    with mlflow.start_run(run_name=f"TimesFM Parent eval for {parent_ticker}") as run:
        mlflow.log_params({
            "ticker": parent_ticker,
            "start_date": start,
            "model_type": "TimesFM-2.5-200M",
            "context_len": config.context_len,
            "pred_len": config.pred_len,
            "features": config.features,
            "timesfm_model_path": config.timesfm_model_path,
        })
        
        try:
            # 1. Data Ingestion
            df = fetch_ohlcv(parent_ticker, start)

            # 2. Fit a scaler (for downstream compat / evaluation metrics)
            scaler = StandardScaler().fit(df[config.features])

            # 3. Load TimesFM model
            model = PrismModel()

            # 4. Evaluate
            out_dir = config.parent_dir
            metrics = evaluate_model_temp(model, df, scaler, out_dir, ticker=parent_ticker)

            # 5. Save scaler for inference pipeline compat
            scaler_path = artifact_path(config, parent_ticker, "parent")
            os.makedirs(scaler_path.parent, exist_ok=True)
            joblib.dump(scaler, scaler_path)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(str(scaler_path), "scaler")
            
            logger.info(f"Parent {parent_ticker} evaluated successfully with TimesFM")
            return {"ticker": parent_ticker, "run_id": run.info.run_id, "metrics": metrics}
        
        except Exception as e:
            logger.error(f"Parent training failed: {e}")
            raise PrismException(f"Parent training failed: {e}", sys)

# Train Child Model
def train_child(ticker: str) -> Dict:
    """
    'Train' child model for a specific ticker.
    
    With TimesFM, this is identical to parent evaluation since the same
    pre-trained model is used for all tickers (no transfer learning needed).
    We still save a scaler per ticker for the inference pipeline.
    """
    config = Config()
    start = config.child_start
    workdir = config.workdir

    with mlflow.start_run(run_name=f"TimesFM Child eval for {ticker}") as run:
        mlflow.log_params({
            "ticker": ticker,
            "start_date": start,
            "model_type": "TimesFM-2.5-200M",
            "context_len": config.context_len,
            "pred_len": config.pred_len,
            "features": config.features,
            "timesfm_model_path": config.timesfm_model_path,
        })
        
        try:
            # 1. Data Ingestion
            df = fetch_ohlcv(ticker, start)

            # 2. Fit a scaler
            scaler = StandardScaler().fit(df[config.features])

            # 3. Load TimesFM model (uses singleton — no re-load)
            model = PrismModel()

            # 4. Evaluate
            child_dir = os.path.join(workdir, ticker)
            metrics = evaluate_model_temp(model, df, scaler, child_dir, ticker=ticker)

            # 5. Save scaler
            scaler_path = artifact_path(config, ticker, "child")
            os.makedirs(scaler_path.parent, exist_ok=True)
            joblib.dump(scaler, scaler_path)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(str(scaler_path), "scaler")
            
            logger.info(f"TRAINING - Child {ticker} evaluated successfully with TimesFM")
            return {"ticker": ticker, "run_id": run.info.run_id, "metrics": metrics}
        
        except Exception as e:
            logger.error(f"TRAINING - Child training failed: {e}")
            raise PrismException(f"TRAINING - Child training failed: {e}", sys)

if __name__ == "__main__":
    try:
        logger.info("TRAINING - Starting Training Pipeline Test")
        
        # Train Parent
        logger.info(f"TRAINING - evaluating parent model for BTC-USD")
        parent_result = train_parent()
        logger.info(f"TRAINING - Parent Result: {parent_result}")
        
        # Train Child (Example)
        child_ticker = "ETH-USD"
        logger.info(f"TRAINING: evaluating child model for {child_ticker}")
        child_result = train_child(child_ticker)
        logger.info(f"TRAINING: Child Result: {child_result}")
        
    except Exception as e:
        logger.error(f"TRAINING: Pipeline Execution Failed: {e}")
        raise
