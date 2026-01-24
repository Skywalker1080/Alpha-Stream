import os
import sys
import mlflow
import joblib
import torch
import pandas as pd
from typing import Dict
from src.config.pipeline_config import Config
from src.data.data_preparation import CryptoData
from src.data.data_ingestion import fetch_ohlcv
from sklearn.preprocessing import StandardScaler
from src.model.model_defination import PrismModel
from src.model.training import fit_model
from torch.utils.data import DataLoader
from src.utils import save_model
from src.train_utils import (_safe_promote_to_production, _get_output_paths)
from src.model.evaluation import evaluate_model_temp
from src.exception.exceptions import PrismException
from logger.logger import get_logger

logger = get_logger()

def train_parent() -> Dict:
    "Train parent model on BTC-USD"
    config = Config()
    parent_ticker = config.parent_ticker
    start = config.start
    epochs = config.parent_epochs
    out_dir = config.parent_dir # add to config

    with mlflow.start_run(run_name=f"Parent training for {parent_ticker}") as run:
        mlflow.log_params({
            "ticker": parent_ticker,
            "start_date": start,
            "epochs": epochs,
            "context_len": config.context_len,
            "pred_len": config.pred_len,
            "features": config.features,
            "batch_size": config.batch_size,
            "input_size": config.input_size
        })
        
        try:
            # 1. Data Ingestion
            df = fetch_ohlcv(parent_ticker, start)

            scaler = StandardScaler().fit(df[config.features])
            
            dataset = CryptoData(df, scaler)

            train_data, val_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

            train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

            model = PrismModel().to(config.device)
            model = fit_model(model, train_loader, val_loader, num_epochs=epochs, lr=1e-3)

            torch_path, scaler_path = _get_output_paths(out_dir, parent_ticker, "parent")

            torch.save(model.state_dict(), torch_path)
            joblib.dump(scaler, scaler_path)

            model.eval()
            metrics = evaluate_model_temp(model, df, scaler, out_dir, ticker=parent_ticker)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(torch_path, "torch_model")
            mlflow.log_artifact(scaler_path, "scaler")
            
            logger.info(f"Parent {parent_ticker} trained successfully")
            return {"ticker": parent_ticker, "run_id": run.info.run_id, "metrics": metrics}
        
        except PrismException as e:
            logger.error(f"Parent training failed: {e}")
            raise PrismException(f"Parent training failed: {e}", sys)

# Train Child Model
def train_child(ticker: str) -> Dict:
    """Train Child model"""
    config = Config()
    start = config.child_start # what if the ticker given was not launched on the given date, fix this later
    epochs = config.child_epochs
    parent_dir = config.parent_dir # add to config
    workdir = config.workdir

    with mlflow.start_run(run_name=f"Child training for {ticker}") as run:
        mlflow.log_params({
            "ticker": ticker,
            "start_date": start,
            "epochs": epochs,
            "context_len": config.context_len,
            "pred_len": config.pred_len,
            "features": config.features,
            "batch_size": config.batch_size,
            "input_size": config.input_size
        })
        
        try:
            # 1. Data Ingestion
            df = fetch_ohlcv(ticker, start)

            scaler = StandardScaler().fit(df[config.features])
            
            dataset = CryptoData(df, scaler)

            train_data, val_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

            train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

            parent_model_path = os.path.join(parent_dir, f"{config.parent_ticker}_parent_model.pt")
            if not os.path.exists(parent_model_path):
                logger.exception(f"Parent model not found at {parent_model_path}")
                raise FileNotFoundError(f"Parent model missing at {parent_model_path}")

            parent_model = PrismModel().to(config.device)
            parent_model.load_state_dict(torch.load(parent_model_path, map_location=config.device))
            logger.info(f"Loaded parent weights from {parent_model_path}")

            # Transfer Learning Strategy
            learning_rate = 3e-4

            if config.transfer_strategy == "freeze":
                logger.info("TRANSFER LEARNING - Freeze LSTM layers")
                for name, param in parent_model.named_parameters():
                    if "lstm" in name:
                        param.requires_grad = False
            elif config.transfer_strategy == "fine_tune":
                logger.info(f"TRANSFER LEARNING - Fine-tune all layers (lr={config.fine_tune_lr})")
                for param in parent_model.parameters():
                    param.requires_grad = True
                learning_rate = config.fine_tune_lr
            else:
                logger.warning(f"TRANSFER LEARNING - Unknown strategy '{config.transfer_strategy}', defaulting to 'freeze'")
                for name, param in parent_model.named_parameters():
                    if "lstm" in name:
                        param.requires_grad = False

            model = fit_model(parent_model, train_loader, val_loader, num_epochs=epochs, lr=learning_rate)

            child_dir = os.path.join(workdir, ticker)
            torch_path, scaler_path = _get_output_paths(child_dir, ticker, "child")
            torch.save(model.state_dict(), torch_path)
            joblib.dump(scaler, scaler_path)

            model.eval()
            metrics = evaluate_model_temp(model, df, scaler, child_dir, ticker=ticker)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(torch_path, "torch_model")
            mlflow.log_artifact(scaler_path, "scaler")
            
            logger.info(f"TRAINING - Child {ticker} trained successfully")
            return {"ticker": ticker, "run_id": run.info.run_id, "metrics": metrics}
        
        except PrismException as e:
            logger.error(f"TRAINING - Child training failed: {e}")
            raise PrismException(f"TRAINING - Child training failed: {e}", sys)

if __name__ == "__main__":
    try:
        logger.info("TRAINING - Starting Training Pipeline Test")
        
        # Train Parent
        logger.info(f"TRAINING - training parent model for BTC-USD")
        parent_result = train_parent()
        logger.info(f"TRAINING - Parent Result: {parent_result}")
        
        # Train Child (Example)
        # Assuming we can test with ETH-USD after parent BTC-USD is done
        child_ticker = "ETH-USD"
        logger.info(f"TRAINING: training child model for {child_ticker}")
        child_result = train_child(child_ticker)
        logger.info(f"TRANING: Child Result: {child_result}")
        
    except Exception as e:
        logger.error(f"TRAINING: Pipeline Execution Failed: {e}")
        raise
