from src.exception.exceptions import PrismException
from sklearn.preprocessing._data import StandardScaler
from pathlib import Path
import os
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
from src.config.pipeline_config import Config
from logger.logger import get_logger
import pandas as pd
import torch
import numpy as np
from src.utils import plot_predictions, plot_residuals, save_metrics
import mlflow

logger = get_logger()

"""
def evaluate_model(model, df: pd.DataFrame, scaler: StandardScaler, out_dir: Path, ticker: str) -> Dict:
    try:
        os.makedirs(out_dir, exist_ok=True)
        config = Config()
        vals = scaler.transform(df[config.features]).astype("float32")
        X, Y = [], []
        for t in range(config.context_len, len(vals) - config.pred_len):
            past = vals[t - config.context_len: t]
            fut = vals[t:t + config.context_len]

            if past.shape == (config.context_len, config.features) and fut.shape == (config.pred_len, config.features):
                X.append(past)
                Y.append(fut)
            else:
                logger.error(f"Skipping invalid evaluation sample at index {t}: past shape {past.shape}, fut shape {fut.shape}")

        if not X:
            logger.error(f"No valid samples for evaluation for {ticker}")
            return {}

        X, Y = np.array(X), np.array(Y)

        with torch.no_grad():
            preds = []
            for x in X:
                x_tensor = torch.tensor(x.reshape(1, config.context_len, config.features), dtype=torch.float32).to(config.device)
                pred = model(x_tensor).cpu().numpy()[0]
                preds.append(pred.flatten())

        preds = np.array(preds)
        y_ohlcv = Y.reshape(-1, config.input_size)[:, :5]
        pred_ohlcv = preds.reshape(-1, config.input_size)[:, :5]

        mse = mean_squared_error(y_ohlcv, pred_ohlcv)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_ohlcv, pred_ohlcv)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
        """

def evaluate_model_temp(model, df: pd.DataFrame, scaler: StandardScaler, temp_dir: str, ticker: str) -> Dict:
    """Evaluate model performance and save metrics directly to MLflow without local persistence."""
    try:
        config = Config()
        vals = scaler.transform(df[config.features]).astype("float32")
        X, Y = [], []
        for t in range(config.context_len, len(vals) - config.pred_len):
            past = vals[t - config.context_len:t]
            fut = vals[t:t + config.pred_len]
            if past.shape == (config.context_len, config.input_size) and fut.shape == (config.pred_len, config.input_size):
                X.append(past)
                Y.append(fut)
            else:
                logger.error(f"MODEL EVALUATION - Skipping invalid evaluation sample at index {t}: past shape {past.shape}, fut shape {fut.shape}")

        if not X:
            logger.error(f"MODEL EVALUATION - No valid samples for evaluation for {ticker}")
            return {}

        X, Y = np.array(X), np.array(Y)
        
        with torch.no_grad():
            preds = []
            for x in X:
                x_tensor = torch.tensor(x.reshape(1, config.context_len, config.input_size), dtype=torch.float32).to(config.device)
                pred = model(x_tensor).cpu().numpy()[0]
                preds.append(pred)
        
        preds = np.array(preds)
        Y_ohlcv = Y.reshape(-1, config.input_size)[:, :5]
        preds_ohlcv = preds.reshape(-1, config.input_size)[:, :5]

        mse = mean_squared_error(Y_ohlcv, preds_ohlcv)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_ohlcv, preds_ohlcv)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
        
        # Save metrics to temporary file and log to MLflow
        metrics_filename = f"{ticker}_metrics.json"
        metrics_path = os.path.join(temp_dir, metrics_filename)
        save_metrics(metrics, temp_dir, ticker)
        
        #logger.info(f"{ticker} -> MSE: {mse:.5f}, RMSE: {rmse:.5f}, R2: {r2:.5f}")

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path, f"metrics/{ticker}")

        # Generate and log plots
        plot_filename = f"{ticker}_predictions.png"
        plot_path = os.path.join(temp_dir, plot_filename)
        plot_predictions(Y_ohlcv, preds_ohlcv, plot_path, ticker)
        mlflow.log_artifact(plot_path, f"plots/{ticker}")

        resid_filename = f"{ticker}_residuals.png"
        resid_path = os.path.join(temp_dir, resid_filename)
        plot_residuals(Y_ohlcv, preds_ohlcv, ticker, resid_path)
        mlflow.log_artifact(resid_path, f"plots/{ticker}")

        return metrics
    except PrismException as e:
        logger.error(f"MODEL EVALUATION - Evaluation failed for {ticker}: {e}")
        raise PrismException(e)