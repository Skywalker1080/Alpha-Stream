import numpy as np
import pandas as pd
import torch
import pickle
import sys
import joblib
from pathlib import Path
from feast import FeatureStore
from sklearn.preprocessing import StandardScaler
from src.config.pipeline_config import Config
from typing import Dict
from src.exception.exceptions import PrismException
from src.model.model_defination import PrismModel
from logger.logger import get_logger

logger = get_logger()
config = Config()

def predict_one_step(model, df: pd.DataFrame, scaler: StandardScaler, ticker: str) -> Dict:
    try:
        logger.info(f"INFERENCE - Starting one-step prediction for ticker: {ticker}")
        
        vals = scaler.transform(df[config.features]).astype('float32')
        logger.debug(f"INFERENCE - Input data transformed. Data shape: {vals.shape}")
        
        X = vals[-config.context_len:].reshape(1, config.context_len, config.input_size)
        logger.debug(f"INFERENCE - Prepared input tensor 'X' with shape: {X.shape}")
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(config.device)
            preds = model(X_tensor).cpu().numpy()[0]
            logger.debug(f"INFERENCE - Model inference completed. Raw predictions shape: {preds.shape}")

        # inverse transform preds
        pred_inv = scaler.inverse_transform(preds.reshape(-1, config.input_size))[:, :5]
        logger.debug("INFERENCE - Inverse transformation of predictions successful.")

        # formatting dates
        last_day = df['date'].iloc[-1]
        next_days = pd.bdate_range(last_day + pd.Timedelta(days=1), periods=config.pred_len)
        logger.info(f"INFERENCE - Generated {len(next_days)} forecast points starting from {last_day}")

        forecast = []
        for i, date in enumerate(next_days):
            forecast.append({
                "date": str(date.date()),
                "open": float(pred_inv[i][0]),
                "high": float(pred_inv[i][1]),
                "low": float(pred_inv[i][2]),
                "close": float(pred_inv[i][3]),
                "volume": float(pred_inv[i][4])
            })

        logger.info(f"INFERENCE - Prediction for {ticker} completed successfully.")
        return {
            "ticker": ticker,
            "last_date": str(last_day.date()),
            "future_window_days": config.pred_len,
            "next_business_days": [str(d.date()) for d in next_days],
            "predictions": {
                "next_day": forecast[0],
                "next_week": {
                    "high": float(np.max([d["high"] for d in forecast])),
                    "low": float(np.min([d["low"] for d in forecast]))
                },
                "full_forecast": forecast
            }
        }
    except Exception as e:
        logger.error(f"INFERENCE - Error in predict_one_step for {ticker}: {str(e)}")
        raise PrismException(f"INFERENCE - Failed to predict one step for {ticker}", str(e))

def safe_load_scaler(path: str):
    """Safely loading scaler via pickle -> fallback to joblib"""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            return joblib.load(path)
        except Exception as e:
            raise PrismException(f"Failed to load model from {path}: {e}", sys)

def safe_load_local_model(ticker: str, model_type: str):
    """Safely load Pytorch model and scaler locally"""
    model_path = None
    try:
        logger.info(f"UTILS - safely loading Pytorch model and scaler")
        if model_type == "parent":
            base_dir = config.parent_dir
            model_path = Path(base_dir) / f"{config.parent_ticker}_parent_model.pt"
            scaler_path = Path(base_dir) / f"{config.parent_ticker}_parent_scaler.pkl"
        else:
            # Child models are stored in outputs/{ticker}/...
            base_dir = Path(config.child_dir) / ticker 
            model_path = base_dir / f"{ticker}_child_model.pt"
            scaler_path = base_dir / f"{ticker}_child_scaler.pkl"
        
        if not Path(model_path).exists():
            logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not Path(scaler_path).exists():
            logger.error(f"Scaler not found at {scaler_path}")
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        model = PrismModel().to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()
        scaler = safe_load_scaler(scaler_path)
        return model, scaler
    except Exception as e:
        msg = f"Failed to load model from {model_path}: {e}" if model_path else f"Failed to load model: {e}"
        raise PrismException(msg, sys)

def get_feature_Store():
    try:
        logger.info("Fetching feature store")
        return FeatureStore(repo_path="feature_store")
    except Exception as e:
        logger.warning(f"Feast Store not initialized or not found at 'feature_store'")
        return None
