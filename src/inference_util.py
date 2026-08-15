import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from src.config.pipeline_config import Config
from src.model.provisioning import scaler_path as artifact_path
from typing import Dict
from src.exception.exceptions import PrismException
from src.model.model_defination import PrismModel
from logger.logger import get_logger

logger = get_logger()
config = Config()


def predict_one_step(model: PrismModel, df: pd.DataFrame, scaler: StandardScaler, ticker: str) -> Dict:
    """
    Generate a multi-step forecast using the TimesFM model.

    Args:
        model: PrismModel wrapping TimesFM 2.5
        df: DataFrame with columns [date, Open, High, Low, Close, Volume, RSI, MACD]
        scaler: StandardScaler (kept for API compat — TimesFM self-normalizes)
        ticker: The ticker symbol

    Returns:
        Dict with forecast results compatible with the existing API contract.
    """
    try:
        logger.info(f"INFERENCE - Starting one-step prediction for ticker: {ticker}")

        # Use only the last context_len rows for prediction
        context_df = df.tail(config.context_len).copy()
        logger.debug(f"INFERENCE - Using last {len(context_df)} rows as context")

        # Predict using TimesFM channel-independent forecasting
        pred_inv = model.predict(context_df, scaler=scaler, horizon=config.pred_len)
        logger.debug(f"INFERENCE - Model inference completed. Predictions shape: {pred_inv.shape}")

        # pred_inv shape: (pred_len, num_features)
        # Features order: [Open, High, Low, Close, Volume, RSI, MACD]
        # We only expose OHLCV (first 5) in the response

        # formatting dates
        last_day = df['date'].iloc[-1]
        # Crypto trades 24/7/365, so forecast on calendar days (not business days).
        next_days = pd.date_range(last_day + pd.Timedelta(days=1), periods=config.pred_len)
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
            "last_date": str(last_day.date()) if hasattr(last_day, 'date') else str(last_day),
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


def safe_load_local_model(ticker: str = None, model_type: str = "parent") -> tuple:
    """
    Load the TimesFM PrismModel.

    TimesFM uses a single pre-trained model (no separate parent/child weights),
    so model_type and ticker are kept for API compatibility but don't affect
    loading. The scaler is returned as None since TimesFM self-normalizes.
    """
    try:
        logger.info(f"UTILS - Loading TimesFM model (model_type={model_type}, ticker={ticker})")
        model = PrismModel()

        # Return None scaler — TimesFM handles normalization internally
        # We still need to return a 2-tuple to match the existing API
        # If a scaler was previously saved, try to load it for backward compat
        scaler = _try_load_scaler(ticker, model_type)

        return model, scaler
    except Exception as e:
        raise PrismException(f"Failed to load TimesFM model: {e}", sys)


def _try_load_scaler(ticker: str, model_type: str):
    """Try to load a previously saved scaler, return None if not found."""
    import joblib
    try:
        scaler_path = artifact_path(config, ticker, model_type)
        if scaler_path.exists():
            return joblib.load(scaler_path)
    except Exception as e:
        logger.warning(f"Could not load scaler from {scaler_path}: {e}")
    return None


def get_feature_Store():
    try:
        from feast import FeatureStore
        logger.info("Fetching feature store")
        return FeatureStore(repo_path="feature_store")
    except Exception as e:
        logger.warning(f"Feast Store not initialized or not found at 'feature_store'")
        return None
