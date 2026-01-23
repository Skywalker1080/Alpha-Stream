from optree.integration import torch
from sklearn.preprocessing._data import StandardScaler
from pathlib import Path
import sys
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
from src.config.pipeline_config import Config
from src.data.data_ingestion import fetch_ohlcv
from logger.logger import get_logger
import pandas as pd
import torch

logger = get_logger()

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