import numpy as np
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt

from src.exception.exceptions import PrismException
from logger.logger import get_logger

logger = get_logger()

def save_metrics(metrics: dict, out_dir: Path, ticker: str):
    try:
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = out_dir / f"{ticker}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        logger.info(f"{ticker} → MSE: {mse:.5f}, RMSE: {rmse:.5f}, R²: {r2:.5f}")
    except PrismException as e:
        logger.exception(f"Failed to save metrics for {ticker}: {e}")
        raise PrismException(e)

def plot_predictions(Y: np.ndarray, preds: np.ndarray, save_path: Path, ticker: str):
    try:
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(12, 8))
        features = ["Open", "High", "Low", "Close", "Volume"]
        for i, feature in enumerate(features):
            plt.subplot(3, 2, i + 1)
            plt.plot(Y[:, i], label="Actual", alpha=0.7)
            plt.plot(preds[:, i], label="Predicted", alpha=0.7)
            plt.title(f"{ticker} - {feature}")
            plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
def plot_residuals(Y: np.ndarray, preds: np.ndarray, ticker: str, save_path: Path):
    """Plot Residuals (Actual - Predicted) for the first 5 dimensions (OHLCV)."""
    try:
        os.makedirs(save_path, exist_ok=True)
        residuals = Y - preds
        plt.figure(figsize=(12, 8))
        features = ["Open", "High", "Low", "Close", "Volume"]
        for i, feature in enumerate(features):
            plt.subplot(3, 2, i + 1)
            plt.plot(residuals[:, i], label="Residuals", alpha=0.7)
            plt.axhline(0, color='r', linestyle='--')
            plt.title(f"{ticker} - {feature} Residuals")
            plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()