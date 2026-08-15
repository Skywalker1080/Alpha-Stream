import numpy as np
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
from src.exception.exceptions import PrismException
from src.config.pipeline_config import Config
from logger.logger import get_logger

logger = get_logger()


def save_metrics(metrics: dict, out_dir: Path, ticker: str):
    try:
        out_dir = Path(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = out_dir / f"{ticker}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        
        mse = metrics.get("MSE", 0)
        rmse = metrics.get("RMSE", 0)
        r2 = metrics.get("R2", 0)
        logger.info(f"{ticker} -> MSE: {mse:.5f}, RMSE: {rmse:.5f}, R2: {r2:.5f}")
    except PrismException as e:
        logger.exception(f"Failed to save metrics for {ticker}: {e}")
        raise PrismException(e)

def plot_predictions(Y: np.ndarray, preds: np.ndarray, save_path: Path, ticker: str):
    try:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
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
    except PrismException as e:
        logger.exception(f"Failed to plot predictions for {ticker}: {e}")
        raise PrismException(e)
        
def plot_residuals(Y: np.ndarray, preds: np.ndarray, ticker: str, save_path: Path):
    """Plot Residuals (Actual - Predicted) for the first 5 dimensions (OHLCV)."""
    try:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
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
    except PrismException as e:
        logger.exception(f"Failed to plot residuals for {ticker}: {e}")
        raise PrismException(e)

def initialize_dirs():
    """Initialize output directories."""
    config = Config()
    os.makedirs(config.parent_dir, exist_ok=True)

