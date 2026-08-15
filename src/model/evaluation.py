from src.exception.exceptions import PrismException
from sklearn.preprocessing._data import StandardScaler
import os
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
from src.config.pipeline_config import Config
from logger.logger import get_logger
import pandas as pd
import numpy as np
from src.utils import plot_predictions, plot_residuals, save_metrics
import mlflow

logger = get_logger()


def evaluate_quick(model, df: pd.DataFrame, scaler: StandardScaler, temp_dir: str, ticker: str) -> Dict:
    """
    Lightweight validation for the zero-shot provisioning step.

    Runs a single forecast on the last window (context_len rows) and compares
    the pred_len-step prediction against the held-out actuals. Produces the
    same MSE/RMSE/R2 + plot artifacts as the full backtest but in one forward
    pass — used by train_parent/train_child, which only need a scaler file as
    the "model exists" marker, not a full-history evaluation.
    """
    try:
        config = Config()
        features = config.features

        n = len(df)
        if n < config.context_len + config.pred_len:
            raise ValueError(
                f"Not enough rows for quick eval: {n} < {config.context_len + config.pred_len}"
            )

        # Last window as context, next pred_len rows as held-out actuals
        context_df = df.iloc[n - config.context_len - config.pred_len:n - config.pred_len].copy()
        actual = df[features].iloc[n - config.pred_len:n].values  # (pred_len, num_features)

        pred = model.predict(context_df, horizon=config.pred_len)  # (pred_len, num_features)

        # Compute metrics on price columns only (OHLC — first 4 features).
        # Volume is orders of magnitude larger in raw units and would swamp
        # the error with a single col (e.g. MSE ~4e10), making RMSE/R2 useless.
        Y_price = actual[:, :4]
        preds_price = pred[:, :4]

        mse = mean_squared_error(Y_price, preds_price)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_price, preds_price)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}

        # Save metrics to temporary file and log to MLflow
        os.makedirs(temp_dir, exist_ok=True)
        metrics_filename = f"{ticker}_metrics.json"
        metrics_path = os.path.join(temp_dir, metrics_filename)
        save_metrics(metrics, temp_dir, ticker)

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path, f"metrics/{ticker}")

        # Generate and log plots
        plot_filename = f"{ticker}_predictions.png"
        plot_path = os.path.join(temp_dir, plot_filename)
        plot_predictions(Y_price, preds_price, plot_path, ticker)
        mlflow.log_artifact(plot_path, f"plots/{ticker}")

        resid_filename = f"{ticker}_residuals.png"
        resid_path = os.path.join(temp_dir, resid_filename)
        plot_residuals(Y_price, preds_price, ticker, resid_path)
        mlflow.log_artifact(resid_path, f"plots/{ticker}")

        return metrics
    except PrismException as e:
        logger.error(f"MODEL EVALUATION - Quick eval failed for {ticker}: {e}")
        raise PrismException(e)


def evaluate_model_temp(model, df: pd.DataFrame, scaler: StandardScaler, temp_dir: str, ticker: str) -> Dict:
    """
    Evaluate TimesFM model performance using rolling-window forecasts.
    
    For each window of context_len rows, we predict pred_len steps ahead
    and compare with actuals. This provides a robust evaluation across
    the entire dataset.
    """
    try:
        config = Config()
        features = config.features

        # Build actuals and predictions using sliding window
        all_actuals = []
        all_preds = []

        step_size = config.pred_len  # non-overlapping evaluation windows
        
        for t in range(config.context_len, len(df) - config.pred_len, step_size):
            context_df = df.iloc[t - config.context_len:t].copy()
            actual = df[features].iloc[t:t + config.pred_len].values  # (pred_len, num_features)

            if actual.shape[0] < config.pred_len:
                break

            try:
                pred = model.predict(context_df, horizon=config.pred_len)  # (pred_len, num_features)
                all_actuals.append(actual)
                all_preds.append(pred)
            except Exception as e:
                logger.warning(f"MODEL EVALUATION - Skipping window at index {t}: {e}")
                continue

        if not all_actuals:
            logger.error(f"MODEL EVALUATION - No valid samples for evaluation for {ticker}")
            return {}

        Y = np.concatenate(all_actuals, axis=0)  # (N, num_features)
        preds = np.concatenate(all_preds, axis=0)  # (N, num_features)

        # Compute metrics on price columns only (OHLC — first 4 features).
        # Volume is orders of magnitude larger in raw units and would swamp
        # the error with a single col (e.g. MSE ~4e10), making RMSE/R2 useless.
        Y_price = Y[:, :4]
        preds_price = preds[:, :4]

        mse = mean_squared_error(Y_price, preds_price)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_price, preds_price)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
        
        # Save metrics to temporary file and log to MLflow
        os.makedirs(temp_dir, exist_ok=True)
        metrics_filename = f"{ticker}_metrics.json"
        metrics_path = os.path.join(temp_dir, metrics_filename)
        save_metrics(metrics, temp_dir, ticker)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path, f"metrics/{ticker}")

        # Generate and log plots
        plot_filename = f"{ticker}_predictions.png"
        plot_path = os.path.join(temp_dir, plot_filename)
        plot_predictions(Y_price, preds_price, plot_path, ticker)
        mlflow.log_artifact(plot_path, f"plots/{ticker}")

        resid_filename = f"{ticker}_residuals.png"
        resid_path = os.path.join(temp_dir, resid_filename)
        plot_residuals(Y_price, preds_price, ticker, resid_path)
        mlflow.log_artifact(resid_path, f"plots/{ticker}")

        return metrics
    except PrismException as e:
        logger.error(f"MODEL EVALUATION - Evaluation failed for {ticker}: {e}")
        raise PrismException(e)