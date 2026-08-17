"""
PrismModel — TimesFM 2.5 200M wrapper for crypto price forecasting.

TimesFM is a univariate foundation model, so we forecast each feature channel
independently and reassemble the results into a multi-variate output that is
compatible with the rest of the pipeline.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional
from src.config.pipeline_config import Config
from logger.logger import get_logger

logger = get_logger()

# Lazy-loaded singleton to avoid loading the ~900MB model multiple times
_TIMESFM_MODEL = None


def _pin_torch_threads():
    """Pin torch to physical cores so GEMMs get uncontended threads.

    The server now runs a single uvicorn worker, so this process owns the
    machine's cores. Using physical (not logical) core count avoids
    hyperthread contention in matrix multiplication.
    """
    import os

    try:
        import psutil

        n = psutil.cpu_count(logical=False)
    except Exception:
        n = None
    n = n or os.cpu_count() or 2
    torch.set_num_threads(max(1, n))
    logger.info(f"MODEL - torch pinned to {torch.get_num_threads()} threads")


def _load_timesfm_model(model_path: Optional[str] = None):
    """Load the TimesFM model from local safetensors checkpoint (singleton)."""
    global _TIMESFM_MODEL
    if _TIMESFM_MODEL is not None:
        return _TIMESFM_MODEL

    import timesfm

    config = Config()
    path = model_path or config.timesfm_model_path

    _pin_torch_threads()
    logger.info(f"MODEL - Loading TimesFM 2.5 200M from {path}")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        path, local_files_only=True
    )

    # Compile for fast batched inference
    forecast_cfg = timesfm.ForecastConfig(
        max_context=config.timesfm_max_context,
        max_horizon=config.timesfm_max_horizon,
        per_core_batch_size=config.batch_size,
        normalize_inputs=True,
        force_flip_invariance=True,
        infer_is_positive=False,   # crypto prices can be volatile, don't clamp
    )
    model.compile(forecast_cfg)

    logger.info("MODEL - TimesFM compiled and ready")
    _TIMESFM_MODEL = model
    return model


class PrismModel:
    """
    Drop-in wrapper around TimesFM 2.5 200M that exposes a predict interface
    compatible with the existing pipeline.

    Unlike the old LSTM PrismModel (nn.Module), this is NOT a PyTorch Module.
    It delegates to the pre-trained TimesFM foundation model for zero-shot
    forecasting.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.config = Config()
        self.model = _load_timesfm_model(model_path)

    # ------------------------------------------------------------------
    # Core inference — channel-independent forecasting
    # ------------------------------------------------------------------
    def predict(self, df, scaler=None, horizon: int = None):
        """
        Forecast future values for all feature channels.

        Args:
            df: DataFrame with columns matching Config().features
            scaler: StandardScaler (unused by TimesFM but kept for API compat)
            horizon: Number of future steps to predict. Defaults to config.pred_len.

        Returns:
            np.ndarray of shape (horizon, num_features) with inverse-transformed
            values (raw price scale).
        """
        horizon = horizon or self.config.pred_len
        features = self.config.features

        # Build univariate inputs — one time series per feature channel
        inputs = []
        for feat in features:
            series = df[feat].values.astype(np.float64)
            inputs.append(series)

        # TimesFM forecast returns (point_forecasts, quantile_forecasts)
        # point_forecasts shape: (num_inputs, horizon)
        point_forecasts, _ = self.model.forecast(horizon=horizon, inputs=inputs)

        # Reassemble into (horizon, num_features) matrix
        result = np.stack(
            [point_forecasts[i, :horizon] for i in range(len(features))],
            axis=1,
        )
        return result

    def eval(self):
        """No-op for API compatibility (TimesFM is always in eval mode)."""
        return self

    def to(self, device):
        """No-op for API compatibility — TimesFM manages its own device."""
        return self

    def state_dict(self):
        """Not applicable for TimesFM — returns empty dict for compat."""
        return {}

    def load_state_dict(self, state_dict, **kwargs):
        """No-op — TimesFM is loaded via from_pretrained."""
        pass

    def parameters(self):
        """Return underlying model parameters for API compat."""
        return self.model.model.parameters()

    def named_parameters(self):
        """Return underlying model named parameters for API compat."""
        return self.model.model.named_parameters()

    def train(self, mode=True):
        """No-op — TimesFM is used for zero-shot inference."""
        return self