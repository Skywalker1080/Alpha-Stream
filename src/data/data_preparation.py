import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from src.exception.exceptions import PrismException
import pandas as pd
import sys
from typing import Tuple
from logger.logger import get_logger

logger = get_logger()

from src.config.pipeline_config import Config

class CryptoData(Dataset):
    """Dataset for Crypto Data, building in sequence"""
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler, context_len: int = Config().context_len, pred_len: int = Config().pred_len):
        self.context_len = context_len
        self.pred_len = pred_len
        try:
            vals = scaler.transform(df[Config().features]).astype("float32")
            self.samples = []
            for t in range(context_len, len(df) - pred_len):
                past = vals[t - context_len:t]
                future = vals[t:t + pred_len]

                # Checking shapes before appending
                if past.shape == (context_len, len(Config().features)) and future.shape == (pred_len, len(Config().features)):
                    self.samples.append((past, future))
                else:
                    print(f"DATA PREPARATION - Ignoring invalid sample at index {t}, where shape: {past.shape} and {future.shape}")

            if not self.samples:
                raise PrismException("DATA PREPARATION - No valid samples found", sys)
        except:
            logger.exception("DATA PREPARATION - Failed to build dataset")
            raise PrismException("DATA PREPARATION - Failed to build dataset", sys)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        past, fut = self.samples[idx]
        return torch.tensor(past), torch.tensor(fut)

        