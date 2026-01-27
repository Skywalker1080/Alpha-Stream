from dataclasses import dataclass, field
import torch
from typing import List

@dataclass
class Config:
    """Configuration for data pipeline"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    context_len: int = 60
    pred_len: int = 5 # forecast step
    features: List[str] = field(default_factory=lambda: ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD"])
    parent_ticker: str = "BTC-USD"
    start: str = "2015-01-01"
    child_start: str = "2022-01-01"
    parent_epochs: int = 20
    child_epochs: int = 10
    transfer_strategy: str = "freeze"
    fine_tune_lr: float = 1e-4
    workdir: str = "outputs"   
    batch_size: int = 32

    
    @property
    def input_size(self) -> int:
        return len(self.features)

    @property
    def parent_dir(self) -> str:
        import os
        return os.path.join(self.workdir, "parent")

    @property
    def child_dir(self) -> str:
        import os
        return os.path.join(self.workdir) # Children are directly in outputs/{ticker} based on earlier listings, but let's check

    

@dataclass
class IndicatorConfig:
    """Config dataclass for all indicators"""
    RSI_WINDOW: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26

