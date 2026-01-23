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
    
    @property
    def input_size(self) -> int:
        return len(self.features)
    

@dataclass
class IndicatorConfig:
    """Config dataclass for all indicators"""
    RSI_WINDOW: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26

