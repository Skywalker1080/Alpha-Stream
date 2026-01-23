import torch.nn as nn
import torch
from src.config.pipeline_config import Config

class PrismModel(nn.Module):
    """A base LSTM model for crypto price forecasting"""
    def __init__(self, input_size: int = Config().input_size, hidden_size: int = 128, num_layers: int = 3, pred_len: int = Config.pred_len,
                dropout: float = 0.2)
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size * pred_len)
        self.pred_len = pred_len
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(-1, self.pred_len, self.input_size)

    