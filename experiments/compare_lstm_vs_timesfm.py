"""
Fair head-to-head: Legacy LSTM (parent->child transfer) vs TimesFM on ETH-USD.

Protocol
--------
- One shared dataset per ticker: daily OHLCV + RSI + MACD from Coinbase.
- Chronological split (no shuffling, no future leakage). Windows are indexed by
  their TARGET timestamps, so a window's context may reach back into earlier
  regions — exactly what live inference does (history up to today). The scaler
  is fit on TRAIN only.
    TRAIN targets: 2020-01-01 .. 2023-12-31
    VAL   targets: 2024-01-01 .. 2024-12-31   (early stopping)
    TEST  targets: 2025-01-01 .. yesterday     (evaluation)
- LSTM parent trained on BTC-USD (TRAIN targets), validated on BTC VAL,
  then weights transferred to ETH-USD child and fine-tuned on ETH TRAIN.
- TimesFM: zero-shot foundation model, no training.
- Walk-forward evaluation on the ETH TEST region (step = pred_len).
- Metrics computed in RAW price space (LSTM outputs inverse-transformed).
"""

import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam

from src.config.pipeline_config import IndicatorConfig
from src.data.data_ingestion import _fetch_coinbase_candles, RSI, MACD
from src.model.model_defination import PrismModel

SEED = 42
FEATURES = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD"]
CLOSE_IDX = FEATURES.index("Close")
CONTEXT_LEN = 512
PRED_LEN = 5
DATA_START = "2020-01-01"
TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END = pd.Timestamp("2024-12-31")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(ticker: str) -> pd.DataFrame:
    """Fetch OHLCV and compute RSI/MACD — same features the pipeline uses."""
    ind = IndicatorConfig()
    end_dt = datetime.utcnow() - timedelta(days=1)  # drop partial current-day candle
    df = _fetch_coinbase_candles(ticker, datetime.strptime(DATA_START, "%Y-%m-%d"), end_dt)
    df["RSI"] = RSI(df["Close"], window=ind.RSI_WINDOW)
    df["MACD"] = MACD(df["Close"], fast=ind.MACD_FAST, slow=ind.MACD_SLOW)
    df = df[["date"] + FEATURES].dropna().reset_index(drop=True)
    return df


def region_bounds(df: pd.DataFrame):
    """Index boundaries: TRAIN targets < val < test."""
    train_end = int((df["date"] <= TRAIN_END).sum())
    val_end = int((df["date"] <= VAL_END).sum())
    return train_end, val_end


# ---------------------------------------------------------------------------
# LSTM model (corrected legacy architecture: 3-layer LSTM, hidden=128)
# ---------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 3, pred_len: int = PRED_LEN, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size * pred_len)
        self.pred_len = pred_len
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(-1, self.pred_len, self.input_size)


def build_samples(df: pd.DataFrame, scaler: StandardScaler, t_start: int, t_end: int,
                  step: int = PRED_LEN):
    """Windows whose TARGET rows are in [t_start, t_end). Context may reach
    back before t_start (strictly past data) — no future leakage."""
    vals = scaler.transform(df[FEATURES]).astype("float32")
    X, Y = [], []
    for t in range(t_start, t_end - PRED_LEN + 1, step):
        past = vals[t - CONTEXT_LEN:t]
        fut = vals[t:t + PRED_LEN]
        if past.shape == (CONTEXT_LEN, len(FEATURES)) and fut.shape == (PRED_LEN, len(FEATURES)):
            X.append(past)
            Y.append(fut)
    X = torch.tensor(np.array(X))
    Y = torch.tensor(np.array(Y))
    return X, Y


def fit_lstm(model: nn.Module, X, Y, Xv, Yv, epochs: int, lr: float,
             batch_size: int = 32, patience: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    crit = nn.MSELoss()
    best_val = float("inf")
    counter = 0
    n = X.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx].to(device), Y[idx].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            xv, yv = Xv.to(device), Yv.to(device)
            vloss = crit(model(xv), yv).item()
        print(f"    epoch {epoch}/{epochs} train={total / n:.5f} val={vloss:.5f}")
        if vloss < best_val:
            best_val = vloss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"    early stop at epoch {epoch}")
                break
    return model


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def predict_lstm_window(model: nn.Module, context_df: pd.DataFrame,
                        scaler: StandardScaler, device) -> np.ndarray:
    vals = scaler.transform(context_df[FEATURES]).astype("float32")
    x = torch.tensor(vals).unsqueeze(0).to(device)  # (1, context, features)
    model.eval()
    with torch.no_grad():
        pred = model(x).detach().cpu().numpy()[0]  # (pred_len, features)
    return scaler.inverse_transform(pred)


def evaluate_walk_forward(predict_fn, test_df: pd.DataFrame, t_start: int,
                          step: int = PRED_LEN):
    """Non-overlapping walk-forward windows whose targets are in TEST region."""
    actuals, preds = [], []
    for t in range(t_start, len(test_df) - PRED_LEN + 1, step):
        context = test_df.iloc[t - CONTEXT_LEN:t].copy()
        actual = test_df[FEATURES].iloc[t:t + PRED_LEN].values
        if actual.shape[0] < PRED_LEN:
            break
        pred = predict_fn(context)
        actuals.append(actual)
        preds.append(pred)
    if not actuals:
        return None
    Y = np.concatenate(actuals, axis=0)
    P = np.concatenate(preds, axis=0)
    return Y, P


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", choices=["fine_tune", "freeze"], default="fine_tune")
    ap.add_argument("--out", default="outputs/experiment")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print("Fetching data...")
    btc = load_data("BTC-USD")
    eth = load_data("ETH-USD")
    print(f"  BTC-USD: {len(btc)} rows | ETH-USD: {len(eth)} rows")

    # ---- 1. Train LSTM parent on BTC-USD (train/val targets only) ----
    btc_train_end, btc_val_end = region_bounds(btc)
    print(f"\n[LSTM] Training parent on BTC-USD "
          f"(train targets < {TRAIN_END.date()}, val targets < {VAL_END.date()})...")
    parent_scaler = StandardScaler().fit(btc.iloc[:btc_train_end][FEATURES])
    Xp, Yp = build_samples(btc, parent_scaler, CONTEXT_LEN, btc_train_end)
    Xpv, Ypv = build_samples(btc, parent_scaler, btc_train_end, btc_val_end)
    print(f"  parent windows train={Xp.shape[0]} val={Xpv.shape[0]}")
    parent = LSTMPredictor(input_size=len(FEATURES))
    fit_lstm(parent, Xp, Yp, Xpv, Ypv, epochs=20, lr=1e-3)

    # ---- 2. Transfer weights -> ETH-USD child, fine-tune on ETH train ----
    eth_train_end, eth_val_end = region_bounds(eth)
    print(f"\n[LSTM] Transferring parent weights to ETH-USD child "
          f"(train targets < {TRAIN_END.date()}, val targets < {VAL_END.date()})...")
    child_scaler = StandardScaler().fit(eth.iloc[:eth_train_end][FEATURES])
    Xc, Yc = build_samples(eth, child_scaler, CONTEXT_LEN, eth_train_end)
    Xcv, Ycv = build_samples(eth, child_scaler, eth_train_end, eth_val_end)
    print(f"  child windows train={Xc.shape[0]} val={Xcv.shape[0]}")

    child = LSTMPredictor(input_size=len(FEATURES))
    child.load_state_dict(parent.state_dict())
    if args.transfer == "freeze":
        print("  [freeze] freezing LSTM layers, training head only")
        for name, p in child.named_parameters():
            if "lstm" in name:
                p.requires_grad = False
    fit_lstm(child, Xc, Yc, Xcv, Ycv, epochs=10,
             lr=1e-4 if args.transfer == "fine_tune" else 3e-4)

    # ---- 3. Walk-forward evaluation on ETH TEST (both models) ----
    test_start = eth_val_end
    print(f"\nWalk-forward evaluation on ETH-USD TEST "
          f"({len(eth) - test_start} rows, {eth['date'].iloc[test_start].date()} -> {eth['date'].iloc[-1].date()})")
    print("  loading TimesFM...")
    tfm = PrismModel()

    def lstm_pred(context_df):
        return predict_lstm_window(child, context_df, child_scaler, device)

    def timesfm_pred(context_df):
        return tfm.predict(context_df, horizon=PRED_LEN)

    print("  evaluating LSTM child...")
    lstm_res = evaluate_walk_forward(lstm_pred, eth, test_start)
    print("  evaluating TimesFM...")
    tfm_res = evaluate_walk_forward(timesfm_pred, eth, test_start)

    # ---- 4. Report ----
    results = []
    for res, name in [(lstm_res, "LSTM_child"), (tfm_res, "TimesFM")]:
        Y, P = res
        yc, pc = Y[:, CLOSE_IDX], P[:, CLOSE_IDX]
        results.append({
            "model": name,
            "windows": int(Y.shape[0] / PRED_LEN),
            "samples": int(Y.shape[0]),
            "close_MSE": float(mean_squared_error(yc, pc)),
            "close_RMSE": float(np.sqrt(mean_squared_error(yc, pc))),
            "close_MAE": float(mean_absolute_error(yc, pc)),
            "close_R2": float(r2_score(yc, pc)),
        })

    print("\n" + "=" * 72)
    print("RESULTS (ETH-USD TEST, raw price space, close channel)")
    print("=" * 72)
    for r in results:
        print(f"{r['model']:<12} windows={r['windows']:<4} "
              f"RMSE=${r['close_RMSE']:>10.2f}  MAE=${r['close_MAE']:>10.2f}  "
              f"R2={r['close_R2']:>6.4f}  MSE={r['close_MSE']:>14.4f}")

    out_path = os.path.join(args.out, "comparison.json")
    with open(out_path, "w") as f:
        json.dump({
            "ticker": "ETH-USD",
            "data_start": DATA_START,
            "train_end": str(TRAIN_END.date()),
            "val_end": str(VAL_END.date()),
            "test_start": str(eth["date"].iloc[test_start].date()),
            "test_end": str(eth["date"].iloc[-1].date()),
            "context_len": CONTEXT_LEN,
            "pred_len": PRED_LEN,
            "transfer_strategy": args.transfer,
            "results": results,
        }, f, indent=2)

    np.savez(os.path.join(args.out, "forecasts.npz"),
             lstm_Y=lstm_res[0], lstm_P=lstm_res[1],
             tfm_Y=tfm_res[0], tfm_P=tfm_res[1])
    print(f"\nSaved to {out_path} and forecasts.npz")


if __name__ == "__main__":
    main()