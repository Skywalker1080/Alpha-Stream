"""
Plot and extend the LSTM-vs-TimesFM comparison from forecasts.npz.
Adds directional accuracy (sign of next-day move) to the raw-space error metrics.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "outputs/experiment"
CLOSE_IDX = 3
PRED_LEN = 5


def directional_accuracy(Y: np.ndarray, P: np.ndarray):
    """% of windows where the sign of the 1-day move is correct."""
    actual = Y[::PRED_LEN, CLOSE_IDX]
    pred = P[::PRED_LEN, CLOSE_IDX]
    actual_move = np.sign(np.roll(actual, -1) - actual)
    pred_move = np.sign(np.roll(pred, -1) - pred)
    valid = actual_move != 0
    return float((actual_move[valid] == pred_move[valid]).mean())


data = np.load(os.path.join(BASE, "forecasts.npz"))
lstm_Y, lstm_P = data["lstm_Y"], data["lstm_P"]
tfm_Y, tfm_P = data["tfm_Y"], data["tfm_P"]
with open(os.path.join(BASE, "comparison.json")) as f:
    meta = json.load(f)

print(f"Directional accuracy (1-day move, {BASE}):")
print(f"  LSTM_child: {directional_accuracy(lstm_Y, lstm_P):.3f}")
print(f"  TimesFM   : {directional_accuracy(tfm_Y, tfm_P):.3f}")

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

for ax, (Y, P, name) in zip(
    axes,
    [
        (lstm_Y, lstm_P, "LSTM child"),
        (tfm_Y, tfm_P, "TimesFM"),
    ],
):
    steps = np.arange(Y.shape[0])
    ax.plot(steps, Y[:, CLOSE_IDX], color="black", alpha=0.85, label="Actual close")
    ax.plot(steps, P[:, CLOSE_IDX], color="tab:blue", alpha=0.7, label=f"{name} forecast")
    rmse = np.sqrt(np.mean((Y[:, CLOSE_IDX] - P[:, CLOSE_IDX]) ** 2))
    ax.set_title(f"{name} — ETH-USD test, RMSE ${rmse:.2f}")
    ax.set_ylabel("Close price ($)")
    ax.legend(loc="upper left")

axes[0].set_xlabel("forecast step index")
plt.tight_layout()
out = os.path.join(BASE, "comparison_plot.png")
plt.savefig(out, dpi=130)
print(f"Saved plot: {out}")