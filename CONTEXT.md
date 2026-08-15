# Crypto Prism Ops

MLOps platform that forecasts cryptocurrency prices. A single pre-trained TimesFM foundation model serves every ticker; per-ticker artifacts ("training") are fitted scalers plus evaluation runs.

## Language

**Ticker**:
A cryptocurrency pair in the form `BTC-USD`, `ETH-USD`.
_Avoid_: coin, symbol

**Parent model**:
The reference model for the market index ticker (`BTC-USD` by default). Conceptually the "parent" that children learn from, though under TimesFM it shares the same foundation weights as children.
_Avoid_: base model

**Child model**:
The per-ticker model, one per ticker, built with the same foundation weights as the parent.
_Avoid_: sub-model

**ModelProvisioner**:
The module that owns the question "is a forecast available for this ticker?" — it decides whether the parent or child for a ticker is provisioned, training, or missing, and enqueues the training that makes it available. One decision, one place to test.
_Avoid_: model service, model manager

**Provisioned**:
The state of a ticker whose required artifacts exist on disk. For the parent that is its scaler; for a child its own scaler. Provisioned is a fact about the filesystem, independent of any in-flight task.
_Avoid_: trained, ready

**Forecast**:
The output a prediction run produces for a ticker: a history window plus a multi-step OHLCV prediction.
_Avoid_: prediction (when meaning the whole result object)

**Task**:
A training or prediction run tracked through a status lifecycle. Tasks have ids and a state (running, completed, failed).
_Avoid_: job, worker