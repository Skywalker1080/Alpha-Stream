# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crypto Prism Ops is an institutional-grade MLOps platform for cryptocurrency price forecasting. It combines:
- A **TimesFM zero-shot forecast pipeline** (TimesFM 2.5 200M + MLflow). The earlier **Parent-Child LSTM** path is **legacy/dead** — kept in the repo as a possible future feature, but not part of the running system.
- A **LangGraph multi-agent** analysis system backed by Ollama
- A **FastAPI** backend with async Redis task queue
- A **Streamlit** frontend dashboard
- **Feast** feature store, **Qdrant** semantic cache, **Prometheus/Grafana** observability

## Running the Stack

### Docker (primary)
```bash
docker-compose up -d --build        # Start all services
docker-compose logs -f              # Stream logs
docker-compose up -d --build --force-recreate  # Full rebuild
```

Ollama must also be running locally (provides the LLM and embeddings):
```bash
ollama serve
```

### Manual (local dev)
```bash
# Install dependencies
pip install uv
uv venv .venv && uv pip install -r Backend/requirements.txt

# Backend (from project root, PYTHONPATH must include root)
PYTHONPATH=. uvicorn Backend.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend
streamlit run frontend/app.py
```

The `Backend/start.sh` also runs `mlflow db upgrade sqlite:///mlflow.db` before starting uvicorn with 4 workers.

### Service URLs
| Service | URL |
|---|---|
| Backend API / Docs | http://localhost:8000 / http://localhost:8000/docs |
| Frontend (Streamlit) | http://localhost:8501 |
| Redis Insights | http://localhost:8001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |
| MLflow | http://localhost:5000 (if port-mapped) |

## Linting & Testing
```bash
ruff check .                        # Linting
pytest tests/                       # Tests (no project-level tests exist yet)
```

## Architecture

### Core Configuration (`src/config/pipeline_config.py`)
Single `Config` dataclass drives the entire pipeline:
- `parent_ticker = "BTC-USD"`, `start = "2020-01-01"` (parent reference start)
- `child_start = "2022-01-01"` (child start)
- `context_len = 512`, `pred_len = 5` (forecast steps; TimesFM supports up to 16384 context)
- `features = [Open, High, Low, Close, Volume, RSI, MACD]`
- `workdir = "outputs"` — parent artifacts at `outputs/parent/`, children at `outputs/{ticker}/`
- `parent_epochs`, `child_epochs`, `transfer_strategy`, `fine_tune_lr` are **LSTM-era dead fields** — unused by the TimesFM pipeline

### ML Model (`src/model/model_defination.py`)
`PrismModel` wraps **TimesFM 2.5 200M** (`model/timesfm-2.5-200m-pytorch/`) for channel-independent zero-shot forecasting: each feature channel is forecast individually, then reassembled into the multi-variate `(pred_len, num_features)` output the pipeline expects. It is **not** a PyTorch module and no weights are trained. The legacy `PrismModel` (3-layer LSTM, hidden=128, transfer learning) is dead; the parent/child distinction survives only as per-ticker scalers fit over the same foundation model.

### Training Pipeline (`src/pipeline/training_pipeline.py`)
`train_parent()` / `train_child(ticker)` — with TimesFM this is a **zero-shot evaluation run**, not weight training: fetch OHLCV → `StandardScaler` fit → load TimesFM → `evaluate_model_temp()` → save the per-ticker scaler → log metrics/artifacts to MLflow. Artifacts land at `outputs/{ticker}/{ticker}_child_scaler.pkl` (parent: `outputs/parent/{parent_ticker}_parent_scaler.pkl`). Whether a model "exists" for a ticker = that scaler exists (see `ModelProvisioner`, `src/model/provisioning.py`). The legacy `.pt` weights and `CryptoData`/`fit_model` path are dead.

### Inference Pipeline (`src/pipeline/inference_pipeline.py`)
`predict_parent()` / `predict_child(ticker)` — load saved model+scaler → fetch fresh OHLCV → optionally pull from Feast online store → `predict_one_step()` → return 30-day history + multi-step forecast.

### Legacy LSTM path (dead, not deleted)
The Parent-Child LSTM system that preceded TimesFM is retained in the repo — `src/model/training.py` (`fit_model`), `src/data/data_preparation.py` (`CryptoData`), `src/train_utils.py`, the stale `.pt` files under `outputs/`, and the dead `Config` fields above. It is not invoked by the running pipeline, but is kept as a candidate for a future trained-model feature. Do not treat these as part of the current architecture.

### Async Task Queue (`Backend/tasks.py` + `Backend/state.py`)
Training runs in a `ThreadPoolExecutor(max_workers=4)` via `asyncio.run_in_executor`. Task status (running/completed/failed) is persisted in Redis with 1-hour TTL. A `chain_fn` can be attached to auto-run inference after training completes. Metrics (training duration, MSE, prediction latency, cache hits) are exposed as Prometheus gauges/counters/histograms.

### LangGraph Agent System (`src/agents/`)
`analyze_stock(ticker)` in `graph.py` orchestrates:
1. **Semantic cache check** — queries Qdrant (`dataset_cache` collection) via `nomic-embed-text` embeddings; returns cached result if score > 0.95
2. **Fetch prediction data** — calls internal predict endpoint
3. **4-node LangGraph graph**: `performance_analyst → market_expert → report_generator → critic`
4. **Cache write** — saves result to Qdrant for future cache hits

LLM is `ChatOllama` with model `gpt-oss:20b-cloud`, connected via `OLLAMA_BASE_URL` env var (defaults to `http://host.docker.internal:11434`). If Ollama is unavailable, a `MockLLM` is used silently.

### Backend API (`Backend/api.py`)
Routes via `APIRouter` included in `Backend/main.py`:
- `POST /train-parent`, `POST /train-child` — trigger async training (rate-limited: 3/60s)
- `POST /predict-parent`, `POST /predict-child` — get predictions; auto-triggers training if model missing (returns HTTP 202)
- `POST /analyze` — full LangGraph analysis with semantic cache
- `GET /status/{task_id}` — poll Redis for training task status
- `POST /monitor/{ticker}`, `GET /monitor/{ticker}/drift` — drift reports from `outputs/{ticker}/drift/`
- `POST /system/reset` — flushes Redis DB and recreates Qdrant collection
- `GET /metrics` — Prometheus scrape endpoint

Rate limiting is Redis-backed sliding window (`Backend/rate_limiter.py`).

### Key Env Vars
| Variable | Default | Purpose |
|---|---|---|
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama endpoint |

### Package Management
Dependencies are managed with `uv` (`pyproject.toml` + `uv.lock`). The Docker build installs from `Backend/requirements.txt` into a venv. `PYTHONPATH` must be set to the project root for all imports to resolve (src, Backend, logger packages).

### Kubernetes (`k8s/`)
Manifests exist for all services. Production deployment targets minikube/k8s. Config via `config-map.yaml` and `secrets.yaml`.
