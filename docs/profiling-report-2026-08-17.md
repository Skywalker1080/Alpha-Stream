# Core Pipeline Profiling Report

- **Date:** 2026-08-17
- **Scope:** Core pure-Python pipeline — TimesFM model load, Coinbase data ingestion, scaler fit, and forecast inference. (Agent/LLM path, Redis/Qdrant caching, and full API latency are out of scope for this pass.)
- **Environment:** Windows 11, 16 logical cores (8 physical, hyperthreading), CPU-only (`torch.cuda.is_available() == False`), Python 3.11 venv, vendored `timesfm` 2.5 package, TimesFM 2.5 200M weights (`model/timesfm-2.5-200m-pytorch`), torch at 8 threads.

## Methodology

- **Stage timing:** `time.perf_counter()` around each pipeline stage in a single process (see below for stage list).
- **Hot-spot analysis:** `cProfile` on one `PrismModel.predict()` call, sorted by cumulative time.
- Data fetched live from the Coinbase Exchange API. Feast/Redis materialization was bypassed for stage timings (see Findings #4) so the pure CPU/network cost could be measured.

## Stage Timings

| Stage | Time | Notes |
|---|---|---|
| Model load (cold) | **2.8–4.1 s** | once per process; ~900 MB RSS per worker |
| Data fetch (Coinbase, 2022→now) | **5.2–8.5 s** | ETH-USD 5.29 s, SOL-USD 8.51 s (network variance) |
| Data fetch (Coinbase, 2020→now, parent) | **11.46 s** | BTC-USD: ~9 × 300-candle pages × (0.35 s sleep + latency) |
| Feature engineering (RSI/MACD) | < 1 s | negligible |
| `StandardScaler` fit (7 features) | ~0 s | negligible |
| **Forecast — `PrismModel.predict` (cold)** | **2.60 s** | first call |
| **Forecast — `PrismModel.predict` (warm)** | **2.47 s** | no warm-up benefit; steady-state |

## Forecast Hot-Spot Breakdown (cProfile, one predict)

```
ncalls  tottime  cumtime  function
   1    0.000    2.459    src/model/model_defination.py:71 (predict)
   1    0.000    2.458    timesfm_2p5_base.py:155 (forecast)
   1    0.010    2.455    timesfm_2p5_torch.py:381 (_compiled_decode)
 672/2  0.012    2.383    torch module _call_impl  (transformer forward)
 178    1.927    1.927    {built-in torch._C._nn.linear}     ← 78% of predict
  40    0.045    1.455    torch/transformer.py:224 (forward)
  40    0.000    0.040    _torch_dot_product_attention       ← 2%, negligible
```

**Takeaway:** ~78% of forecast time is `torch._C._nn.linear` matrix multiplication in the transformer feed-forward layers — pure CPU GEMM on the 200M model. Attention and everything else are negligible.

## Findings

1. **Primary bottleneck — CPU-bound TimesFM inference (~2.5 s/predict).**
   Cold (2.60 s) ≈ warm (2.47 s), so there is no compile-on-first-call or in-process cache to unlock: every forecast pays a full CPU forward pass. The prediction result is Redis-cached for 24 h, so *repeat* requests are cheap; the 2.5 s floor hits cold predictions and every provisioning validation pass (`evaluate_quick`).

2. **Secondary — Coinbase pagination sleep.** `_fetch_coinbase_candles` sleeps a hard-coded `0.35 s` per 300-candle page. The parent's 2020 history is ~9 pages, so ~3 s of the ~11.5 s fetch is pure sleep. Fetch time scales linearly with the requested history range.

3. **Operational — model load × worker count.** `Backend/start.sh` runs `uvicorn --workers 4`; each worker lazily loads the ~900 MB model (2.8–4.1 s). Cold start ≈ 12 s and steady-state RSS ≈ 3.6 GB just for the foundation model. Additionally, 4 workers × 8 torch threads = 32 threads competing for 16 logical cores → GEMM oversubscription that can *hurt* single-request latency.

4. **Coupling / robustness (not a speed issue).** `fetch_ohlcv` **hard-fails without Redis**: the Feast online-store materialization (`redis:6379`) is mandatory, so the "core" pipeline is not actually standalone. Also, `evaluate_quick`'s MLflow logging crashed locally on a stale `mlflow.db` (alembic revision mismatch) — masked in Docker because `start.sh` runs `mlflow db upgrade`.

## High-Leverage Fix (targets #1 and #3)

**Run the backend as a single uvicorn worker and let torch use the full core count.**

- `Backend/start.sh`: `--workers 4` → `--workers 1` (the app is already async with an internal `ThreadPoolExecutor(max_workers=4)`, so concurrency is preserved; FastAPI + asyncio handles concurrent HTTP, blocking work stays in the thread pool, and torch releases the GIL during GEMM).
- In `_load_timesfm_model` (or at backend startup): set `torch.set_num_threads()` to the physical core count so the single process's GEMMs get uncontended cores.

**Measured impact:** see [Experiment: Single-Worker Consolidation](#experiment-single-worker-consolidation) below. The latency hypothesis was confirmed — and then some.

**Alternative (keep 4 workers):** `uvicorn --workers 4 --preload` and load the model at import time so the master loads once and workers inherit pages via fork COW. More complex; single worker is the simpler, higher-leverage change.

## Experiment: Single-Worker Consolidation (2026-08-17)

**Hypothesis:** reducing `--workers 4` → `--workers 1` removes redundant 900 MB model loads and torch-thread oversubscription (4 workers × 8 threads = 32 threads, up to 128 under load, on 16 logical cores), cutting cold start, RAM, and per-request latency.

**Method (controlled):**
- Branch: `experiment/single-worker-torch-cores` (recoverable; changes: `start.sh` workers → 1, `_pin_torch_threads()` added to model load).
- Driver: `exp_driver.py` (temp scratch) spawns real worker processes, each mirroring a uvicorn worker — load `PrismModel` once (timed + RSS), run 3 serial predicts, then 4 concurrent predicts through a 4-thread pool (matches the backend `ThreadPoolExecutor`).
- Input: identical ETH-USD frame (1677 rows, 2022→now) fetched once and reused across configs. Same machine, sequential runs.
- Configs: A = 4 workers × 8 threads (baseline), B = 1 worker × 8 threads (physical), C = 1 worker × 16 threads (all logical).

**Results:**

| Config | workers | threads | model load (sum) | RSS (sum) | serial mean | concurrent mean | concurrent p95 |
|---|---|---|---|---|---|---|---|
| **A baseline** | 4 | 8 | **30.5 s** | **3589 MB** | **13.83 s** | **8.84 s** | **36.88 s** |
| **B treatment** | 1 | 8 | 3.78 s | 1146 MB | 2.46 s | 2.69 s | 9.16 s |
| **C variant** | 1 | 16 | 2.48 s | 1146 MB | 2.40 s | 2.61 s | 8.76 s |

**Interpretation:**
- The 4-worker baseline is catastrophically oversubscribed: concurrent predicts contend for CPU so heavily that **p95 latency hits 37 s** and the mean concurrent predict is **8.8 s** (vs the 2.5 s isolated GEMM cost).
- Single worker restores near-isolated latency: **serial 2.46 s, concurrent mean 2.69 s** — a ~3× mean and ~4× p95 improvement under load.
- **RSS drops 3.1× (3.6 GB → 1.1 GB)**; aggregate model-load CPU work drops ~8–12× (30.5 s → 2.5–3.8 s).
- Threads 8 vs 16 are effectively tied on latency (2.40–2.46 s serial); the branch pins physical cores (8) as the safe default to keep oversubscription headroom under heavier concurrency.
- Caveat: single run per config; the baseline effect is large enough (≈5.6× serial) that variance cannot reverse the conclusion.

## Follow-ups / open questions

- ✅ Worker consolidation measured (see experiment above) — pending decision to merge `experiment/single-worker-torch-cores` into `main`.
- Consider GPU / `torch.compile` / bf16 if the ~2.5 s CPU GEMM floor still matters after the worker change (the single-worker fix removes oversubscription, not the hardware-bound GEMM cost).
- Consider removing or tuning the 0.35 s/page fetch sleep (e.g., concurrent page fetches) — separate from #1/#3.
- Decide whether Feast materialization should be degradable when Redis is down (currently it aborts the whole ingest).