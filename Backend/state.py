import redis
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, REGISTRY

# Shared State
Redis_client: Optional[redis.Redis] = None
executor = ThreadPoolExecutor(max_workers=4)

# Metrics
registry = REGISTRY

SYSTEM_CPU = Gauge("system_cpu_percent", "CPU percent", registry=registry)
SYSTEM_RAM = Gauge("system_ram_used_mb", "RAM MB", registry=registry)
SYSTEM_DISK = Gauge("system_disk_used_mb", "Disk Used MB", registry=registry)
REDIS_STATUS = Gauge("redis_up", "Redis up=1/down=0", registry=registry)
REDIS_KEYS = Gauge("redis_keys_total", "Number of keys in Redis", registry=registry)
QDRANT_STATUS = Gauge("qdrant_up", "Qdrant up=1/down=0", registry=registry)
OLLAMA_STATUS = Gauge("ollama_up", "Ollama up=1/down=0", registry=registry)
TRAINING_STATUS = Gauge("training_status", "0=idle 1=running 2=completed", registry=registry)
TRAINING_MSE = Gauge("training_mse_last", "Last training MSE", registry=registry)
TRAINING_DURATION = Histogram("training_duration_seconds", "Training duration in seconds", registry=registry)
PREDICTION_COUNTER = Counter("prediction_total", "Total predictions", ["type", "ticker"], registry=registry)
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", ["type", "ticker"], registry=registry)
CACHE_HIT = Counter("redis_cache_hit_total", "Cache hits", ["key"], registry=registry)
CACHE_MISS = Counter("redis_cache_miss_total", "Cache misses", ["key"], registry=registry)

# Drift Metrics
DRIFT_SCORE = Gauge("model_drift_score", "Drift score for ticker", ["ticker"], registry=registry)
VOLATILITY_INDEX = Gauge("model_volatility_index", "Volatility index for ticker", ["ticker"], registry=registry)

# Agent path (LangGraph /analyze) + LLM
_LATENCY_BUCKETS = [0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120]
AGENT_ANALYSIS_LATENCY = Histogram(
    "agent_analysis_latency_seconds",
    "End-to-end /analyze latency", ["ticker"],
    buckets=_LATENCY_BUCKETS, registry=registry,
)
AGENT_ANALYSIS_TOTAL = Counter("agent_analysis_total", "Total /analyze requests", ["ticker"], registry=registry)
AGENT_ANALYSIS_ERRORS = Counter("agent_analysis_errors_total", "/analyze errors", ["ticker"], registry=registry)

LLM_NODE_LATENCY = Histogram(
    "llm_node_latency_seconds",
    "Per-node LLM invoke latency (incl. retries)", ["node"],
    buckets=_LATENCY_BUCKETS, registry=registry,
)
LLM_CALLS = Counter("llm_calls_total", "LLM invokes by outcome", ["node", "outcome"], registry=registry)
LLM_RETRIES = Counter("llm_retries_total", "LLM invoke retry attempts", ["node"], registry=registry)

SEMANTIC_CACHE_HIT = Counter("semantic_cache_hit_total", "Qdrant semantic cache hits", [], registry=registry)
SEMANTIC_CACHE_MISS = Counter("semantic_cache_miss_total", "Qdrant semantic cache misses", [], registry=registry)