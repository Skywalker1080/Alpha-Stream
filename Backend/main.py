import asyncio
import os
import redis
import uvicorn
import psutil
import shutil
import urllib.request
from logger.logger import get_logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from Backend.telemetry import init_telemetry, current_trace_id, _OTEL_AVAILABLE
from src.utils import initialize_dirs
import Backend.state as app_state
from Backend.state import (
    REDIS_STATUS,
    QDRANT_STATUS,
    OLLAMA_STATUS,
    SYSTEM_CPU,
    SYSTEM_RAM,
    SYSTEM_DISK,
    REDIS_KEYS
)
from Backend.api import router

logger = get_logger()

init_telemetry()

if _OTEL_AVAILABLE:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        RequestsInstrumentor().instrument()
        RedisInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        logger.info("OpenTelemetry client instrumentors enabled (requests/redis/httpx)")
    except Exception as e:
        logger.warning(f"OpenTelemetry client instrumentation failed: {e}")

app = FastAPI(title="Alpha Stream API", description="Backend for Alpha Stream", version="0.1.0")
app.include_router(router)


@app.middleware("http")
async def add_trace_id_header(request, call_next):
    """Echo the active trace id back on the response for easy correlation."""
    response = await call_next(request)
    trace_id = current_trace_id()
    if trace_id:
        response.headers["X-Trace-Id"] = trace_id
    return response


if _OTEL_AVAILABLE:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry FastAPI instrumentation enabled")
    except Exception as e:
        logger.warning(f"OpenTelemetry FastAPI instrumentation failed: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus Instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[".*admin.*", "/metrics"],
    env_var_name="ENABLE_METRICS",
)
instrumentator.instrument(app)

from prometheus_client import generate_latest
from starlette.responses import Response as StarletteResponse

@app.get("/metrics")
async def metrics_route():
    try:
        data = generate_latest()
        return StarletteResponse(content=data, media_type="text/plain; version=1.0.0; charset=utf-8")
    except Exception as e:
        logger.error(f"/metrics generate_latest error: {type(e).__name__}: {e}")
        return StarletteResponse(content=b"", media_type="text/plain; version=1.0.0; charset=utf-8", status_code=500)


@app.on_event("startup")
async def startup():
    initialize_dirs()
    
    # retry logic for redis
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    
    connected = False
    for i in range(10):
        try:
            client = redis.Redis(host=redis_host, port=redis_port, db=0)
            client.ping()
            app_state.Redis_client = client
            logger.info(f"BACKEND - System Online (FastAPI, Redis at {redis_host}:{redis_port}, Mlflow)")
            REDIS_STATUS.set(1)
            connected = True
            break
        except Exception as e:
            logger.warning(f"BACKEND - Waiting for Redis connection... attempting {i+1}/10. Error: {str(e)}")
            await asyncio.sleep(5)

    if not connected:
        REDIS_STATUS.set(0)
        logger.error("BACKEND - Failed to connect to Redis")
    
    # Start background task for metrics
    asyncio.create_task(metrics_updater())

def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


async def metrics_updater():
    """Background task to update system and redis metrics every 15 seconds."""
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    while True:
        try:
            # CPU
            SYSTEM_CPU.set(psutil.cpu_percent())
            
            # RAM
            ram = psutil.virtual_memory()
            SYSTEM_RAM.set(ram.used / (1024 * 1024)) # MB
            
            # Disk
            total, used, free = shutil.disk_usage("/")
            SYSTEM_DISK.set(used / (1024 * 1024)) # MB

            # Qdrant / Ollama uptime
            QDRANT_STATUS.set(1 if await asyncio.to_thread(_http_ok, f"http://{qdrant_host}:6333/") else 0)
            OLLAMA_STATUS.set(1 if await asyncio.to_thread(_http_ok, f"{ollama_base}/api/tags") else 0)
            
            # Redis Keys
            if app_state.Redis_client:
                try:
                    num_keys = app_state.Redis_client.dbsize()
                    REDIS_KEYS.set(num_keys)
                    REDIS_STATUS.set(1)
                except Exception:
                    REDIS_STATUS.set(0)
            else:
                REDIS_STATUS.set(0)
                
        except Exception as e:
            logger.error(f"Metrics updater error: {str(e)}")
            
        await asyncio.sleep(15)

if __name__=="__main__":
    uvicorn.run("Backend.main:app", host="0.0.0.0", port=8000, reload=True)

    