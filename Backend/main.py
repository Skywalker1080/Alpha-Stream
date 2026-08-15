import asyncio
import os
import redis
import uvicorn
import psutil
import shutil
from logger.logger import get_logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.utils import initialize_dirs
import Backend.state as app_state
from Backend.state import (
    REDIS_STATUS, 
    SYSTEM_CPU, 
    SYSTEM_RAM, 
    SYSTEM_DISK, 
    REDIS_KEYS
)
from Backend.api import router

logger = get_logger()

app = FastAPI(title="Alpha Stream API", description="Backend for Alpha Stream", version="0.1.0")
app.include_router(router)

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
instrumentator.instrument(app).expose(app, endpoint="/metrics")

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

async def metrics_updater():
    """Background task to update system and redis metrics every 15 seconds."""
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

    