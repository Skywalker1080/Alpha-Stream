import asyncio
import contextvars
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

from logger.logger import get_logger

import Backend.state as app_state
from Backend.state import (
    executor,
    TRAINING_DURATION, TRAINING_STATUS, TRAINING_MSE,
    CACHE_HIT, CACHE_MISS
)

logger = get_logger()

def _redis():
    return app_state.Redis_client

def refresh_system_metrics():
    pass # in monitoring run

def _json_default(o):
    """JSON serializer fallback for pandas/numpy types in cached payloads."""
    import numpy as np
    if hasattr(o, "isoformat"):  # pandas Timestamp / datetime
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)

def get_or_set_cache(key: str, compute_fn, expire: int = 86400):
    """Helper to check Redis cache or compute and cache."""
    refresh_system_metrics()
    redis = _redis()
    if redis:
        try:
            val = redis.get(key)
            if val:
                logger.info(f"CACHE HIT (redis): {key}")
                CACHE_HIT.labels(key).inc()
                return json.loads(val), True
            logger.info(f"CACHE MISS (redis): {key}")
        except Exception as e:
            logger.warning(f"CACHE READ ERROR: {key}: {str(e)}")
    else:
        logger.warning(f"CACHE NO-REDIS (redis client unavailable): {key}")

    try:
        result = compute_fn()
    except Exception as e:
        logger.error(f"CACHE COMPUTE FAILED: {key}: {str(e)}")
        raise

    if redis:
        try:
            redis.set(key, json.dumps(result, default=_json_default), ex=expire)
            logger.info(f"CACHE SET OK: {key} (ttl={expire}s)")
            CACHE_MISS.labels(key).inc()
        except Exception as e:
            logger.error(f"CACHE SET FAILED: {key}: {str(e)} -> result will NOT be cached")
    return result, False

# Task status tracking Redis

def get_task_key(task_id: str) -> str:
    return f"task_status: {task_id.lower()}"

def save_task_status(task_id: str, status_data: Dict[str, Any], ttl: int = 3600):
    """Save task status to Redis"""
    try:
        redis = _redis()
        if redis:
            redis.set(get_task_key(task_id), json.dumps(status_data), ex=ttl)
    except Exception as e:
        logger.error(f"Failed to save task status for {task_id}: {str(e)}")

def get_task_status_redis(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status from Redis"""
    try:
        redis = _redis()
        if redis:
            val = redis.get(get_task_key(task_id))
            if val:
                return json.loads(val)
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {str(e)}")
    return None

def delete_task_status(task_id: str):
    """Delete task status from Redis"""
    try:
        redis = _redis()
        if redis:
            redis.delete(get_task_key(task_id))
    except Exception as e:
        logger.error(f"Failed to delete task status for {task_id}: {str(e)}")

# Training task execution

async def run_training_worker(task_id: str, fn, *args, chain_fn=None):
    """Actual training worker that runs in background. (Thread Pool)"""
    loop = asyncio.get_event_loop()
    start_time = time.time()
    ctx = contextvars.copy_context()
    try:
        result = await loop.run_in_executor(executor, ctx.run, fn, *args)
        if chain_fn:
            logger.info(f"Task {task_id}: Training complete, running chained task...")
            await loop.run_in_executor(executor, ctx.run, chain_fn)
            logger.info(f"Task {task_id}: Chained Task Complete")
        
        duration = time.time() - start_time
        TRAINING_DURATION.observe(duration)

        status_data = {"status": "completed", "result": result, "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        save_task_status(task_id, status_data, ttl=3600)

        TRAINING_STATUS.set(2)

        if isinstance(result, dict) and "mse" in result:
            TRAINING_MSE.set(result["mse"])
    except Exception as e:
        status_data = {"status": "failed", "error": str(e), "failed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        save_task_status(task_id, status_data, ttl=3600)

        TRAINING_STATUS.set(0)
        logger.error(f"Training failed for {task_id}: {str(e)}")

async def run_blocking_fn(fn, *args):
    """Run a blocking function in the thread pool, preserving trace context."""
    loop = asyncio.get_event_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(executor, ctx.run, fn, *args)

async def run_training(task_id: str, fn, *args, chain_fn=None):
    """Start training in background and return immediately."""
    task_id = task_id.lower()

    current_status = get_task_status_redis(task_id)
    if current_status and current_status.get("status") == "running":
        return

    status_data = {"status": "running", "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    save_task_status(task_id, status_data, ttl=7200)

    TRAINING_STATUS.set(1)

    asyncio.create_task(run_training_worker(task_id, fn, *args, chain_fn=chain_fn))

