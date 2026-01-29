import asyncio
import os
import redis
import uvicorn
from logger.logger import get_logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.exceptions import HTTPException
from src.utils import initialize_dirs
import Backend.state as app_state
from Backend.state import Redis_client, REDIS_STATUS, registry
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

@app.on_event("startup")
async def startup():
    initialize_dirs()
    
    # retry logic for redis
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    
    for i in range(10):
        try:
            client = redis.Redis(host=redis_host, port=redis_port, db=0)
            client.ping()
            app_state.Redis_client = client
            logger.info(f"BACKEND - System Online (FastAPI, Redis at {redis_host}:{redis_port}, Mlflow)")
            REDIS_STATUS.set(1)
            return
        except Exception as e:
            logger.warning(f"BACKEND - Waiting for Redis connection... attempting {i+1}/10. Error: {str(e)}")
            await asyncio.sleep(5)

    REDIS_STATUS.set(0)
    logger.error("BACKEND - Failed to connect to Redis")

if __name__=="__main__":
    uvicorn.run("Backend.main:app", host="0.0.0.0", port=8000, reload=True)

    