import asyncio
import redis
import uvicorn
from logger.logger import get_logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.exceptions import HTTPException
from backend.state import Redis_client, REDIS_STATUS

logger = get_logger()

app = FastAPI(title="Alpha Stream API", description="Backend for Alpha Stream", version="0.1.0")

app.middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    # initalize_dirs()    implement this later in utils
    
    # retry logic for redis
    for i in range(10):
        try:
            client = redis.Redis(host="redis", port=6379, db=0)
            client.ping()
            app_state.Redis_client = client
            logger.info("BACKEND - System Online (FastAPI, Redis, Mlflow)")
            REDIS_STATUS.set(1)
            return
        except Exception as e:
            logger.warning(f"BACKEND - Waiting for Redis connection... attempting {i+1}/10")
            await asyncio.sleep(5)

    REDIS_STATUS.set(0)
    logger.error("BACKEND - Failed to connect to Redis")

if __name__=="__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

    