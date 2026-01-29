from cProfile import label
from Backend.state import PREDICTION_LATENCY
from Backend.state import PREDICTION_COUNTER
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from src.exception.exceptions import PrismException
import sys
from src.config.pipeline_config import Config
from pathlib import Path
from Backend.tasks import (
    run_training, run_blocking_fn, save_task_status,
    get_or_set_cache, get_task_key, get_task_status_redis,
    delete_task_status
)

from src.pipeline.training_pipeline import train_child, train_parent
from src.pipeline.inference_pipeline import predict_child, predict_parent
from src.utils import check_model_exists

from logger.logger import get_logger

logger = get_logger()
router = APIRouter()
config = Config()

@router.get("/")
def home():
    """Returns information and commands available"""
    return {"message": "Welcome to the Alpha Stream API"} # implement later

@router.get("/health")
def health():
    """Returns health status of the system"""
    return {"status": "healthy"}

@router.post("/analyze")
def analyze():
    pass # endpoint to call AI agent, to be implemented later

# Training Endpoints

@router.post("/train-parent")
async def train_parent_model():
    """Start parent model training"""
    task_id = "parent_training" # later implement task_id generation 
    
    if check_model_exists(model_type="parent"):
        return {"status": "completed", "task_id": task_id, "detail": "Parent model already exists"}

    if get_task_status_redis(task_id) and get_task_status_redis(task_id).get("status") == "running":
        return {"status": "already running", "task_id": task_id}

    await run_training(task_id, train_parent)
    return {"status": "started", "task_id": task_id}

@router.post("/train-child")
async def train_child_model(request: Request):
    """Start child model training"""
    data = await request.json()

    ticker = data.get("ticker", "").strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    task_id = f"train_child_{ticker}"

    parent_path = Path(config.parent_dir) / f"{config.parent_ticker}_parent_model.pt"

    if not parent_path.exists():
        logger.warning("Parent Model Missing, initiating parent training")
        parent_status = get_task_status_redis("parent_training")
        if not parent_status or parent_status.get("status") != "completed":
            await run_training("parent_training", train_parent)
            parent_status = get_task_status_redis("parent_training")
            if parent_status and parent_status.get("status") == "running":
                return {"status": "running_parent", "task_id": "parent_training", "detail": "Parent model is currently training"}

    if check_model_exists(ticker, "child"):
        return {"status": "completed", "task_id": task_id, "detail": "Child model already exists"}

    curr_status = get_task_status_redis(task_id)
    if curr_status and curr_status.get("status") == "running":
        return {"status": "running", "task_id": task_id, "detail": "Training already in progress"}

    def chain_predict():
        logger.info(f"Auto-predicting for {ticker} after training...")
        get_or_set_cache(f"predict_child_{ticker.lower()}", lambda: predict_child(ticker), expire=86400)

    await run_training(task_id, train_child, ticker, chain_fn=chain_predict)
    return {"status": "started", "task_id": task_id}
    
# Prediciton Endpoints

@router.post("/predict-parent")
async def predict_parent_endpoint():
    """Get parent predictions"""
    PREDICTION_COUNTER.labels(type="parent").inc()
    start_time = time.time()
    try:
        result = await run_blocking_fn(predict_parent)
        PREDICTION_LATENCY.labels(type="parent").observe(time.time() - start_time)
        return {"result": result}
    except Exception as e:
        logger.error(f"Parent prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-child")
async def predict_child_endpoint(request: Request):
    """Get child predictions"""
    data = await request.json()
    ticker = data.get("ticker", "").strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
    
    task_id = ticker.lower()
    PREDICTION_COUNTER.labels(type="child").inc()
    start_time = time.time()
    try:
        def get_preds():
            return get_or_set_cache(f"predict_child_{ticker.lower()}", lambda: predict_child(ticker), expire=86400)

        preds, _ = await run_blocking_fn(get_preds)
        PREDICTION_LATENCY.labels(type="child").observe(time.time() - start_time)
        return {"result": preds}
    except (FileNotFoundError, PrismException) as e:
        if "Missing" in str(e) or "not found" in str(e):
            logger.info(f"Model missing for {ticker}, triggering auto-training.")

            # Check if Parent Model exists
            if not check_model_exists("parent", "parent"):
                logger.warning("Parent model missing. Triggering parent training first.")
                parent_status = get_task_status_redis("parent_training")
                if not parent_status or parent_status.get("status") != "completed":
                    await run_training("parent_training", train_parent)
                    response.status_code = 202
                    return {"status": "training", "detail": "Parent model missing. Training parent first.", "task_id": "parent_training"}

            status = get_task_status_redis(task_id)
            if status and status.get("status") == "running":
                 response.status_code = 202
                 return {"status": "training", "detail": "Training in progress. Please retry later.", "task_id": task_id}

            def chain_predict():
                # Chain prediction and caching after training
                logger.info(f"Auto-predicting for {ticker} after auto-training...")
                get_or_set_cache(f"predict_child_{ticker.lower()}", lambda: predict_child(ticker), expire=86400)

            await run_training(task_id, train_child, ticker, chain_fn=chain_predict)
            response.status_code = 202
            return {"status": "training", "detail": "Model missing. Training child model...", "task_id": task_id}
        
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
