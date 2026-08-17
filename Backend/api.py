from Backend.state import PREDICTION_LATENCY
from Backend.state import PREDICTION_COUNTER
from Backend.state import AGENT_ANALYSIS_LATENCY, AGENT_ANALYSIS_TOTAL, AGENT_ANALYSIS_ERRORS
from fastapi import APIRouter, HTTPException, Request, Response
import datetime
import json
import time
from src.agents.graph import analyze_stock  
import os
from src.config.pipeline_config import Config
from Backend.schema import AnalyzeRequest
from Backend.tasks import (
    run_training, run_blocking_fn,
    get_or_set_cache, get_task_status_redis,
)
from src.monitoring.drift_check import check_drift
from src.agents.memory import SemanticCache

from Backend.rate_limiter import rate_limit_decorator

from src.pipeline.training_pipeline import train_child, train_parent
from src.pipeline.inference_pipeline import predict_child, predict_parent
from src.model.provisioning import (
    ModelProvisioner, ProvisionState, PARENT_TASK_ID, child_task_id, CHILD_TASK_PREFIX
)

BASE_PATH = "outputs"

from logger.logger import get_logger

logger = get_logger()
router = APIRouter()
config = Config()


def _prime_prediction_cache(ticker: str):
    get_or_set_cache(
        f"predict_child_{ticker.lower()}",
        lambda: predict_child(ticker),
        expire=86400,
    )


provisioner = ModelProvisioner(
    config=config,
    task_status=get_task_status_redis,
    start_task=run_training,
    train_parent=train_parent,
    train_child=train_child,
    prime_child=_prime_prediction_cache,
)

@router.get("/")
def home():
    """Returns information and commands available"""
    return {"message": "Welcome to the Alpha Stream API"} # implement later

@router.get("/health")
def health():
    """Returns health status of the system"""
    return {"status": "healthy"}

@router.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.ticker:
        raise HTTPException(400, "ticker required")

    ticker = req.ticker.upper()
    AGENT_ANALYSIS_TOTAL.labels(ticker=ticker).inc()
    start_time = time.time()
    try:
        result = analyze_stock(req.ticker, thread_id= req.thread_id)
        AGENT_ANALYSIS_LATENCY.labels(ticker=ticker).observe(time.time() - start_time)
        return result
    except Exception as e:
        AGENT_ANALYSIS_ERRORS.labels(ticker=ticker).inc()
        AGENT_ANALYSIS_LATENCY.labels(ticker=ticker).observe(time.time() - start_time)
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    

# Training Endpoints

@router.post("/train-parent")
@rate_limit_decorator(limit=3, window_sec=60, key_prefix="train_parent")
async def train_parent_model():
    """Start parent model training"""
    task_id = PARENT_TASK_ID

    result = await provisioner.ensure(config.parent_ticker, "parent")
    if result.state == ProvisionState.PROVISIONED:
        return {"status": "completed", "task_id": task_id, "detail": "Parent model already exists"}
    if result.state == ProvisionState.TRAINING:
        return {"status": "already running", "task_id": task_id}
    return {"status": "started", "task_id": task_id}

@router.post("/train-child")
@rate_limit_decorator(limit=3, window_sec=60, key_prefix="train_child")
async def train_child_model(request: Request):
    """Start child model training"""
    data = await request.json()

    ticker = data.get("ticker", "").strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    task_id = child_task_id(ticker)
    result = await provisioner.ensure(ticker, "child")

    if result.state == ProvisionState.PROVISIONED:
        return {"status": "completed", "task_id": task_id, "detail": "Child model already exists"}
    if result.state == ProvisionState.TRAINING and result.task_id == PARENT_TASK_ID:
        return {"status": "running_parent", "task_id": PARENT_TASK_ID, "detail": "Parent model is currently training"}
    if result.state == ProvisionState.TRAINING:
        return {"status": "running", "task_id": task_id, "detail": "Training already in progress"}
    if result.task_id == PARENT_TASK_ID:
        return {"status": "running_parent", "task_id": PARENT_TASK_ID, "detail": "Parent model is currently training"}
    return {"status": "started", "task_id": task_id}
    
# Prediciton Endpoints

@router.post("/predict-parent")
@rate_limit_decorator(limit=3, window_sec=60, key_prefix="predict_parent")
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
@rate_limit_decorator(limit=3, window_sec=60, key_prefix="predict_child")
async def predict_child_endpoint(request: Request, response: Response):
    """Get child predictions"""
    data = await request.json()
    ticker = data.get("ticker", "").strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    PREDICTION_COUNTER.labels(type="child").inc()
    start_time = time.time()

    result = await provisioner.ensure(ticker, "child")
    if result.state != ProvisionState.PROVISIONED:
        response.status_code = 202
        if result.task_id == PARENT_TASK_ID:
            detail = "Parent model missing. Training parent first."
        else:
            detail = "Training in progress. Please retry later."
        return {"status": "training", "task_id": result.task_id, "detail": detail}

    try:
        def get_preds():
            return get_or_set_cache(f"predict_child_{ticker.lower()}", lambda: predict_child(ticker), expire=86400)

        preds, _ = await run_blocking_fn(get_preds)
        PREDICTION_LATENCY.labels(type="child").observe(time.time() - start_time)
        return {"result": preds}
    except Exception as e:
        logger.error(f"Child prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# System monitoring Endpoints

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Check the status of the task"""
    task = task_id.lower()
    if task == "parent":
        task = PARENT_TASK_ID

    status = get_task_status_redis(task)

    if task == PARENT_TASK_ID:
        ticker_for_disk, model_type = config.parent_ticker, "parent"
    elif task.startswith(CHILD_TASK_PREFIX):
        ticker_for_disk, model_type = task[len(CHILD_TASK_PREFIX):].upper(), "child"
    else:
        ticker_for_disk, model_type = task.upper(), "child"
    file_exists = provisioner.is_provisioned(ticker_for_disk, model_type)

    if not status:
        if file_exists:
            return {"status": "completed", "detail": "Model already exists", "task_id": task_id}
        raise HTTPException(404, f"Model {ticker_for_disk} not found")

    if status.get("status") not in ["running", "failed"] and file_exists:
        status["status"] = "completed"

    response = status.copy()
    if response.get("status") == "running" and "start_time" in response:
        try:
            start_dt = datetime.datetime.strptime(response["start_time"], "%Y-%m-%d %H:%M:%S")
            response["elapsed_seconds"] = int((datetime.datetime.now() - start_dt).total_seconds())
        except Exception:
            pass

    response.pop("start_time", None)
    return response

@router.get("/system/logs")
async def get_system_logs(lines: int = 100):
    """Get system logs"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        return {"detail":"log directory not found"}

    files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    if not files:
        return {"detail":"no log files found"}

    latest_file = sorted(files)[-1]
    path = os.path.join(log_dir, latest_file)
    
    try:
        with open(path, "r") as f:
            content = f.readlines()
            last_lines = content[-lines:]
            return {"logs": "".join(last_lines), "filename": latest_file}
    except Exception as e:
        return {"error": f"Failed to read logs: {e}"}

@router.post("/system/reset")
async def reset_system():
    """Resets all the cache from Qdrant and Redis"""
    results = {}
    
    # 1. Reset Redis
    try:
        from Backend.state import Redis_client
        if Redis_client:
            Redis_client.flushdb()
            results["redis"] = "success (flushed db)"
        else:
            results["redis"] = "skipped (no client)"
    except Exception as e:
        logger.error(f"Failed to reset Redis: {e}")
        results["redis"] = f"failed: {str(e)}"

    # 2. Reset Qdrant
    try:
        # Initialize SemanticCache - it will use default collection name "dataset_cache"
        cache = SemanticCache()
        cache.clear_cache(delete_collection=True)
        results["qdrant"] = "success (recreated collection)"
    except Exception as e:
        logger.error(f"Failed to reset Qdrant: {e}")
        results["qdrant"] = f"failed: {str(e)}"

    return {
        "status": "reset complete",
        "timestamp": datetime.datetime.now().isoformat(),
        "results": results
    }

@router.post("/monitor/parent")
async def monitor_parent():
    """Monitor parent model"""

    config = Config()
    ticker = config.parent_ticker

    ### Implement Later
    
    try:
        drift_result = check_drift(ticker, BASE_PATH) # implement this function
    except Exception as e:
        drift_result = {"status": "failed", "error": str(e)}
    
    ### Implement Later
    """
    try:
        evaluator = AgentEvaluator(BASE_PATH) # implement this function
        eval_res = evaluator.evaluate_live(ticker)
    except Exception as e:
        eval_res = {"status": "failed", "error": str(e)}"""
    
    return {
        "ticker": ticker,
        "type": "Parent Model (Market Index)",
        "drift": drift_result,
        "agent_eval": "not implemented yet",
        "links": {
            "get_drift_json": f"/monitor/{ticker}/drift",
            "get_eval_json": f"/monitor/{ticker}/eval"
        }
    }

@router.post("/monitor/{ticker}")
async def trigger_monitoring(ticker: str):
    """Trigger live monitoring for a ticker."""
    cfg = Config()
    clean_ticker = ticker.strip().upper()
    is_parent = (clean_ticker == cfg.parent_ticker)
    
    try:
        drift_res = check_drift(clean_ticker, BASE_PATH)
    except Exception as e:
        drift_res = {"status": "failed", "error": str(e)}
    """ 
    try:
        #evaluator = AgentEvaluator(BASE_PATH)
        eval_res = evaluator.evaluate_live(clean_ticker)
    except Exception as e:
        #eval_res = {"status": "failed", "error": str(e)}"""
            
    return {
        "ticker": clean_ticker,
        "is_parent": is_parent,
        "drift": drift_res,
        "agent_eval": "not implemented yet"
    }

@router.get("/monitor/{ticker}/drift")
def get_drift_result(ticker: str):
    t = ticker.lower()
    drift_dir = os.path.join(BASE_PATH, t, "drift")
    if not os.path.exists(drift_dir):
        raise HTTPException(404, detail=f"No drift report for {ticker} found")

    json_path = os.path.join(drift_dir, "latest_drift.json")
    if not os.path.exists(json_path):
        raise HTTPException(404, detail=f"No drift report for {ticker} found")

    with open(json_path, "r") as f:
        data = json.load(f)

    return {
        "ticker": ticker,
        "data": data,
        "history": os.listdir(drift_dir)
    }