from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from src.config.pipeline_config import Config
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

@app.post("/train-parent")
def train_parent_model():
    """Start parent model training"""
    task_id = "train_parent_task" # later implement task_id generation 
    
    if check_model_exists(model_type="parent"):
        return {"status": "completed", "task_id": task_id, "detail": "Parent model already exists"}

    if get_task_status(task_id) and get_task_status(task_id).get("status") == "running":
        return {"status": "already running", "task_id": task_id}

    await run_training(task_id, train_parent)
    return {"status": "started", "task_id": task_id}

@app.post("/train-child")
def train_child_model():
    """Start child model training"""
    pass
    
