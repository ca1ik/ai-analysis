"""
Training Routes — Dataset upload, training execution, model management.
"""
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config.settings import TRAINING_SETS_DIR, MODELS_DIR
from src.training.pipeline import TrainingPipeline, PipelineConfig

router = APIRouter()
pipeline = TrainingPipeline()

# In-memory training status tracker
_training_status: dict[str, dict] = {}


class TrainRequest(BaseModel):
    bot_name: str
    system_prompt: str
    description: str = ""
    target_audience: str = ""
    social_goal: str = ""
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    dataset_name: Optional[str] = None  # Use existing uploaded dataset


class TrainFromTemplateRequest(BaseModel):
    bot_name: str
    template: str  # e.g., "social_good_base_tr" or "social_good_base_en"
    system_prompt: str = ""
    social_goal: str = ""
    num_epochs: int = 3


@router.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
):
    """Upload a training dataset (JSONL, JSON, CSV)."""
    allowed_extensions = {".jsonl", ".json", ".csv"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(400, f"Unsupported format: {suffix}. Use: {allowed_extensions}")

    save_path = TRAINING_SETS_DIR / f"{name}{suffix}"
    with open(save_path, "wb") as f:
        content = await file.read()
        # Basic size limit: 500MB
        if len(content) > 500 * 1024 * 1024:
            raise HTTPException(400, "Dataset too large (max 500MB)")
        f.write(content)

    return {"message": f"Dataset uploaded: {name}", "path": str(save_path), "size_mb": len(content) / (1024*1024)}


@router.get("/datasets")
async def list_datasets():
    """List available training datasets."""
    datasets = []
    for f in TRAINING_SETS_DIR.iterdir():
        if f.is_file() and f.suffix in {".jsonl", ".json", ".csv"}:
            datasets.append({
                "name": f.stem,
                "format": f.suffix,
                "size_mb": round(f.stat().st_size / (1024*1024), 2),
            })
    return {"datasets": datasets}


def _run_training(config: PipelineConfig, job_id: str):
    """Background training task."""
    def progress_fn(update):
        _training_status[job_id].update(update)

    _training_status[job_id] = {"status": "running", "progress": 0}

    result = pipeline.run(config, progress_fn=progress_fn)

    _training_status[job_id] = {
        "status": "completed" if result.success else "failed",
        "progress": 1.0 if result.success else 0,
        "model_path": result.model_path,
        "error": result.error,
        "training_loss": result.training_result.training_loss if result.training_result else None,
    }


@router.post("/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Start a training job."""
    # Find dataset
    dataset_path = None
    if req.dataset_name:
        for ext in [".jsonl", ".json", ".csv"]:
            candidate = TRAINING_SETS_DIR / f"{req.dataset_name}{ext}"
            if candidate.exists():
                dataset_path = str(candidate)
                break
    if not dataset_path:
        raise HTTPException(404, f"Dataset '{req.dataset_name}' not found. Upload one first.")

    import uuid
    job_id = f"train_{uuid.uuid4().hex[:8]}"

    config = PipelineConfig(
        dataset_path=dataset_path,
        bot_name=req.bot_name,
        system_prompt=req.system_prompt,
        description=req.description,
        target_audience=req.target_audience,
        social_goal=req.social_goal,
        num_epochs=req.num_epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
    )

    background_tasks.add_task(_run_training, config, job_id)
    return {"job_id": job_id, "status": "started", "bot_name": req.bot_name}


@router.post("/train-from-template")
async def train_from_template(req: TrainFromTemplateRequest, background_tasks: BackgroundTasks):
    """Train using built-in TR/EN social good templates."""
    template_path = TRAINING_SETS_DIR / f"{req.template}.jsonl"
    if not template_path.exists():
        available = [f.stem for f in TRAINING_SETS_DIR.glob("social_good_base_*.jsonl")]
        raise HTTPException(404, f"Template '{req.template}' not found. Available: {available}")

    import uuid
    job_id = f"train_{uuid.uuid4().hex[:8]}"

    config = PipelineConfig(
        dataset_path=str(template_path),
        bot_name=req.bot_name,
        system_prompt=req.system_prompt or f"Social good chatbot trained from {req.template}",
        social_goal=req.social_goal,
        num_epochs=req.num_epochs,
    )

    background_tasks.add_task(_run_training, config, job_id)
    return {"job_id": job_id, "status": "started", "template": req.template}


@router.get("/status/{job_id}")
async def training_status(job_id: str):
    """Check training job status."""
    if job_id not in _training_status:
        raise HTTPException(404, "Training job not found")
    return _training_status[job_id]


@router.get("/models")
async def list_models():
    """List all trained models."""
    return {"models": pipeline.list_trained_models()}


@router.get("/history")
async def training_history():
    """Get training history."""
    return {"history": pipeline.get_training_history()}
