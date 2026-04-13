"""
System Routes — GPU status, health checks, platform info.
"""
from fastapi import APIRouter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.utils.gpu_manager import gpu_manager

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy"}


@router.get("/gpu")
async def gpu_status():
    return gpu_manager.get_status_report()


@router.post("/gpu/clear-cache")
async def clear_gpu_cache():
    gpu_manager.clear_cache()
    return {"message": "GPU cache cleared"}
