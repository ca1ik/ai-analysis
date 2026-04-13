"""
Export Routes — Package chatbots for deployment (Docker, Widget, API, GGUF).
"""
import json
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config.settings import MODELS_DIR, EXPORTS_DIR
from src.generator.bot_factory import bot_factory
from src.export.docker_exporter import DockerExporter
from src.export.widget_exporter import WidgetExporter
from src.export.api_exporter import APIExporter

router = APIRouter()

docker_exporter = DockerExporter()
widget_exporter = WidgetExporter()
api_exporter = APIExporter()


class ExportRequest(BaseModel):
    bot_id: str
    format: str  # "docker", "widget", "api", "gguf"
    include_model: bool = True


@router.post("/package")
async def export_bot(req: ExportRequest, background_tasks: BackgroundTasks):
    """Export a chatbot as a deployable package."""
    bots = bot_factory.list_bots()
    bot = next((b for b in bots if b["bot_id"] == req.bot_id), None)
    if not bot:
        raise HTTPException(404, "Bot not found")

    exporters = {
        "docker": docker_exporter,
        "widget": widget_exporter,
        "api": api_exporter,
    }

    exporter = exporters.get(req.format)
    if not exporter:
        raise HTTPException(400, f"Unknown format: {req.format}. Options: {list(exporters.keys())}")

    try:
        output_path = exporter.export(bot, include_model=req.include_model)
        return {
            "message": f"Bot exported as {req.format}",
            "output_path": str(output_path),
            "bot_id": req.bot_id,
            "format": req.format,
        }
    except Exception as e:
        raise HTTPException(500, f"Export failed: {str(e)}")


@router.get("/formats")
async def list_formats():
    return {
        "formats": [
            {"id": "docker", "name": "Docker Container", "description": "Standalone Docker container with API"},
            {"id": "widget", "name": "Web Widget", "description": "Embeddable HTML/JS chat widget"},
            {"id": "api", "name": "Standalone API", "description": "FastAPI server package"},
        ]
    }


@router.get("/list")
async def list_exports():
    """List all exported packages."""
    exports = []
    if EXPORTS_DIR.exists():
        for d in EXPORTS_DIR.iterdir():
            if d.is_dir():
                meta_path = d / "export_meta.json"
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        exports.append(json.load(f))
    return {"exports": exports}
