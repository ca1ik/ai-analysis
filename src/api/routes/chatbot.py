"""
Chatbot Management Routes — Create, list, configure, delete bots.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.generator.bot_factory import bot_factory, BotSpec

router = APIRouter()


class CreateBotRequest(BaseModel):
    name: str
    template: str = "custom"
    system_prompt: str = ""
    description: str = ""
    target_audience: str = ""
    social_goal: str = ""
    language: str = "tr"
    temperature: float = 0.7
    model_path: Optional[str] = None


class UpdateBotRequest(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    description: Optional[str] = None
    target_audience: Optional[str] = None
    social_goal: Optional[str] = None
    temperature: Optional[float] = None


@router.post("/create")
async def create_bot(req: CreateBotRequest):
    """Create a new chatbot from template or custom config."""
    spec = BotSpec(
        name=req.name,
        template=req.template,
        system_prompt=req.system_prompt,
        description=req.description,
        target_audience=req.target_audience,
        social_goal=req.social_goal,
        language=req.language,
        temperature=req.temperature,
        model_path=req.model_path,
    )
    try:
        bot = bot_factory.create_bot(spec)
        return {
            "bot_id": bot.bot_id,
            "name": bot.name,
            "persona": bot.persona,
            "message": f"Bot '{bot.name}' created successfully",
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/list")
async def list_bots():
    """List all created bots."""
    return {"bots": bot_factory.list_bots()}


@router.get("/templates")
async def list_templates():
    """List available chatbot templates."""
    return {"templates": bot_factory.list_templates()}


@router.get("/{bot_id}")
async def get_bot(bot_id: str):
    """Get bot details."""
    bots = bot_factory.list_bots()
    bot = next((b for b in bots if b["bot_id"] == bot_id), None)
    if not bot:
        raise HTTPException(404, "Bot not found")
    return bot


@router.put("/{bot_id}")
async def update_bot(bot_id: str, req: UpdateBotRequest):
    """Update bot configuration."""
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not bot_factory.update_bot(bot_id, updates):
        raise HTTPException(404, "Bot not found")
    return {"message": "Bot updated", "bot_id": bot_id}


@router.delete("/{bot_id}")
async def delete_bot(bot_id: str):
    """Delete a bot."""
    if not bot_factory.delete_bot(bot_id):
        raise HTTPException(404, "Bot not found")
    return {"message": "Bot deleted", "bot_id": bot_id}
