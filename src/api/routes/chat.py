"""
Chat Routes — Real-time chat interaction with bots.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.core.engine import engine
from src.core.social_guard import social_guard
from src.generator.bot_factory import bot_factory

router = APIRouter()


class ChatRequest(BaseModel):
    bot_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    disclaimer: str | None = None
    risk_level: str = "safe"


@router.post("/send", response_model=ChatResponse)
async def send_message(req: ChatRequest):
    """Send a message to a chatbot and get a response."""
    # Ensure bot is loaded
    bot = engine.get_bot(req.bot_id)
    if bot is None:
        bot = bot_factory.load_bot(req.bot_id)
        if bot is None:
            raise HTTPException(404, f"Bot '{req.bot_id}' not found")

    # Safety check on input
    input_check = social_guard.check_input(req.message)
    if not input_check.is_safe:
        return ChatResponse(
            response=f"Bu mesaj güvenlik filtresine takıldı: {input_check.reason}",
            risk_level=input_check.risk_level.value,
        )

    # Generate response
    try:
        response = engine.generate(req.bot_id, req.message)
    except Exception as e:
        raise HTTPException(500, f"Generation error: {str(e)}")

    # Safety check on output
    output_check = social_guard.check_output(response, req.message)
    if not output_check.is_safe:
        response = (
            "Üzgünüm, bu konuda güvenli bir yanıt üretemiyorum. "
            "Lütfen sorunuzu farklı şekilde sormayı deneyin."
        )
        return ChatResponse(response=response, risk_level=output_check.risk_level.value)

    final_response = output_check.modified_content or response
    return ChatResponse(
        response=final_response,
        disclaimer=output_check.disclaimer,
        risk_level=output_check.risk_level.value,
    )


@router.post("/stream")
async def stream_message(req: ChatRequest):
    """Stream a chatbot response in real-time."""
    bot = engine.get_bot(req.bot_id)
    if bot is None:
        bot = bot_factory.load_bot(req.bot_id)
        if bot is None:
            raise HTTPException(404, f"Bot '{req.bot_id}' not found")

    input_check = social_guard.check_input(req.message)
    if not input_check.is_safe:
        async def blocked_stream():
            yield f"data: {input_check.reason}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    async def generate_stream():
        try:
            for token in engine.generate_stream(req.bot_id, req.message):
                yield f"data: {token}\n\n"
            # Add disclaimer if crisis detected
            if input_check.disclaimer:
                yield f"data: \n\n---\n⚠️ {input_check.disclaimer}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@router.post("/{bot_id}/clear-history")
async def clear_history(bot_id: str):
    """Clear conversation history for a bot."""
    bot = engine.get_bot(bot_id)
    if bot is None:
        raise HTTPException(404, "Bot not found")
    bot.conversation_history.clear()
    return {"message": "Conversation history cleared"}


@router.get("/{bot_id}/history")
async def get_history(bot_id: str):
    """Get conversation history for a bot."""
    bot = engine.get_bot(bot_id)
    if bot is None:
        raise HTTPException(404, "Bot not found")
    return {
        "bot_id": bot_id,
        "history": [
            {"role": m.role, "content": m.content}
            for m in bot.conversation_history
        ]
    }
