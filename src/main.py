"""
FastAPI Main Application — Social Good Chatbot Platform API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import api_config
from src.api.routes.training import router as training_router
from src.api.routes.chatbot import router as chatbot_router
from src.api.routes.chat import router as chat_router
from src.api.routes.export import router as export_router
from src.api.routes.system import router as system_router

app = FastAPI(
    title="Social Good Chatbot Platform",
    description="Build, train, and deploy chatbots for social impact",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router, prefix="/api/system", tags=["System"])
app.include_router(training_router, prefix="/api/training", tags=["Training"])
app.include_router(chatbot_router, prefix="/api/bots", tags=["Chatbot Management"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(export_router, prefix="/api/export", tags=["Export"])


@app.get("/")
async def root():
    return {
        "name": "Social Good Chatbot Platform",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }
