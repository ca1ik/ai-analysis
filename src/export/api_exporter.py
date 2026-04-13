"""
API Exporter — Packages chatbot as a standalone FastAPI server.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import EXPORTS_DIR


class APIExporter:
    """Exports chatbot as a standalone Python API package."""

    def export(self, bot_config: dict, include_model: bool = True) -> Path:
        bot_name = bot_config.get("name", "Social Good Bot")
        bot_id = bot_config["bot_id"]
        export_dir = EXPORTS_DIR / f"{bot_id}_api"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Generate standalone server
        server_py = f'''"""
Standalone Chatbot API — {bot_name}
Run: uvicorn server:app --host 0.0.0.0 --port 8080
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

app = FastAPI(title="{bot_name} API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """{bot_config.get("system_prompt", "")}"""
MODEL_PATH = "./model"  # Path to the fine-tuned model
TEMPERATURE = {bot_config.get("temperature", 0.7)}
MAX_HISTORY = 20

model = None
tokenizer = None
conversations: dict[str, list] = {{}}


@app.on_event("startup")
async def startup():
    global model, tokenizer
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    print(f"[{{"{bot_name}"}}] Model loaded and ready")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.session_id not in conversations:
        conversations[req.session_id] = []

    history = conversations[req.session_id]
    messages = [{{"role": "system", "content": SYSTEM_PROMPT}}]
    messages.extend(history[-MAX_HISTORY:])
    messages.append({{"role": "user", "content": req.message}})

    ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = ids.to(device)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=512,
            temperature=TEMPERATURE,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    resp = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

    history.append({{"role": "user", "content": req.message}})
    history.append({{"role": "assistant", "content": resp}})

    return ChatResponse(response=resp, session_id=req.session_id)


@app.delete("/session/{{session_id}}")
async def clear_session(session_id: str):
    conversations.pop(session_id, None)
    return {{"message": "Session cleared"}}


@app.get("/health")
async def health():
    return {{"status": "healthy", "name": "{bot_name}", "model_loaded": model is not None}}
'''

        requirements_txt = """fastapi==0.115.0
uvicorn[standard]==0.30.0
transformers>=4.44.0
torch>=2.1.0
accelerate>=0.33.0
bitsandbytes>=0.43.0
pydantic>=2.0.0
"""

        readme = f"""# {bot_name} — Standalone API

## Quick Start

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8080
```

## Endpoints

- `POST /chat` — Send message, get response
- `DELETE /session/{{session_id}}` — Clear session
- `GET /health` — Health check
- `GET /docs` — API documentation

## Chat Example

```bash
curl -X POST http://localhost:8080/chat \\
  -H "Content-Type: application/json" \\
  -d '{{"message": "Merhaba!", "session_id": "user1"}}'
```

## Social Good Mission
{bot_config.get("social_goal", "Bu bot topluma fayda sağlamak amacıyla geliştirilmiştir.")}
"""

        # Write files
        (export_dir / "server.py").write_text(server_py, encoding="utf-8")
        (export_dir / "requirements.txt").write_text(requirements_txt, encoding="utf-8")
        (export_dir / "README.md").write_text(readme, encoding="utf-8")

        # Copy model if available
        if include_model and bot_config.get("model_path"):
            model_src = Path(bot_config["model_path"])
            if model_src.exists():
                model_dest = export_dir / "model"
                if model_dest.exists():
                    shutil.rmtree(model_dest)
                shutil.copytree(model_src, model_dest)

        # Export metadata
        meta = {
            "bot_id": bot_id,
            "name": bot_name,
            "format": "api",
            "exported_at": datetime.now().isoformat(),
            "path": str(export_dir),
        }
        with open(export_dir / "export_meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return export_dir
