"""
Docker Exporter — Packages chatbot as a standalone Docker container.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import EXPORTS_DIR, MODELS_DIR


class DockerExporter:
    """Exports a chatbot as a Docker container with embedded API."""

    def export(self, bot_config: dict, include_model: bool = True) -> Path:
        bot_name = bot_config["bot_id"]
        export_dir = EXPORTS_DIR / f"{bot_name}_docker"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Generate Dockerfile
        dockerfile_content = f"""FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
"""

        # Generate lightweight inference server
        server_content = f'''"""Auto-generated chatbot server — {bot_config.get("name", "Social Good Bot")}"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, json

app = FastAPI(title="{bot_config.get("name", "Chatbot")}")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SYSTEM_PROMPT = """{bot_config.get("system_prompt", "")}"""
MODEL_PATH = "./model"
TEMPERATURE = {bot_config.get("temperature", 0.7)}

model = None
tokenizer = None
history = []

@app.on_event("startup")
async def load():
    global model, tokenizer
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
                                                  device_map="auto", torch_dtype=torch.float16)
    model.eval()

class ChatReq(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatReq):
    messages = [{{"role": "system", "content": SYSTEM_PROMPT}}]
    messages.extend(history[-20:])
    messages.append({{"role": "user", "content": req.message}})
    ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=512, temperature=TEMPERATURE,
                             top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    resp = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    history.append({{"role": "user", "content": req.message}})
    history.append({{"role": "assistant", "content": resp}})
    return {{"response": resp}}

@app.get("/health")
async def health():
    return {{"status": "healthy", "name": "{bot_config.get("name", "Bot")}"}}
'''

        requirements_content = """fastapi==0.115.0
uvicorn[standard]==0.30.0
transformers>=4.44.0
torch>=2.1.0
accelerate>=0.33.0
bitsandbytes>=0.43.0
pydantic>=2.0.0
"""

        # Write files
        (export_dir / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")
        (export_dir / "server.py").write_text(server_content, encoding="utf-8")
        (export_dir / "requirements.txt").write_text(requirements_content, encoding="utf-8")

        # Copy model if exists and requested
        if include_model and bot_config.get("model_path"):
            model_src = Path(bot_config["model_path"])
            if model_src.exists():
                model_dest = export_dir / "model"
                if model_dest.exists():
                    shutil.rmtree(model_dest)
                shutil.copytree(model_src, model_dest)

        # Export metadata
        meta = {
            "bot_id": bot_config["bot_id"],
            "name": bot_config.get("name"),
            "format": "docker",
            "exported_at": datetime.now().isoformat(),
            "path": str(export_dir),
        }
        with open(export_dir / "export_meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return export_dir
