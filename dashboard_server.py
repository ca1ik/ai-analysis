"""
AI Training Dashboard — Backend Server
Real-time training monitoring, control, GPU stats, model chat testing.
"""
import os, sys, json, time, subprocess, asyncio, threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

os.environ["PYTHONUTF8"] = "1"

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="AI Training Dashboard", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Mount static files
STATIC_DIR = ROOT / "dashboard" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── State ────────────────────────────────────────────────
training_process: Optional[subprocess.Popen] = None
training_status = {
    "running": False,
    "pid": None,
    "started_at": None,
    "current_step": 0,
    "total_steps": 1701,
    "current_loss": None,
    "best_loss": None,
    "chunks_completed": 0,
    "errors": 0,
}

LOG_FILE = ROOT / "training_log.txt"
MODELS_DIR = ROOT / "data" / "models" / "social_good_v1"


# ─── Pydantic Models ─────────────────────────────────────
class TrainingRequest(BaseModel):
    steps_per_chunk: int = 50
    total_steps: int = 1701
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 5e-5
    max_seq_length: int = 1024
    gpu_power_limit: int = 175


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7


# ─── Helper Functions ─────────────────────────────────────
def get_gpu_stats() -> Dict[str, Any]:
    """Get GPU stats via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,power.draw,power.limit,"
             "memory.used,memory.total,utilization.gpu,fan.speed,clocks.gr",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "name": parts[0] if len(parts) > 0 else "N/A",
                "temperature": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
                "power_draw": float(parts[2]) if len(parts) > 2 else 0,
                "power_limit": float(parts[3]) if len(parts) > 3 else 250,
                "memory_used": int(parts[4]) if len(parts) > 4 else 0,
                "memory_total": int(parts[5]) if len(parts) > 5 else 12227,
                "utilization": int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else 0,
                "fan_speed": parts[7] if len(parts) > 7 else "N/A",
                "clock_speed": int(parts[8]) if len(parts) > 8 and parts[8].isdigit() else 0,
            }
    except Exception:
        pass
    return {"name": "N/A", "temperature": 0, "power_draw": 0, "power_limit": 250,
            "memory_used": 0, "memory_total": 12227, "utilization": 0,
            "fan_speed": "N/A", "clock_speed": 0}


def get_checkpoints() -> List[Dict]:
    """List all checkpoints with metadata."""
    checkpoints = []
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir() and d.name.startswith("checkpoint-"):
                try:
                    step = int(d.name.split("-")[1])
                    size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)
                    mtime = datetime.fromtimestamp(d.stat().st_mtime)
                    state_file = d / "trainer_state.json"
                    loss = None
                    if state_file.exists():
                        state = json.loads(state_file.read_text(encoding="utf-8"))
                        if state.get("log_history"):
                            loss = state["log_history"][-1].get("loss")
                    checkpoints.append({
                        "name": d.name,
                        "step": step,
                        "size_mb": round(size_mb, 1),
                        "created": mtime.isoformat(),
                        "loss": loss,
                    })
                except (ValueError, OSError):
                    pass
    return sorted(checkpoints, key=lambda x: x["step"])


def get_training_logs(last_n: int = 200) -> List[str]:
    """Read last N lines from training log."""
    if not LOG_FILE.exists():
        return []
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").strip().split("\n")
        return lines[-last_n:]
    except Exception:
        return []


def parse_log_for_metrics() -> Dict[str, List]:
    """Parse training log for loss/step metrics over time."""
    steps, losses, timestamps = [], [], []
    if not LOG_FILE.exists():
        return {"steps": steps, "losses": losses, "timestamps": timestamps}
    try:
        for line in LOG_FILE.read_text(encoding="utf-8", errors="replace").split("\n"):
            line = line.strip()
            # Pattern: "[HH:MM:SS]   Chunk tamamlandi! Step: 250, Loss: 1.2345"
            if "Step:" in line and "Loss:" in line:
                try:
                    ts = line.split("]")[0].replace("[", "").strip()
                    step_part = line.split("Step:")[1].split(",")[0].strip()
                    loss_part = line.split("Loss:")[1].strip()
                    steps.append(int(step_part))
                    losses.append(float(loss_part))
                    timestamps.append(ts)
                except (ValueError, IndexError):
                    pass
            # HuggingFace trainer log pattern: {'loss': 1.23, 'step': 210, ...}
            if "'loss'" in line and "'step'" in line:
                try:
                    data = eval(line.split("]")[-1].strip())
                    if isinstance(data, dict):
                        steps.append(data["step"])
                        losses.append(data["loss"])
                        timestamps.append(line.split("]")[0].replace("[", "").strip())
                except Exception:
                    pass
    except Exception:
        pass
    return {"steps": steps, "losses": losses, "timestamps": timestamps}


def calculate_level(current_step: int, total_steps: int) -> Dict:
    """Calculate training level (1-99) based on progress."""
    if total_steps <= 0:
        progress = 0
    else:
        progress = min(current_step / total_steps, 1.0)

    level = max(1, min(99, int(progress * 99)))

    # Level tier names
    if level < 10:
        tier = "Novice"
        tier_color = "#888888"
    elif level < 25:
        tier = "Apprentice"
        tier_color = "#4ecdc4"
    elif level < 50:
        tier = "Specialist"
        tier_color = "#45b7d1"
    elif level < 75:
        tier = "Expert"
        tier_color = "#f9ca24"
    elif level < 90:
        tier = "Master"
        tier_color = "#f0932b"
    elif level < 99:
        tier = "Grandmaster"
        tier_color = "#e74c3c"
    else:
        tier = "Legendary"
        tier_color = "#e056fd"

    xp_in_level = (progress * 99) - int(progress * 99)

    return {
        "level": level,
        "tier": tier,
        "tier_color": tier_color,
        "progress": round(progress * 100, 2),
        "xp_progress": round(xp_in_level * 100, 1),
        "current_step": current_step,
        "total_steps": total_steps,
    }


def get_latest_step() -> int:
    """Get the highest completed step from checkpoints."""
    if not MODELS_DIR.exists():
        return 0
    max_step = 0
    for d in MODELS_DIR.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                max_step = max(max_step, step)
            except ValueError:
                pass
    return max_step


# ─── API Routes ───────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard."""
    html_path = ROOT / "dashboard" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found</h1>")


@app.get("/api/status")
async def get_status():
    """Full status: training, GPU, level, checkpoints."""
    current_step = get_latest_step()
    gpu = get_gpu_stats()
    level = calculate_level(current_step, training_status["total_steps"])
    checkpoints = get_checkpoints()

    # Check if training process is still running
    is_running = False
    if training_process and training_process.poll() is None:
        is_running = True
    else:
        # Check for cmd.exe running run_training.bat
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                capture_output=True, text=True, timeout=5
            )
            if "train_chunk" in result.stdout.lower():
                is_running = True
        except Exception:
            pass

    training_status["running"] = is_running
    training_status["current_step"] = current_step

    return {
        "training": {**training_status, "current_step": current_step},
        "gpu": gpu,
        "level": level,
        "checkpoints": checkpoints,
        "logs": get_training_logs(50),
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get training metrics for charts."""
    return parse_log_for_metrics()


@app.get("/api/gpu")
async def gpu_stats():
    """Get current GPU stats."""
    return get_gpu_stats()


@app.get("/api/gpu/stream")
async def gpu_stream():
    """SSE stream for real-time GPU monitoring."""
    async def event_generator():
        while True:
            data = json.dumps(get_gpu_stats())
            yield f"data: {data}\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/level")
async def get_level():
    """Get current training level."""
    current_step = get_latest_step()
    return calculate_level(current_step, training_status["total_steps"])


@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """Get training logs."""
    return {"logs": get_training_logs(lines)}


@app.get("/api/checkpoints")
async def list_checkpoints():
    """List all training checkpoints."""
    return {"checkpoints": get_checkpoints()}


@app.get("/api/infrastructure")
async def get_infrastructure():
    """Get infrastructure info for sidebar."""
    gpu = get_gpu_stats()
    import platform
    try:
        import torch
        torch_ver = torch.__version__
        cuda_ver = torch.version.cuda or "N/A"
    except ImportError:
        torch_ver = "N/A"
        cuda_ver = "N/A"
    try:
        import transformers
        tf_ver = transformers.__version__
    except ImportError:
        tf_ver = "N/A"
    try:
        import peft
        peft_ver = peft.__version__
    except ImportError:
        peft_ver = "N/A"
    try:
        import trl
        trl_ver = trl.__version__
    except ImportError:
        trl_ver = "N/A"

    return {
        "system": {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "cpu": platform.processor(),
        },
        "gpu": {
            "name": gpu["name"],
            "vram": f"{gpu['memory_total']} MiB",
            "driver": "595.71",
            "cuda": cuda_ver,
        },
        "ml_stack": {
            "pytorch": torch_ver,
            "transformers": tf_ver,
            "peft": peft_ver,
            "trl": trl_ver,
        },
        "model": {
            "base": "Qwen/Qwen2.5-7B-Instruct",
            "method": "QLoRA (4-bit NF4)",
            "lora_r": 64,
            "lora_alpha": 128,
            "target_modules": "q/k/v/o/gate/up/down_proj",
            "effective_batch": "1 × 8 = 8",
            "precision": "BFloat16",
            "max_seq_length": 1024,
            "optimizer": "Paged AdamW 32-bit",
            "scheduler": "Cosine",
        },
        "strategy": {
            "name": "Micro-Chunk Training",
            "description": "Her 50 step'te process yeniden başlatılır. GPU driver crash önlenir.",
            "steps_per_chunk": 50,
            "gpu_power_limit": "175W (70%)",
            "auto_resume": True,
            "max_retries": 3,
        }
    }


@app.post("/api/training/start")
async def start_training(req: TrainingRequest):
    """Start training via run_training.bat (detached from VS Code)."""
    global training_process

    # Check if already running
    if training_process and training_process.poll() is None:
        raise HTTPException(400, "Training already running")

    # Update total steps
    training_status["total_steps"] = req.total_steps

    # Set GPU power limit
    try:
        subprocess.run(
            ["nvidia-smi", "-pl", str(req.gpu_power_limit)],
            capture_output=True, timeout=5
        )
    except Exception:
        pass

    # Launch training in separate process
    bat_path = ROOT / "run_training.bat"
    training_process = subprocess.Popen(
        ["cmd.exe", "/c", str(bat_path)],
        cwd=str(ROOT),
        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    training_status["running"] = True
    training_status["pid"] = training_process.pid
    training_status["started_at"] = datetime.now().isoformat()

    return {"status": "started", "pid": training_process.pid}


@app.post("/api/training/stop")
async def stop_training():
    """Stop training gracefully."""
    global training_process
    stopped = False

    # Kill python training processes
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "python" in line.lower():
                parts = line.strip().strip('"').split('","')
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1].strip('"'))
                        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                                     capture_output=True, timeout=5)
                        stopped = True
                    except (ValueError, subprocess.SubprocessError):
                        pass
    except Exception:
        pass

    if training_process:
        try:
            training_process.kill()
            stopped = True
        except Exception:
            pass
        training_process = None

    training_status["running"] = False

    # Reset GPU power limit
    try:
        subprocess.run(["nvidia-smi", "-pl", "250"], capture_output=True, timeout=5)
    except Exception:
        pass

    return {"status": "stopped" if stopped else "not_running"}


@app.post("/api/chat")
async def chat_with_model(req: ChatRequest):
    """Chat with the trained model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        # Find best checkpoint
        checkpoints = get_checkpoints()
        if not checkpoints:
            raise HTTPException(404, "No trained model found. Train first!")

        best_ckpt = checkpoints[-1]  # Latest
        ckpt_path = str(MODELS_DIR / best_ckpt["name"])

        token = os.environ.get("HF_TOKEN")
        base_model = "Qwen/Qwen2.5-7B-Instruct"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16, token=token,
        )
        model = PeftModel.from_pretrained(model, ckpt_path)

        messages = [
            {"role": "system", "content": "Sen topluma yararlı konularda yardımcı olan bir AI asistansın. Türkçe ve İngilizce konuşabilirsin."},
            {"role": "user", "content": req.message}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Clean up
        del model, tokenizer, inputs, output
        torch.cuda.empty_cache()

        return {
            "response": response,
            "model": f"social_good_v1/{best_ckpt['name']}",
            "step": best_ckpt["step"],
        }

    except ImportError:
        raise HTTPException(503, "PyTorch/transformers not available")
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000, log_level="info")
