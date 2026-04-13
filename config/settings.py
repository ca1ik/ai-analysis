"""
Global Settings — Social Good Chatbot Platform
RTX 5070 optimized configuration
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
TRAINING_SETS_DIR = DATA_DIR / "training_sets"
EXPORTS_DIR = BASE_DIR / "exports"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, TRAINING_SETS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    # Switch models here — Qwen needs no token, LLaMA needs HF_TOKEN env var
    # base_model: str = "Qwen/Qwen2.5-7B-Instruct"      # Open, no token
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # Needs HF_TOKEN
    hf_token: Optional[str] = os.environ.get("HF_TOKEN")  # NEVER hardcode
    quantization: str = "4bit"  # 4bit QLoRA for RTX 5070 12GB
    max_seq_length: int = 2048
    dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class TrainingConfig:
    # QLoRA Parameters — RTX 5070 12GB optimized
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training hyperparameters — 3-hour session optimized
    num_epochs: int = 15
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "cosine"

    # Memory optimization
    fp16: bool = True
    bf16: bool = False  # RTX 5070 supports bf16 but fp16 more stable for QLoRA
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"

    # Output
    output_dir: str = str(MODELS_DIR)
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


@dataclass
class SocialGuardConfig:
    """Guardrails for social good compliance"""
    blocked_categories: list = field(default_factory=lambda: [
        "hate_speech", "violence", "discrimination",
        "misinformation", "exploitation", "self_harm"
    ])
    required_disclaimers: dict = field(default_factory=lambda: {
        "medical": "Bu bilgi tıbbi tavsiye yerine geçmez. Lütfen bir sağlık profesyoneline danışın.",
        "legal": "Bu bilgi hukuki tavsiye niteliğinde değildir. Bir avukata danışmanız önerilir.",
        "crisis": "Acil durumdaysanız 112'yi arayın. Destek hattı: 182 (ALO 182)"
    })
    max_response_length: int = 1024
    enable_bias_detection: bool = True
    enable_fact_checking: bool = True
    supported_languages: list = field(default_factory=lambda: ["tr", "en"])


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = os.environ.get("PLATFORM_API_KEY")


# Singleton instances
model_config = ModelConfig()
training_config = TrainingConfig()
social_guard_config = SocialGuardConfig()
api_config = APIConfig()
