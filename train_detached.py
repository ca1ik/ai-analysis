"""
Detached Training Script — VS Code'dan bağımsız çalışır.
- Windows Task Scheduler / Start-Process ile başlatılır
- Her chunk sonunda checkpoint kaydeder
- BSOD/crash sonrası tekrar çalıştırılınca kaldığı yerden devam eder
- Tüm loglar dosyaya yazılır (terminal bağımsız)
"""
import os
import sys
import json
import time
import math
import traceback
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUTF8"] = "1"

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "training.log"
LOCK_FILE = ROOT / "training.lock"

# ─── CONFIG ──────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = str(ROOT / "data" / "models" / "social_good_v1")

# QLoRA
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training — Conservative for stability
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 1024
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.3
TARGET_EPOCHS = 189
SAVE_EVERY_STEPS = 100  # Frequent saves for crash recovery


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_latest_checkpoint():
    """Find the highest-numbered checkpoint directory."""
    model_dir = Path(OUTPUT_DIR)
    if not model_dir.exists():
        return None, 0
    
    checkpoints = []
    for d in model_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                # Must have adapter_model to be valid
                if (d / "adapter_model.safetensors").exists():
                    checkpoints.append((step, str(d)))
            except ValueError:
                pass
    
    if not checkpoints:
        return None, 0
    
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    best_step, best_path = checkpoints[0]
    
    # Ensure trainer_state.json exists (may be missing from BSOD)
    state_file = Path(best_path) / "trainer_state.json"
    if not state_file.exists():
        state = {
            "best_metric": None,
            "best_model_checkpoint": None,
            "epoch": best_step / 9.0,
            "global_step": best_step,
            "is_hyper_param_search": False,
            "is_local_process_zero": True,
            "is_world_process_zero": True,
            "log_history": [],
            "logging_steps": 25,
            "max_steps": -1,
            "num_input_tokens_seen": 0,
            "num_train_epochs": TARGET_EPOCHS,
            "save_steps": SAVE_EVERY_STEPS,
            "stateful_callbacks": {},
            "total_flos": 0,
            "train_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "trial_name": None,
            "trial_params": None
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        log(f"  Created missing trainer_state.json for checkpoint-{best_step}")
    else:
        # Fix train_batch_size if missing
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("train_batch_size") is None:
            state["train_batch_size"] = BATCH_SIZE * GRAD_ACCUM
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            log(f"  Fixed train_batch_size in checkpoint-{best_step}")
    
    return best_path, best_step


def prepare_data():
    """Ensure training data exists."""
    train_path = str(ROOT / "data" / "processed" / "train_2h.jsonl")
    eval_path = str(ROOT / "data" / "processed" / "eval_2h.jsonl")
    
    if not Path(train_path).exists():
        merged_path = ROOT / "data" / "training_sets" / "social_good_all.jsonl"
        if not merged_path.exists():
            log("Running data generator...")
            exec(open(ROOT / "generate_training_data.py", encoding="utf-8").read())
        
        import random
        with open(merged_path, "r", encoding="utf-8") as f:
            all_data = [json.loads(l.strip()) for l in f if l.strip()]
        random.seed(42)
        random.shuffle(all_data)
        
        eval_count = max(3, int(len(all_data) * 0.1))
        train_data = all_data[:-eval_count]
        eval_data = all_data[-eval_count:]
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        with open(train_path, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(eval_path, "w", encoding="utf-8") as f:
            for item in eval_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    train_count = sum(1 for line in open(train_path, encoding="utf-8") if line.strip())
    eval_count = sum(1 for line in open(eval_path, encoding="utf-8") if line.strip())
    return train_path, eval_path, train_count, eval_count


def run_training():
    """Single training run — loads from latest checkpoint, trains until done."""
    import torch
    
    if not torch.cuda.is_available():
        log("FATAL: CUDA not available!")
        return False
    
    torch.cuda.empty_cache()
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"GPU: {gpu} ({vram:.1f}GB)")
    
    token = os.environ.get("HF_TOKEN")
    
    # Find latest checkpoint
    resume_from, last_step = find_latest_checkpoint()
    
    total_steps = math.ceil(72 / BATCH_SIZE) * TARGET_EPOCHS // GRAD_ACCUM
    remaining = total_steps - last_step
    
    if remaining <= 0:
        log(f"Training already complete! ({last_step}/{total_steps} steps)")
        return True
    
    log(f"Resume: checkpoint-{last_step} | Remaining: {remaining}/{total_steps} steps")
    
    # Data
    train_path, eval_path, train_count, eval_count = prepare_data()
    log(f"Data: train={train_count}, eval={eval_count}")
    
    # Model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    used = torch.cuda.memory_allocated() / 1024**3
    log(f"Model loaded: {used:.1f}GB VRAM")
    
    # LoRA
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"LoRA trainable: {trainable:,} params")
    
    # Dataset
    dataset = load_dataset("json", data_files={"train": train_path, "eval": eval_path})
    
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    
    # Training config
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TARGET_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=0 if last_step > 0 else 20,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=25,
        save_steps=SAVE_EVERY_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_EVERY_STEPS,
        save_total_limit=3,
        load_best_model_at_end=False,  # Simpler — avoid extra eval at end
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_pin_memory=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        formatting_func=formatting_func,
        args=sft_config,
    )
    
    # Train
    log("=" * 50)
    log("TRAINING STARTED")
    log("=" * 50)
    
    if resume_from:
        result = trainer.train(resume_from_checkpoint=resume_from)
    else:
        result = trainer.train()
    
    # Save final
    log("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    eval_results = trainer.evaluate()
    
    metadata = {
        "base_model": MODEL_NAME,
        "total_steps": result.global_step,
        "training_loss": result.training_loss,
        "eval_loss": eval_results.get("eval_loss"),
        "completed_at": datetime.now().isoformat(),
    }
    with open(Path(OUTPUT_DIR) / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    log(f"TRAINING COMPLETE — loss={result.training_loss:.4f}, steps={result.global_step}")
    return True


def main():
    # Prevent double-run
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            import psutil
            if psutil.pid_exists(pid):
                log(f"Another training is running (PID {pid}). Exiting.")
                return
        except Exception:
            pass
    
    LOCK_FILE.write_text(str(os.getpid()))
    
    try:
        log("")
        log("=" * 60)
        log("DETACHED TRAINING SESSION START")
        log(f"PID: {os.getpid()}")
        log("=" * 60)
        
        start = time.time()
        success = run_training()
        elapsed = time.time() - start
        h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
        
        if success:
            log(f"Session finished successfully in {h}h {m}m")
        else:
            log(f"Session ended with issues after {h}h {m}m")
    
    except Exception as e:
        log(f"CRASH: {e}")
        traceback.print_exc()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)
    
    finally:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()


if __name__ == "__main__":
    main()
