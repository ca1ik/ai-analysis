"""
Safe 2-Hour Resume Training — GPU %60-70 Load
Checkpoint-102'den devam eder. BSOD riski minimize edilmiştir.
"""
import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─── CONFIG — GPU SAFE MODE ──────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = str(ROOT / "data" / "models" / "social_good_v1")
CHECKPOINT_DIR = str(ROOT / "data" / "models" / "social_good_v1" / "checkpoint-102")
TRAINING_HOURS = 2.0

# QLoRA
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ── GPU-SAFE Training params ──
# Batch=1 + grad_accum=8 = effective 8 (same quality, less peak VRAM)
# max_seq=1024 instead of 2048 → ~40% less VRAM per sample
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 5e-5  # Lower LR since we're resuming (already warmed up)
WARMUP_STEPS = 0      # No warmup needed on resume
MAX_SEQ_LENGTH = 1024  # Halved from 2048 — reduces compute per step
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.3
# GPU power limit set to 175W (70%) via nvidia-smi -pl 175


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def main():
    print("=" * 60)
    print("  SOCIAL GOOD CHATBOT — SAFE 2H RESUME TRAINING")
    print("  GPU Load: ~60-70% | Resume from checkpoint-102")
    print("=" * 60)
    print()

    start_time = time.time()

    # ─── GPU Memory Limit ─────────────────────────────────
    import torch
    if not torch.cuda.is_available():
        log("FAIL: CUDA not available!")
        return

    torch.cuda.empty_cache()

    gpu = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"GPU: {gpu} ({vram_total:.1f}GB, power limited to 175W = 70%)")

    token = os.environ.get("HF_TOKEN")
    if token:
        log(f"HF_TOKEN: {token[:8]}...{token[-4:]}")

    # Check checkpoint exists
    if not Path(CHECKPOINT_DIR).exists():
        log(f"WARN: Checkpoint not found at {CHECKPOINT_DIR}, training from scratch")
        resume_from = None
    else:
        log(f"Checkpoint found: checkpoint-102 (will resume)")
        resume_from = CHECKPOINT_DIR

    try:
        # ─── Data ─────────────────────────────────────────
        log("\n── STEP 1: Data ──")
        train_path = str(ROOT / "data" / "processed" / "train_2h.jsonl")
        eval_path = str(ROOT / "data" / "processed" / "eval_2h.jsonl")

        if not Path(train_path).exists():
            log("Processed data not found, regenerating...")
            merged_path = ROOT / "data" / "training_sets" / "social_good_all.jsonl"
            if not merged_path.exists():
                log("Running data generator...")
                exec(open(ROOT / "generate_training_data.py", encoding="utf-8").read())

            import random
            all_data = []
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

        train_count = sum(1 for _ in open(train_path, encoding="utf-8") if _.strip())
        eval_count = sum(1 for _ in open(eval_path, encoding="utf-8") if _.strip())
        log(f"Data: Train={train_count} | Eval={eval_count}")

        # ─── Epoch Calculation ────────────────────────────
        log("\n── STEP 2: Training Plan ──")
        import math
        steps_per_epoch = math.ceil(train_count / BATCH_SIZE) / GRAD_ACCUM

        # With batch=1, seq=1024: estimate ~4s per step (lighter)
        SEC_PER_STEP = 4.5
        target_seconds = TRAINING_HOURS * 3600
        remaining_steps = int(target_seconds / SEC_PER_STEP)

        # We already did 102 steps. Total new target:
        already_done = 102
        total_target_steps = already_done + remaining_steps
        target_epochs = max(10, int(total_target_steps / steps_per_epoch))

        log(f"  Steps/epoch: {steps_per_epoch:.1f}")
        log(f"  Already completed: {already_done} steps")
        log(f"  Remaining steps: ~{remaining_steps}")
        log(f"  Total epochs: {target_epochs}")
        log(f"  Estimated remaining time: {remaining_steps * SEC_PER_STEP / 3600:.1f}h")

        # ─── Model Loading ────────────────────────────────
        log("\n── STEP 3: Loading Model (GPU-safe mode) ──")
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

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, token=token,
        )
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
        log(f"Model loaded. GPU VRAM: {used:.1f}GB / {vram_total:.1f}GB")

        # LoRA
        log("Applying LoRA adapters...")
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ─── Dataset ──────────────────────────────────────
        log("Loading datasets...")
        dataset = load_dataset("json", data_files={"train": train_path, "eval": eval_path})
        train_dataset = dataset["train"]
        eval_dataset = dataset["eval"]

        def formatting_func(example):
            return tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )

        # ─── Training Config — GPU-SAFE ───────────────────
        log("\n── STEP 4: Training (GPU-safe) ──")
        log(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = effective {BATCH_SIZE * GRAD_ACCUM}")
        log(f"  Max seq: {MAX_SEQ_LENGTH}")
        log(f"  GPU power limit: 175W (70%)")
        log(f"  Resume from: {'checkpoint-102' if resume_from else 'scratch'}")

        log_steps = max(1, remaining_steps // 40)
        save_steps = max(20, remaining_steps // 8)
        eval_steps = save_steps

        sft_config = SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=target_epochs,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,
            lr_scheduler_type="cosine",
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=log_steps,
            save_steps=save_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_pin_memory=False,  # Reduce memory pressure
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_func,
            args=sft_config,
        )

        # ─── Resume Training ─────────────────────────────
        log("=" * 50)
        log("TRAINING STARTED (GPU-SAFE MODE)")
        log("=" * 50)

        if resume_from:
            log(f"Resuming from {resume_from}...")
            train_result = trainer.train(resume_from_checkpoint=resume_from)
        else:
            train_result = trainer.train()

        log("=" * 50)
        log("TRAINING COMPLETE!")
        log("=" * 50)

        # ─── Save ─────────────────────────────────────────
        log(f"Saving final model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        eval_results = trainer.evaluate()
        eval_loss = eval_results.get("eval_loss", None)

        metadata = {
            "base_model": MODEL_NAME,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "num_epochs": target_epochs,
            "total_steps": train_result.global_step,
            "training_loss": train_result.training_loss,
            "eval_loss": eval_loss,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "max_seq_length": MAX_SEQ_LENGTH,
            "learning_rate": LEARNING_RATE,
            "gpu_power_limit_watts": 175,
            "bf16": True,
            "resumed_from": "checkpoint-102",
            "created_at": datetime.now().isoformat(),
        }
        with open(Path(OUTPUT_DIR) / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)

        print()
        print("=" * 60)
        log(f"TRAINING COMPLETE")
        log(f"  Model: {OUTPUT_DIR}")
        log(f"  Loss: {train_result.training_loss:.4f}")
        if eval_loss:
            log(f"  Eval loss: {eval_loss:.4f}")
        log(f"  Steps: {train_result.global_step}")
        log(f"  Duration: {hours}h {minutes}m")
        print("=" * 60)
        log("Test: python -m uvicorn src.main:app --reload --port 8000")

    except torch.cuda.OutOfMemoryError:
        log("GPU MEMORY EXCEEDED — but system is safe (no BSOD)")
        log("Try reducing MAX_SEQ_LENGTH to 512 or LORA_R to 32")
        torch.cuda.empty_cache()
        traceback.print_exc()

    except Exception as e:
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print()
        log(f"FAILED: {e}")
        log(f"Duration: {hours}h {minutes}m")
        traceback.print_exc()


if __name__ == "__main__":
    main()
