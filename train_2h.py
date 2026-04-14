"""
Robust 2-Hour Training Script — Social Good Chatbot
Self-contained, bypasses all pipeline issues.
Handles BFloat16/fp16, TRL 1.1.0 API changes, and error recovery.
"""
import os
import sys
import json
import time
import random
import traceback
from pathlib import Path
from datetime import datetime

# CRITICAL: Fix Windows Turkish encoding (cp1254)
os.environ["PYTHONUTF8"] = "1"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─── CONFIG ───────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = str(ROOT / "data" / "models" / "social_good_v1")
DATA_DIR = ROOT / "data" / "training_sets"
TRAINING_HOURS = 2.0

# QLoRA config
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training hyperparams — will be adjusted for 2-hour target
BATCH_SIZE = 2
GRAD_ACCUM = 4  # Reduced from 8 for more steps/epoch
LEARNING_RATE = 1e-4
WARMUP_STEPS = 10
MAX_SEQ_LENGTH = 2048
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.3


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def step1_generate_data():
    """Generate expanded training data if not already done."""
    merged_path = DATA_DIR / "social_good_all.jsonl"

    if merged_path.exists() and sum(1 for _ in open(merged_path, "r", encoding="utf-8")) > 50:
        count = sum(1 for _ in open(merged_path, "r", encoding="utf-8") if _.strip())
        log(f"Using existing merged data: {count} samples")
        return str(merged_path), count

    log("Generating extended training data...")
    gen_script = ROOT / "generate_training_data.py"
    if gen_script.exists():
        exec(open(gen_script, encoding="utf-8").read())
        count = sum(1 for _ in open(merged_path, "r", encoding="utf-8") if _.strip())
        return str(merged_path), count
    else:
        # Fallback: merge existing files
        all_lines = []
        for fname in DATA_DIR.glob("social_good_base_*.jsonl"):
            with open(fname, "r", encoding="utf-8") as f:
                all_lines.extend([l.strip() for l in f if l.strip()])
        random.seed(42)
        random.shuffle(all_lines)
        with open(merged_path, "w", encoding="utf-8") as f:
            for line in all_lines:
                f.write(line + "\n")
        return str(merged_path), len(all_lines)


def step2_split_data(merged_path, total_count):
    """Split data into train/eval."""
    from datasets import load_dataset

    eval_ratio = 0.1
    eval_count = max(3, int(total_count * eval_ratio))
    train_count = total_count - eval_count

    all_data = []
    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line.strip()))

    random.seed(42)
    random.shuffle(all_data)

    train_data = all_data[:train_count]
    eval_data = all_data[train_count:]

    train_path = str(ROOT / "data" / "processed" / "train_2h.jsonl")
    eval_path = str(ROOT / "data" / "processed" / "eval_2h.jsonl")

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    log(f"Data split: Train={len(train_data)} | Eval={len(eval_data)}")
    return train_path, eval_path, len(train_data)


def step3_calculate_epochs(train_count):
    """Calculate exact epochs needed for 2-hour training."""
    # Estimate: steps_per_epoch = ceil(train_count / batch_size) / grad_accum
    import math
    steps_per_epoch = math.ceil(train_count / BATCH_SIZE) / GRAD_ACCUM

    # Estimate seconds per optimizer step on RTX 5070 with 7B 4-bit model
    # Conservative estimate: 5-10 seconds per step
    SEC_PER_STEP = 7.0

    target_seconds = TRAINING_HOURS * 3600
    target_steps = target_seconds / SEC_PER_STEP
    target_epochs = max(10, int(target_steps / steps_per_epoch))

    total_steps = int(target_epochs * steps_per_epoch)

    log(f"Training plan:")
    log(f"  Steps/epoch: {steps_per_epoch:.1f}")
    log(f"  Target epochs: {target_epochs}")
    log(f"  Total optimizer steps: {total_steps}")
    log(f"  Estimated time: {total_steps * SEC_PER_STEP / 3600:.1f}h")

    return target_epochs, total_steps


def step4_train(train_path, eval_path, num_epochs, total_steps):
    """Execute QLoRA training with all TRL 1.1.0 fixes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    # ─── Model Loading ────────────────────────────────────
    log(f"Loading model: {MODEL_NAME}")

    token = os.environ.get("HF_TOKEN")

    # BFloat16 compute — Qwen2.5 native dtype is BFloat16
    # Using bf16 avoids the GradScaler issue with mixed BFloat16 tensors
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        token=token,
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

    log(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # ─── LoRA ─────────────────────────────────────────────
    log("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ─── Data ─────────────────────────────────────────────
    log("Loading datasets...")
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "eval": eval_path,
    })
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
    log(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # ─── Formatting ───────────────────────────────────────
    def formatting_func(example):
        """Format single example to chat template string."""
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    # ─── Training Config ──────────────────────────────────
    # Key fix: bf16=True, fp16=False
    # BFloat16 training does NOT use GradScaler, avoiding the BFloat16 unscale error
    log_steps = max(1, total_steps // 50)  # ~50 log points
    save_steps = max(10, total_steps // 10)  # ~10 checkpoints
    eval_steps = save_steps

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=num_epochs,
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
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
    )

    # ─── Trainer ──────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        args=sft_config,
    )

    # ─── Train ────────────────────────────────────────────
    log("=" * 50)
    log("TRAINING STARTED")
    log("=" * 50)

    train_result = trainer.train()

    log("=" * 50)
    log("TRAINING COMPLETE")
    log("=" * 50)

    # ─── Save ─────────────────────────────────────────────
    log(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Evaluate
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", None)

    # Save metadata
    metadata = {
        "base_model": MODEL_NAME,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "num_epochs": num_epochs,
        "total_steps": train_result.global_step,
        "training_loss": train_result.training_loss,
        "eval_loss": eval_loss,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "bf16": True,
        "created_at": datetime.now().isoformat(),
    }
    with open(Path(OUTPUT_DIR) / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return train_result, eval_loss


def main():
    print("=" * 60)
    print("  SOCIAL GOOD CHATBOT — 2-HOUR ROBUST TRAINING")
    print("=" * 60)
    print()

    start_time = time.time()

    # Prerequisites
    import torch
    if not torch.cuda.is_available():
        log("FAIL: CUDA not available!")
        return

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"GPU: {gpu} ({vram:.1f}GB VRAM)")

    token = os.environ.get("HF_TOKEN")
    if token:
        log(f"HF_TOKEN: {token[:8]}...{token[-4:]}")
    else:
        log("HF_TOKEN not set — will try without token")

    try:
        # Step 1: Data
        log("\n── STEP 1: Training Data ──")
        merged_path, total_count = step1_generate_data()

        # Step 2: Split
        log("\n── STEP 2: Data Split ──")
        train_path, eval_path, train_count = step2_split_data(merged_path, total_count)

        # Step 3: Plan
        log("\n── STEP 3: Training Plan ──")
        num_epochs, total_steps = step3_calculate_epochs(train_count)

        # Step 4: Train
        log("\n── STEP 4: QLoRA Training ──")
        log(f"  Model: {MODEL_NAME}")
        log(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
        log(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} = effective {BATCH_SIZE * GRAD_ACCUM}")
        log(f"  Epochs: {num_epochs}")
        log(f"  LR: {LEARNING_RATE}")
        log(f"  BF16: True (native Qwen2.5 dtype)")
        log("")

        train_result, eval_loss = step4_train(train_path, eval_path, num_epochs, total_steps)

        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)

        print()
        print("=" * 60)
        print("  TRAINING RESULTS")
        print("=" * 60)
        log(f"✅ Training COMPLETE")
        log(f"  Model saved: {OUTPUT_DIR}")
        log(f"  Training loss: {train_result.training_loss:.4f}")
        if eval_loss:
            log(f"  Eval loss: {eval_loss:.4f}")
        log(f"  Total steps: {train_result.global_step}")
        log(f"  Duration: {hours}h {minutes}m")
        print()
        log("Sonraki adım:")
        log("  python -m uvicorn src.main:app --reload --port 8000")
        log("  http://localhost:8000/docs")

    except Exception as e:
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)

        print()
        print("=" * 60)
        log(f"❌ Training FAILED: {e}")
        log(f"Duration: {hours}h {minutes}m")
        traceback.print_exc()
        print("=" * 60)


if __name__ == "__main__":
    main()
