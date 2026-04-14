"""
Chunked Training — BSOD-Proof
Her chunk 15 dakika çalışır, checkpoint kaydeder, GPU'yu soğutur, devam eder.
BSOD olursa en fazla 15 dk kayıp.
"""
import os
import sys
import json
import time
import math
import gc
import traceback
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUTF8"] = "1"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─── CONFIG ───────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = str(ROOT / "data" / "models" / "social_good_v1")

# QLoRA
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.3

# Chunked training
CHUNK_MINUTES = 15           # Her chunk kaç dakika çalışsın
COOLDOWN_SECONDS = 30        # Chunk arası GPU soğutma
TOTAL_EPOCHS = 200           # Hedef epoch sayısı
SAVE_STEPS = 50              # Her 50 step'te checkpoint


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    # Log to file too
    with open(ROOT / "training_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_latest_checkpoint():
    """En son checkpoint'u bul."""
    output = Path(OUTPUT_DIR)
    checkpoints = []
    for d in output.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                checkpoints.append((step, str(d)))
            except ValueError:
                pass
    if not checkpoints:
        return None, 0
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1], checkpoints[0][0]


def ensure_trainer_state(checkpoint_dir, step_num):
    """trainer_state.json yoksa oluştur (BSOD sonrası)."""
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.exists():
        log(f"  trainer_state.json oluşturuluyor (step {step_num})...")
        state = {
            "best_metric": None,
            "best_model_checkpoint": None,
            "epoch": step_num / 9.0,  # 72 samples / 8 effective batch = 9 steps/epoch
            "global_step": step_num,
            "is_hyper_param_search": False,
            "is_local_process_zero": True,
            "is_world_process_zero": True,
            "log_history": [],
            "logging_steps": 10,
            "max_steps": -1,
            "num_input_tokens_seen": 0,
            "num_train_epochs": TOTAL_EPOCHS,
            "save_steps": SAVE_STEPS,
            "stateful_callbacks": {},
            "total_flos": 0,
            "train_batch_size": BATCH_SIZE,
            "trial_name": None,
            "trial_params": None
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        log(f"  trainer_state.json oluşturuldu.")


def prepare_data():
    """Training verisini hazırla."""
    train_path = str(ROOT / "data" / "processed" / "train_2h.jsonl")
    eval_path = str(ROOT / "data" / "processed" / "eval_2h.jsonl")

    if not Path(train_path).exists():
        merged_path = ROOT / "data" / "training_sets" / "social_good_all.jsonl"
        if not merged_path.exists():
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
    return train_path, eval_path, train_count, eval_count


def run_training_chunk(resume_checkpoint, chunk_num, total_target_steps):
    """Tek bir training chunk'ı çalıştır. max CHUNK_MINUTES dakika."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import TrainerCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    train_path, eval_path, train_count, eval_count = prepare_data()

    # ─── Time-based stop callback ─────────────────────────
    class TimeLimitCallback(TrainerCallback):
        def __init__(self, max_seconds):
            self.max_seconds = max_seconds
            self.start_time = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self.start_time
            if elapsed >= self.max_seconds:
                log(f"  ⏱ Süre doldu ({elapsed/60:.1f} dk). Durduruluyor...")
                control.should_training_stop = True
                control.should_save = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                elapsed = time.time() - self.start_time
                log(f"  Step {state.global_step} | Loss: {logs['loss']:.4f} | {elapsed/60:.1f}dk")

    # ─── Model ────────────────────────────────────────────
    torch.cuda.empty_cache()
    gc.collect()

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

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    used = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log(f"  VRAM: {used:.1f}GB / {total:.1f}GB")

    # ─── Dataset ──────────────────────────────────────────
    dataset = load_dataset("json", data_files={"train": train_path, "eval": eval_path})

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    # ─── Config ───────────────────────────────────────────
    steps_per_epoch = math.ceil(train_count / BATCH_SIZE) / GRAD_ACCUM

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TOTAL_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=10 if not resume_checkpoint else 0,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=False,
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
        callbacks=[TimeLimitCallback(CHUNK_MINUTES * 60)],
    )

    # ─── Train ────────────────────────────────────────────
    if resume_checkpoint:
        log(f"  Resuming from {Path(resume_checkpoint).name}...")
        result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        result = trainer.train()

    # ─── Save ─────────────────────────────────────────────
    final_step = result.global_step
    log(f"  Chunk tamamlandı. Step: {final_step}, Loss: {result.training_loss:.4f}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ─── Cleanup GPU ──────────────────────────────────────
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return final_step, result.training_loss


def main():
    print("=" * 60)
    print("  CHUNKED BSOD-PROOF TRAINING")
    print(f"  {CHUNK_MINUTES}dk chunks | {COOLDOWN_SECONDS}s cooldown")
    print("=" * 60)

    import torch
    if not torch.cuda.is_available():
        log("CUDA not available!")
        return

    gpu = torch.cuda.get_device_name(0)
    log(f"GPU: {gpu}")

    start_time = time.time()
    steps_per_epoch = 9  # 72 samples / 8 effective batch
    total_target_steps = TOTAL_EPOCHS * steps_per_epoch  # 200 * 9 = 1800

    chunk_num = 0
    while True:
        chunk_num += 1
        checkpoint_path, last_step = find_latest_checkpoint()

        if last_step >= total_target_steps:
            log(f"Training tamamlandı! {last_step}/{total_target_steps} steps")
            break

        log(f"\n{'='*50}")
        log(f"CHUNK #{chunk_num}")
        log(f"  Son checkpoint: step {last_step}")
        log(f"  Kalan: {total_target_steps - last_step} steps")
        log(f"{'='*50}")

        if checkpoint_path:
            ensure_trainer_state(checkpoint_path, last_step)

        try:
            final_step, loss = run_training_chunk(checkpoint_path, chunk_num, total_target_steps)

            elapsed_total = (time.time() - start_time) / 60
            log(f"  Chunk #{chunk_num} bitti: step {final_step}, loss {loss:.4f}")
            log(f"  Toplam geçen süre: {elapsed_total:.0f} dk")

            if final_step >= total_target_steps:
                log("TRAINING TAMAMLANDI!")
                break

            # Cooldown
            log(f"  GPU soğutma: {COOLDOWN_SECONDS}s...")
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            time.sleep(COOLDOWN_SECONDS)

        except Exception as e:
            log(f"  HATA: {e}")
            traceback.print_exc()
            log(f"  30s bekleniyor, tekrar denenecek...")
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
            time.sleep(30)

    # ─── Final Summary ────────────────────────────────────
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    log(f"\n{'='*60}")
    log(f"TRAINING COMPLETE")
    log(f"  Chunks: {chunk_num}")
    log(f"  Duration: {hours}h {minutes}m")
    log(f"  Model: {OUTPUT_DIR}")
    log(f"{'='*60}")

    # Save metadata
    metadata = {
        "base_model": MODEL_NAME,
        "total_chunks": chunk_num,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "max_seq_length": MAX_SEQ_LENGTH,
        "learning_rate": LEARNING_RATE,
        "total_epochs": TOTAL_EPOCHS,
        "bf16": True,
        "gpu_clock_limit": "1500MHz",
        "gpu_power_limit": "175W",
        "chunk_minutes": CHUNK_MINUTES,
        "created_at": datetime.now().isoformat(),
    }
    with open(Path(OUTPUT_DIR) / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
