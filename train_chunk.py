"""
Micro-Chunk Trainer — Runs exactly N steps then exits cleanly.
Called repeatedly by run_training.bat with fresh CUDA context each time.
Avoids RTX 5070 driver crash on sustained load.
"""
import os, sys, json, time, gc, traceback
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUTF8"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

ROOT = Path(__file__).resolve().parent

# ─── CONFIG ───────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR       = str(ROOT / "data" / "models" / "social_good_v1")
TRAIN_PATH       = str(ROOT / "data" / "processed" / "train_2h.jsonl")
EVAL_PATH        = str(ROOT / "data" / "processed" / "eval_2h.jsonl")

STEPS_PER_CHUNK  = 50       # Run this many steps then EXIT
TOTAL_TARGET     = 1701     # Total training steps

LORA_R           = 64
LORA_ALPHA       = 128
LORA_DROPOUT     = 0.05
TARGET_MODULES   = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

BATCH_SIZE       = 1
GRAD_ACCUM       = 8
LEARNING_RATE    = 5e-5
MAX_SEQ_LENGTH   = 1024
WEIGHT_DECAY     = 0.01
MAX_GRAD_NORM    = 0.3
NUM_EPOCHS       = 189


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(ROOT / "training_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_latest_checkpoint():
    """Find highest-numbered checkpoint directory."""
    ckpt_dir = Path(OUTPUT_DIR)
    checkpoints = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                checkpoints.append((step, d))
            except ValueError:
                pass
    if not checkpoints:
        return 0, None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][0], str(checkpoints[-1][1])


def ensure_trainer_state(ckpt_path, step):
    """Create trainer_state.json if missing (BSOD leaves incomplete checkpoints)."""
    state_path = Path(ckpt_path) / "trainer_state.json"
    if not state_path.exists():
        log(f"  trainer_state.json eksik, olusturuluyor (step {step})...")
        state = {
            "best_metric": None,
            "best_model_checkpoint": None,
            "epoch": step / 9.0,
            "global_step": step,
            "is_hyper_param_search": False,
            "is_local_process_zero": True,
            "is_world_process_zero": True,
            "log_history": [],
            "logging_steps": 10,
            "max_steps": -1,
            "num_input_tokens_seen": 0,
            "num_train_epochs": NUM_EPOCHS,
            "save_steps": STEPS_PER_CHUNK,
            "stateful_callbacks": {},
            "total_flos": 0,
            "train_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "trial_name": None,
            "trial_params": None
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        log(f"  trainer_state.json olusturuldu.")


def main():
    start = time.time()

    import torch
    if not torch.cuda.is_available():
        log("CUDA yok!")
        return 1

    # Find where we left off
    last_step, ckpt_path = find_latest_checkpoint()

    if last_step >= TOTAL_TARGET:
        log(f"TAMAMLANDI! {last_step}/{TOTAL_TARGET} steps done.")
        return 0

    remaining = TOTAL_TARGET - last_step
    chunk_steps = min(STEPS_PER_CHUNK, remaining)

    log("=" * 50)
    log(f"CHUNK: step {last_step} -> {last_step + chunk_steps}")
    log(f"  Kalan: {remaining} steps")
    log("=" * 50)

    if ckpt_path:
        ensure_trainer_state(ckpt_path, last_step)

    try:
        torch.cuda.empty_cache()
        gc.collect()

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset

        token = os.environ.get("HF_TOKEN")

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
            MODEL_NAME, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16, token=token,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        vram = torch.cuda.memory_allocated() / 1024**3
        log(f"  Model loaded: {vram:.1f}GB VRAM")

        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "eval": EVAL_PATH})

        def formatting_func(example):
            return tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )

        # Save every chunk_steps, so at the END of this chunk we get a checkpoint
        sft_config = SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            warmup_steps=0,
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,
            lr_scheduler_type="cosine",
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_steps=chunk_steps,       # Save at end of this chunk
            save_total_limit=3,
            eval_strategy="no",           # Skip eval for speed in chunks
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            max_steps=last_step + chunk_steps,  # HARD STOP after N steps
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

        log(f"  Training basliyor...")
        if ckpt_path:
            result = trainer.train(resume_from_checkpoint=ckpt_path)
        else:
            result = trainer.train()

        log(f"  Chunk tamamlandi! Step: {result.global_step}, Loss: {result.training_loss:.4f}")

        # Explicit save
        save_dir = Path(OUTPUT_DIR) / f"checkpoint-{result.global_step}"
        if not save_dir.exists():
            trainer.save_model(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            trainer.state.save_to_json(str(save_dir / "trainer_state.json"))
            log(f"  Checkpoint kaydedildi: {save_dir.name}")

        elapsed = time.time() - start
        log(f"  Sure: {elapsed/60:.1f}dk")

        # Check if done
        if result.global_step >= TOTAL_TARGET:
            log("=" * 50)
            log("TRAINING TAMAMLANDI!")
            log("=" * 50)
            # Save final model to root
            trainer.save_model(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            meta = {
                "base_model": MODEL_NAME, "total_steps": result.global_step,
                "training_loss": result.training_loss, "lora_r": LORA_R,
                "completed_at": datetime.now().isoformat(),
            }
            with open(Path(OUTPUT_DIR) / "training_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            return 0

        return 2  # More chunks needed

    except Exception as e:
        log(f"  HATA: {e}")
        traceback.print_exc()
        elapsed = time.time() - start
        log(f"  Crash after {elapsed/60:.1f}dk")
        return 1

    finally:
        # Aggressively clean GPU
        try:
            import torch
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass


if __name__ == "__main__":
    code = main()
    sys.exit(code)
