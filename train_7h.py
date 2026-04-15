"""Ironclad Trainer v3 - Anti-crash for RTX 5070 WDDM + driver 595.71.

v3 critical fixes:
  - StabilityCallback: torch.cuda.synchronize() EVERY step (prevents TDR kill)
  - 0.3s sleep between steps (software GPU throttle - nvidia-smi -pl needs admin)
  - gc.collect + empty_cache every 5 steps (prevent VRAM fragmentation)
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (anti-fragmentation)
  - 15 steps per chunk (minimal crash window)
  - LoRA r=64 preserved (resume from checkpoint-225)
  - grad_accum=4, seq=512, fp16 (from v2)
"""
import os
import sys
import json
import time
import gc
import traceback
import subprocess
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.7"

ROOT = Path(__file__).resolve().parent

# --- CONFIG ---
MODEL_NAME       = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR       = str(ROOT / "data" / "models" / "social_good_v1")
TRAIN_PATH       = str(ROOT / "data" / "processed" / "train_2h.jsonl")
EVAL_PATH        = str(ROOT / "data" / "processed" / "eval_2h.jsonl")

STEPS_PER_CHUNK  = 15       # v3: very small chunks = minimal crash window
TOTAL_TARGET     = 1701

LORA_R           = 64
LORA_ALPHA       = 128
LORA_DROPOUT     = 0.05
TARGET_MODULES   = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

BATCH_SIZE       = 1
GRAD_ACCUM       = 4
LEARNING_RATE    = 5e-5
MAX_SEQ_LENGTH   = 512
WEIGHT_DECAY     = 0.01
MAX_GRAD_NORM    = 0.3
NUM_EPOCHS       = 189

MAX_GPU_TEMP     = 75
STEP_SLEEP       = 0.3      # v3: seconds to sleep EVERY step (software throttle)
CLEANUP_EVERY    = 5        # v3: gc + empty_cache every N steps


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    print(line, flush=True)
    try:
        with open(ROOT / "training_log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def get_gpu_temp():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip())
    except Exception:
        return None


def find_latest_checkpoint():
    ckpt_dir = Path(OUTPUT_DIR)
    if not ckpt_dir.exists():
        return 0, None
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
    state_path = Path(ckpt_path) / "trainer_state.json"
    if not state_path.exists():
        log("  [REPAIR] trainer_state.json missing at step {}, creating...".format(step))
        state = {
            "best_metric": None,
            "best_model_checkpoint": None,
            "epoch": step / 9.0,
            "global_step": step,
            "is_hyper_param_search": False,
            "is_local_process_zero": True,
            "is_world_process_zero": True,
            "log_history": [],
            "logging_steps": 5,
            "max_steps": -1,
            "num_input_tokens_seen": 0,
            "num_train_epochs": NUM_EPOCHS,
            "save_steps": STEPS_PER_CHUNK,
            "stateful_callbacks": {},
            "total_flos": 0,
            "train_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "trial_name": None,
            "trial_params": None,
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        log("  [REPAIR] Created.")


def main():
    start = time.time()

    temp = get_gpu_temp()
    if temp is not None:
        log("  GPU Temperature: {}C".format(temp))
        if temp > MAX_GPU_TEMP:
            log("  [WAIT] GPU too hot ({}C > {}C). Exiting for cooldown.".format(temp, MAX_GPU_TEMP))
            return 3

    import torch

    if not torch.cuda.is_available():
        log("[FATAL] CUDA not available!")
        return 1

    # === v3: Maximum Stability Settings ===
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # v3: Verify CUDA works before loading anything
    try:
        test_tensor = torch.zeros(1, device="cuda")
        del test_tensor
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log("  CUDA health check: OK")
    except Exception as e:
        log("  [FATAL] CUDA health check failed: {}".format(e))
        return 1

    gc.collect()

    last_step, ckpt_path = find_latest_checkpoint()

    if last_step >= TOTAL_TARGET:
        log("[DONE] Training complete! {}/{}".format(last_step, TOTAL_TARGET))
        return 0

    remaining = TOTAL_TARGET - last_step
    chunk_steps = min(STEPS_PER_CHUNK, remaining)

    log("=" * 60)
    log("  CHUNK START: step {} -> {}".format(last_step, last_step + chunk_steps))
    log("  Remaining: {} steps | Target: {}".format(remaining, TOTAL_TARGET))
    log("  GPU Temp: {}C | Time: {}".format(temp, datetime.now().strftime("%H:%M")))
    log("=" * 60)

    if ckpt_path:
        ensure_trainer_state(ckpt_path, last_step)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers import TrainerCallback
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset

        # === v3: CRITICAL - StabilityCallback ===
        class StabilityCallback(TrainerCallback):
            """Prevents TDR kills and CUDA driver crashes on WDDM Windows.
            
            After every optimizer step:
            1. torch.cuda.synchronize() - forces GPU to complete all ops
               This prevents Windows TDR from thinking GPU is hung
            2. time.sleep(0.3) - software power throttle since nvidia-smi -pl needs admin
            3. Every 5 steps: gc.collect + empty_cache to prevent fragmentation
            """
            def on_step_end(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                time.sleep(STEP_SLEEP)
                if state.global_step % CLEANUP_EVERY == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    step_num = state.global_step
                    cur_temp = get_gpu_temp()
                    vram = torch.cuda.memory_allocated() / 1024**3
                    log("  [Step {}] loss={} temp={}C vram={:.1f}GB".format(
                        step_num, logs.get("loss", "?"), cur_temp, vram))
                    if cur_temp is not None and cur_temp > 80:
                        log("  [EMERGENCY] GPU {}C > 80C! Requesting stop.".format(cur_temp))
                        control.should_training_stop = True

        token = os.environ.get("HF_TOKEN")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        log("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, token=token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        log("  Loading model (4-bit QLoRA)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            token=token,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        # v3: sync after model load to ensure clean state
        torch.cuda.synchronize()

        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log("  Model loaded: {:.1f}/{:.1f} GB VRAM".format(vram_used, vram_total))

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log("  LoRA params: {:,} / {:,} ({:.2f}%)".format(trainable, total, 100 * trainable / total))

        dataset = load_dataset(
            "json", data_files={"train": TRAIN_PATH, "eval": EVAL_PATH},
        )

        def formatting_func(example):
            return tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,
            )

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
            fp16=True,
            bf16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_32bit",
            logging_steps=5,
            save_steps=chunk_steps,
            save_total_limit=5,
            eval_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            max_steps=last_step + chunk_steps,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            formatting_func=formatting_func,
            args=sft_config,
            callbacks=[StabilityCallback()],
        )

        torch.cuda.empty_cache()
        gc.collect()

        log("  Training starting... ({} steps, sync every step, sleep {}s)".format(
            chunk_steps, STEP_SLEEP))
        train_start = time.time()

        if ckpt_path:
            result = trainer.train(resume_from_checkpoint=ckpt_path)
        else:
            result = trainer.train()

        # v3: sync before declaring success
        torch.cuda.synchronize()

        train_elapsed = time.time() - train_start
        log("  Chunk done! Step: {}, Loss: {:.4f}".format(result.global_step, result.training_loss))
        log("  Training time: {:.1f}min".format(train_elapsed / 60))

        torch.cuda.empty_cache()

        save_dir = Path(OUTPUT_DIR) / "checkpoint-{}".format(result.global_step)
        if not save_dir.exists():
            trainer.save_model(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            trainer.state.save_to_json(str(save_dir / "trainer_state.json"))
            log("  Checkpoint saved: {}".format(save_dir.name))

        post_temp = get_gpu_temp()
        log("  Post-training GPU temp: {}C".format(post_temp))

        elapsed = time.time() - start
        log("  Total chunk time: {:.1f}min".format(elapsed / 60))

        if result.global_step >= TOTAL_TARGET:
            log("=" * 60)
            log("  TRAINING COMPLETE!")
            log("=" * 60)
            trainer.save_model(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            meta = {
                "base_model": MODEL_NAME,
                "total_steps": result.global_step,
                "training_loss": result.training_loss,
                "lora_r": LORA_R,
                "completed_at": datetime.now().isoformat(),
            }
            with open(Path(OUTPUT_DIR) / "training_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            return 0

        return 2  # More chunks needed

    except Exception as e:
        log("  [ERROR] {}: {}".format(type(e).__name__, e))
        traceback.print_exc()
        elapsed = time.time() - start
        log("  Crashed after {:.1f}min".format(elapsed / 60))
        return 1

    finally:
        try:
            import torch as _torch
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


if __name__ == "__main__":
    code = main()
    sys.exit(code)
"""Ironclad 7-Hour Trainer v2 - Maximum stability for RTX 5070 + driver 595.71.
Runs as a CHILD of run_7h_training.py orchestrator.
Each invocation = 1 chunk (25 steps), then exits cleanly.

v2 hardening (post cudaErrorUnknown crash):
  - fp16 instead of bf16 (more stable driver code path)
  - TF32 disabled (avoids tensor core instability)
  - Smaller chunks (25 steps = less exposure time)
  - Shorter seq length (512 = less VRAM pressure)
  - CUDA memory fraction capped at 85%
  - Periodic torch.cuda.synchronize() for clean GPU state
  - Power limit 130W (orchestrator controls this)
"""
import os, sys, json, time, gc, traceback, subprocess
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUTF8"] = "1"

ROOT = Path(__file__).resolve().parent

# ─── CONFIG ───────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR       = str(ROOT / "data" / "models" / "social_good_v1")
TRAIN_PATH       = str(ROOT / "data" / "processed" / "train_2h.jsonl")
EVAL_PATH        = str(ROOT / "data" / "processed" / "eval_2h.jsonl")

STEPS_PER_CHUNK  = 25       # v2: smaller chunks = less crash window
TOTAL_TARGET     = 1701

LORA_R           = 64
LORA_ALPHA       = 128
LORA_DROPOUT     = 0.05
TARGET_MODULES   = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

BATCH_SIZE       = 1
GRAD_ACCUM       = 4        # v2: reduced from 8 → less VRAM spikes
LEARNING_RATE    = 5e-5
MAX_SEQ_LENGTH   = 512      # v2: reduced from 1024 → significant VRAM relief
WEIGHT_DECAY     = 0.01
MAX_GRAD_NORM    = 0.3
NUM_EPOCHS       = 189

MAX_GPU_TEMP     = 78  # Don't start if GPU hotter than this


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(ROOT / "training_log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


def get_gpu_temp():
    """Get GPU temperature via nvidia-smi. Returns int or None."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip())
    except:
        return None


def find_latest_checkpoint():
    ckpt_dir = Path(OUTPUT_DIR)
    if not ckpt_dir.exists():
        return 0, None
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
    state_path = Path(ckpt_path) / "trainer_state.json"
    if not state_path.exists():
        log(f"  [REPAIR] trainer_state.json missing at step {step}, creating...")
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
            "trial_params": None,
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        log(f"  [REPAIR] Created.")


def main():
    start = time.time()

    # ─── Temperature Guard ────────────────────────────
    temp = get_gpu_temp()
    if temp is not None:
        log(f"  GPU Temperature: {temp}°C")
        if temp > MAX_GPU_TEMP:
            log(f"  [WAIT] GPU too hot ({temp}°C > {MAX_GPU_TEMP}°C). Exiting for cooldown.")
            return 3  # Special code: "too hot, retry after cooldown"

    import torch

    if not torch.cuda.is_available():
        log("[FATAL] CUDA not available!")
        return 1

    # === Maximum Stability Settings for RTX 5070 ===
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False   # v2: disable TF32 tensor cores
    torch.backends.cudnn.allow_tf32 = False          # v2: disable TF32 in cuDNN
    # No memory fraction cap - checkpoint resume needs full VRAM
    torch.cuda.empty_cache()
    gc.collect()

    last_step, ckpt_path = find_latest_checkpoint()

    if last_step >= TOTAL_TARGET:
        log(f"[DONE] Training complete! {last_step}/{TOTAL_TARGET}")
        return 0

    remaining = TOTAL_TARGET - last_step
    chunk_steps = min(STEPS_PER_CHUNK, remaining)

    log("=" * 60)
    log(f"  CHUNK START: step {last_step} → {last_step + chunk_steps}")
    log(f"  Remaining: {remaining} steps | Target: {TOTAL_TARGET}")
    log(f"  GPU Temp: {temp}°C | Time: {datetime.now().strftime('%H:%M')}")
    log("=" * 60)

    if ckpt_path:
        ensure_trainer_state(ckpt_path, last_step)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset

        token = os.environ.get("HF_TOKEN")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # v2: fp16 instead of bf16
            bnb_4bit_use_double_quant=True,
        )

        log("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, token=token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        log("  Loading model (4-bit QLoRA)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # v2: fp16 for driver stability
            token=token,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"  Model loaded: {vram_used:.1f}/{vram_total:.1f} GB VRAM")

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"  LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        dataset = load_dataset(
            "json", data_files={"train": TRAIN_PATH, "eval": EVAL_PATH},
        )

        def formatting_func(example):
            return tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,
            )

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
            fp16=True,             # v2: fp16 instead of bf16 for stability
            bf16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},  # v2: more stable
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_steps=chunk_steps,
            save_total_limit=5,         # Keep more checkpoints for safety
            eval_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            max_steps=last_step + chunk_steps,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,   # No multiprocessing overhead on Windows
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            formatting_func=formatting_func,
            args=sft_config,
        )

        # Clear VRAM before training starts
        torch.cuda.empty_cache()
        gc.collect()

        log(f"  Training starting...")
        train_start = time.time()

        if ckpt_path:
            result = trainer.train(resume_from_checkpoint=ckpt_path)
        else:
            result = trainer.train()

        train_elapsed = time.time() - train_start
        log(f"  Chunk done! Step: {result.global_step}, Loss: {result.training_loss:.4f}")
        log(f"  Training time: {train_elapsed/60:.1f}min")

        # Clear VRAM after training
        torch.cuda.empty_cache()

        # Explicit save
        save_dir = Path(OUTPUT_DIR) / f"checkpoint-{result.global_step}"
        if not save_dir.exists():
            trainer.save_model(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            trainer.state.save_to_json(str(save_dir / "trainer_state.json"))
            log(f"  Checkpoint saved: {save_dir.name}")

        # Post-training temp check
        post_temp = get_gpu_temp()
        log(f"  Post-training GPU temp: {post_temp}°C")

        elapsed = time.time() - start
        log(f"  Total chunk time: {elapsed/60:.1f}min")

        if result.global_step >= TOTAL_TARGET:
            log("=" * 60)
            log("  TRAINING COMPLETE!")
            log("=" * 60)
            trainer.save_model(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            meta = {
                "base_model": MODEL_NAME,
                "total_steps": result.global_step,
                "training_loss": result.training_loss,
                "lora_r": LORA_R,
                "completed_at": datetime.now().isoformat(),
            }
            with open(Path(OUTPUT_DIR) / "training_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            return 0

        return 2  # More chunks needed

    except Exception as e:
        log(f"  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        elapsed = time.time() - start
        log(f"  Crashed after {elapsed/60:.1f}min")
        return 1

    finally:
        try:
            import torch
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass


if __name__ == "__main__":
    code = main()
    sys.exit(code)
