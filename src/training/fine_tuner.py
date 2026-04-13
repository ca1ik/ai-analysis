"""
QLoRA Fine-Tuner — Trains LLaMA 3 8B with 4-bit quantization on RTX 5070.
Optimized for 12GB VRAM with gradient checkpointing and paged optimizer.
"""
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import model_config, training_config, MODELS_DIR


@dataclass
class TrainingResult:
    success: bool
    model_path: Optional[str]
    training_loss: float
    eval_loss: Optional[float]
    total_steps: int
    duration_seconds: float
    error: Optional[str] = None


class TrainingProgressCallback(TrainerCallback):
    """Reports training progress to the platform."""

    def __init__(self, progress_fn: Optional[Callable] = None):
        self.progress_fn = progress_fn
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        self._report(0, 0, "Training started")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            loss = logs.get("loss", 0)
            progress = state.global_step / state.max_steps if state.max_steps else 0
            self._report(progress, loss, f"Step {state.global_step}/{state.max_steps}")

    def on_train_end(self, args, state, control, **kwargs):
        self._report(1.0, 0, "Training complete")

    def _report(self, progress: float, loss: float, message: str):
        if self.progress_fn:
            self.progress_fn({
                "progress": progress,
                "loss": loss,
                "message": message,
            })
        print(f"[Training] {progress:.1%} | {message} | Loss: {loss:.4f}")


class QLoRAFineTuner:
    """QLoRA fine-tuning engine for LLaMA 3 8B on RTX 5070 12GB."""

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _load_base_model(self):
        """Load base model with 4-bit quantization."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        token = model_config.hf_token

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_config.base_model,
            trust_remote_code=model_config.trust_remote_code,
            token=token,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch.float16,
            token=token,
        )
        self._model = prepare_model_for_kbit_training(
            self._model, use_gradient_checkpointing=True
        )

    def _apply_lora(self):
        """Apply LoRA adapter to the model."""
        lora_config = LoraConfig(
            r=training_config.lora_r,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            target_modules=training_config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self._model = get_peft_model(self._model, lora_config)
        self._model.print_trainable_parameters()

    def _load_datasets(self, train_path: str, eval_path: str):
        """Load processed JSONL datasets."""
        dataset = load_dataset("json", data_files={
            "train": train_path,
            "eval": eval_path,
        })
        return dataset["train"], dataset["eval"]

    def _formatting_func(self, examples):
        """Format dataset examples to chat template strings."""
        texts = []
        for messages in examples["messages"]:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return texts

    def train(
        self,
        train_path: str,
        eval_path: str,
        output_name: str,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        progress_fn: Optional[Callable] = None,
    ) -> TrainingResult:
        """Execute QLoRA fine-tuning pipeline."""
        start_time = datetime.now()
        output_dir = str(MODELS_DIR / output_name)

        try:
            print(f"[FineTuner] Loading base model: {model_config.base_model}")
            self._load_base_model()

            print("[FineTuner] Applying LoRA adapters...")
            self._apply_lora()

            print("[FineTuner] Loading datasets...")
            train_dataset, eval_dataset = self._load_datasets(train_path, eval_path)
            print(f"[FineTuner] Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

            # Training arguments — RTX 5070 optimized
            sft_config = SFTConfig(
                output_dir=output_dir,
                num_train_epochs=num_epochs or training_config.num_epochs,
                per_device_train_batch_size=batch_size or training_config.batch_size,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                learning_rate=learning_rate or training_config.learning_rate,
                warmup_ratio=training_config.warmup_ratio,
                weight_decay=training_config.weight_decay,
                max_grad_norm=training_config.max_grad_norm,
                lr_scheduler_type=training_config.lr_scheduler_type,
                fp16=training_config.fp16,
                bf16=training_config.bf16,
                gradient_checkpointing=training_config.gradient_checkpointing,
                optim=training_config.optim,
                logging_steps=training_config.logging_steps,
                save_steps=training_config.save_steps,
                eval_strategy="steps",
                eval_steps=training_config.eval_steps,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="none",
                max_seq_length=model_config.max_seq_length,
                dataset_text_field=None,  # Using formatting_func
            )

            # Trainer
            trainer = SFTTrainer(
                model=self._model,
                tokenizer=self._tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                formatting_func=self._formatting_func,
                args=sft_config,
                callbacks=[TrainingProgressCallback(progress_fn)],
            )

            print("[FineTuner] Starting training...")
            train_result = trainer.train()

            # Save LoRA adapter
            print(f"[FineTuner] Saving model to {output_dir}")
            trainer.save_model(output_dir)
            self._tokenizer.save_pretrained(output_dir)

            # Save training metadata
            metadata = {
                "base_model": model_config.base_model,
                "output_name": output_name,
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "num_epochs": num_epochs or training_config.num_epochs,
                "lora_r": training_config.lora_r,
                "lora_alpha": training_config.lora_alpha,
                "training_loss": train_result.training_loss,
                "total_steps": train_result.global_step,
                "created_at": datetime.now().isoformat(),
            }
            with open(Path(output_dir) / "training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            duration = (datetime.now() - start_time).total_seconds()

            # Evaluate
            eval_results = trainer.evaluate()
            eval_loss = eval_results.get("eval_loss")

            return TrainingResult(
                success=True,
                model_path=output_dir,
                training_loss=train_result.training_loss,
                eval_loss=eval_loss,
                total_steps=train_result.global_step,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TrainingResult(
                success=False,
                model_path=None,
                training_loss=0,
                eval_loss=None,
                total_steps=0,
                duration_seconds=duration,
                error=str(e),
            )
        finally:
            self._cleanup()

    def _cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def merge_and_export(self, adapter_path: str, output_path: str):
        """Merge LoRA adapter with base model for standalone deployment."""
        from peft import PeftModel

        print(f"[FineTuner] Merging adapter: {adapter_path}")

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=model_config.trust_remote_code,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()

        print(f"[FineTuner] Saving merged model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        del model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[FineTuner] Merge complete")
