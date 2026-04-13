"""
Training Pipeline Orchestrator — Coordinates data processing, training, and evaluation.
Single entry point for the entire training workflow.
"""
import json
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MODELS_DIR, TRAINING_SETS_DIR
from src.training.data_processor import DataProcessor, DatasetStats
from src.training.fine_tuner import QLoRAFineTuner, TrainingResult


@dataclass
class PipelineConfig:
    dataset_path: str
    bot_name: str
    system_prompt: str
    description: str = ""
    target_audience: str = ""
    social_goal: str = ""
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    eval_ratio: float = 0.1


@dataclass
class PipelineResult:
    success: bool
    bot_name: str
    model_path: Optional[str]
    dataset_stats: Optional[DatasetStats]
    training_result: Optional[TrainingResult]
    error: Optional[str] = None


class TrainingPipeline:
    """Orchestrates the full training pipeline from raw data to fine-tuned model."""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.fine_tuner = QLoRAFineTuner()
        self._training_history: list[dict] = []
        self._load_history()

    def _load_history(self):
        history_path = MODELS_DIR / "training_history.json"
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                self._training_history = json.load(f)

    def _save_history(self, result: PipelineResult):
        entry = {
            "bot_name": result.bot_name,
            "model_path": result.model_path,
            "success": result.success,
            "timestamp": datetime.now().isoformat(),
            "dataset_stats": asdict(result.dataset_stats) if result.dataset_stats else None,
            "training_loss": result.training_result.training_loss if result.training_result else None,
            "eval_loss": result.training_result.eval_loss if result.training_result else None,
            "error": result.error,
        }
        self._training_history.append(entry)
        history_path = MODELS_DIR / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self._training_history, f, indent=2, ensure_ascii=False)

    def run(
        self,
        config: PipelineConfig,
        progress_fn: Optional[Callable] = None,
    ) -> PipelineResult:
        """Execute the full training pipeline."""
        print(f"\n{'='*60}")
        print(f"  TRAINING PIPELINE: {config.bot_name}")
        print(f"  Social Goal: {config.social_goal}")
        print(f"{'='*60}\n")

        try:
            # Phase 1: Data Processing
            print("[Pipeline] Phase 1: Processing training data...")
            if progress_fn:
                progress_fn({"phase": "data_processing", "progress": 0})

            train_path, eval_path, stats = self.data_processor.prepare_dataset(
                data_path=config.dataset_path,
                system_prompt=config.system_prompt,
                dataset_name=config.bot_name,
                eval_ratio=config.eval_ratio,
            )

            if stats.total_samples < 10:
                raise ValueError(
                    f"Minimum 10 eğitim örneği gerekli, {stats.total_samples} bulundu. "
                    "Daha fazla veri ekleyin."
                )

            # Phase 2: Fine-Tuning
            print("[Pipeline] Phase 2: QLoRA Fine-Tuning...")
            if progress_fn:
                progress_fn({"phase": "training", "progress": 0})

            training_result = self.fine_tuner.train(
                train_path=str(train_path),
                eval_path=str(eval_path),
                output_name=config.bot_name,
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                progress_fn=progress_fn,
            )

            if not training_result.success:
                raise RuntimeError(f"Training failed: {training_result.error}")

            # Phase 3: Validation
            print("[Pipeline] Phase 3: Validation...")
            if progress_fn:
                progress_fn({"phase": "validation", "progress": 0.9})

            # Save bot configuration alongside model
            bot_config = {
                "bot_name": config.bot_name,
                "system_prompt": config.system_prompt,
                "description": config.description,
                "target_audience": config.target_audience,
                "social_goal": config.social_goal,
                "model_path": training_result.model_path,
                "created_at": datetime.now().isoformat(),
            }
            config_path = Path(training_result.model_path) / "bot_config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(bot_config, f, indent=2, ensure_ascii=False)

            result = PipelineResult(
                success=True,
                bot_name=config.bot_name,
                model_path=training_result.model_path,
                dataset_stats=stats,
                training_result=training_result,
            )

            print(f"\n[Pipeline] ✅ Training complete!")
            print(f"  Model: {training_result.model_path}")
            print(f"  Loss: {training_result.training_loss:.4f}")
            print(f"  Duration: {training_result.duration_seconds:.0f}s")

        except Exception as e:
            result = PipelineResult(
                success=False,
                bot_name=config.bot_name,
                model_path=None,
                dataset_stats=None,
                training_result=None,
                error=str(e),
            )
            print(f"\n[Pipeline] ❌ Failed: {e}")

        self._save_history(result)
        return result

    def list_trained_models(self) -> list[dict]:
        """List all trained models with their metadata."""
        models = []
        if not MODELS_DIR.exists():
            return models

        for model_dir in MODELS_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            config_path = model_dir / "bot_config.json"
            metadata_path = model_dir / "training_metadata.json"

            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    bot_config = json.load(f)
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    bot_config["training_metadata"] = metadata
                models.append(bot_config)

        return models

    def get_training_history(self) -> list[dict]:
        return self._training_history
