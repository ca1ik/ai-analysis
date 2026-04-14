"""
3-Hour Training Launcher — Social Good Chatbot Platform
Tek komutla: python train_3h.py
HF_TOKEN environment variable gerekli.
"""
import os
import sys

# Fix Windows Turkish encoding (cp1254) → force UTF-8
os.environ["PYTHONUTF8"] = "1"

import json
import time
from pathlib import Path
from datetime import datetime

# Ensure project root in path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import model_config, training_config, TRAINING_SETS_DIR, MODELS_DIR, DATA_DIR


def check_prerequisites():
    """Tüm gereksinimleri kontrol et."""
    print("=" * 60)
    print("  SOCIAL GOOD CHATBOT — 3-HOUR TRAINING SESSION")
    print("=" * 60)
    print()

    # 1. HF_TOKEN
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[FAIL] HF_TOKEN environment variable bulunamadı!")
        print("       Çözüm: setx HF_TOKEN \"hf_xxx...\"")
        print("       Sonra terminali kapat-aç.")
        return False
    print(f"[OK] HF_TOKEN: {token[:8]}...{token[-4:]}")

    # 2. CUDA
    import torch
    if not torch.cuda.is_available():
        print("[FAIL] CUDA kullanılamıyor!")
        return False
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[OK] GPU: {gpu} ({vram:.1f}GB VRAM)")

    # 3. Training data
    tr_data = TRAINING_SETS_DIR / "social_good_base_tr.jsonl"
    en_data = TRAINING_SETS_DIR / "social_good_base_en.jsonl"
    if not tr_data.exists() or not en_data.exists():
        print("[FAIL] Eğitim verileri bulunamadı!")
        return False

    tr_count = sum(1 for _ in open(tr_data, "r", encoding="utf-8") if _.strip())
    en_count = sum(1 for _ in open(en_data, "r", encoding="utf-8") if _.strip())
    print(f"[OK] Training data: {tr_count} TR + {en_count} EN = {tr_count + en_count} samples")

    # 4. Model access test
    print(f"[..] Base model: {model_config.base_model}")
    print()
    return True


def merge_training_data():
    """TR ve EN eğitim verilerini birleştir."""
    tr_path = TRAINING_SETS_DIR / "social_good_base_tr.jsonl"
    en_path = TRAINING_SETS_DIR / "social_good_base_en.jsonl"
    merged_path = TRAINING_SETS_DIR / "social_good_merged.jsonl"

    all_data = []
    for path in [tr_path, en_path]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(line)

    # Shuffle for better training
    import random
    random.seed(42)
    random.shuffle(all_data)

    with open(merged_path, "w", encoding="utf-8") as f:
        for line in all_data:
            f.write(line + "\n")

    print(f"[OK] Merged dataset: {len(all_data)} samples → {merged_path.name}")
    return merged_path


def run_training():
    """3 saatlik QLoRA eğitimini başlat."""
    from src.training.data_processor import DataProcessor
    from src.training.fine_tuner import QLoRAFineTuner

    # Phase 1: Merge & Process Data
    print("\n" + "─" * 60)
    print("  PHASE 1: Data Processing")
    print("─" * 60)

    merged_path = merge_training_data()
    processor = DataProcessor()

    system_prompt = (
        "Sen topluma faydalı bir yapay zeka asistanısın. "
        "Empati, doğruluk ve sosyal fayda önceliklerindir. "
        "Türkçe ve İngilizce bilirsin. Her zaman yardımsever, doğru bilgi veren, "
        "kriz anlarında destek hattlarını paylaşan, "
        "önyargısız ve kapsayıcı bir dil kullanan bir asistansın."
    )

    train_path, eval_path, stats = processor.prepare_dataset(
        data_path=merged_path,
        system_prompt=system_prompt,
        dataset_name="social_good_3h",
        eval_ratio=0.1,
    )

    print(f"\n  Train: {stats.train_samples} | Eval: {stats.eval_samples}")
    print(f"  Languages: {stats.languages}")

    # Phase 2: QLoRA Fine-Tuning
    print("\n" + "─" * 60)
    print("  PHASE 2: QLoRA Fine-Tuning (3-hour session)")
    print("─" * 60)
    print(f"  Model: {model_config.base_model}")
    print(f"  LoRA r={training_config.lora_r}, alpha={training_config.lora_alpha}")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Batch: {training_config.batch_size} × {training_config.gradient_accumulation_steps} accum = effective {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print(f"  LR: {training_config.learning_rate}")
    print()

    start_time = time.time()
    tuner = QLoRAFineTuner()

    result = tuner.train(
        train_path=str(train_path),
        eval_path=str(eval_path),
        output_name="social_good_v1",
        num_epochs=training_config.num_epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
    )

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    # Phase 3: Results
    print("\n" + "─" * 60)
    print("  PHASE 3: Results")
    print("─" * 60)

    if result.success:
        print(f"  ✅ Training COMPLETE")
        print(f"  Model saved: {result.model_path}")
        print(f"  Training loss: {result.training_loss:.4f}")
        print(f"  Eval loss: {result.eval_loss:.4f}" if result.eval_loss else "")
        print(f"  Total steps: {result.total_steps}")
        print(f"  Duration: {hours}h {minutes}m")
        print()
        print("  Sonraki adım:")
        print("    python -m uvicorn src.main:app --reload --port 8000")
        print("    http://localhost:8000/docs → Chat endpoint ile test et")
    else:
        print(f"  ❌ Training FAILED: {result.error}")
        print(f"  Duration: {hours}h {minutes}m")

    return result


if __name__ == "__main__":
    if not check_prerequisites():
        print("\n[!] Eksikler tamamlanmadan training başlatılamaz.")
        sys.exit(1)

    print("\n[START] Training başlıyor...")
    print("[INFO] İptal etmek için: Ctrl+C\n")

    try:
        result = run_training()
        sys.exit(0 if result.success else 1)
    except KeyboardInterrupt:
        print("\n\n[STOP] Training kullanıcı tarafından durduruldu.")
        print("[INFO] Son checkpoint data/models/ altında kayıtlıdır.")
        sys.exit(0)
