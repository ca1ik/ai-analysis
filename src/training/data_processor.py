"""
Data Processor — Prepares training datasets for social good chatbot fine-tuning.
Handles JSONL, CSV, conversation formats. TR/EN bilingual support.
"""
import json
import csv
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import TRAINING_SETS_DIR, DATA_DIR


@dataclass
class DatasetStats:
    total_samples: int
    train_samples: int
    eval_samples: int
    avg_input_length: float
    avg_output_length: float
    languages: dict  # {"tr": count, "en": count}


class DataProcessor:
    """Processes and validates training data for chatbot fine-tuning."""

    SUPPORTED_FORMATS = {".jsonl", ".json", ".csv", ".txt"}

    def __init__(self):
        self.processed_dir = DATA_DIR / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_jsonl(self, path: Path) -> list[dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Line {line_num}: JSON parse error — {e}")
        return data

    def load_json(self, path: Path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        raise ValueError("JSON must be a list or contain a 'data' key")

    def load_csv(self, path: Path) -> list[dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data

    def load_data(self, path: str | Path) -> list[dict]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        if path.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}")

        loaders = {
            ".jsonl": self.load_jsonl,
            ".json": self.load_json,
            ".csv": self.load_csv,
        }
        return loaders[path.suffix](path)

    def normalize_to_chat_format(self, data: list[dict], system_prompt: str = "") -> list[dict]:
        """
        Normalize various input formats to HuggingFace chat format:
        [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]

        Supports input formats:
        - {"instruction": ..., "output": ...}
        - {"input": ..., "output": ...}
        - {"question": ..., "answer": ...}
        - {"messages": [...]}  (already in chat format)
        - {"prompt": ..., "completion": ...}
        """
        normalized = []
        for item in data:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if "messages" in item:
                # Already in chat format
                msgs = item["messages"]
                if system_prompt and msgs and msgs[0].get("role") != "system":
                    messages.extend(msgs)
                else:
                    messages = msgs
            elif "instruction" in item and "output" in item:
                user_content = item["instruction"]
                if item.get("input"):
                    user_content += f"\n\n{item['input']}"
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": item["output"]})
            elif "question" in item and "answer" in item:
                messages.append({"role": "user", "content": item["question"]})
                messages.append({"role": "assistant", "content": item["answer"]})
            elif "prompt" in item and "completion" in item:
                messages.append({"role": "user", "content": item["prompt"]})
                messages.append({"role": "assistant", "content": item["completion"]})
            elif "input" in item and "output" in item:
                messages.append({"role": "user", "content": item["input"]})
                messages.append({"role": "assistant", "content": item["output"]})
            else:
                continue  # Skip unrecognized format

            if len(messages) >= 2:
                normalized.append({"messages": messages})

        return normalized

    def validate_data(self, data: list[dict]) -> tuple[list[dict], list[str]]:
        """Validate normalized data. Returns (valid_data, warnings)."""
        valid = []
        warnings = []
        for i, item in enumerate(data):
            msgs = item.get("messages", [])
            if len(msgs) < 2:
                warnings.append(f"Sample {i}: Less than 2 messages, skipped")
                continue

            has_user = any(m["role"] == "user" for m in msgs)
            has_assistant = any(m["role"] == "assistant" for m in msgs)
            if not has_user or not has_assistant:
                warnings.append(f"Sample {i}: Missing user/assistant role, skipped")
                continue

            # Content length check
            for m in msgs:
                if len(m.get("content", "")) < 2:
                    warnings.append(f"Sample {i}: Very short content in {m['role']}")

            valid.append(item)

        return valid, warnings

    def split_data(
        self, data: list[dict], eval_ratio: float = 0.1, seed: int = 42
    ) -> tuple[list[dict], list[dict]]:
        random.seed(seed)
        shuffled = data.copy()
        random.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1 - eval_ratio)))
        return shuffled[:split_idx], shuffled[split_idx:]

    def compute_stats(self, train_data: list[dict], eval_data: list[dict]) -> DatasetStats:
        all_data = train_data + eval_data
        input_lengths = []
        output_lengths = []
        lang_count = {"tr": 0, "en": 0}

        turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        for item in all_data:
            for msg in item.get("messages", []):
                content = msg.get("content", "")
                if msg["role"] == "user":
                    input_lengths.append(len(content))
                elif msg["role"] == "assistant":
                    output_lengths.append(len(content))
                if any(c in turkish_chars for c in content):
                    lang_count["tr"] += 1
                else:
                    lang_count["en"] += 1

        return DatasetStats(
            total_samples=len(all_data),
            train_samples=len(train_data),
            eval_samples=len(eval_data),
            avg_input_length=sum(input_lengths) / max(len(input_lengths), 1),
            avg_output_length=sum(output_lengths) / max(len(output_lengths), 1),
            languages=lang_count,
        )

    def save_processed(
        self, train_data: list[dict], eval_data: list[dict], name: str
    ) -> tuple[Path, Path]:
        train_path = self.processed_dir / f"{name}_train.jsonl"
        eval_path = self.processed_dir / f"{name}_eval.jsonl"

        for path, data in [(train_path, train_data), (eval_path, eval_data)]:
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return train_path, eval_path

    def prepare_dataset(
        self,
        data_path: str | Path,
        system_prompt: str = "",
        dataset_name: str = "custom",
        eval_ratio: float = 0.1,
    ) -> tuple[Path, Path, DatasetStats]:
        """Full pipeline: load → normalize → validate → split → save."""
        raw_data = self.load_data(data_path)
        normalized = self.normalize_to_chat_format(raw_data, system_prompt)
        valid_data, warnings = self.validate_data(normalized)

        if warnings:
            print(f"[DataProcessor] {len(warnings)} warnings:")
            for w in warnings[:10]:
                print(f"  - {w}")

        if not valid_data:
            raise ValueError("No valid training samples after processing")

        train_data, eval_data = self.split_data(valid_data, eval_ratio)
        train_path, eval_path = self.save_processed(train_data, eval_data, dataset_name)
        stats = self.compute_stats(train_data, eval_data)

        print(f"[DataProcessor] Dataset '{dataset_name}' ready:")
        print(f"  Train: {stats.train_samples} | Eval: {stats.eval_samples}")
        print(f"  Avg input: {stats.avg_input_length:.0f} chars | Avg output: {stats.avg_output_length:.0f} chars")
        print(f"  Languages: {stats.languages}")

        return train_path, eval_path, stats
