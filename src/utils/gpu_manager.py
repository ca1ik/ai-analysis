"""
GPU Manager — Monitors GPU resources and manages VRAM allocation.
"""
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    name: str
    total_vram_gb: float
    used_vram_gb: float
    free_vram_gb: float
    utilization_percent: float
    temperature: Optional[float] = None


class GPUManager:
    """Monitors and manages GPU resources for training and inference."""

    def __init__(self):
        self._cuda_available = torch.cuda.is_available()

    @property
    def is_available(self) -> bool:
        return self._cuda_available

    def get_info(self) -> Optional[GPUInfo]:
        if not self._cuda_available:
            return None

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total = props.total_mem / (1024**3)
        used = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        free = total - reserved

        return GPUInfo(
            name=props.name,
            total_vram_gb=round(total, 2),
            used_vram_gb=round(used, 2),
            free_vram_gb=round(free, 2),
            utilization_percent=round(used / total * 100, 1) if total > 0 else 0,
        )

    def can_train(self, model_size_b: float = 8.0) -> tuple[bool, str]:
        """Check if GPU can handle training for given model size."""
        if not self._cuda_available:
            return False, "CUDA is not available"

        info = self.get_info()
        if info is None:
            return False, "Cannot retrieve GPU info"

        # 4-bit QLoRA memory requirements (approximate)
        # 8B model: ~5-6GB base + ~2-3GB for training overhead
        required_gb = model_size_b * 0.75  # 4-bit ≈ 0.5 bytes/param + overhead
        if info.free_vram_gb < required_gb:
            return False, (
                f"Insufficient VRAM: {info.free_vram_gb:.1f}GB free, "
                f"~{required_gb:.1f}GB required for {model_size_b}B model"
            )

        return True, f"OK: {info.free_vram_gb:.1f}GB free, ~{required_gb:.1f}GB required"

    def can_infer(self) -> tuple[bool, str]:
        """Check if GPU can handle inference."""
        if not self._cuda_available:
            return False, "CUDA not available — will use CPU (slower)"

        info = self.get_info()
        if info is None:
            return False, "Cannot retrieve GPU info"

        # 4-bit inference: ~5GB for 8B model
        if info.free_vram_gb < 5.0:
            return False, f"Low VRAM for inference: {info.free_vram_gb:.1f}GB free"

        return True, f"OK: {info.free_vram_gb:.1f}GB free"

    def clear_cache(self):
        if self._cuda_available:
            torch.cuda.empty_cache()

    def get_status_report(self) -> dict:
        info = self.get_info()
        if info is None:
            return {"cuda": False, "message": "No CUDA GPU detected"}

        can_train, train_msg = self.can_train()
        can_infer, infer_msg = self.can_infer()

        return {
            "cuda": True,
            "gpu_name": info.name,
            "total_vram_gb": info.total_vram_gb,
            "used_vram_gb": info.used_vram_gb,
            "free_vram_gb": info.free_vram_gb,
            "utilization_percent": info.utilization_percent,
            "can_train": can_train,
            "train_message": train_msg,
            "can_infer": can_infer,
            "infer_message": infer_msg,
        }


# Singleton
gpu_manager = GPUManager()
