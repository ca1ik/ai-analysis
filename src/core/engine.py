"""
Core Chatbot Engine — Manages model loading, inference, and conversation state.
Optimized for RTX 5070 12GB VRAM with 4-bit quantization.
"""
import torch
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import model_config, MODELS_DIR


@dataclass
class ConversationMessage:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatbotInstance:
    bot_id: str
    name: str
    persona: str
    system_prompt: str
    model_path: Optional[str] = None
    conversation_history: list = field(default_factory=list)
    max_history: int = 20
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
    repetition_penalty: float = 1.15


class ChatbotEngine:
    """Core engine that loads models and runs inference."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._current_model_path: Optional[str] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._active_bots: dict[str, ChatbotInstance] = {}

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    def get_quantization_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    def load_model(self, model_path: Optional[str] = None):
        """Load base or fine-tuned model with 4-bit quantization."""
        target_path = model_path or model_config.base_model
        if self._current_model_path == target_path and self._model is not None:
            return  # Already loaded

        # Free previous model
        self.unload_model()

        quantization = self.get_quantization_config()

        self._tokenizer = AutoTokenizer.from_pretrained(
            target_path,
            trust_remote_code=model_config.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            target_path,
            quantization_config=quantization,
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch.float16,
        )
        self._model.eval()
        self._current_model_path = target_path

    def unload_model(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._current_model_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def register_bot(self, bot: ChatbotInstance):
        self._active_bots[bot.bot_id] = bot
        # Load model if bot has custom fine-tuned path
        model_path = bot.model_path or model_config.base_model
        if self._current_model_path != model_path:
            self.load_model(model_path)

    def get_bot(self, bot_id: str) -> Optional[ChatbotInstance]:
        return self._active_bots.get(bot_id)

    def remove_bot(self, bot_id: str):
        self._active_bots.pop(bot_id, None)

    def _build_prompt(self, bot: ChatbotInstance, user_message: str) -> list[dict]:
        messages = [{"role": "system", "content": bot.system_prompt}]
        # Add conversation history (windowed)
        history_window = bot.conversation_history[-(bot.max_history * 2):]
        for msg in history_window:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_message})
        return messages

    def generate(self, bot_id: str, user_message: str) -> str:
        bot = self._active_bots.get(bot_id)
        if bot is None:
            raise ValueError(f"Bot '{bot_id}' not registered")
        if self._model is None:
            self.load_model(bot.model_path)

        messages = self._build_prompt(bot, user_message)
        input_ids = self._tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=bot.max_new_tokens,
                temperature=bot.temperature,
                top_p=bot.top_p,
                repetition_penalty=bot.repetition_penalty,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only new tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Update conversation history
        bot.conversation_history.append(ConversationMessage(role="user", content=user_message))
        bot.conversation_history.append(ConversationMessage(role="assistant", content=response))

        return response

    def generate_stream(self, bot_id: str, user_message: str) -> Generator[str, None, None]:
        """Streaming generation for real-time chat UX."""
        bot = self._active_bots.get(bot_id)
        if bot is None:
            raise ValueError(f"Bot '{bot_id}' not registered")
        if self._model is None:
            self.load_model(bot.model_path)

        messages = self._build_prompt(bot, user_message)
        input_ids = self._tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self._device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": bot.max_new_tokens,
            "temperature": bot.temperature,
            "top_p": bot.top_p,
            "repetition_penalty": bot.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        full_response = []
        for text in streamer:
            full_response.append(text)
            yield text

        thread.join()

        # Update history
        response = "".join(full_response)
        bot.conversation_history.append(ConversationMessage(role="user", content=user_message))
        bot.conversation_history.append(ConversationMessage(role="assistant", content=response))

    def list_bots(self) -> list[dict]:
        return [
            {"bot_id": b.bot_id, "name": b.name, "persona": b.persona}
            for b in self._active_bots.values()
        ]


# Singleton engine
engine = ChatbotEngine()
