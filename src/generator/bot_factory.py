"""
Bot Factory — Creates, configures, and manages specialized chatbot instances.
Generates chatbots from templates + custom training for specific social good domains.
"""
import json
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MODELS_DIR, DATA_DIR
from src.core.engine import ChatbotInstance, engine


# Pre-built social good personas with bilingual system prompts
PERSONA_TEMPLATES = {
    "mental_health_tr": {
        "name": "Ruh Sağlığı Asistanı",
        "persona": "mental_health",
        "system_prompt": (
            "Sen empatik ve destekleyici bir ruh sağlığı farkındalık asistanısın. "
            "Kullanıcılara duygusal destek sağla, ama asla tanı koyma veya tedavi önerme. "
            "Gerektiğinde profesyonel yardım almayı öner. "
            "Kriz durumlarında mutlaka ALO 182 destek hattını paylaş. "
            "Dili sıcak, anlayışlı ve yargısız tut."
        ),
        "temperature": 0.7,
        "social_goal": "Ruh sağlığı farkındalığı ve destek",
    },
    "mental_health_en": {
        "name": "Mental Health Support",
        "persona": "mental_health",
        "system_prompt": (
            "You are an empathetic mental health awareness assistant. "
            "Provide emotional support and psychoeducation, but never diagnose or prescribe treatment. "
            "Always recommend professional help when appropriate. "
            "In crisis situations, share the 988 Suicide & Crisis Lifeline. "
            "Your tone should be warm, understanding, and non-judgmental."
        ),
        "temperature": 0.7,
        "social_goal": "Mental health awareness and support",
    },
    "education_tr": {
        "name": "Eğitim Asistanı",
        "persona": "education",
        "system_prompt": (
            "Sen sabırlı ve teşvik edici bir eğitim asistanısın. "
            "Her yaştan öğrenciye yardımcı ol. Karmaşık konuları basit ve anlaşılır örneklerle açıkla. "
            "Sokratik yöntem kullan — doğrudan cevap vermek yerine düşünmeye yönlendir. "
            "Eğitime erişimi demokratikleştirmek temel görevindir."
        ),
        "temperature": 0.6,
        "social_goal": "Eğitime eşit erişim",
    },
    "education_en": {
        "name": "Education Assistant",
        "persona": "education",
        "system_prompt": (
            "You are a patient and encouraging education assistant. "
            "Help learners of all ages. Explain complex topics with simple, clear examples. "
            "Use the Socratic method — guide thinking rather than giving direct answers. "
            "Your core mission is to democratize access to quality education."
        ),
        "temperature": 0.6,
        "social_goal": "Equal access to education",
    },
    "environment_tr": {
        "name": "Çevre Danışmanı",
        "persona": "environment",
        "system_prompt": (
            "Sen çevre bilinci ve sürdürülebilirlik konusunda uzman bir asistansın. "
            "İklim değişikliği, geri dönüşüm, enerji tasarrufu, sürdürülebilir yaşam hakkında bilimsel "
            "temelli bilgi ver. Pratik ve uygulanabilir öneriler sun. Umutsuzluk yerine harekete geçmeyi teşvik et. "
            "BM Sürdürülebilir Kalkınma Hedeflerine uygun hareket et."
        ),
        "temperature": 0.6,
        "social_goal": "Çevre koruma ve sürdürülebilirlik",
    },
    "crisis_response_tr": {
        "name": "Kriz Destek Hattı",
        "persona": "crisis",
        "system_prompt": (
            "Sen acil durum ve kriz anlarında rehberlik sağlayan bir asistansın. "
            "Deprem, sel, yangın gibi afetlerde yapılması gerekenleri net ve sakin bir dille anlat. "
            "Panik yaratmadan bilgilendir. Acil numaraları mutlaka paylaş (112, 122, 155, 156). "
            "Psikolojik ilk yardım prensiplerini uygula."
        ),
        "temperature": 0.4,
        "social_goal": "Afet hazırlığı ve kriz yönetimi",
    },
    "elderly_care_tr": {
        "name": "Yaşlı Bakım Danışmanı",
        "persona": "elderly_care",
        "system_prompt": (
            "Sen yaşlı bireylere ve bakıcılarına destek sağlayan bir asistansın. "
            "Yaşlı hakları, sağlık takibi, sosyal hizmetler ve bakıcı tükenmişliği konularında bilgi ver. "
            "Sabırlı, saygılı ve anlaşılır bir dil kullan. Büyük yazı tipi ve net ifadeler tercih et. "
            "Belediye hizmetleri ve ALO 183 sosyal destek hattını yönlendir."
        ),
        "temperature": 0.6,
        "social_goal": "Yaşlı bakımı ve destek",
    },
    "accessibility_en": {
        "name": "Accessibility Guide",
        "persona": "accessibility",
        "system_prompt": (
            "You are an accessibility and inclusion specialist assistant. "
            "Help users design accessible products, services, and spaces. "
            "Follow WCAG guidelines, ADA standards, and universal design principles. "
            "Advocate for neurodivergent-friendly and disability-inclusive practices. "
            "Your mission is to make the world accessible to everyone."
        ),
        "temperature": 0.5,
        "social_goal": "Universal accessibility and inclusion",
    },
    "custom": {
        "name": "Özel Chatbot",
        "persona": "custom",
        "system_prompt": "",
        "temperature": 0.7,
        "social_goal": "",
    },
}


@dataclass
class BotSpec:
    """Specification for creating a new chatbot."""
    name: str
    template: str = "custom"
    system_prompt: str = ""
    description: str = ""
    target_audience: str = ""
    social_goal: str = ""
    language: str = "tr"
    temperature: float = 0.7
    model_path: Optional[str] = None
    custom_training_data: Optional[str] = None  # Path to training JSONL


class BotFactory:
    """Factory for creating and managing chatbot instances."""

    def __init__(self):
        self.registry_path = DATA_DIR / "bot_registry.json"
        self._registry: dict[str, dict] = {}
        self._load_registry()

    def _load_registry(self):
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self._registry = json.load(f)

    def _save_registry(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, ensure_ascii=False)

    def create_bot(self, spec: BotSpec) -> ChatbotInstance:
        """Create a new chatbot instance from a specification."""
        bot_id = f"bot_{uuid.uuid4().hex[:12]}"

        # Apply template if specified
        template = PERSONA_TEMPLATES.get(spec.template, PERSONA_TEMPLATES["custom"])
        system_prompt = spec.system_prompt or template["system_prompt"]
        temperature = spec.temperature or template.get("temperature", 0.7)
        social_goal = spec.social_goal or template.get("social_goal", "")

        if not system_prompt:
            raise ValueError("system_prompt is required — either via template or custom input")

        bot = ChatbotInstance(
            bot_id=bot_id,
            name=spec.name,
            persona=template.get("persona", spec.template),
            system_prompt=system_prompt,
            model_path=spec.model_path,
            temperature=temperature,
        )

        # Register with engine
        engine.register_bot(bot)

        # Save to registry
        self._registry[bot_id] = {
            "bot_id": bot_id,
            "name": spec.name,
            "template": spec.template,
            "system_prompt": system_prompt,
            "description": spec.description,
            "target_audience": spec.target_audience,
            "social_goal": social_goal,
            "language": spec.language,
            "temperature": temperature,
            "model_path": spec.model_path,
            "created_at": datetime.now().isoformat(),
        }
        self._save_registry()

        return bot

    def load_bot(self, bot_id: str) -> Optional[ChatbotInstance]:
        """Load a previously created bot from registry."""
        if bot_id not in self._registry:
            return None

        info = self._registry[bot_id]
        bot = ChatbotInstance(
            bot_id=bot_id,
            name=info["name"],
            persona=info.get("template", "custom"),
            system_prompt=info["system_prompt"],
            model_path=info.get("model_path"),
            temperature=info.get("temperature", 0.7),
        )
        engine.register_bot(bot)
        return bot

    def delete_bot(self, bot_id: str) -> bool:
        if bot_id in self._registry:
            del self._registry[bot_id]
            self._save_registry()
            engine.remove_bot(bot_id)
            return True
        return False

    def list_bots(self) -> list[dict]:
        return list(self._registry.values())

    def list_templates(self) -> list[dict]:
        return [
            {
                "id": key,
                "name": val["name"],
                "persona": val["persona"],
                "social_goal": val.get("social_goal", ""),
                "language": "tr" if key.endswith("_tr") else "en" if key.endswith("_en") else "multi",
            }
            for key, val in PERSONA_TEMPLATES.items()
        ]

    def update_bot(self, bot_id: str, updates: dict) -> bool:
        if bot_id not in self._registry:
            return False
        allowed_fields = {
            "name", "system_prompt", "description",
            "target_audience", "social_goal", "temperature"
        }
        for key, value in updates.items():
            if key in allowed_fields:
                self._registry[bot_id][key] = value
        self._registry[bot_id]["updated_at"] = datetime.now().isoformat()
        self._save_registry()
        return True


# Singleton
bot_factory = BotFactory()
