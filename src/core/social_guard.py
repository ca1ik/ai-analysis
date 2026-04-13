"""
Social Good Guardrails — Content safety, bias detection, and ethical compliance.
Filters both user input and model output.
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import social_guard_config


class RiskLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


@dataclass
class SafetyResult:
    is_safe: bool
    risk_level: RiskLevel
    flagged_categories: list[str]
    modified_content: Optional[str] = None
    disclaimer: Optional[str] = None
    reason: Optional[str] = None


class SocialGuard:
    """Multi-layer content safety and ethical compliance system."""

    def __init__(self):
        self.config = social_guard_config
        self._build_patterns()

    def _build_patterns(self):
        # Hate speech / discrimination patterns (TR + EN)
        self._hate_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r'\b(kill|murder|attack)\s+(all|every)\s+\w+',
                r'\b(öldür|yok et|imha et)\s+.*(hepsi|tümü|herkes)',
                r'\b(inferior|superior)\s+race',
                r'\b(üstün|aşağı)\s+ırk',
            ]
        ]
        # Self-harm / crisis patterns
        self._crisis_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r'\b(intihar|kendime zarar|yaşamak istemiyorum)',
                r'\b(suicide|self.?harm|end my life|kill myself)',
                r'\b(want to die|don\'t want to live)',
            ]
        ]
        # Medical advice patterns
        self._medical_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r'\b(ilaç|tedavi|tanı|hastalık|semptom)',
                r'\b(medication|treatment|diagnosis|prescription|symptom)',
                r'\b(doktor|hastane|ameliyat)',
            ]
        ]
        # Legal advice patterns
        self._legal_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r'\b(dava|mahkeme|avukat|hukuk|ceza)',
                r'\b(lawsuit|court|attorney|legal\s+advice|sue)',
            ]
        ]
        # Misinformation high-risk topics
        self._misinfo_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r'\b(aşı|vaccine)\s+.*(zehir|poison|öldür|kill|tehlike|danger)',
                r'\b(düz\s+dünya|flat\s+earth)',
                r'\b(5g)\s+.*(virus|virüs|corona|covid)',
            ]
        ]

    def check_input(self, text: str) -> SafetyResult:
        """Check user input for safety before sending to model."""
        flagged = []

        # Crisis detection — highest priority, don't block but add resources
        for pattern in self._crisis_patterns:
            if pattern.search(text):
                return SafetyResult(
                    is_safe=True,  # Don't block, but flag
                    risk_level=RiskLevel.HIGH,
                    flagged_categories=["crisis"],
                    disclaimer=self.config.required_disclaimers["crisis"],
                    reason="Kriz tespit edildi — destek bilgisi eklendi"
                )

        # Hate speech — block
        for pattern in self._hate_patterns:
            if pattern.search(text):
                flagged.append("hate_speech")

        if flagged:
            return SafetyResult(
                is_safe=False,
                risk_level=RiskLevel.BLOCKED,
                flagged_categories=flagged,
                reason="Nefret söylemi / şiddet içerikli ifade tespit edildi"
            )

        return SafetyResult(
            is_safe=True,
            risk_level=RiskLevel.SAFE,
            flagged_categories=[]
        )

    def check_output(self, text: str, input_text: str = "") -> SafetyResult:
        """Check model output for safety, add disclaimers where needed."""
        flagged = []
        disclaimers = []

        # Hate speech in output
        for pattern in self._hate_patterns:
            if pattern.search(text):
                flagged.append("hate_speech")

        if flagged:
            return SafetyResult(
                is_safe=False,
                risk_level=RiskLevel.BLOCKED,
                flagged_categories=flagged,
                reason="Model çıktısında uygunsuz içerik tespit edildi"
            )

        # Medical disclaimer
        combined = f"{input_text} {text}"
        for pattern in self._medical_patterns:
            if pattern.search(combined):
                disclaimers.append(self.config.required_disclaimers["medical"])
                flagged.append("medical_advice")
                break

        # Legal disclaimer
        for pattern in self._legal_patterns:
            if pattern.search(combined):
                disclaimers.append(self.config.required_disclaimers["legal"])
                flagged.append("legal_advice")
                break

        # Misinformation check
        for pattern in self._misinfo_patterns:
            if pattern.search(text):
                flagged.append("misinformation")
                return SafetyResult(
                    is_safe=False,
                    risk_level=RiskLevel.BLOCKED,
                    flagged_categories=flagged,
                    reason="Yanlış bilgi yayma riski tespit edildi"
                )

        # Length guard
        if len(text) > self.config.max_response_length * 4:  # char estimate
            text = text[:self.config.max_response_length * 4] + "..."

        # Build final output with disclaimers
        modified = text
        if disclaimers:
            disclaimer_block = "\n\n---\n" + "\n".join(f"⚠️ {d}" for d in disclaimers)
            modified = text + disclaimer_block

        risk = RiskLevel.LOW if flagged else RiskLevel.SAFE
        return SafetyResult(
            is_safe=True,
            risk_level=risk,
            flagged_categories=flagged,
            modified_content=modified if modified != text else None,
            disclaimer="\n".join(disclaimers) if disclaimers else None,
        )

    def detect_language(self, text: str) -> str:
        """Simple heuristic language detection for TR/EN routing."""
        turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        turkish_count = sum(1 for c in text if c in turkish_chars)
        # Also check common Turkish words
        tr_words = ["bir", "ve", "bu", "için", "ile", "olan", "ben", "sen", "nasıl", "merhaba"]
        word_set = set(text.lower().split())
        tr_word_count = sum(1 for w in tr_words if w in word_set)

        if turkish_count > 0 or tr_word_count >= 2:
            return "tr"
        return "en"


# Singleton
social_guard = SocialGuard()
