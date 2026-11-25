from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from memory.base import BaseMemory, register_memory_type


@dataclass
class PreferenceMemory(BaseMemory):
    subject: str = ""
    preference: str = ""
    rationale: str = ""
    scope: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="preference")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_text(self) -> str:
        return (
            f"Subject: {self.subject}\n"
            f"Preference: {self.preference}\n"
            f"Rationale: {self.rationale}\n"
            f"Scope: {self.scope}"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "PreferenceMemory":
        created_at = data.get("created_at", datetime.utcnow())
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            memory_type=data.get("memory_type", "preference"),
            created_at=created_at,
            metadata=data.get("metadata", {}),
            subject=data.get("subject", ""),
            preference=data.get("preference", ""),
            rationale=data.get("rationale", ""),
            scope=data.get("scope", ""),
        )


register_memory_type("preference", PreferenceMemory)
