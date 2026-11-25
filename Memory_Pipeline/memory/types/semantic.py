from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from memory.base import BaseMemory, register_memory_type


@dataclass
class SemanticMemory(BaseMemory):
    concept: str = ""
    description: str = ""
    sources: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="semantic")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_text(self) -> str:
        source_text = ", ".join(self.sources)
        return (
            f"Concept: {self.concept}\n"
            f"Description: {self.description}\n"
            f"Sources: {source_text}"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticMemory":
        created_at = data.get("created_at", datetime.utcnow())
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            memory_type=data.get("memory_type", "semantic"),
            created_at=created_at,
            metadata=data.get("metadata", {}),
            concept=data.get("concept", ""),
            description=data.get("description", ""),
            sources=data.get("sources", []),
        )


register_memory_type("semantic", SemanticMemory)
