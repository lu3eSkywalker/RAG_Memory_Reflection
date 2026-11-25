from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

from memory.base import BaseMemory, register_memory_type


@dataclass
class EntityMemory(BaseMemory):
    entity_type: str = ""
    name: str = ""
    attributes: Dict = field(default_factory=dict)
    description: str = ""
    context: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="entity")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_text(self) -> str:
        attribute_text = "; ".join(
            f"{key}: {value}" for key, value in self.attributes.items()
        )
        return (
            f"Entity type: {self.entity_type}\n"
            f"Name: {self.name}\n"
            f"Attributes: {attribute_text}\n"
            f"Description: {self.description}\n"
            f"Context: {self.context}"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "EntityMemory":
        created_at = data.get("created_at", datetime.utcnow())
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            memory_type=data.get("memory_type", "entity"),
            created_at=created_at,
            metadata=data.get("metadata", {}),
            entity_type=data.get("entity_type", ""),
            name=data.get("name", ""),
            attributes=data.get("attributes", {}),
            description=data.get("description", ""),
            context=data.get("context", ""),
        )


register_memory_type("entity", EntityMemory)
