from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from memory.base import BaseMemory, register_memory_type


@dataclass
class TemporalMemory(BaseMemory):
    event: str = ""
    occurred_at: str = ""
    sequence: int = 0
    context: str = ""
    outcome: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="temporal")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_text(self) -> str:
        return (
            f"Event: {self.event}\n"
            f"Occurred at: {self.occurred_at}\n"
            f"Sequence: {self.sequence}\n"
            f"Context: {self.context}\n"
            f"Outcome: {self.outcome}"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "TemporalMemory":
        created_at = data.get("created_at", datetime.utcnow())
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            memory_type=data.get("memory_type", "temporal"),
            created_at=created_at,
            metadata=data.get("metadata", {}),
            event=data.get("event", ""),
            occurred_at=data.get("occurred_at", ""),
            sequence=data.get("sequence", 0),
            context=data.get("context", ""),
            outcome=data.get("outcome", ""),
        )


register_memory_type("temporal", TemporalMemory)
