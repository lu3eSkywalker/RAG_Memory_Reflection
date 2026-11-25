from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from memory.base import BaseMemory, register_memory_type


@dataclass
class WorkingMemory(BaseMemory):
    task: str = ""
    context: str = ""
    state: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="working")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_text(self) -> str:
        return (
            f"Task: {self.task}\n"
            f"Context: {self.context}\n"
            f"State: {self.state}"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "WorkingMemory":
        created_at = data.get("created_at", datetime.utcnow())
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            memory_type=data.get("memory_type", "working"),
            created_at=created_at,
            metadata=data.get("metadata", {}),
            task=data.get("task", ""),
            context=data.get("context", ""),
            state=data.get("state", ""),
        )


register_memory_type("working", WorkingMemory)
