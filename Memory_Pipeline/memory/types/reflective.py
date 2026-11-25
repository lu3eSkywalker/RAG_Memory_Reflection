from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from memory.base import BaseMemory, register_memory_type


@dataclass
class ReflectiveMemory(BaseMemory):
    lesson: str = ""
    reflection: str = ""
    feedback: str = ""
    improvement: str = ""
    context: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="reflective")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_text(self) -> str:
        return (
            f"Lesson: {self.lesson}\n"
            f"Reflection: {self.reflection}\n"
            f"Feedback: {self.feedback}\n"
            f"Improvement: {self.improvement}\n"
            f"Context: {self.context}"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ReflectiveMemory":
        created_at = data.get("created_at", datetime.utcnow())
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            memory_type=data.get("memory_type", "reflective"),
            created_at=created_at,
            metadata=data.get("metadata", {}),
            lesson=data.get("lesson", ""),
            reflection=data.get("reflection", ""),
            feedback=data.get("feedback", ""),
            improvement=data.get("improvement", ""),
            context=data.get("context", ""),
        )


register_memory_type("reflective", ReflectiveMemory)
