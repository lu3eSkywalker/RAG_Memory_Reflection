from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from memory.base import BaseMemory, register_memory_type


@dataclass
class ProceduralMemory(BaseMemory):
    procedure: str = ""
    steps: List[str] = field(default_factory=list)
    preconditions: str = ""
    result: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="procedural")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_text(self) -> str:
        step_text = "\n".join(f"- {step}" for step in self.steps)
        return (
            f"Procedure: {self.procedure}\n"
            f"Preconditions: {self.preconditions}\n"
            f"Steps: {step_text}\n"
            f"Result: {self.result}"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ProceduralMemory":
        created_at = data.get("created_at", datetime.utcnow())
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            memory_type=data.get("memory_type", "procedural"),
            created_at=created_at,
            metadata=data.get("metadata", {}),
            procedure=data.get("procedure", ""),
            steps=data.get("steps", []),
            preconditions=data.get("preconditions", ""),
            result=data.get("result", ""),
        )


register_memory_type("procedural", ProceduralMemory)
