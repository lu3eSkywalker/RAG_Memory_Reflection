"""
memory/types/episodic.py — Episodic Memory
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict
import uuid

from memory.base import BaseMemory, register_memory_type


@dataclass
class EpisodicMemory(BaseMemory):
    """Past coding experience: Task / Context / Action / Result"""
    task: str = ""
    context: str = ""
    action: str = ""
    result: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="episodic")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict = field(default_factory=dict)

    def to_text(self) -> str:
        return (
            f"Task: {self.task}\n"
            f"Context: {self.context}\n"
            f"Action: {self.action}\n"
            f"Result: {self.result}"
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        # ensure dataclass fields are all present (asdict already does)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodicMemory":
        # Normalize datetime
        if isinstance(data.get("created_at"), str):
            try:
                data = dict(data)
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except Exception:
                pass
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            memory_type=data.get("memory_type", "episodic"),
            created_at=data.get("created_at", datetime.utcnow()),
            metadata=data.get("metadata", {}),
            task=data.get("task", ""),
            context=data.get("context", ""),
            action=data.get("action", ""),
            result=data.get("result", ""),
        )


# Auto-register
register_memory_type("episodic", EpisodicMemory)
