"""
memory/types/bug.py — Bug Memory
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict
import uuid

from memory.base import BaseMemory, register_memory_type


@dataclass
class BugMemory(BaseMemory):
    """Failure, its cause, and its successful fix."""
    bug: str = ""
    cause: str = ""
    fix: str = ""
    context: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="bug")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict = field(default_factory=dict)

    def to_text(self) -> str:
        return (
            f"Bug: {self.bug}\n"
            f"Cause: {self.cause}\n"
            f"Fix: {self.fix}\n"
            f"Context: {self.context}"
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "BugMemory":
        if isinstance(data.get("created_at"), str):
            try:
                data = dict(data)
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except Exception:
                pass
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            memory_type=data.get("memory_type", "bug"),
            created_at=data.get("created_at", datetime.utcnow()),
            metadata=data.get("metadata", {}),
            bug=data.get("bug", ""),
            cause=data.get("cause", ""),
            fix=data.get("fix", ""),
            context=data.get("context", ""),
        )


register_memory_type("bug", BugMemory)
