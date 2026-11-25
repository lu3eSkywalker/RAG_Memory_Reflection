from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from memory.base import BaseMemory, register_memory_type


@dataclass
class SummarizationMemory(BaseMemory):
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    source_count: int = 0
    context: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="summarization")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_text(self) -> str:
        key_point_text = "\n".join(f"- {point}" for point in self.key_points)
        source_text = ", ".join(self.source_ids)
        return (
            f"Summary: {self.summary}\n"
            f"Key points: {key_point_text}\n"
            f"Source IDs: {source_text}\n"
            f"Source count: {self.source_count}\n"
            f"Context: {self.context}"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SummarizationMemory":
        created_at = data.get("created_at", datetime.utcnow())
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            memory_type=data.get("memory_type", "summarization"),
            created_at=created_at,
            metadata=data.get("metadata", {}),
            summary=data.get("summary", ""),
            key_points=data.get("key_points", []),
            source_ids=data.get("source_ids", []),
            source_count=data.get("source_count", 0),
            context=data.get("context", ""),
        )


register_memory_type("summarization", SummarizationMemory)
