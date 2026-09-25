"""memory/config.py — Configuration"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MemoryConfig:
    enabled: bool = True
    top_k: int = 5
    enabled_memory_types: List[str] = field(default_factory=lambda: ["episodic", "bug"])
    # Future extension points (not implemented now)
    # similarity_threshold: float = 0.0
    # deduplication: bool = False
    # decay: bool = False
