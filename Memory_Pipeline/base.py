"""
memory/base.py — Core memory abstraction and registry.

Implements BaseMemory and the extensible memory-type registry.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Type, Optional


@dataclass
class BaseMemory(ABC):
    """
    Generic base memory representation.
    Every memory item contains common metadata.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="base")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict. Handles datetime."""
        d = asdict(self)
        # datetime -> isoformat
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        else:
            # already string from asdict? ensure isoformat
            try:
                d["created_at"] = self.created_at.isoformat()  # type: ignore
            except Exception:
                pass
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "BaseMemory":
        """Deserialize from dict. Dispatches via registry if memory_type differs.

        If called on BaseMemory directly, it will look up the correct subclass
        via the registry.
        """
        # copy to avoid mutation
        data = dict(data)
        # parse created_at
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            try:
                data["created_at"] = datetime.fromisoformat(created_at)
            except ValueError:
                # try handling Z suffix
                try:
                    data["created_at"] = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except Exception:
                    pass
        # If called on BaseMemory, dispatch to correct subclass
        if cls is BaseMemory:
            mem_type = data.get("memory_type", "base")
            target_cls = MEMORY_TYPES.get(mem_type, BaseMemory)
            if target_cls is BaseMemory:
                return BaseMemory(**{k: v for k, v in data.items() if k in {"id", "memory_type", "created_at", "metadata"}})  # type: ignore
            return target_cls.from_dict(data)  # type: ignore
        # Subclass case: filter to fields that belong to that class
        # Let dataclass handle it; pass through relevant keys
        return cls(**data)  # type: ignore

    @abstractmethod
    def to_text(self) -> str:
        """Meaningful textual representation suitable for embedding."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r}, type={self.memory_type!r})"


# ---------------------------------------------------------------------------
# Registry / Factory
# ---------------------------------------------------------------------------
MEMORY_TYPES: Dict[str, Type[BaseMemory]] = {}


def register_memory_type(name: str, cls: Type[BaseMemory]) -> None:
    """
    Register a new memory type.

    Example:
        register_memory_type("semantic", SemanticMemory)
    """
    if not issubclass(cls, BaseMemory):
        raise TypeError(f"cls must be subclass of BaseMemory, got {cls}")
    MEMORY_TYPES[name] = cls


def get_memory_class(name: str) -> Optional[Type[BaseMemory]]:
    return MEMORY_TYPES.get(name)


def create_memory_from_dict(data: dict) -> BaseMemory:
    """Factory helper: create correct memory subclass from dict."""
    mem_type = data.get("memory_type", "base")
    cls = MEMORY_TYPES.get(mem_type)
    if cls is None:
        raise ValueError(f"Unknown memory_type: {mem_type!r}. Registered: {list(MEMORY_TYPES)}")
    return cls.from_dict(data)
