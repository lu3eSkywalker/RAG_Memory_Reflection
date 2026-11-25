"""
memory/retriever.py — Retrieval abstraction
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from memory.base import BaseMemory
from memory.stores.vector_store import VectorMemoryStore, MemoryStore


class MemoryRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[BaseMemory]:
        pass


class VectorMemoryRetriever(MemoryRetriever):
    """
    Retriever backed by VectorMemoryStore.
    Supports filtering by memory_type and top_k.
    Generic — does not hardcode memory types.
    """
    def __init__(self, store: VectorMemoryStore):
        if not isinstance(store, VectorMemoryStore):
            # Allow any store that implements .search()
            # but log warning
            if not hasattr(store, "search"):
                raise TypeError("store must be VectorMemoryStore or implement search(query, top_k, memory_type)")
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[BaseMemory]:
        # Delegate to store's search
        return self.store.search(query=query, top_k=top_k, memory_type=memory_type)


class SimpleMemoryRetriever(MemoryRetriever):
    """
    Fallback retriever for InMemoryMemoryStore or generic MemoryStore
    using brute-force keyword overlap if no vector search available.
    """
    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve(self, query: str, top_k: int = 5, memory_type: Optional[str] = None) -> List[BaseMemory]:
        if hasattr(self.store, "search"):
            return self.store.search(query=query, top_k=top_k, memory_type=memory_type)  # type: ignore
        # Brute-force: rank by token overlap
        all_mems = getattr(self.store, "all_memories", lambda: [])()
        if not all_mems:
            # try _memories dict fallback
            try:
                all_mems = list(self.store._memories.values())  # type: ignore
            except Exception:
                all_mems = []
        q_tokens = set(query.lower().split())
        scored = []
        for mem in all_mems:
            if memory_type and mem.memory_type != memory_type:
                continue
            text = mem.to_text().lower()
            t_tokens = set(text.split())
            overlap = len(q_tokens & t_tokens)
            # also consider substring bonus
            substr = 1 if query.lower() in text else 0
            scored.append((overlap + substr, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scored[:top_k] if s > 0] or [m for _, m in scored[:top_k]]
