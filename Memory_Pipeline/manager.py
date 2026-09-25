"""
memory/manager.py — High-level orchestration
"""
from __future__ import annotations

import logging
from typing import List, Optional, Any

from memory.base import BaseMemory
from memory.extractor import MemoryExtractor, BasicMemoryExtractor
from memory.stores.vector_store import VectorMemoryStore, MemoryStore
from memory.retriever import MemoryRetriever, VectorMemoryRetriever
from memory.config import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Coordinates Extractor -> Store and Query -> Retriever.
    Generic over memory types — knows nothing about internal representation.
    """

    def __init__(
        self,
        extractor: Optional[MemoryExtractor] = None,
        store: Optional[MemoryStore] = None,
        retriever: Optional[MemoryRetriever] = None,
        config: Optional[MemoryConfig] = None,
    ):
        self.config = config or MemoryConfig()
        self.extractor = extractor or BasicMemoryExtractor()
        self.store: MemoryStore = store or VectorMemoryStore()
        # Build retriever if not provided and store supports search
        if retriever is not None:
            self.retriever = retriever
        else:
            if hasattr(self.store, "search"):
                self.retriever = VectorMemoryRetriever(self.store)  # type: ignore
            else:
                # Defer import to avoid cycle
                from memory.retriever import SimpleMemoryRetriever
                self.retriever = SimpleMemoryRetriever(self.store)

    # ---- Write path ----
    def remember(
        self,
        task: Any,
        context: Any,
        action: Any,
        result: Any,
        test_result: Optional[str] = None,
        reflection: Optional[Any] = None,
    ) -> List[BaseMemory]:
        """
        Extract memories and store them. Errors are logged but do not raise
        to avoid breaking the coding pipeline.
        Returns list of stored memories (may be empty on failure).
        """
        if not self.config.enabled:
            logger.info("[Memory] Memory disabled — skipping remember()")
            return []
        try:
            memories = self.extractor.extract(
                task=task,
                context=context,
                action=action,
                result=result,
                test_result=test_result,
                reflection=reflection,
            )
            if not memories:
                logger.info("[Memory] No memories extracted")
                return []

            # Filter by enabled types if configured
            filtered = [
                m for m in memories if m.memory_type in self.config.enabled_memory_types
            ]
            if len(filtered) != len(memories):
                logger.info(f"[Memory] Filtered {len(memories)-len(filtered)} memories by config.enabled_memory_types")

            # Store
            for mem in filtered:
                try:
                    self.store.add(mem)
                except Exception as e:
                    logger.error(f"[Memory] Failed to store memory id={mem.id}: {e}", exc_info=True)

            logger.info(f"[Memory] Extracted {len(filtered)} memories (types={[m.memory_type for m in filtered]})")
            return filtered
        except Exception as e:
            # Must not break pipeline
            logger.error(f"[Memory] Extraction/storage failed: {e}", exc_info=True)
            return []

    # ---- Read path ----
    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[BaseMemory]:
        if not self.config.enabled:
            logger.info("[Memory] Memory disabled — skipping recall()")
            return []
        try:
            # Use config top_k as default if not overridden
            k = top_k if top_k is not None else self.config.top_k
            memories = self.retriever.retrieve(query=query, top_k=k, memory_type=memory_type)
            # Apply enabled_memory_types filter if memory_type not specified
            if memory_type is None and self.config.enabled_memory_types:
                memories = [m for m in memories if m.memory_type in self.config.enabled_memory_types]
            logger.info(f"[Memory] Retrieved {len(memories)} memories for query={query!r}")
            return memories
        except Exception as e:
            logger.error(f"[Memory] Retrieval failed: {e}", exc_info=True)
            return []

    # ---- Direct store access helpers ----
    def get(self, memory_id: str) -> Optional[BaseMemory]:
        return self.store.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)

    def clear(self) -> None:
        self.store.clear()

    def save(self, path: str) -> None:
        if hasattr(self.store, "save"):
            self.store.save(path)  # type: ignore

    def load(self, path: str) -> None:
        if hasattr(self.store, "load"):
            self.store.load(path)  # type: ignore
