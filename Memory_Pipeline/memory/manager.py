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
from memory.config import (
    MemoryConfig,
    DEFAULT_STORE_PATH,
    EMBEDDING_DIMENSION,
    build_default_store,
    resolve_store_path,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Coordinates Extractor -> Store and Query -> Retriever.
    Generic over memory types — knows nothing about internal representation.

    Persistence:
        Pass ``store_path`` to bind the manager to the shared store file
        (defaults to :data:`memory.config.DEFAULT_STORE_PATH` when you pass
        ``use_default_store=True``). When bound, the manager loads existing
        memories on construction and persists to the same file after every
        successful ``remember()``. Without ``store_path`` the manager stays
        purely in-memory, which is the historical behaviour.
    """

    def __init__(
        self,
        extractor: Optional[MemoryExtractor] = None,
        store: Optional[MemoryStore] = None,
        retriever: Optional[MemoryRetriever] = None,
        config: Optional[MemoryConfig] = None,
        store_path: Optional[Any] = None,
        use_default_store: bool = False,
        persist: bool = True,
    ):
        self.config = config or MemoryConfig()
        self.extractor = extractor or BasicMemoryExtractor()
        self.persist = persist
        self.store_path: Optional[Any] = None

        if store is not None:
            self.store: MemoryStore = store
            if store_path is not None:
                self.store_path = resolve_store_path(store_path)
        else:
            if use_default_store and store_path is None:
                store_path = DEFAULT_STORE_PATH
            if store_path is not None:
                self.store_path = resolve_store_path(store_path)
                self.store = build_default_store()
                try:
                    self.store.load_if_exists(str(self.store_path))
                except Exception as error:
                    # Never break the pipeline because of a bad store file.
                    logger.error(
                        "[Memory] Could not load store %s: %s", self.store_path, error
                    )
            else:
                self.store = build_default_store()
                logger.debug("[Memory] Manager running in-memory (no store_path given)")

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
            stored = 0
            for mem in filtered:
                try:
                    self.store.add(mem)
                    stored += 1
                except Exception as e:
                    logger.error(f"[Memory] Failed to store memory id={mem.id}: {e}", exc_info=True)

            # Persist so accepted memories survive this process (best effort).
            if stored and self.store_path is not None and self.persist:
                try:
                    self.save(str(self.store_path))
                except Exception as e:
                    logger.error(
                        "[Memory] Stored %d memories in memory but persistence to %s "
                        "failed: %s", stored, self.store_path, e, exc_info=True,
                    )

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

    def save(self, path: str = None) -> None:
        target = path or (str(self.store_path) if self.store_path else None)
        if not target:
            raise ValueError("No store path: pass one or construct the manager with store_path")
        if hasattr(self.store, "save"):
            self.store.save(target)  # type: ignore

    def load(self, path: str = None) -> None:
        target = path or (str(self.store_path) if self.store_path else None)
        if not target:
            raise ValueError("No store path: pass one or construct the manager with store_path")
        if hasattr(self.store, "load_if_exists"):
            self.store.load_if_exists(target)  # type: ignore
        elif hasattr(self.store, "load"):
            self.store.load(target)  # type: ignore
