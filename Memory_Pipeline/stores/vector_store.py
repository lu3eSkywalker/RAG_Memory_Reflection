"""
memory/stores/vector_store.py — Embedder + MemoryStore + VectorMemoryStore
"""
from __future__ import annotations

import hashlib
import math
import re
import pickle
import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging

from memory.base import BaseMemory, create_memory_from_dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedder abstraction
# ---------------------------------------------------------------------------
class MemoryEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        pass


class HashEmbedder(MemoryEmbedder):
    """
    Simple deterministic embedder without external dependencies.
    Uses hashed bag-of-words → fixed dimension L2-normalized vector.
    Suitable for unit tests and offline usage. Replace with real model via DI.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self._token_re = re.compile(r"\w+")

    def _text_to_vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = self._token_re.findall(text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            # stable hash via hashlib
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            # use second hash for sign/value to avoid collisions bias
            sign_hash = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
            sign = 1.0 if (sign_hash % 2 == 0) else -1.0
            # tf-like increment
            vec[idx] += sign * (1.0 + math.log(1 + tokens.count(tok)))
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        return vec

    def embed(self, text: str) -> List[float]:
        return self._text_to_vector(text)

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(t) for t in texts]


# Backward-compatible alias
SimpleEmbedder = HashEmbedder
DummyEmbedder = HashEmbedder

# ---------------------------------------------------------------------------
# MemoryStore abstraction
# ---------------------------------------------------------------------------
class MemoryStore(ABC):
    @abstractmethod
    def add(self, memory: BaseMemory) -> str:
        pass

    @abstractmethod
    def add_many(self, memories: List[BaseMemory]) -> List[str]:
        pass

    @abstractmethod
    def get(self, memory_id: str) -> Optional[BaseMemory]:
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class InMemoryMemoryStore(MemoryStore):
    """Simple dict-backed store without vectors — useful for testing store interface."""
    def __init__(self):
        self._memories: Dict[str, BaseMemory] = {}

    def add(self, memory: BaseMemory) -> str:
        self._memories[memory.id] = memory
        logger.info(f"[Memory] Stored memory id={memory.id} type={memory.memory_type}")
        return memory.id

    def add_many(self, memories: List[BaseMemory]) -> List[str]:
        return [self.add(m) for m in memories]

    def get(self, memory_id: str) -> Optional[BaseMemory]:
        return self._memories.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

    def clear(self) -> None:
        self._memories.clear()

    def all_memories(self) -> List[BaseMemory]:
        return list(self._memories.values())


class VectorMemoryStore(MemoryStore):
    """
    Vector-store adapter.

    Stores: Embedding + Memory text + Metadata + Memory ID
    Delegates embedding to MemoryEmbedder. In-memory vector store with cosine similarity.
    Replaceable: pass any MemoryEmbedder; persistence via save()/load().
    """
    def __init__(self, embedder: Optional[MemoryEmbedder] = None, dim: Optional[int] = None):
        self.embedder: MemoryEmbedder = embedder or HashEmbedder(dim=dim or 128)
        self._memories: Dict[str, BaseMemory] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._dim = getattr(self.embedder, "dim", dim or 128)

    # ---- MemoryStore interface ----
    def add(self, memory: BaseMemory) -> str:
        text = memory.to_text()
        vec = self.embedder.embed(text)
        self._memories[memory.id] = memory
        self._embeddings[memory.id] = vec
        logger.info(f"[Memory] Stored memory id={memory.id} type={memory.memory_type}")
        return memory.id

    def add_many(self, memories: List[BaseMemory]) -> List[str]:
        if not memories:
            return []
        texts = [m.to_text() for m in memories]
        vecs = self.embedder.embed_many(texts)
        ids: List[str] = []
        for mem, vec in zip(memories, vecs):
            self._memories[mem.id] = mem
            self._embeddings[mem.id] = vec
            ids.append(mem.id)
            logger.info(f"[Memory] Stored memory id={mem.id} type={mem.memory_type}")
        return ids

    def get(self, memory_id: str) -> Optional[BaseMemory]:
        return self._memories.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._embeddings.pop(memory_id, None)
            return True
        return False

    def clear(self) -> None:
        self._memories.clear()
        self._embeddings.clear()

    # ---- Vector search ----
    def search(self, query: str, top_k: int = 5, memory_type: Optional[str] = None) -> List[BaseMemory]:
        """
        Cosine similarity search over stored embeddings.
        Filters by memory_type if provided.
        """
        if not self._memories:
            return []
        qvec = self.embedder.embed(query)
        scored = []
        for mid, mem in self._memories.items():
            if memory_type is not None and mem.memory_type != memory_type:
                continue
            vec = self._embeddings.get(mid)
            if vec is None:
                continue
            sim = self._cosine(qvec, vec)
            scored.append((sim, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = [m for _, m in scored[:top_k]]
        logger.info(f"[Memory] Retrieved {len(result)} memories for query={query!r} type={memory_type} top_k={top_k}")
        return result

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        # vectors are already L2-normalized for HashEmbedder, but generic fallback
        dot = sum(x * y for x, y in zip(a, b))
        # if not normalized, normalize dot
        # Since HashEmbedder normalizes, dot is cosine. For others, compute properly.
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        # If vectors were normalized, dot ≈ cosine already; but recompute if not
        # Detect if already normalized (norm≈1): then dot is cosine
        if abs(norm_a - 1.0) < 1e-6 and abs(norm_b - 1.0) < 1e-6:
            return dot
        return dot / (norm_a * norm_b)

    # ---- Helpers ----
    def all_memories(self) -> List[BaseMemory]:
        return list(self._memories.values())

    def count(self) -> int:
        return len(self._memories)

    # ---- Persistence ----
    def save(self, path: str) -> None:
        """Persist memories + embeddings via pickle."""
        data = {
            "memories": {mid: mem.to_dict() for mid, mem in self._memories.items()},
            "embeddings": self._embeddings,
            "dim": self._dim,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[Memory] Saved {len(self._memories)} memories to {path}")

    def load(self, path: str) -> None:
        """Load persisted data."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        memories_raw: Dict[str, dict] = data.get("memories", {})
        embeddings: Dict[str, List[float]] = data.get("embeddings", {})
        self._memories = {}
        for mid, mdict in memories_raw.items():
            try:
                mem = create_memory_from_dict(mdict)
                self._memories[mid] = mem
            except Exception as e:
                logger.warning(f"[Memory] Failed to load memory {mid}: {e}")
        self._embeddings = embeddings
        logger.info(f"[Memory] Loaded {len(self._memories)} memories from {path}")

    # Alternative JSON persistence for debugging
    def save_json(self, path: str) -> None:
        data = {
            "memories": [m.to_dict() for m in self._memories.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for mdict in data.get("memories", []):
            mem = create_memory_from_dict(mdict)
            self._memories[mem.id] = mem
            self._embeddings[mem.id] = self.embedder.embed(mem.to_text())
