"""memory/config.py — Configuration"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Shared persistence configuration
# ---------------------------------------------------------------------------
# One canonical store file shared by the ingestor CLI, the retriever CLI, the
# MemoryManager and the generation/reflection pipeline. Every component must
# import this constant instead of hardcoding a path, so a second memory_store.pkl
# is never created next to a script.
#
# MEMORY_PIPELINE_DIR may be overridden (e.g. via the MEMORY_PIPELINE_DIR
# environment variable) for portability; the default resolves to
# ~/Downloads/memory pipeline/memory_store.pkl
# ---------------------------------------------------------------------------
DEFAULT_STORE_DIRNAME = "memory pipeline"
DEFAULT_STORE_FILENAME = "memory_store.pkl"

MEMORY_PIPELINE_DIR = Path(
    __import__("os").environ.get(
        "MEMORY_PIPELINE_DIR", str(Path.home() / "Downloads" / DEFAULT_STORE_DIRNAME)
    )
).expanduser()

DEFAULT_STORE_PATH = MEMORY_PIPELINE_DIR / DEFAULT_STORE_FILENAME

# Embedding width of the persisted pickle schema; changing it invalidates old
# vectors, so it is pinned here and reused by every component.
EMBEDDING_DIMENSION = 128


def resolve_store_path(path: Optional[Path | str] = None) -> Path:
    """Return an explicit path when given, otherwise the shared default.

    An explicitly supplied path always overrides the default (used by the
    `--store` option of both CLI tools and by tests).
    """
    if path is None:
        return DEFAULT_STORE_PATH
    return Path(path).expanduser()


def build_default_store(path: Optional[Path | str] = None, embedder=None):
    """VectorMemoryStore configured with the shared embedder width.

    Does NOT load existing data — callers decide whether to load, so that a
    missing store is never silently created/overwritten.
    """
    from memory.stores.vector_store import HashEmbedder, VectorMemoryStore

    store = VectorMemoryStore(
        embedder=embedder or HashEmbedder(dim=EMBEDDING_DIMENSION)
    )
    return store


@dataclass
class MemoryConfig:
    enabled: bool = True
    top_k: int = 5
    enabled_memory_types: List[str] = field(
        default_factory=lambda: [
            "episodic",
            "bug",
            "semantic",
            "procedural",
            "working",
            "preference",
            "entity",
            "reflective",
            "temporal",
            "summarization",
        ]
    )
    # Future extension points (not implemented now)
    # similarity_threshold: float = 0.0
    # deduplication: bool = False
    # decay: bool = False
