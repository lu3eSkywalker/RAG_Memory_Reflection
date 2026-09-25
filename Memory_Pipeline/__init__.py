"""
memory — Basic Extensible Memory Architecture
"""
from memory.base import BaseMemory, MEMORY_TYPES, register_memory_type, get_memory_class, create_memory_from_dict
from memory.models import EpisodicMemory, BugMemory  # noqa: F401 ensure registration
from memory.types.episodic import EpisodicMemory  # noqa: F401
from memory.types.bug import BugMemory  # noqa: F401
from memory.extractor import MemoryExtractor, BasicMemoryExtractor, LLMMemoryExtractor
from memory.stores.vector_store import (
    MemoryStore,
    MemoryEmbedder,
    VectorMemoryStore,
    InMemoryMemoryStore,
    HashEmbedder,
    SimpleEmbedder,
    DummyEmbedder,
)
from memory.retriever import MemoryRetriever, VectorMemoryRetriever, SimpleMemoryRetriever
from memory.manager import MemoryManager
from memory.formatter import MemoryFormatter
from memory.config import MemoryConfig

# Ensure types are imported (trigger registration)
import memory.types.episodic  # noqa: F401
import memory.types.bug  # noqa: F401

__all__ = [
    "BaseMemory",
    "EpisodicMemory",
    "BugMemory",
    "MEMORY_TYPES",
    "register_memory_type",
    "get_memory_class",
    "create_memory_from_dict",
    "MemoryExtractor",
    "BasicMemoryExtractor",
    "LLMMemoryExtractor",
    "MemoryStore",
    "MemoryEmbedder",
    "VectorMemoryStore",
    "InMemoryMemoryStore",
    "HashEmbedder",
    "SimpleEmbedder",
    "DummyEmbedder",
    "MemoryRetriever",
    "VectorMemoryRetriever",
    "SimpleMemoryRetriever",
    "MemoryManager",
    "MemoryFormatter",
    "MemoryConfig",
]
