"""
memory/models.py — Convenience re-exports.

Spec mentions models.py as alternative location for BaseMemory.
This module re-exports core models for backward compatibility.
"""
from memory.base import BaseMemory, MEMORY_TYPES, register_memory_type, get_memory_class, create_memory_from_dict
from memory.types.episodic import EpisodicMemory
from memory.types.bug import BugMemory

__all__ = [
    "BaseMemory",
    "EpisodicMemory",
    "BugMemory",
    "MEMORY_TYPES",
    "register_memory_type",
    "get_memory_class",
    "create_memory_from_dict",
]
