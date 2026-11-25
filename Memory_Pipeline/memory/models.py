"""
memory/models.py — Convenience re-exports.

Spec mentions models.py as alternative location for BaseMemory.
This module re-exports core models for backward compatibility.
"""
from memory.base import BaseMemory, MEMORY_TYPES, register_memory_type, get_memory_class, create_memory_from_dict
from memory.types.episodic import EpisodicMemory
from memory.types.bug import BugMemory
from memory.types.semantic import SemanticMemory
from memory.types.procedural import ProceduralMemory
from memory.types.working import WorkingMemory
from memory.types.preference import PreferenceMemory
from memory.types.entity import EntityMemory
from memory.types.reflective import ReflectiveMemory
from memory.types.temporal import TemporalMemory
from memory.types.summarization import SummarizationMemory

__all__ = [
    "BaseMemory",
    "EpisodicMemory",
    "BugMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "WorkingMemory",
    "PreferenceMemory",
    "EntityMemory",
    "ReflectiveMemory",
    "TemporalMemory",
    "SummarizationMemory",
    "MEMORY_TYPES",
    "register_memory_type",
    "get_memory_class",
    "create_memory_from_dict",
]
