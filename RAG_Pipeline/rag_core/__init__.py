"""
Modular RAG Pipeline Core
Inspired by RepoCoder's architecture with pluggable components.
"""

from .base import (
    Chunker,
    Embedder,
    Retriever,
    PromptBuilder,
    Generator,
    IterationStrategy,
    Evaluator,
    PipelineComponent,
)
from .registry import ComponentRegistry, PipelineRegistry
from .pipeline import RAGPipeline, PipelineConfig
from .context import PipelineContext

__all__ = [
    "Chunker",
    "Embedder", 
    "Retriever",
    "PromptBuilder",
    "Generator",
    "IterationStrategy",
    "Evaluator",
    "PipelineComponent",
    "ComponentRegistry",
    "PipelineRegistry",
    "RAGPipeline",
    "PipelineConfig",
    "PipelineContext",
]