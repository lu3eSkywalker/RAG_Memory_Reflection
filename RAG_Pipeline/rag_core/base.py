"""
Base interfaces for all RAG pipeline components.
Each component is a pluggable module with a standard interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, List, Dict, Optional
import json

T = TypeVar('T')
U = TypeVar('U')


@dataclass
class PipelineComponent(ABC, Generic[T, U]):
    """Base class for all pipeline components."""
    component_id: str
    config: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    def process(self, input_data: T, context: 'PipelineContext') -> U:
        """Process input and return output."""
        pass
    
    def validate_config(self) -> bool:
        """Validate component configuration."""
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return component metadata for logging/debugging."""
        return {
            "component_id": self.component_id,
            "type": self.__class__.__name__,
            "config": self.config
        }


@dataclass
class Document:
    """A document/chunk with content and metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "score": self.score
        }


@dataclass
class RetrievalResult:
    """Result of retrieval with documents and scores."""
    query: str
    documents: List[Document]
    retriever_id: str
    retrieval_time: float = 0.0


@dataclass
class GenerationResult:
    """Result of generation."""
    prompt: str
    completions: List[str]
    generator_id: str
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluation."""
    metrics: Dict[str, float]
    evaluator_id: str
    details: Dict[str, Any] = field(default_factory=dict)


class Chunker(PipelineComponent[str, List[Document]], ABC):
    """Split documents into chunks."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        pass
    
    def process(self, input_data: str, context: 'PipelineContext') -> List[Document]:
        return self.chunk(input_data, context.get("metadata", {}))


class Embedder(PipelineComponent[List[Document], List[Document]], ABC):
    """Generate embeddings for documents."""
    
    @abstractmethod
    def embed(self, documents: List[Document]) -> List[Document]:
        pass
    
    def process(self, input_data: List[Document], context: 'PipelineContext') -> List[Document]:
        return self.embed(input_data)


class Retriever(PipelineComponent[str, RetrievalResult], ABC):
    """Retrieve relevant documents for a query."""
    
    @abstractmethod
    def retrieve(self, query: str, index: Any, context: 'PipelineContext') -> RetrievalResult:
        pass
    
    def process(self, input_data: str, context: 'PipelineContext') -> RetrievalResult:
        index = context.get("index")
        return self.retrieve(input_data, index, context)


class PromptBuilder(PipelineComponent[RetrievalResult, str], ABC):
    """Build prompt from query and retrieved documents."""
    
    @abstractmethod
    def build_prompt(self, query: str, retrieval_result: RetrievalResult, context: 'PipelineContext') -> str:
        pass
    
    def process(self, input_data: RetrievalResult, context: 'PipelineContext') -> str:
        query = context.get("original_query", "")
        return self.build_prompt(query, input_data, context)


class Generator(PipelineComponent[str, GenerationResult], ABC):
    """Generate completions from prompt."""
    
    @abstractmethod
    def generate(self, prompt: str, context: 'PipelineContext') -> GenerationResult:
        pass
    
    def process(self, input_data: str, context: 'PipelineContext') -> GenerationResult:
        return self.generate(input_data, context)


class IterationStrategy(PipelineComponent[Dict[str, Any], Dict[str, Any]], ABC):
    """Control multi-round retrieval-generation loops."""
    
    @abstractmethod
    def run(self, initial_input: Dict[str, Any], pipeline: 'RAGPipeline', context: 'PipelineContext') -> Dict[str, Any]:
        """Execute the iteration strategy."""
        pass
    
    def process(self, input_data: Dict[str, Any], context: 'PipelineContext') -> Dict[str, Any]:
        pipeline = context.get("pipeline")
        return self.run(input_data, pipeline, context)


class Evaluator(PipelineComponent[Dict[str, Any], EvaluationResult], ABC):
    """Evaluate predictions against ground truth."""
    
    @abstractmethod
    def evaluate(self, predictions: List[str], ground_truths: List[str], context: 'PipelineContext') -> EvaluationResult:
        pass
    
    def process(self, input_data: Dict[str, Any], context: 'PipelineContext') -> EvaluationResult:
        return self.evaluate(
            input_data.get("predictions", []),
            input_data.get("ground_truths", []),
            context
        )


class VectorStore(PipelineComponent[List[Document], Any], ABC):
    """Vector store for indexing and searching document embeddings."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document], context: 'PipelineContext') -> Any:
        """Add documents to the index."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int, context: 'PipelineContext') -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        pass
    
    def process(self, input_data: List[Document], context: 'PipelineContext') -> Any:
        return self.add_documents(input_data, context)