"""
Pipeline execution context - carries state between components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


@dataclass
class PipelineContext:
    """Mutable context passed through pipeline execution."""
    
    # Run identification
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pipeline_id: str = ""
    
    # Input data
    original_query: str = ""
    ground_truth: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Repository/index data
    index: Any = None
    vector_store: Any = None
    documents: List[Any] = field(default_factory=list)
    
    # Intermediate results (populated during execution)
    chunks: List[Any] = field(default_factory=list)
    embeddings: List[Any] = field(default_factory=list)
    retrieval_result: Any = None
    prompt: str = ""
    generation_result: Any = None
    predictions: List[str] = field(default_factory=list)
    
    # Iteration state
    iteration: int = 0
    max_iterations: int = 1
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Evaluation results
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    start_time: float = field(default_factory=time.time)
    step_timings: Dict[str, float] = field(default_factory=dict)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Pipeline reference (for iteration strategies)
    pipeline: Optional['RAGPipeline'] = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context (supports nested keys with dot notation)."""
        keys = key.split('.')
        value = self.__dict__
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return getattr(value, k, default)
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in context (supports nested keys with dot notation)."""
        keys = key.split('.')
        target = self.__dict__
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
    
    def add_timing(self, step: str, duration: float) -> None:
        self.step_timings[step] = duration
    
    def get_total_time(self) -> float:
        return time.time() - self.start_time
    
    def add_error(self, error: str) -> None:
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for logging/debugging."""
        return {
            "run_id": self.run_id,
            "pipeline_id": self.pipeline_id,
            "original_query": self.original_query[:100] + "..." if len(self.original_query) > 100 else self.original_query,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "step_timings": self.step_timings,
            "total_time": self.get_total_time(),
            "errors": self.errors,
            "warnings": self.warnings,
            "evaluation_results": self.evaluation_results,
        }