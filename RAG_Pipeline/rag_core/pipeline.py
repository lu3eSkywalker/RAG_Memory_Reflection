"""
Main RAG Pipeline Executor.
Orchestrates component execution with context passing.
"""

from typing import Dict, Any, List, Optional, Callable
import time
import logging
from dataclasses import dataclass

from .base import PipelineComponent
from .context import PipelineContext
from .registry import ComponentRegistry, PipelineRegistry, PipelineSpec, PipelineStep

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Runtime configuration for pipeline execution."""
    cache_dir: str = "./cache"
    parallel_workers: int = 4
    batch_size: int = 32
    timeout: float = 300.0
    save_intermediate: bool = True
    log_level: str = "INFO"


class RAGPipeline:
    """Main pipeline executor that runs a sequence of components."""
    
    def __init__(
        self,
        pipeline_spec: PipelineSpec,
        component_registry: ComponentRegistry,
        config: PipelineConfig = None
    ):
        self.spec = pipeline_spec
        self.component_registry = component_registry
        self.config = config or PipelineConfig()
        
        # Instantiate all components
        self.components: List[PipelineComponent] = []
        self.step_output_keys: List[str] = []
        
        for step in pipeline_spec.steps:
            component = component_registry.instantiate(
                step.component_type,
                step.component_id,
                **step.override_config
            )
            self.components.append(component)
            self.step_output_keys.append(step.output_key or f"step_{len(self.components)}")
        
        logger.info(f"Initialized pipeline '{pipeline_spec.pipeline_id}' with {len(self.components)} steps")
    
    def run(
        self,
        query: str,
        ground_truth: str = "",
        metadata: Dict[str, Any] = None,
        index: Any = None,
        documents: List[Any] = None,
        **context_kwargs
    ) -> PipelineContext:
        """Execute the full pipeline."""
        
        # Initialize context
        context = PipelineContext(
            pipeline_id=self.spec.pipeline_id,
            original_query=query,
            ground_truth=ground_truth,
            metadata=metadata or {},
            index=index,
            documents=documents or [],
            config=self.config.__dict__,
        )
        context.pipeline = self  # For iteration strategies
        
        # Apply any additional context kwargs
        for key, value in context_kwargs.items():
            context.set(key, value)
        
        logger.info(f"Starting pipeline run: {context.run_id}")
        start_time = time.time()
        
        try:
            # Execute each component in sequence
            for i, (component, output_key) in enumerate(zip(self.components, self.step_output_keys)):
                step_start = time.time()
                
                logger.debug(f"Executing step {i+1}/{len(self.components)}: {component.component_id}")
                
                # Get input for this component
                input_data = self._get_component_input(component, context, i)
                
                # Execute component
                output = component.process(input_data, context)
                
                # Store output in context
                context.set(output_key, output)
                
                # Also store by component type for easy access
                self._store_by_type(context, component, output)
                
                step_duration = time.time() - step_start
                context.add_timing(component.component_id, step_duration)
                logger.debug(f"Step {component.component_id} completed in {step_duration:.3f}s")
            
            total_time = time.time() - start_time
            context.add_timing("total", total_time)
            logger.info(f"Pipeline completed in {total_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            context.add_error(str(e))
            raise
        
        return context
    
    def run_batch(
        self,
        queries: List[str],
        ground_truths: List[str] = None,
        metadatas: List[Dict[str, Any]] = None,
        **context_kwargs
    ) -> List[PipelineContext]:
        """Execute pipeline for multiple queries."""
        ground_truths = ground_truths or [""] * len(queries)
        metadatas = metadatas or [{}] * len(queries)
        
        results = []
        for query, gt, meta in zip(queries, ground_truths, metadatas):
            context = self.run(query, gt, meta, **context_kwargs)
            results.append(context)
        
        return results
    
    def _get_component_input(self, component: PipelineComponent, context: PipelineContext, step_index: int) -> Any:
        """Determine input for a component based on its type and previous outputs."""
        from .base import Chunker, Embedder, Retriever, PromptBuilder, Generator, IterationStrategy, Evaluator, VectorStore
        
        # For first step, use original query
        if step_index == 0:
            return context.original_query
        
        # Type-specific input extraction
        if isinstance(component, Chunker):
            return context.original_query
        elif isinstance(component, Embedder):
            return context.get("chunks", [])
        elif isinstance(component, VectorStore):
            return context.get("chunks", [])
        elif isinstance(component, Retriever):
            return context.original_query
        elif isinstance(component, PromptBuilder):
            return context.get("retrieval_result")
        elif isinstance(component, Generator):
            return context.get("prompt", "")
        elif isinstance(component, IterationStrategy):
            return {
                "query": context.original_query,
                "ground_truth": context.ground_truth,
                "metadata": context.metadata,
            }
        elif isinstance(component, Evaluator):
            return {
                "predictions": context.predictions,
                "ground_truths": [context.ground_truth] if context.ground_truth else [],
            }
        
        # Default: pass full context
        return context
    
    def _store_by_type(self, context: PipelineContext, component: PipelineComponent, output: Any) -> None:
        """Store output in context with type-based keys."""
        from .base import Chunker, Embedder, Retriever, PromptBuilder, Generator, IterationStrategy, Evaluator, VectorStore
        
        if isinstance(component, Chunker):
            context.chunks = output
        elif isinstance(component, Embedder):
            context.embeddings = output
            # Store embedder instance for query embedding
            context.set("embedder_instance", component)
        elif isinstance(component, VectorStore):
            context.vector_store = output
            # Also store documents for retrieval
            if hasattr(component, '_documents'):
                context.documents = component._documents
        elif isinstance(component, Retriever):
            context.retrieval_result = output
        elif isinstance(component, PromptBuilder):
            context.prompt = output
        elif isinstance(component, Generator):
            context.generation_result = output
            if hasattr(output, 'completions'):
                context.predictions = output.completions
        elif isinstance(component, IterationStrategy):
            # Iteration strategy returns final results
            if isinstance(output, dict):
                context.predictions = output.get("predictions", context.predictions)
                context.iteration_history = output.get("iteration_history", [])
        elif isinstance(component, Evaluator):
            context.evaluation_results[component.component_id] = output
    
    def get_component(self, component_id: str) -> Optional[PipelineComponent]:
        """Get a component by ID."""
        for comp in self.components:
            if comp.component_id == component_id:
                return comp
        return None


class PipelineRunner:
    """High-level runner for managing multiple pipelines."""
    
    def __init__(
        self,
        component_registry: ComponentRegistry,
        pipeline_registry: PipelineRegistry,
        config: PipelineConfig = None
    ):
        self.component_registry = component_registry
        self.pipeline_registry = pipeline_registry
        self.config = config or PipelineConfig()
        self._pipeline_cache: Dict[str, RAGPipeline] = {}
    
    def get_pipeline(self, pipeline_id: str) -> RAGPipeline:
        """Get or create pipeline instance."""
        if pipeline_id not in self._pipeline_cache:
            spec = self.pipeline_registry.get_spec(pipeline_id)
            errors = self.pipeline_registry.validate_pipeline(pipeline_id)
            if errors:
                raise ValueError(f"Pipeline validation failed: {errors}")
            self._pipeline_cache[pipeline_id] = RAGPipeline(spec, self.component_registry, self.config)
        return self._pipeline_cache[pipeline_id]
    
    def run(
        self,
        pipeline_id: str,
        query: str,
        ground_truth: str = "",
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> PipelineContext:
        """Run a single query through a pipeline."""
        pipeline = self.get_pipeline(pipeline_id)
        return pipeline.run(query, ground_truth, metadata, **kwargs)
    
    def run_batch(
        self,
        pipeline_id: str,
        queries: List[str],
        ground_truths: List[str] = None,
        metadatas: List[Dict[str, Any]] = None,
        **kwargs
    ) -> List[PipelineContext]:
        """Run multiple queries through a pipeline."""
        pipeline = self.get_pipeline(pipeline_id)
        return pipeline.run_batch(queries, ground_truths, metadatas, **kwargs)
    
    def compare_pipelines(
        self,
        pipeline_ids: List[str],
        queries: List[str],
        ground_truths: List[str] = None,
        metadatas: List[Dict[str, Any]] = None
    ) -> Dict[str, List[PipelineContext]]:
        """Run same queries through multiple pipelines for comparison."""
        results = {}
        for pid in pipeline_ids:
            logger.info(f"Running pipeline: {pid}")
            results[pid] = self.run_batch(pid, queries, ground_truths, metadatas)
        return results