"""
Component and Pipeline Registries.
Allows dynamic registration and discovery of pipeline components.
"""

from typing import Dict, Type, Any, List, Optional, Callable
from dataclasses import dataclass, field
import importlib
import yaml
import logging

from .base import PipelineComponent, Chunker, Embedder, Retriever, PromptBuilder, Generator, IterationStrategy, Evaluator, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ComponentSpec:
    """Specification for a registered component."""
    component_id: str
    component_type: str  # chunker, embedder, retriever, prompt_builder, generator, iteration_strategy, evaluator
    class_path: str  # Full import path: "module.ClassName"
    config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: List[str] = field(default_factory=list)


class ComponentRegistry:
    """Registry for all pipeline component types."""
    
    COMPONENT_TYPES = [
        "chunker", "embedder", "retriever", 
        "prompt_builder", "generator", 
        "iteration_strategy", "evaluator",
        "vector_store"
    ]
    
    # Base classes for validation
    BASE_CLASSES = {
        "chunker": Chunker,
        "embedder": Embedder,
        "retriever": Retriever,
        "prompt_builder": PromptBuilder,
        "generator": Generator,
        "iteration_strategy": IterationStrategy,
        "evaluator": Evaluator,
        "vector_store": VectorStore,
    }
    
    def __init__(self):
        self._components: Dict[str, Dict[str, ComponentSpec]] = {
            ct: {} for ct in self.COMPONENT_TYPES
        }
        self._class_cache: Dict[str, Type] = {}
    
    def register(self, spec: ComponentSpec) -> None:
        """Register a component specification."""
        if spec.component_type not in self.COMPONENT_TYPES:
            raise ValueError(f"Unknown component type: {spec.component_type}")
        
        if spec.component_id in self._components[spec.component_type]:
            logger.warning(f"Overwriting component: {spec.component_type}.{spec.component_id}")
        
        self._components[spec.component_type][spec.component_id] = spec
        logger.info(f"Registered {spec.component_type}: {spec.component_id}")
    
    def register_from_dict(self, data: Dict[str, Any]) -> None:
        """Register from dictionary (e.g., loaded from YAML)."""
        spec = ComponentSpec(**data)
        self.register(spec)
    
    def get_spec(self, component_type: str, component_id: str) -> ComponentSpec:
        """Get component specification."""
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        if component_id not in self._components[component_type]:
            raise KeyError(f"Component not found: {component_type}.{component_id}")
        
        return self._components[component_type][component_id]
    
    def get_class(self, component_type: str, component_id: str) -> Type[PipelineComponent]:
        """Get component class (loads dynamically)."""
        cache_key = f"{component_type}.{component_id}"
        
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]
        
        spec = self.get_spec(component_type, component_id)
        class_path = spec.class_path
        
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            
            # Validate inheritance
            base_class = self.BASE_CLASSES[component_type]
            if not issubclass(cls, base_class):
                raise TypeError(f"{class_path} does not inherit from {base_class.__name__}")
            
            self._class_cache[cache_key] = cls
            return cls
            
        except (ImportError, AttributeError, ValueError) as e:
            raise RuntimeError(f"Failed to load {class_path}: {e}")
    
    def instantiate(self, component_type: str, component_id: str, **override_config) -> PipelineComponent:
        """Instantiate a component with optional config overrides."""
        spec = self.get_spec(component_type, component_id)
        cls = self.get_class(component_type, component_id)
        
        # Merge config with overrides
        config = {**spec.config, **override_config}
        
        instance = cls(component_id=component_id, config=config)
        
        if not instance.validate_config():
            raise ValueError(f"Invalid config for {component_type}.{component_id}")
        
        return instance
    
    def list_components(self, component_type: str = None) -> Dict[str, List[str]]:
        """List all registered components."""
        if component_type:
            return {component_type: list(self._components.get(component_type, {}).keys())}
        return {ct: list(comps.keys()) for ct, comps in self._components.items()}
    
    def load_from_yaml(self, yaml_path: str) -> None:
        """Load component specifications from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if "components" in data:
            for comp_type, components in data["components"].items():
                for comp_data in components:
                    comp_data["component_type"] = comp_type
                    self.register_from_dict(comp_data)
    
    def load_extensions(self, extension_paths: Dict[str, str]) -> None:
        """Load components from extension modules."""
        for comp_type, module_path in extension_paths.items():
            if comp_type not in self.COMPONENT_TYPES:
                logger.warning(f"Unknown component type in extensions: {comp_type}")
                continue
            
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, "REGISTER_COMPONENTS"):
                    for comp_data in module.REGISTER_COMPONENTS:
                        comp_data["component_type"] = comp_type
                        self.register_from_dict(comp_data)
                logger.info(f"Loaded extensions for {comp_type} from {module_path}")
            except ImportError as e:
                logger.warning(f"Failed to load extensions from {module_path}: {e}")


@dataclass
class PipelineStep:
    """A single step in a pipeline."""
    component_type: str
    component_id: str
    override_config: Dict[str, Any] = field(default_factory=dict)
    output_key: str = ""  # Where to store output in context


@dataclass
class PipelineSpec:
    """Complete pipeline specification."""
    pipeline_id: str
    name: str
    description: str = ""
    steps: List[PipelineStep] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class PipelineRegistry:
    """Registry for complete pipeline definitions."""
    
    def __init__(self, component_registry: ComponentRegistry):
        self.component_registry = component_registry
        self._pipelines: Dict[str, PipelineSpec] = {}
    
    def register(self, spec: PipelineSpec) -> None:
        """Register a pipeline specification."""
        if spec.pipeline_id in self._pipelines:
            logger.warning(f"Overwriting pipeline: {spec.pipeline_id}")
        self._pipelines[spec.pipeline_id] = spec
        logger.info(f"Registered pipeline: {spec.pipeline_id}")
    
    def get_spec(self, pipeline_id: str) -> PipelineSpec:
        """Get pipeline specification."""
        if pipeline_id not in self._pipelines:
            raise KeyError(f"Pipeline not found: {pipeline_id}")
        return self._pipelines[pipeline_id]
    
    def list_pipelines(self) -> List[str]:
        """List all registered pipeline IDs."""
        return list(self._pipelines.keys())
    
    def load_from_yaml(self, yaml_path: str) -> None:
        """Load pipeline specifications from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if "pipelines" in data:
            for pipe_data in data["pipelines"]:
                steps = [
                    PipelineStep(**step) for step in pipe_data.get("steps", [])
                ]
                spec = PipelineSpec(
                    pipeline_id=pipe_data["id"],
                    name=pipe_data["name"],
                    description=pipe_data.get("description", ""),
                    steps=steps,
                    config=pipe_data.get("config", {})
                )
                self.register(spec)
    
    def validate_pipeline(self, pipeline_id: str) -> List[str]:
        """Validate that all components in pipeline exist."""
        spec = self.get_spec(pipeline_id)
        errors = []
        
        for step in spec.steps:
            try:
                self.component_registry.get_spec(step.component_type, step.component_id)
            except KeyError:
                errors.append(f"Missing component: {step.component_type}.{step.component_id}")
        
        return errors