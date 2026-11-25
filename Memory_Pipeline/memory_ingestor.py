from __future__ import annotations

import argparse
from pathlib import Path

from memory import (
    BugMemory,
    EntityMemory,
    EpisodicMemory,
    HashEmbedder,
    PreferenceMemory,
    ProceduralMemory,
    ReflectiveMemory,
    SemanticMemory,
    SummarizationMemory,
    TemporalMemory,
    VectorMemoryStore,
    WorkingMemory,
)
from memory.config import (
    DEFAULT_STORE_PATH,
    EMBEDDING_DIMENSION,
    build_default_store,
    resolve_store_path,
)


DEFAULT_STORE_PATH = DEFAULT_STORE_PATH  # re-exported for backwards compatibility
MEMORY_TYPE_NAMES = (
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
)


def _build_memory(memory_type: str, text: str):
    metadata = {
        "source": "memory_ingestor",
        "content_type": "raw_text",
    }
    if memory_type == "episodic":
        return EpisodicMemory(task=text, metadata=metadata)
    if memory_type == "bug":
        return BugMemory(bug=text, metadata=metadata)
    if memory_type == "semantic":
        return SemanticMemory(concept=text, metadata=metadata)
    if memory_type == "procedural":
        return ProceduralMemory(procedure=text, metadata=metadata)
    if memory_type == "working":
        return WorkingMemory(task=text, metadata=metadata)
    if memory_type == "preference":
        return PreferenceMemory(preference=text, metadata=metadata)
    if memory_type == "entity":
        return EntityMemory(description=text, metadata=metadata)
    if memory_type == "reflective":
        return ReflectiveMemory(lesson=text, metadata=metadata)
    if memory_type == "temporal":
        return TemporalMemory(event=text, metadata=metadata)
    if memory_type == "summarization":
        return SummarizationMemory(summary=text, metadata=metadata)
    raise ValueError(f"Unsupported memory type: {memory_type}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Store one memory as a vector in the shared memory store."
    )
    parser.add_argument(
        "memory",
        nargs="?",
        help="Memory text to store. If omitted, the script prompts for it.",
    )
    parser.add_argument(
        "--memory-type",
        choices=MEMORY_TYPE_NAMES,
        default="episodic",
        help="Memory category to create (default: episodic).",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE_PATH,
        help=f"Shared vector store path (default: {DEFAULT_STORE_PATH}).",
    )
    args = parser.parse_args(argv)

    memory_text = args.memory
    if memory_text is None:
        memory_text = input("Memory to store: ")
    memory_text = memory_text.strip()
    if not memory_text:
        parser.error("memory text cannot be empty")

    store = build_default_store()
    store_path = resolve_store_path(args.store)
    if store_path.exists():
        store.load(str(store_path))

    memory = _build_memory(args.memory_type, memory_text)
    store.add(memory)
    store.save(str(store_path))

    print(f"Stored {memory.memory_type} memory {memory.id} in {store_path}")
    print(f"Total memories: {store.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
