from __future__ import annotations

import argparse
from pathlib import Path

from memory import (
    HashEmbedder,
    MEMORY_TYPES,
    MemoryFormatter,
    MemoryManager,
    VectorMemoryStore,
)
from memory.config import (
    DEFAULT_STORE_PATH,
    EMBEDDING_DIMENSION,
    build_default_store,
    resolve_store_path,
)


DEFAULT_STORE_PATH = DEFAULT_STORE_PATH  # re-exported for backwards compatibility


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve related memories from the shared vector store."
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Query used to find related memories. If omitted, the script prompts for it.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE_PATH,
        help=f"Shared vector store path (default: {DEFAULT_STORE_PATH}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of memories to retrieve (default: 5).",
    )
    parser.add_argument(
        "--memory-type",
        choices=sorted(MEMORY_TYPES),
        help="Optional memory type filter.",
    )
    args = parser.parse_args(argv)

    query = args.query
    if query is None:
        query = input("Search query: ")
    query = query.strip()
    if not query:
        parser.error("query cannot be empty")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    store_path = resolve_store_path(args.store)
    if not store_path.is_file():
        print(f"No memory store found at {store_path}")
        print("Run memory_ingestor.py first.")
        return 1

    store = build_default_store()
    store.load(str(store_path))
    manager = MemoryManager(store=store)
    memories = manager.recall(
        query=query,
        top_k=args.top_k,
        memory_type=args.memory_type,
    )

    if not memories:
        print(f"No memories found for query: {query}")
        return 0

    print(f"Memories found for query: {query}")
    print(MemoryFormatter().format(memories))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
