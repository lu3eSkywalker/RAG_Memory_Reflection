#!/usr/bin/env python3
"""
Run every RAG technique end to end: steps 1-6 per technique.

    Step 1  Chunk      sliding-window chunking of the target repository
    Step 2  Embed      SentenceTransformer embeddings for every chunk
    Step 3  VectorStore FAISS index over the embeddings
    Step 4  Retrieve   one technique (semantic, AST, iterative, adaptive,
                       cross-file, multi-query, BM25, hybrid)
    Step 5  Prompt     RepoCoder-style prompt (commented code fragments)
    Step 6  Generate   Gemini (if GEMINI_API_KEY is set) or a mock generator

Steps 1-3 run once and are shared by every technique; steps 4-6 then run per
technique so their outputs can be compared directly.

Usage:
    python run_all_techniques.py
    python run_all_techniques.py --repo /path/to/repo --query "add auth middleware"
    python run_all_techniques.py --techniques ast,cross_file,hybrid --skip-generate
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.base import Document  # noqa: E402

import gemini_client  # noqa: E402

from retrieval.adaptive_retriever import AdaptiveRetriever  # noqa: E402
from retrieval.ast_retriever import (  # noqa: E402
    ASTBasedRetriever,
    ASTIndex,
    build_index,
)
from retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from retrieval.cross_file_retriever import CrossFileRetriever  # noqa: E402
from retrieval.hybrid_retriever import (  # noqa: E402
    HybridRetrievalConfig,
    HybridRetriever,
)
from retrieval.iterative_retriever import (  # noqa: E402
    IterativeRetriever,
    query_concepts,
)
from retrieval.multiquery_retriever import MultiQueryRetriever  # noqa: E402
from retrieval.semantic_retriever import (  # noqa: E402
    FaissVectorStore,
    SentenceTransformerEmbedder,
    SemanticRetriever,
)

DEFAULT_REPO = Path("/home/sooraj/Downloads/embedding_and_chunking_result")
DEFAULT_QUERY = "def fibonacci_iterative(n: int) -> int:"
DEFAULT_OUTDIR = PROJECT_ROOT / "run_all_artifacts"

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "site-packages", "node_modules",
    "build", "dist", ".mypy_cache", ".pytest_cache", "run_all_artifacts",
}

TECHNIQUES = [
    "semantic",
    "ast",
    "iterative",
    "adaptive",
    "cross_file",
    "multiquery",
    "bm25",
    "hybrid",
]


# ============================================================
# Step 1: chunking
# ============================================================

def collect_python_files(repo_root: Path) -> List[Path]:
    return [
        path
        for path in sorted(Path(repo_root).rglob("*.py"))
        if not (SKIP_DIRS & set(path.parts))
    ]


def chunk_file(path: Path, repo_root: Path, window: int = 20, stride: int = 10) -> List[Document]:
    """RepoCoder-style sliding window over one file."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    documents: List[Document] = []
    relative = str(path.relative_to(repo_root))

    for start in range(0, max(len(lines), 1), stride):
        end = min(start + window, len(lines))
        chunk = lines[start:end]
        if len(chunk) < 3:
            continue
        documents.append(
            Document(
                content="\n".join(chunk),
                metadata={
                    "file_path": relative,
                    "repo": repo_root.name,
                    "start_line": start,
                    "end_line": end,
                },
            )
        )
        if end >= len(lines):
            break
    return documents


def run_chunking(repo_root: Path, outdir: Path, window: int, stride: int) -> Path:
    print("=" * 70)
    print("STEP 1  CHUNK")
    print("=" * 70)

    files = collect_python_files(repo_root)
    documents: List[Document] = []
    for path in files:
        documents.extend(chunk_file(path, repo_root, window, stride))

    output = outdir / "chunks.json"
    with open(output, "w") as handle:
        json.dump(
            [
                {"content": d.content, "metadata": d.metadata}
                for d in documents
            ],
            handle,
            indent=2,
        )
    print(f"Repository : {repo_root}")
    print(f"Files      : {len(files)}")
    print(f"Chunks     : {len(documents)}  -> {output}")
    return output


# ============================================================
# Step 2: embedding
# ============================================================

def run_embedding(chunks_file: Path, outdir: Path, model: str, device: str) -> Path:
    print()
    print("=" * 70)
    print("STEP 2  EMBED")
    print("=" * 70)

    with open(chunks_file, "r") as handle:
        chunks = json.load(handle)
    documents = [
        Document(content=item["content"], metadata=item.get("metadata", {}))
        for item in chunks
    ]

    embedder = SentenceTransformerEmbedder(model=model, device=device)
    embedded = embedder.embed_documents(documents)

    output = outdir / "embeddings.json"
    with open(output, "w") as handle:
        json.dump(
            [
                {
                    "content": d.content,
                    "metadata": d.metadata,
                    "embedding": d.embedding,
                }
                for d in embedded
            ],
            handle,
            indent=2,
        )

    dimension = len(embedded[0].embedding) if embedded else 0
    print(f"Chunks embedded : {len(embedded)}")
    print(f"Model           : {model if embedder.available else 'mock (random)'}")
    print(f"Dimension       : {dimension}")
    print(f"Saved           : {output}")
    return output


# ============================================================
# Step 3: vector store
# ============================================================

def run_vector_store(embeddings_file: Path, outdir: Path) -> Path:
    print()
    print("=" * 70)
    print("STEP 3  VECTOR STORE")
    print("=" * 70)

    with open(embeddings_file, "r") as handle:
        payload = json.load(handle)
    documents = [
        Document(
            content=item["content"],
            metadata=item.get("metadata", {}),
            embedding=item.get("embedding"),
        )
        for item in payload
    ]

    store = FaissVectorStore()
    store.add_documents(documents)
    index_path = outdir / "vector_store.faiss"
    store.save(index_path)

    backend = "FAISS IndexFlatIP" if store.use_faiss and store.index is not None else "numpy fallback"
    print(f"Backend    : {backend}")
    print(f"Documents  : {store.count}")
    print(f"Dimension  : {store.dimension}")
    print(f"Saved      : {index_path}")
    return index_path


# ============================================================
# Step 4: retrievers
# ============================================================

def ensure_ast_index(repo_root: Path, outdir: Path) -> Path:
    index_path = outdir / "ast_index.json"
    index = build_index(repo_root)
    index.save(index_path)
    kinds: Dict[str, int] = {}
    for element in index.elements:
        kinds[element.kind] = kinds.get(element.kind, 0) + 1
    print(f"AST index built: {len(index.elements)} units {kinds} -> {index_path}")
    return index_path


def build_retriever(
    technique: str,
    args: argparse.Namespace,
    ast_index_path: Path,
) -> Optional[Any]:
    embeddings = Path(args.embeddings)
    vector_store = Path(args.vector_store)
    repo = getattr(args, "repo", None)      # optional repository scope

    if technique == "semantic":
        return SemanticRetriever(embeddings, vector_store, top_k=args.top_k, repo=repo)

    if technique == "ast":
        return ASTBasedRetriever(
            ASTIndex.load(ast_index_path), top_k=args.top_k, hybrid=False
        )

    if technique == "iterative":
        return IterativeRetriever(
            semantic=SemanticRetriever(embeddings, vector_store, top_k=args.top_k,
                                        repo=repo),
            rounds=args.rounds,
            top_k=args.top_k,
            coverage_threshold=args.coverage_threshold,
        )

    if technique == "adaptive":
        return AdaptiveRetriever(
            semantic=SemanticRetriever(embeddings, vector_store, top_k=args.max_k,
                                        repo=repo),
            min_k=args.min_k,
            max_k=args.max_k,
            step=args.step,
            coverage_threshold=args.coverage_threshold,
        )

    if technique == "cross_file":
        return CrossFileRetriever(
            ASTIndex.load(ast_index_path),
            semantic=SemanticRetriever(embeddings, vector_store, repo=repo),
            hops=args.hops,
            max_per_file=args.max_per_file,
        )

    if technique == "multiquery":
        return MultiQueryRetriever(
            semantic=SemanticRetriever(embeddings, vector_store, top_k=args.top_k,
                                       repo=repo),
            structural=ASTBasedRetriever(
                ASTIndex.load(ast_index_path), top_k=args.top_k, hybrid=False
            ),
            top_k=args.top_k,
            index=ASTIndex.load(ast_index_path),
        )

    if technique == "bm25":
        return BM25Retriever(
            corpus=getattr(args, "corpus", "both"),
            chunks_file=Path(args.chunks),
            ast_index_file=ast_index_path,
            top_k=args.top_k,
        )

    if technique == "hybrid":
        return HybridRetriever(
            SemanticRetriever(embeddings, vector_store),
            BM25Retriever(
                corpus=getattr(args, "corpus", "both"),
                chunks_file=Path(args.chunks),
                ast_index_file=ast_index_path,
            ),
            HybridRetrievalConfig(top_k=args.top_k),
        )

    return None


# ============================================================
# Step 5: prompt
# ============================================================

def build_prompt(query: str, documents: Sequence[Document], max_examples: int = 6,
                 max_chars: int = 6000) -> str:
    """RepoCoder-style prompt: commented code fragments + the query."""
    separator = "# " + "-" * 50
    blocks: List[str] = []
    budget = max_chars

    for document in documents[:max_examples]:
        file_path = (document.metadata or {}).get("file_path", "unknown")
        body = "\n".join(f"# {line}" for line in document.content.splitlines())
        block = f"# the below code fragment can be found in:\n# {file_path}\n{separator}\n{body}\n{separator}\n"
        if len(block) > budget:
            break
        blocks.append(block)
        budget -= len(block)

    header = "# Here are some relevant code fragments from other files of the repo:\n"
    fragments = header + "\n".join(reversed(blocks))
    instruction = (
        "\n# Complete the code below using the repository conventions above.\n"
        "# Reply with code only.\n\n"
    )
    return fragments + instruction + query


# ============================================================
# Step 6: generation
# ============================================================

class MockGenerator:
    def __init__(self) -> None:
        self.model = "mock"

    def generate(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "fibonacci" in lowered:
            return "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"
        return "    # mock completion\n    pass"


class GeminiGenerator:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash",
                 temperature: float = 0.2, max_tokens: int = 1024) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        return gemini_client.generate(
            prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


def build_generator(skip_generate: bool) -> Any:
    if skip_generate:
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return GeminiGenerator(api_key)
    print("Note: GEMINI_API_KEY not set; using the mock generator.")
    return MockGenerator()


def load_reflection_layer(store_path: Optional[str], model: str) -> Optional[Any]:
    """Load scripts/09_reflection.py lazily; never break the pipeline."""
    if store_path is None and not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "reflection_layer", str(PROJECT_ROOT / "scripts" / "09_reflection.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not module.MEMORY_AVAILABLE:
            print(f"Note: reflection disabled (memory package unavailable: "
                  f"{module.MEMORY_IMPORT_ERROR})")
            return None
        return module.Reflector(model=model, store_path=store_path)
    except Exception as error:
        print(f"Note: reflection disabled ({error})")
        return None


# ============================================================
# Runner
# ============================================================

def run_technique(
    technique: str,
    query: str,
    retriever: Any,
    outdir: Path,
    generator: Optional[Any],
    reflector: Optional[Any] = None,
) -> Dict[str, Any]:
    print()
    print("=" * 70)
    print(f"TECHNIQUE  {technique}")
    print("=" * 70)

    # Step 4
    step_start = time.time()
    documents = retriever.retrieve_documents(query)
    retrieval_time = time.time() - step_start
    files = {(d.metadata or {}).get("file_path") for d in documents}
    print(f"STEP 4 RETRIEVE  {len(documents)} chunks from {len(files)} file(s) "
          f"in {retrieval_time:.2f}s")

    retrieval_path = outdir / f"retrieval_{technique}.json"
    with open(retrieval_path, "w") as handle:
        json.dump(
            {
                "query": query,
                "retriever_id": technique,
                "documents": [
                    {
                        "content": d.content,
                        "metadata": d.metadata,
                        "score": d.score,
                    }
                    for d in documents
                ],
            },
            handle,
            indent=2,
        )
    for rank, document in enumerate(documents[:5], start=1):
        metadata = document.metadata or {}
        print(f"   {rank}. {metadata.get('file_path')}"
              f":{metadata.get('start_line')}-{metadata.get('end_line')} "
              f"score={document.score}")

    # Step 5
    step_start = time.time()
    prompt = build_prompt(query, documents)
    prompt_time = time.time() - step_start
    prompt_path = outdir / f"prompt_{technique}.txt"
    prompt_path.write_text(prompt)
    print(f"STEP 5 PROMPT     ~{len(prompt) // 4} tokens in {prompt_time:.2f}s -> {prompt_path.name}")

    # Step 6
    completion = ""
    generation_time = 0.0
    model = "skipped"
    if generator is not None:
        step_start = time.time()
        try:
            completion = generator.generate(prompt)
            model = getattr(generator, "model", "unknown")
        except Exception as error:
            completion = f"# generation failed: {error}"
            model = "error"
        generation_time = time.time() - step_start
        (outdir / f"generation_{technique}.txt").write_text(completion)
        print(f"STEP 6 GENERATE   model={model} in {generation_time:.2f}s")
        print("   " + completion.replace("\n", "\n   ")[:400])
    else:
        print("STEP 6 GENERATE   skipped (--skip-generate)")

    # Reflection & memory extraction (post-generation, best effort)
    reflection_decision = "skipped"
    reflection_time = 0.0
    stored_ids: List[str] = []
    if reflector is not None and completion:
        step_start = time.time()
        try:
            outcome = reflector.reflect(
                task=query,
                retrieved_context=documents,
                generated_result=completion,
                store=True,
            )
            reflection_time = time.time() - step_start
            reflection_decision = outcome.decision
            stored_ids = outcome.stored_ids
            header = (
                f"# technique={technique}  generation_model={model}  "
                f"memory_store={outcome.store_path}"
            )
            (outdir / f"reflection_{technique}.txt").write_text(
                outcome.render(header) + "\n"
            )
            memory_line = outcome.memory_text or "(none)"
            print(f"REFLECTION       {outcome.decision} in {reflection_time:.2f}s")
            print(f"   memory: {memory_line[:160]}")
            if outcome.error:
                print(f"   note: {outcome.error}")
        except Exception as error:
            reflection_decision = "failed"
            print(f"REFLECTION       failed gracefully: {error}")

    return {
        "technique": technique,
        "documents": len(documents),
        "files": len(files),
        "retrieval_time": round(retrieval_time, 3),
        "prompt_tokens": len(prompt) // 4,
        "prompt_time": round(prompt_time, 3),
        "model": model,
        "generation_time": round(generation_time, 3),
        "completion_preview": completion[:200],
        "reflection": reflection_decision,
        "reflection_time": round(reflection_time, 3),
        "memories_stored": len(stored_ids),
        "retrieval_file": retrieval_path.name,
        "prompt_file": prompt_path.name,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run every RAG technique, steps 1-6")
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--techniques", default=",".join(TECHNIQUES))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max-per-file", type=int, default=2)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--no-reflect", action="store_true",
                        help="Skip the post-generation reflection/memory stage")
    parser.add_argument("--reflect-model", default="gemini-2.5-flash")
    parser.add_argument("--memory-store", default=None,
                        help="Override the shared memory store path "
                             "(default: memory pipeline's memory_store.pkl)")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo).resolve()
    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]

    if not repo_root.exists():
        print(f"Error: repository not found: {repo_root}")
        return

    print("=" * 70)
    print("RAG TECHNIQUE SUITE  (chunk -> embed -> store -> retrieve -> prompt -> generate)")
    print("=" * 70)
    print(f"Repository  : {repo_root}")
    print(f"Query       : {args.query!r}")
    print(f"Techniques  : {', '.join(techniques)}")
    print(f"Artifacts   : {outdir}")

    # Steps 1-3, shared by every technique
    chunks_file = run_chunking(repo_root, outdir, args.window, args.stride)
    embeddings_file = run_embedding(chunks_file, outdir, args.model, args.device)
    index_path = run_vector_store(embeddings_file, outdir)
    ast_index_path = ensure_ast_index(repo_root, outdir)

    # Point every technique at this run's artifacts.
    args.chunks = str(chunks_file)
    args.embeddings = str(embeddings_file)
    args.vector_store = str(index_path)

    generator = build_generator(args.skip_generate)
    reflector = None
    if not args.no_reflect:
        reflector = load_reflection_layer(args.memory_store, args.reflect_model)
        if reflector is None:
            print("Note: reflection stage not available; continuing without it.")
    summary: List[Dict[str, Any]] = []

    for technique in techniques:
        retriever = build_retriever(technique, args, ast_index_path)
        if retriever is None:
            print(f"Unknown technique: {technique}")
            continue
        try:
            summary.append(
                run_technique(technique, args.query, retriever, outdir, generator, reflector)
            )
        except Exception as error:
            print(f"TECHNIQUE {technique} FAILED: {error}")
            summary.append({"technique": technique, "error": str(error)})

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = (
        f"{'technique':<12}{'docs':>6}{'files':>7}{'tokens':>8}{'gen(s)':>9}"
        f"{'refl(s)':>9}{'mem':>5}  status"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        if "error" in row:
            print(f"{row['technique']:<12}{'-':>6}{'-':>7}{'-':>8}{'-':>9}{'-':>9}{'-':>5}  "
                  f"FAILED: {row['error']}")
            continue
        print(f"{row['technique']:<12}{row['documents']:>6}{row['files']:>7}"
              f"{row['prompt_tokens']:>8}{row['generation_time']:>9.2f}"
              f"{row.get('reflection_time', 0.0):>9.2f}"
              f"{row.get('memories_stored', 0):>5}  "
              f"{row['model']} / {row.get('reflection', 'skipped')}")

    with open(outdir / "summary.json", "w") as handle:
        json.dump({"query": args.query, "repo": str(repo_root), "results": summary}, handle, indent=2)
    print(f"\nAll artifacts written to {outdir}")


if __name__ == "__main__":
    main()