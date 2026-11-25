#!/usr/bin/env python3
"""
11_retrieve_questions.py — retrieve context for every question in a folder.

For each row of every *.jsonl in the questions folder:

    query (tail of the prompt, the code before the hole)
        -> embed + retrieve top-k from the existing index
        -> restricted to the question's own repository (default)
        -> build the RepoCoder-style prompt
        -> append a readable block to questions_test.txt
        -> append a machine-readable row to retrieval_all.jsonl

The corpus index is whatever 01/02/03 already produced (chunks.json,
embeddings.json, vector_store.faiss), so this stage never re-embeds the corpus.

Usage:
    python scripts/11_retrieve_questions.py \
        --questions /home/sooraj/Downloads/sample_data_to_test/questions \
        --out questions_test.txt --technique hybrid --top-k 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.base import Document  # noqa: E402

import run_all_techniques as pipeline  # noqa: E402

logger = logging.getLogger("retrieve_questions")

DEFAULT_QUESTIONS = Path("/home/sooraj/Downloads/sample_data_to_test/questions")
DEFAULT_CHUNKS = PROJECT_ROOT / "chunks.json"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "embeddings.json"
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "vector_store.faiss"

TECHNIQUES = ("semantic", "bm25", "ast", "hybrid", "multiquery", "cross_file",
              "iterative", "adaptive")


# ============================================================
# Question loading
# ============================================================

def load_questions(folder: Path) -> List[Dict[str, Any]]:
    """One record per JSONL row across every file in the folder."""
    folder = Path(folder)
    files = sorted(folder.glob("*.jsonl")) or sorted(folder.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl files found in {folder}")

    cases: List[Dict[str, Any]] = []
    for path in files:
        with open(path, "r") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                metadata = row.get("metadata", {}) or {}
                cases.append(
                    {
                        "source_file": path.name,
                        "row": number,
                        "prompt": row.get("prompt", ""),
                        "metadata": metadata,
                        "task_id": metadata.get("task_id", f"{path.name}:{number}"),
                        "repo": repo_of(metadata, path.name),
                        "ground_truth": metadata.get("ground_truth", ""),
                    }
                )
    return cases


def repo_of(metadata: Dict[str, Any], fallback: str = "") -> str:
    """Repository folder for a question: fpath_tuple[0] else task_id prefix."""
    fpath = metadata.get("fpath_tuple")
    if isinstance(fpath, (list, tuple)) and fpath:
        return str(fpath[0])
    task_id = str(metadata.get("task_id", ""))
    if "/" in task_id:
        return task_id.split("/")[0]
    return fallback or "unknown"


# ============================================================
# Query construction
# ============================================================

def build_query(prompt: str, query_lines: int) -> str:
    """Retrieval query = the code immediately before the hole (prompt tail).

    The full prompt is thousands of characters; embedding all of it dilutes the
    signal and blows past the model's token window, so only the tail is used.
    """
    lines = (prompt or "").splitlines()
    tail = lines[-query_lines:] if query_lines and len(lines) > query_lines else lines
    return "\n".join(tail).strip()


def completion_context(prompt: str, context_lines: int) -> str:
    """The code the model must continue."""
    lines = (prompt or "").splitlines()
    return "\n".join(lines[-context_lines:]) if context_lines else "\n".join(lines)


# ============================================================
# Retrieval
# ============================================================

def known_repos(chunks_file: Path) -> List[str]:
    if not chunks_file.exists():
        return []
    with open(chunks_file, "r") as handle:
        repos = {
            (item.get("metadata", {}) or {}).get("repo", "?")
            for item in json.load(handle)
        }
    return sorted(repos)


def filter_by_repo(documents: Sequence[Document], repo: str, top_k: int,
                  strict: bool) -> List[Document]:
    """Keep same-repo chunks first, then optionally backfill with others."""
    same = [d for d in documents if (d.metadata or {}).get("repo") == repo]
    if strict:
        return same[:top_k]
    others = [d for d in documents if (d.metadata or {}).get("repo") != repo]
    return (same + others)[:top_k]


def retrieve_for_case(retriever, query: str, repo: str, top_k: int,
                      pool_multiplier: int, strict: bool) -> List[Document]:
    pool = retriever.retrieve_documents(query, top_k=top_k * pool_multiplier)
    return filter_by_repo(pool, repo, top_k, strict)


# Dense-only techniques: scope the corpus to the question's repository *before*
# ranking. Ranking first and filtering afterwards can leave nothing usable,
# because a purely semantic query can fill the entire top-k with lookalike code
# from another repository (observed: 12/12 awslabs_fortuna chunks for a
# huggingface_evaluate question).
DENSE_TECHNIQUES = {"semantic", "iterative", "adaptive"}
_SCOPED_RETRIEVERS: Dict[str, Any] = {}


def scoped_retriever(retriever_args, technique: str, repo: str, ast_index_path: Path):
    """One retriever per repository, built once and reused."""
    if repo in _SCOPED_RETRIEVERS:
        return _SCOPED_RETRIEVERS[repo]
    scoped_args = argparse.Namespace(**vars(retriever_args))
    scoped_args.repo = repo
    retriever = pipeline.build_retriever(technique, scoped_args, ast_index_path)
    _SCOPED_RETRIEVERS[repo] = retriever
    return retriever


# ============================================================
# Prompt assembly
# ============================================================

def build_case_prompt(documents: Sequence[Document], query: str,
                      context: str) -> str:
    """RepoCoder-style prompt that asks for the missing lines only."""
    separator = "# " + "-" * 50
    blocks: List[str] = []
    for document in documents:
        metadata = document.metadata or {}
        body = "\n".join(f"# {line}" for line in (document.content or "").splitlines())
        blocks.append(
            f"# the below code fragment can be found in:\n"
            f"# {metadata.get('file_path', 'unknown')}"
            f":{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}\n"
            f"{separator}\n{body}\n{separator}\n"
        )

    fragments = (
        "# Here are relevant code fragments from the repository:\n"
        + "\n".join(reversed(blocks))
    )
    return (
        fragments
        + "\n# Complete the code below using the repository conventions above.\n"
          "# Reply with ONLY the missing code lines: no explanation, no markdown fences,\n"
          "# no repeated lines that are already shown.\n\n"
          "# Existing code (the block to continue):\n"
        + context
        + "\n\n# Missing code:\n"
    )


# ============================================================
# Output
# ============================================================

def write_txt(path: Path, header: Dict[str, Any], case: Dict[str, Any],
              documents: Sequence[Document], prompt: str) -> None:
    metadata = case["metadata"]
    with open(path, "a") as handle:
        handle.write("=" * 78 + "\n")
        handle.write(f"CASE {case['index']}/{header['total_cases']}  "
                     f"{case['source_file']}:{case['row']}\n")
        handle.write("=" * 78 + "\n")
        handle.write(f"task_id              : {case['task_id']}\n")
        handle.write(f"repo                 : {case['repo']}\n")
        handle.write(f"file                 : {'/'.join(metadata.get('fpath_tuple', []))}\n")
        handle.write(f"line_no              : {metadata.get('line_no')}\n")
        handle.write(f"context_start_lineno : {metadata.get('context_start_lineno')}\n")
        handle.write(f"retrieval technique  : {header['technique']}\n")
        handle.write(f"retrieved            : {len(documents)} chunk(s)\n\n")

        handle.write("--- RETRIEVED CONTEXT ---\n")
        for rank, document in enumerate(documents, start=1):
            m = document.metadata or {}
            handle.write(
                f"#{rank} {m.get('file_path')}:{m.get('start_line')}-{m.get('end_line')}"
                f"  score={document.score}  kind={m.get('kind', 'chunk')}\n"
            )
            handle.write(f"{document.content}\n\n")

        handle.write("--- RETRIEVAL QUERY (prompt tail) ---\n")
        handle.write(f"{case['query']}\n\n")

        handle.write("--- PROMPT SENT TO THE LLM ---\n")
        handle.write(f"{prompt}\n\n")

        handle.write("--- GROUND TRUTH (for reference; never sent to the LLM) ---\n")
        handle.write(f"{case['ground_truth']}\n\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieve context for every question")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS),
                        help="Folder containing *.jsonl question files")
    parser.add_argument("--technique", default="hybrid", choices=sorted(TECHNIQUES))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool-multiplier", type=int, default=20,
                        help="Fetch this many x top-k candidates before repo filtering")
    parser.add_argument("--query-lines", type=int, default=20,
                        help="Prompt tail lines used as the retrieval query")
    parser.add_argument("--context-lines", type=int, default=40,
                        help="Prompt tail lines given to the LLM as the block to continue")
    parser.add_argument("--all-repos", action="store_true",
                        help="Search the whole corpus instead of the question's repo")
    parser.add_argument("--scope-retrieval", default="dense",
                        choices=["dense", "none"],
                        help="dense: restrict the candidate pool to the question's "
                             "repo before ranking (dense techniques only); "
                             "none: rank globally then filter")
    parser.add_argument("--corpus", default="chunks", choices=["chunks", "ast", "both"],
                        help="BM25 corpus; 'chunks' avoids mixing an AST index built "
                             "from a different corpus")
    parser.add_argument("--min-k", type=int, default=2,
                        help="adaptive_retriever: first batch size")
    parser.add_argument("--step", type=int, default=4,
                        help="adaptive_retriever: batch growth step")
    parser.add_argument("--max-k", type=int, default=0,
                        help="adaptive_retriever: k budget (0 = auto, max(top-k,12))")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="questions_test.txt")
    parser.add_argument("--jsonl-out", default="retrieval_all.jsonl")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--vector-store", default=str(DEFAULT_VECTOR_STORE))
    parser.add_argument("--ast-index", default=str(pipeline.DEFAULT_OUTDIR / "ast_index.json")
                        if hasattr(pipeline, "DEFAULT_OUTDIR")
                        else "ast_index.json")
    parser.add_argument("--reuse", action="store_true",
                        help="Append to existing files instead of starting fresh")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    questions_folder = Path(args.questions).expanduser()
    if not questions_folder.is_dir():
        print(f"Error: questions folder not found: {questions_folder}")
        return 1

    cases = load_questions(questions_folder)
    if args.limit:
        cases = cases[: args.limit]
    for position, case in enumerate(cases, start=1):
        case["index"] = position

    print("=" * 70)
    print("RETRIEVAL FOR EVERY QUESTION")
    print("=" * 70)
    print(f"Questions  : {questions_folder} ({len(cases)} cases)")
    print(f"Technique  : {args.technique}   top-k: {args.top_k}")
    print(f"Repo filter: {'off (whole corpus)' if args.all_repos else 'on (per repository)'}")
    print(f"Corpus repos: {', '.join(known_repos(Path(args.chunks)))}")

    # Reuse the tested retriever factory from the technique suite.
    retriever_args = argparse.Namespace(
        top_k=args.top_k, max_k=args.max_k or max(args.top_k, 12),
        min_k=args.min_k, step=args.step,
        rounds=2, coverage_threshold=0.8, hops=2, max_per_file=args.top_k,
        chunks=args.chunks, embeddings=args.embeddings, vector_store=args.vector_store,
        ast_index=args.ast_index, corpus=args.corpus,
    )
    retriever = pipeline.build_retriever(args.technique, retriever_args, Path(args.ast_index))
    if retriever is None or not getattr(retriever, "ready", False):
        print("Error: retriever unavailable. Run 01/02/03 first (and build the AST index "
              "for AST-based techniques).")
        return 1

    txt_path = Path(args.out)
    jsonl_path = Path(args.jsonl_out)
    if not args.reuse:
        for path in (txt_path, jsonl_path):
            if path.exists():
                path.unlink()

    header = {
        "total_cases": len(cases),
        "technique": args.technique,
        "top_k": args.top_k,
        "questions_folder": str(questions_folder),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    started = time.time()
    use_scope = (
        not args.all_repos
        and args.scope_retrieval == "dense"
        and args.technique in DENSE_TECHNIQUES
    )
    if use_scope:
        print(f"Scope        : per-repository index ({args.technique})")

    with open(jsonl_path, "w") as jsonl:
        for case in cases:
            case["query"] = build_query(case["prompt"], args.query_lines)
            context = completion_context(case["prompt"], args.context_lines)

            if use_scope:
                scoped = scoped_retriever(
                    retriever_args, args.technique, case["repo"], Path(args.ast_index)
                )
                documents = scoped.retrieve_documents(case["query"], top_k=args.top_k)
            else:
                documents = retrieve_for_case(
                    retriever, case["query"], case["repo"],
                    args.top_k, args.pool_multiplier, args.all_repos,
                )
            prompt = build_case_prompt(documents, case["query"], context)

            case["retrieved"] = [
                {
                    "file_path": (d.metadata or {}).get("file_path"),
                    "start_line": (d.metadata or {}).get("start_line"),
                    "end_line": (d.metadata or {}).get("end_line"),
                    "repo": (d.metadata or {}).get("repo"),
                    "kind": (d.metadata or {}).get("kind"),
                    "score": d.score,
                    "content": d.content,
                }
                for d in documents
            ]

            write_txt(txt_path, header, case, documents, prompt)

            jsonl.write(
                json.dumps(
                    {
                        "metadata": {
                            "task_id": case["task_id"],
                            "source_file": case["source_file"],
                            "row": case["row"],
                            "repo": case["repo"],
                            "fpath_tuple": case["metadata"].get("fpath_tuple"),
                            "line_no": case["metadata"].get("line_no"),
                            "context_start_lineno": case["metadata"].get("context_start_lineno"),
                            "ground_truth": case["ground_truth"],
                            "query": case["query"],
                            "prompt": prompt,
                            "retrieved": case["retrieved"],
                        }
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            jsonl.flush()

            same_repo = sum(1 for d in case["retrieved"] if d["repo"] == case["repo"])
            print(f"  case {case['index']:>2}/{len(cases)}  {case['task_id']:<32}"
                  f" repo={case['repo']:<22} chunks={len(documents)} (same-repo {same_repo})")

    elapsed = time.time() - started
    print()
    print(f"Retrieval done in {elapsed:.2f}s")
    print(f"Readable report -> {txt_path}")
    print(f"Machine readable -> {jsonl_path}")
    print(f"Next: python scripts/12_generate_answers.py --input {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())