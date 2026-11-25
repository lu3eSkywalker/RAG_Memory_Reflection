#!/usr/bin/env python3
"""
01_run_chunking.py — chunk every embeddable file in the corpus.

The corpus location and the filter rules live in `code_corpus.py`, which is
shared with 02_run_embedding.py, so both stages always agree on what is embedded:

    root      : /home/sooraj/Downloads/data_for_rag/sample_testing_new/data_to_embed
                (scanned recursively - every repository inside is picked up
                 automatically, no folder needs to be named)
    embed     : *.py only, minus tests/ minus docs/ minus .github/ minus CI/config
    skip      : LICENSE, Makefile, *.txt, *.yml, *.yaml, *.md, *.json, *.ipynb
    chunking  : one chunk per function (AST based), fallback 50-line windows

Writes `chunks.json` (schema unchanged, consumed by 02/03/04/05/06).

Usage:
    python scripts/01_run_chunking.py
    python scripts/01_run_chunking.py --root /path/to/corpus --chunker window
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.base import Document  # noqa: E402

import code_corpus  # noqa: E402


def save_chunks(chunks, output_path: Path):
    with open(output_path, "w") as handle:
        json.dump(chunks, handle, indent=2)


def print_report(stats, chunks, output_path: Path):
    print("=" * 70)
    print("STEP 1  CHUNKING")
    print("=" * 70)
    print(f"Corpus root   : {stats['root']}")
    print(f"Repositories  : {', '.join(stats['repositories'])}")
    print(f"Chunker       : {stats['chunker']} (one chunk per function; "
          f"{code_corpus.WINDOW_SIZE}-line window fallback)")
    print(f"Files embedded: {stats['total_files']} "
          f"(parsed {stats['files_parsed']}, failed {stats['files_failed']})")
    print(f"Chunks created: {stats['total_chunks']}")

    print()
    print(f"{'repository':<28}{'files':>7}{'functions':>11}{'windows':>9}{'chunks':>8}")
    print("-" * 63)
    for repo, repo_stats in sorted(stats["per_repo"].items()):
        print(f"{repo:<28}{repo_stats['files']:>7}{repo_stats['functions']:>11}"
              f"{repo_stats['windows']:>9}{repo_stats['chunks']:>8}")

    print()
    print("Sample chunks:")
    for chunk in chunks[:5]:
        metadata = chunk["metadata"]
        print(f"  [{metadata['kind']:<8}] {metadata['qualified_name']}  "
              f"({metadata['repo']}/{metadata['file_path']}:"
              f"{metadata['start_line'] + 1}-{metadata['end_line']})")

    print()
    print(f"Saved -> {output_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Chunk the RAG corpus (one chunk per function)")
    parser.add_argument("--root", default=str(code_corpus.DEFAULT_EMBED_ROOT),
                        help="Corpus root; scanned recursively")
    parser.add_argument("--chunker", default="ast_function",
                        choices=sorted(code_corpus.CHUNKERS),
                        help="ast_function (default) or window")
    parser.add_argument("--output", default="chunks.json")
    parser.add_argument("--limit-per-repo", type=int, default=None,
                        help="Only chunk N files per repository (debugging)")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"Error: corpus root not found: {root}")
        return 1

    chunks, stats = code_corpus.chunk_corpus(
        root=root,
        chunker=args.chunker,
        limit_per_repo=args.limit_per_repo,
    )

    if not chunks:
        print(f"Error: no chunks produced from {root} (check the filter rules)")
        return 1

    output_path = Path(args.output)
    save_chunks(chunks, output_path)
    print_report(stats, chunks, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

