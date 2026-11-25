"""Shared rules for what gets chunked and embedded, plus the chunkers.

Both `scripts/01_run_chunking.py` and `scripts/02_run_embedding.py` import this
module, so the corpus location and the "what to embed / what to skip" rules exist
exactly once.

What to actually embed
    embed:  *.py only, minus tests/ minus docs/ minus .github/ minus CI/config files
    skip:   LICENSE, Makefile, *.txt, *.yml, *.yaml, *.md, *.json, *.ipynb
    chunks: one per function (ast-based), fallback 50-line windows

The corpus root is scanned recursively, so every repository inside it is picked
up automatically - there is no need to name individual folders.
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Corpus location and filters
# --------------------------------------------------------------------------
DEFAULT_EMBED_ROOT = Path(
    "/home/sooraj/Downloads/data_for_rag/combined_for_exp"
)

EMBED_SUFFIXES = {".py"}

SKIP_SUFFIXES = {
    ".txt", ".yml", ".yaml", ".md", ".json", ".ipynb",
    ".rst", ".mdx", ".toml", ".cfg", ".ini", ".cfg.py",
}

SKIP_FILENAMES = {
    "license", "licence", "copying", "makefile", "readme", "notice",
    "authors", "changelog", "contributing", "codeowners",
}

# Python files that are really project/CI configuration rather than source code.
SKIP_PY_FILENAMES = {"setup.py", "conftest.py", "noxfile.py", "fabfile.py"}
CI_PY_PATTERN = re.compile(r"(^ci_|_ci$|^_ci_|ci_.*)\.py$", re.I)

SKIP_DIR_NAMES = {
    "tests", "test", "testing", "docs", "doc", "examples_ci",
    ".github", ".circleci", ".gitlab", ".travis",
    "venv", ".venv", "env", "__pycache__", "node_modules",
    "build", "dist", "site-packages", ".mypy_cache", ".pytest_cache", ".tox",
}

WINDOW_SIZE = 50
MIN_CHUNK_LINES = 3


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def should_embed(path: Path, root: Path) -> bool:
    """True when the file belongs in the corpus (suffix + directory + name rules)."""
    path = Path(path)
    if path.suffix.lower() not in EMBED_SUFFIXES:
        return False

    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    for part in relative.parts[:-1]:
        if part.lower() in SKIP_DIR_NAMES:
            return False
        if part.startswith("."):          # hidden dirs (.github, .circleci, ...)
            return False

    stem = path.stem.lower()
    if stem in SKIP_FILENAMES or path.name.lower() in SKIP_FILENAMES:
        return False
    if stem in SKIP_PY_FILENAMES or CI_PY_PATTERN.search(path.name):
        return False

    return True


def discover_files(root: Path = DEFAULT_EMBED_ROOT) -> List[Tuple[str, Path]]:
    """(repo_name, path) for every embeddable file under root, sorted."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus root not found: {root}")

    found: List[Tuple[str, Path]] = []
    for path in sorted(root.rglob("*.py")):
        if not path.is_file() or not should_embed(path, root):
            continue
        relative = path.relative_to(root)
        repo = relative.parts[0] if len(relative.parts) > 1 else root.name
        found.append((repo, path))
    return found


def repo_names(files: Sequence[Tuple[str, Path]]) -> List[str]:
    seen: List[str] = []
    for repo, _ in files:
        if repo not in seen:
            seen.append(repo)
    return seen


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def _source_segment(lines: Sequence[str], node: ast.AST) -> Tuple[int, str]:
    """(start_line, text) for a node including its decorators."""
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    end = getattr(node, "end_lineno", node.lineno)
    return start, "\n".join(lines[start - 1: end])


def window_chunks(
    source: str,
    metadata: Dict[str, Any],
    size: int = WINDOW_SIZE,
) -> List[Dict[str, Any]]:
    """Fallback chunking: fixed line windows."""
    lines = source.splitlines()
    chunks: List[Dict[str, Any]] = []
    if not lines:
        return chunks

    for start in range(0, len(lines), size):
        end = min(start + size, len(lines))
        block = lines[start:end]
        if len(block) < min(MIN_CHUNK_LINES, len(lines)):
            continue
        chunks.append(
            {
                "content": "\n".join(block),
                "metadata": {
                    **metadata,
                    "name": f"{metadata.get('file_path', 'file')}:{start + 1}-{end}",
                    "qualified_name": f"{metadata.get('file_path', 'file')}::{start + 1}-{end}",
                    "kind": "window",
                    "start_line": start,
                    "end_line": end,
                    "parent": None,
                    "chunker": f"window_{size}",
                },
            }
        )
    return chunks


def ast_function_chunks(source: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One chunk per function/method (nested included); windows if none found."""
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return window_chunks(source, metadata)

    chunks: List[Dict[str, Any]] = []
    scope: List[str] = []

    def qualify(name: str, parent: Optional[str]) -> str:
        return f"{parent}.{name}" if parent else name

    def collect(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                scope.append(child.name)
                collect(child)
                scope.pop()
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent = ".".join(scope) or None
                start, text = _source_segment(lines, child)
                chunks.append(
                    {
                        "content": text,
                        "metadata": {
                            **metadata,
                            "name": child.name,
                            "qualified_name": qualify(child.name, parent),
                            "kind": "method" if parent else "function",
                            "start_line": start,
                            "end_line": child.end_lineno,
                            "parent": parent,
                            "chunker": "ast_function",
                        },
                    }
                )
                scope.append(child.name)
                collect(child)
                scope.pop()
            else:
                collect(child)

    collect(tree)
    if not chunks:
        return window_chunks(source, metadata)
    return chunks


CHUNKERS = {
    "ast_function": ast_function_chunks,
    "window": lambda source, metadata: window_chunks(source, metadata, WINDOW_SIZE),
}


def chunk_corpus(
    root: Path = DEFAULT_EMBED_ROOT,
    chunker: str = "ast_function",
    limit_per_repo: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Chunk every embeddable file under root.

    Returns (chunks, stats) where stats carries per-repo counts for reporting.
    """
    root = Path(root)
    chunk_function = CHUNKERS.get(chunker, ast_function_chunks)

    files = discover_files(root)
    per_repo: Dict[str, Dict[str, int]] = {}
    chunks: List[Dict[str, Any]] = []
    parsed = 0
    failed = 0

    for repo, path in files:
        repo_stats = per_repo.setdefault(repo, {"files": 0, "chunks": 0, "functions": 0,
                                                 "windows": 0})
        if limit_per_repo is not None and repo_stats["files"] >= limit_per_repo:
            continue
        repo_stats["files"] += 1

        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            failed += 1
            continue

        metadata = {
            "file_path": str(path.relative_to(root)),
            "absolute_path": str(path),
            "repo": repo,
            "repo_root": str(root),
        }

        try:
            file_chunks = chunk_function(source, metadata)
            parsed += 1
        except Exception:
            failed += 1
            file_chunks = window_chunks(source, metadata)

        chunks.extend(file_chunks)
        repo_stats["chunks"] += len(file_chunks)
        for chunk in file_chunks:
            if chunk["metadata"].get("kind") == "window":
                repo_stats["windows"] += 1
            else:
                repo_stats["functions"] += 1

    stats = {
        "root": str(root),
        "chunker": chunker,
        "repositories": repo_names(files),
        "total_files": sum(v["files"] for v in per_repo.values()),
        "total_chunks": len(chunks),
        "files_parsed": parsed,
        "files_failed": failed,
        "per_repo": per_repo,
    }
    return chunks, stats