"""AST-based (structural) retrieval.

Indexes a repository with Python's `ast` module and retrieves whole code units -
functions, classes, methods, imports, module-level variables - ranked by
structural signals instead of raw text similarity:

  * symbol / qualified-name match against the query
  * function-call overlap (which callees the query names)
  * inheritance (base classes, decorators) and imports
  * docstring overlap
  * shallow body-token overlap
  * structural expansion (parent class of a matched method, callees of a match)

Optionally fuses those scores with text-retrieval results (`retrieval_data.json`)
for a hybrid AST + text ranking, and falls back to those candidates when the
query names nothing that exists in the AST.

Build once:  python -m retrieval.ast_retriever --index-only
"""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.base import Document, RetrievalResult  # noqa: E402

import ast  # noqa: E402

DEFAULT_REPO = Path("/home/sooraj/Downloads/embedding_and_chunking_result")
DEFAULT_QUERY = "Add authentication middleware to the API"
DEFAULT_INDEX = PROJECT_ROOT / "ast_index.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "ast_retrieval_data.json"
DEFAULT_TEXT_RETRIEVAL = PROJECT_ROOT / "retrieval_data.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "chunks.json"

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "site-packages", "node_modules",
    "build", "dist", ".mypy_cache", ".pytest_cache",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "add", "implement", "create", "make", "code", "file", "files", "this",
    "that", "it", "is", "are", "be", "should", "must", "can", "will", "how",
    "using", "use", "used", "into", "from", "by", "at", "as", "we", "i",
}


# ============================================================
# Tokenisation
# ============================================================

def split_identifier(name: str) -> List[str]:
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", name)
    return [part.lower() for part in parts] or [name.lower()]


def tokenize(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or ""):
        tokens.update(split_identifier(raw))
    return tokens - STOPWORDS


def dotted_name(node: ast.AST) -> str:
    parts: List[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


# ============================================================
# Index
# ============================================================

@dataclass
class CodeElement:
    element_id: str
    kind: str
    name: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    source: str
    docstring: str = ""
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None


class ASTIndex:
    def __init__(self) -> None:
        self.elements: List[CodeElement] = []
        self.by_id: Dict[str, CodeElement] = {}
        self.skipped_files: List[str] = []

    def add_many(self, elements: Sequence[CodeElement]) -> None:
        for element in elements:
            if element.element_id not in self.by_id:
                self.elements.append(element)
                self.by_id[element.element_id] = element

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_type": "ast_structural",
            "elements": [asdict(element) for element in self.elements],
            "skipped_files": self.skipped_files,
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ASTIndex":
        with open(path, "r") as handle:
            payload = json.load(handle)
        index = cls()
        index.skipped_files = payload.get("skipped_files", [])
        for raw in payload.get("elements", []):
            element = CodeElement(**raw)
            index.elements.append(element)
            index.by_id[element.element_id] = element
        return index


class ElementCollector(ast.NodeVisitor):
    def __init__(self, file_path: str, source: str) -> None:
        self.file_path = file_path
        self.lines = source.splitlines()
        self.elements: List[CodeElement] = []
        self.scope: List[Tuple[str, Optional[str]]] = []

    @property
    def parent_id(self) -> Optional[str]:
        return self.scope[-1][1] if self.scope else None

    def _qualified(self, node_name: str) -> str:
        return ".".join([scope[0] for scope in self.scope] + [node_name])

    def _segment(self, node: ast.AST) -> str:
        start = min(
            [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]
        )
        return "\n".join(self.lines[start - 1: node.end_lineno])

    def _calls(self, node: ast.AST) -> List[str]:
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = dotted_name(child.func)
                if name:
                    names.add(name)
        return sorted(names)

    def _make(
        self, node: ast.AST, kind: str, name: str, qualified_name: str, source: str
    ) -> CodeElement:
        start = min(
            [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]
        )
        return CodeElement(
            element_id=f"{self.file_path}::{qualified_name}#{start}",
            kind=kind,
            name=name,
            qualified_name=qualified_name,
            file_path=self.file_path,
            start_line=start,
            end_line=node.end_lineno,
            source=source,
            docstring=(ast.get_docstring(node) or "").strip(),
            bases=[dotted_name(base) for base in getattr(node, "bases", [])],
            decorators=[dotted_name(d) for d in getattr(node, "decorator_list", [])],
            calls=self._calls(node),
            parent_id=self.parent_id,
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        qualified = self._qualified(node.name)
        element = self._make(node, "class", node.name, qualified, self._segment(node))
        self.elements.append(element)

        self.scope.append((qualified, element.element_id))
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(statement)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        qualified = self._qualified(node.name)
        kind = "method" if self.scope else "function"
        element = self._make(node, kind, node.name, qualified, self._segment(node))
        self.elements.append(element)

        self.scope.append((qualified, element.element_id))
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(statement)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        names = [alias.name for alias in node.names]
        self.elements.append(
            CodeElement(
                element_id=f"{self.file_path}::import:{node.lineno}",
                kind="import",
                name=", ".join(names),
                qualified_name=f"import {', '.join(names)}",
                file_path=self.file_path,
                start_line=node.lineno,
                end_line=node.end_lineno,
                source=self._segment(node),
                imports=names,
                parent_id=self.parent_id,
            )
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or "."
        names = [f"{module}.{alias.name}" for alias in node.names]
        self.elements.append(
            CodeElement(
                element_id=f"{self.file_path}::import_from:{node.lineno}",
                kind="import",
                name=module,
                qualified_name=(
                    f"from {module} import {', '.join(alias.name for alias in node.names)}"
                ),
                file_path=self.file_path,
                start_line=node.lineno,
                end_line=node.end_lineno,
                source=self._segment(node),
                imports=names,
                parent_id=self.parent_id,
            )
        )

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if self.scope:
            return
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not names:
            return
        name = ", ".join(names)
        self.elements.append(
            CodeElement(
                element_id=f"{self.file_path}::{name}={node.lineno}",
                kind="variable",
                name=name,
                qualified_name=name,
                file_path=self.file_path,
                start_line=node.lineno,
                end_line=node.end_lineno,
                source=self._segment(node),
                calls=self._calls(node),
            )
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if self.scope or not isinstance(node.target, ast.Name):
            return
        name = node.target.id
        self.elements.append(
            CodeElement(
                element_id=f"{self.file_path}::{name}:{node.lineno}",
                kind="variable",
                name=name,
                qualified_name=name,
                file_path=self.file_path,
                start_line=node.lineno,
                end_line=node.end_lineno,
                source=self._segment(node),
                calls=self._calls(node),
            )
        )


def collect_python_files(repo_root: Path) -> List[Path]:
    return [
        path
        for path in sorted(Path(repo_root).rglob("*.py"))
        if not (SKIP_DIRS & set(path.parts))
    ]


def build_index(repo_root: Path) -> ASTIndex:
    index = ASTIndex()
    for path in collect_python_files(repo_root):
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            index.skipped_files.append(f"{path}: {error}")
            continue
        collector = ElementCollector(str(path), source)
        collector.visit(tree)
        index.add_many(collector.elements)
    return index


# ============================================================
# Scoring
# ============================================================

@dataclass
class ScoredElement:
    element: CodeElement
    score: float = 0.0
    text_score: float = 0.0
    reasons: List[str] = field(default_factory=list)


def _overlap(query_tokens: Set[str], tokens: Set[str]) -> float:
    if not query_tokens or not tokens:
        return 0.0
    return len(query_tokens & tokens) / len(query_tokens)


def score_element(element: CodeElement, query: str, query_tokens: Set[str]) -> ScoredElement:
    score = 0.0
    reasons: List[str] = []

    name_tokens = tokenize(element.name)
    qualified_tokens = tokenize(element.qualified_name)
    doc_tokens = tokenize(element.docstring)
    body_tokens = tokenize(element.source)
    call_tokens: Set[str] = set()
    for call in element.calls:
        call_tokens |= tokenize(call)
    struct_tokens: Set[str] = set()
    for reference in element.bases + element.decorators + element.imports:
        struct_tokens |= tokenize(reference)

    qualified_hit = _overlap(query_tokens, qualified_tokens)
    if qualified_hit:
        score += 3.0 * qualified_hit
        reasons.append(f"name match ({qualified_hit:.2f})")

    if element.qualified_name.lower() in query.lower() or element.name.lower() in query.lower():
        score += 3.0
        reasons.append("exact symbol mentioned in query")

    name_hit = _overlap(query_tokens, name_tokens)
    if name_hit:
        score += 2.0 * name_hit
        reasons.append(f"symbol token match ({name_hit:.2f})")

    call_hit = _overlap(query_tokens, call_tokens)
    if call_hit:
        score += 1.5 * call_hit
        reasons.append(f"call graph match ({call_hit:.2f})")

    struct_hit = _overlap(query_tokens, struct_tokens)
    if struct_hit:
        score += 1.0 * struct_hit
        reasons.append(f"inheritance/import match ({struct_hit:.2f})")

    doc_hit = _overlap(query_tokens, doc_tokens)
    if doc_hit:
        score += 1.0 * doc_hit
        reasons.append(f"docstring match ({doc_hit:.2f})")

    body_hit = _overlap(query_tokens, body_tokens)
    if body_hit:
        score += 0.5 * body_hit
        reasons.append(f"body token match ({body_hit:.2f})")

    if element.kind in query.lower():
        score += 0.3
        reasons.append(f"query mentions '{element.kind}'")

    return ScoredElement(element=element, score=score, reasons=reasons)


def expand_structurally(results: Sequence[ScoredElement], index: ASTIndex) -> List[ScoredElement]:
    """Add parent classes and same-file callees of the matched units."""
    selected = {result.element.element_id: result for result in results}

    for scored in list(results):
        parent_id = scored.element.parent_id
        if parent_id and parent_id in index.by_id and parent_id not in selected:
            selected[parent_id] = ScoredElement(
                element=index.by_id[parent_id],
                score=scored.score * 0.6,
                reasons=[f"owner of {scored.element.qualified_name}"],
            )

        called = set(scored.element.calls)
        if not called:
            continue
        for candidate in index.elements:
            if candidate.element_id in selected:
                continue
            if candidate.kind not in {"function", "method"}:
                continue
            if candidate.name in called:
                selected[candidate.element_id] = ScoredElement(
                    element=candidate,
                    score=scored.score * 0.4,
                    reasons=[f"callee of {scored.element.qualified_name}"],
                )

    return list(selected.values())


def is_future_context(element: CodeElement, hole_line: Optional[int]) -> bool:
    """RepoCoder-style filter: drop units that live after the hole in the file."""
    if not hole_line:
        return False
    return element.end_line > hole_line


# ============================================================
# Retriever
# ============================================================

class ASTBasedRetriever:
    def __init__(
        self,
        index: ASTIndex,
        top_k: int = 8,
        hybrid: bool = True,
        ast_weight: float = 0.75,
        hole_line: Optional[int] = None,
    ) -> None:
        self.index = index
        self.top_k = top_k
        self.hybrid = hybrid
        self.ast_weight = ast_weight
        self.hole_line = hole_line

    @property
    def ready(self) -> bool:
        return bool(self.index.elements)

    # -- helpers -------------------------------------------------------
    @staticmethod
    def load_text_scores(path: Path) -> List[Dict[str, Any]]:
        if not Path(path).exists():
            return []
        with open(path, "r") as handle:
            return json.load(handle).get("documents", [])

    @staticmethod
    def text_score_for_element(
        element: CodeElement, text_docs: Sequence[Dict[str, Any]]
    ) -> Tuple[float, List[str]]:
        if not text_docs:
            return 0.0, []

        total = len(text_docs)
        best = 0.0
        reasons: List[str] = []
        for rank, doc in enumerate(text_docs):
            rank_score = 1.0 - rank / total
            metadata = doc.get("metadata", {})
            if metadata.get("file_path") != element.file_path:
                continue

            start, end = metadata.get("start_line"), metadata.get("end_line")
            if isinstance(start, int) and isinstance(end, int):
                if not (start <= element.end_line and end >= element.start_line):
                    continue
                value = rank_score
                if value > best:
                    best = value
                    reasons = [f"overlaps text chunk at rank {rank + 1}"]
            else:
                value = rank_score * 0.3
                if value > best:
                    best = value
                    reasons = [f"same file as text chunk at rank {rank + 1}"]
        return best, reasons

    def _fallback_from_text(self, text_docs: Sequence[Dict[str, Any]]) -> List[ScoredElement]:
        """Map text/embedding chunks onto the AST units that contain them."""
        fallback: Dict[str, ScoredElement] = {}
        total = max(len(text_docs), 1)

        for rank, doc in enumerate(text_docs):
            metadata = doc.get("metadata", {})
            file_path = metadata.get("file_path")
            if not file_path:
                continue
            rank_score = 1.0 - rank / total
            start, end = metadata.get("start_line"), metadata.get("end_line")

            for element in self.index.elements:
                if element.file_path != file_path:
                    continue
                if is_future_context(element, self.hole_line):
                    continue
                if isinstance(start, int) and isinstance(end, int):
                    if not (start <= element.end_line and end >= element.start_line):
                        continue
                    value = rank_score
                    reason = f"contains text chunk at rank {rank + 1}"
                else:
                    value = rank_score * 0.3
                    reason = f"same file as text chunk at rank {rank + 1}"

                existing = fallback.get(element.element_id)
                if existing is None or value > existing.score:
                    fallback[element.element_id] = ScoredElement(
                        element=element, score=value, reasons=[reason]
                    )
        return list(fallback.values())

    # -- retrieval -----------------------------------------------------
    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        top_k = top_k or self.top_k
        query_tokens = tokenize(query)

        scored = [
            score_element(element, query, query_tokens)
            for element in self.index.elements
        ]
        scored = [
            item for item in scored
            if item.score > 0 and not is_future_context(item.element, self.hole_line)
        ]

        text_docs = self.load_text_scores(DEFAULT_TEXT_RETRIEVAL) if self.hybrid else []
        if not scored and text_docs:
            scored = self._fallback_from_text(text_docs)

        if self.hybrid and text_docs:
            for item in scored:
                item.text_score, reasons = self.text_score_for_element(item.element, text_docs)
                item.reasons.extend(reasons)

        max_ast = max((item.score for item in scored), default=0.0) or 1.0
        for item in scored:
            normalised = item.score / max_ast
            item.score = (
                self.ast_weight * normalised + (1 - self.ast_weight) * item.text_score
                if self.hybrid and text_docs
                else normalised
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        scored = expand_structurally(scored, self.index)
        scored = [
            item for item in scored
            if not is_future_context(item.element, self.hole_line) and item.score > 0
        ]
        scored.sort(key=lambda item: item.score, reverse=True)

        documents: List[Document] = []
        for item in scored[:top_k]:
            element = item.element
            documents.append(
                Document(
                    content=element.source,
                    metadata={
                        "file_path": element.file_path,
                        "start_line": element.start_line,
                        "end_line": element.end_line,
                        "element_id": element.element_id,
                        "kind": element.kind,
                        "name": element.name,
                        "qualified_name": element.qualified_name,
                        "parent_id": element.parent_id,
                        "docstring": element.docstring,
                        "bases": element.bases,
                        "decorators": element.decorators,
                        "calls": element.calls,
                        "match_reasons": item.reasons,
                    },
                    score=round(item.score, 4),
                )
            )
        return documents

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            documents=self.retrieve_documents(query, top_k),
            retriever_id="ast_based_retrieval",
        )


# ============================================================
# CLI
# ============================================================

def infer_repo(chunks_file: Path, explicit_repo: Optional[str]) -> Path:
    if explicit_repo:
        return Path(explicit_repo)
    if chunks_file.exists():
        with open(chunks_file, "r") as handle:
            for item in json.load(handle):
                file_path = item.get("metadata", {}).get("file_path")
                if file_path and Path(file_path).exists():
                    return Path(file_path).parent
    return DEFAULT_REPO


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="AST-based structural retrieval")
    parser.add_argument("--repo", default=None, help="Repository root to index")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--retrieval", default=str(DEFAULT_TEXT_RETRIEVAL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--ast-weight", type=float, default=0.75)
    parser.add_argument("--hole-line", type=int, default=None)
    parser.add_argument("--index-only", action="store_true",
                        help="Only (re)build the AST index and exit")
    args = parser.parse_args(argv)

    index_path = Path(args.index)
    output_path = Path(args.output)
    repo_root = infer_repo(DEFAULT_CHUNKS, args.repo)

    print("=" * 60)
    print("AST-Based Retrieval")
    print("=" * 60)
    print(f"Repository: {repo_root}")

    if not repo_root.exists():
        print(f"Error: repository not found: {repo_root}")
        return

    if index_path.exists():
        index = ASTIndex.load(index_path)
        print(f"Loaded cached index with {len(index.elements)} elements")
    else:
        index = build_index(repo_root)
        index.save(index_path)
        print(f"Indexed {len(collect_python_files(repo_root))} python files")

    kinds = Counter(element.kind for element in index.elements)
    print(f"Indexed elements: {len(index.elements)} {dict(kinds)}")
    print(f"Index saved to {index_path}")

    if args.index_only:
        return

    retriever = ASTBasedRetriever(
        index,
        top_k=args.top_k,
        hybrid=not args.no_hybrid,
        ast_weight=args.ast_weight,
        hole_line=args.hole_line,
    )
    result = retriever.retrieve(args.query)

    text_docs = ASTBasedRetriever.load_text_scores(Path(args.retrieval)) if not args.no_hybrid else []
    if text_docs:
        print(f"Hybrid fusion with {len(text_docs)} text-retrieval candidates")
    print(f"\n{'=' * 60}")
    print(f"Query: {result.query}")
    print("=" * 60)
    print(f"Retrieved {len(result.documents)} code units")

    structural_only = result.documents and all(
        any("text chunk" in reason for reason in (doc.metadata or {}).get("match_reasons", []))
        for doc in result.documents
    )

    for rank, doc in enumerate(result.documents, start=1):
        metadata = doc.metadata
        print(f"\n--- #{rank} [{metadata['kind']}] score={doc.score:.4f} ---")
        print(f"{metadata['qualified_name']}  ({Path(metadata['file_path']).name}:"
              f"{metadata['start_line']}-{metadata['end_line']})")
        if metadata.get("match_reasons"):
            print(f"match: {', '.join(metadata['match_reasons'][:4])}")
        print(f"{doc.content[:300]}")

    if structural_only:
        print("\nNote: no structural symbol matched this query; results fall back "
              "to text-retriever candidates mapped onto AST units.")

    payload = {
        "query": result.query,
        "retriever_id": result.retriever_id,
        "documents": [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "score": doc.score,
            }
            for doc in result.documents
        ],
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nResults saved to {output_path}")
    print("Next: copy it over retrieval_data.json to run scripts/05_run_prompt.py")


if __name__ == "__main__":
    main()