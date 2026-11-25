"""Cross-file relationship index.

Builds a directed graph of code units from the AST index so retrieval can follow
relationships between files:

    calls        a function/method calls a symbol defined in another file
    extends      a class inherits from a symbol defined in another file
    imports      a file imports a symbol defined in another file
    references   source text mentions a symbol defined in another file
    test-for     a test file exercises a symbol defined in another file
    uses-imports an element depends on what its file imports

Every edge keeps a weight and a human readable `via` string; reverse traversal
(with a penalty) exposes callers, tests and importers.
"""

import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.base import Document  # noqa: E402

from retrieval.ast_retriever import ASTIndex  # noqa: E402

RELATION_WEIGHTS = {
    "calls": 1.0,
    "extends": 1.0,
    "imports": 0.9,
    "test-for": 0.8,
    "references": 0.6,
    "uses-imports": 0.5,
}

REVERSE_PENALTY = 0.7
DEFINITION_KINDS = {"function", "class"}
CALLABLE_KINDS = {"function", "method"}


def is_test_file(path: str) -> bool:
    name = Path(path).name
    stem = Path(path).stem
    parts = {part.lower() for part in Path(path).parts}
    return (
        stem.startswith("test")
        or stem.endswith("_test")
        or "tests" in parts
        or "test" in parts
    )


class CrossFileGraph:
    def __init__(self, index: ASTIndex, include_same_file: bool = False) -> None:
        self.index = index
        self.include_same_file = include_same_file
        self.edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.reverse_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.by_name: Dict[str, List[Any]] = defaultdict(list)
        self.relation_counts: Dict[str, int] = defaultdict(int)
        self.files: Set[str] = set()
        self._build()

    # -- construction --------------------------------------------------
    def _resolve(self, name: str, source_file: str) -> List[Any]:
        base = name.rsplit(".", 1)[-1]
        candidates = self.by_name.get(base, [])
        if self.include_same_file:
            return candidates
        return [c for c in candidates if c.file_path != source_file]

    def _link(
        self,
        source: Any,
        target: Any,
        relation: str,
        via: str,
        allow_same_file: bool = False,
    ) -> None:
        if source.element_id == target.element_id:
            return
        if not allow_same_file and not self.include_same_file:
            if source.file_path == target.file_path:
                return

        weight = RELATION_WEIGHTS.get(relation, 0.5)
        self.edges[source.element_id].append(
            {
                "target": target.element_id,
                "relation": relation,
                "via": via,
                "weight": weight,
            }
        )
        self.reverse_edges[target.element_id].append(
            {
                "target": source.element_id,
                "relation": relation,
                "via": f"{via} (reverse: used by {Path(source.file_path).name})",
                "weight": weight * REVERSE_PENALTY,
            }
        )
        self.relation_counts[relation] += 1

    def _build(self) -> None:
        for element in self.index.elements:
            self.files.add(element.file_path)
            if element.kind in DEFINITION_KINDS:
                self.by_name[element.name].append(element)

        definition_names = sorted(
            {e.name for e in self.index.elements if e.kind in DEFINITION_KINDS}
        )
        name_patterns = {
            name: re.compile(r"\b" + re.escape(name) + r"\b") for name in definition_names
        }

        imports_by_file: Dict[str, List[Any]] = defaultdict(list)
        for element in self.index.elements:
            if element.kind == "import":
                imports_by_file[element.file_path].append(element)

        # Only imports resolving to another file's definition are meaningful.
        external_import_ids: Set[str] = set()
        for element in self.index.elements:
            if element.kind != "import":
                continue
            if any(self._resolve(name, element.file_path) for name in element.imports):
                external_import_ids.add(element.element_id)

        for element in self.index.elements:
            source_file = element.file_path

            if element.kind in CALLABLE_KINDS:
                for call in element.calls:
                    for target in self._resolve(call, source_file):
                        self._link(element, target, "calls",
                                   f"{element.qualified_name}() calls {call}")

            if element.kind == "class":
                for base in element.bases:
                    for target in self._resolve(base, source_file):
                        self._link(element, target, "extends",
                                   f"{element.name} extends {base}")

            if element.kind == "import":
                for imported in element.imports:
                    for target in self._resolve(imported, source_file):
                        self._link(element, target, "imports",
                                   f"{Path(source_file).name} imports {imported}")

            if element.kind in {"function", "method", "class"}:
                for name, pattern in name_patterns.items():
                    for target in self.by_name.get(name, []):
                        if target.file_path == source_file and not self.include_same_file:
                            continue
                        if not pattern.search(element.source):
                            continue
                        relation = (
                            "test-for" if is_test_file(source_file) else "references"
                        )
                        self._link(element, target, relation,
                                   f"{element.qualified_name} mentions {name}")

            for import_element in imports_by_file.get(source_file, []):
                if import_element.element_id not in external_import_ids:
                    continue
                self._link(
                    element,
                    import_element,
                    "uses-imports",
                    f"{Path(source_file).name} imports {import_element.name}",
                    allow_same_file=True,
                )

    # -- queries -------------------------------------------------------
    def out_edges(self, element_id: str) -> List[Dict[str, Any]]:
        return self.edges.get(element_id, [])

    def in_edges(self, element_id: str) -> List[Dict[str, Any]]:
        return self.reverse_edges.get(element_id, [])

    def stats(self) -> Dict[str, Any]:
        return {
            "files": len(self.files),
            "units": len(self.index.elements),
            "edges": dict(self.relation_counts),
        }


def build_relationship_index(
    ast_index_path: Path, include_same_file: bool = False
) -> CrossFileGraph:
    return CrossFileGraph(ASTIndex.load(Path(ast_index_path)), include_same_file)


# ============================================================
# Seeds and expansion
# ============================================================

def seed_from_dense(
    dense: Any, index: ASTIndex, query: str, seeds_wanted: int
) -> List[Tuple[Any, float, str]]:
    """Map dense hits onto the AST units that contain them."""
    if dense is None or not getattr(dense, "ready", False):
        return []

    documents = dense.retrieve_documents(query, top_k=max(seeds_wanted * 3, 10))
    if not documents:
        return []

    ordered = sorted(documents, key=lambda d: d.score or 0.0, reverse=True)
    seeds: List[Tuple[Any, float, str]] = []
    for rank, document in enumerate(ordered):
        metadata = document.metadata or {}
        file_path = metadata.get("file_path")
        start, end = metadata.get("start_line"), metadata.get("end_line")
        for element in index.elements:
            if element.file_path != file_path:
                continue
            if element.kind not in {"function", "method", "class"}:
                continue
            if isinstance(start, int) and isinstance(end, int):
                if not (start <= element.end_line and end >= element.start_line):
                    continue
            seeds.append(
                (
                    element,
                    1.0 - rank / max(len(ordered), 1),
                    f"seed: dense rank {rank + 1} covers "
                    f"{Path(file_path).name}:{element.start_line}-{element.end_line}",
                )
            )
    return seeds[: seeds_wanted * 3]


def seed_from_structure(
    index: ASTIndex, query: str, seeds_wanted: int
) -> List[Tuple[Any, float, str]]:
    """Lexical/structural seeds when dense artifacts are unavailable."""
    from retrieval.ast_retriever import score_element, tokenize

    query_tokens = tokenize(query)
    scored: List[Tuple[Any, float, str]] = []
    for element in index.elements:
        if element.kind not in {"function", "method", "class"}:
            continue
        item = score_element(element, query, query_tokens)
        if item.score > 0:
            scored.append(
                (element, item.score, f"seed: {', '.join(item.reasons[:2]) or 'structural match'}")
            )
    scored.sort(key=lambda entry: entry[1], reverse=True)
    if scored:
        top = scored[0][1]
        scored = [(element, score / top, reason) for element, score, reason in scored]
    return scored[:seeds_wanted]


def expand(
    graph: CrossFileGraph,
    seeds: Sequence[Tuple[Any, float, str]],
    hops: int = 2,
    decay: float = 0.6,
    max_results: int = 10,
    max_per_file: int = 2,
    include_imports: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Breadth-first expansion with score decay and per-file caps."""
    best: Dict[str, Dict[str, Any]] = {}
    frontier: deque = deque()

    for element, score, reason in seeds:
        if element.element_id in best:
            continue
        best[element.element_id] = {
            "element": element,
            "score": score,
            "hop": 0,
            "path": reason,
            "relation": "seed",
        }
        frontier.append((element.element_id, score, 0))

    while frontier:
        element_id, score, hop = frontier.popleft()
        if hop >= hops:
            continue
        for edge in graph.out_edges(element_id) + graph.in_edges(element_id):
            target = graph.index.by_id.get(edge["target"])
            if target is None:
                continue
            new_score = score * decay * edge["weight"]
            if new_score <= 0.01:
                continue

            current = best.get(target.element_id)
            if current is None or new_score > current["score"]:
                source_entry = best[element_id]
                best[target.element_id] = {
                    "element": target,
                    "score": new_score,
                    "hop": hop + 1,
                    "path": f"{source_entry['path']} -> {edge['via']}",
                    "relation": edge["relation"],
                }
                frontier.append((target.element_id, new_score, hop + 1))

    ranked = sorted(best.values(), key=lambda entry: entry["score"], reverse=True)
    if not include_imports:
        ranked = [entry for entry in ranked if entry["element"].kind != "import"]

    kept: List[Dict[str, Any]] = []
    per_file: Dict[str, int] = defaultdict(int)
    for entry in ranked:
        file_path = entry["element"].file_path
        if per_file[file_path] >= max_per_file:
            continue
        per_file[file_path] += 1
        kept.append(entry)
        if len(kept) >= max_results:
            break

    relation_stats: Dict[str, int] = defaultdict(int)
    for entry in kept:
        relation_stats[entry["relation"]] += 1
    return kept, dict(relation_stats)


def to_documents(entries: Sequence[Dict[str, Any]]) -> List[Document]:
    documents = []
    for entry in entries:
        element = entry["element"]
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
                    "bases": element.bases,
                    "calls": element.calls,
                    "relation": entry["relation"],
                    "hop": entry["hop"],
                    "path": entry["path"],
                    "score": round(entry["score"], 6),
                },
                score=round(entry["score"], 6),
            )
        )
    return documents