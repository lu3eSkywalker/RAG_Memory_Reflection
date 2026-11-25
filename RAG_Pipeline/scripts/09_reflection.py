#!/usr/bin/env python3
"""
09_reflection.py — Reflection & Memory Extraction Layer.

Runs *after* generation and decides whether the episode contains reusable
knowledge worth remembering for future coding tasks. It reuses the existing
memory architecture — it never re-implements extraction, storage, retrieval or
formatting:

    task + retrieved context + generated result (+ test feedback)
        -> reflection LLM  ->  STORE | NO_MEMORY
        -> MemoryManager (MemoryExtractor -> BaseMemory subclass)
        -> VectorMemoryStore (+ HashEmbedder)
        -> <shared memory_store.pkl>

Design rules honoured here:
  * `extract_memory(...)` returns a structured candidate or None (NO_MEMORY).
    It never writes to the store.
  * Persistence goes through MemoryManager/extractor/store only.
  * Only memory types registered in MEMORY_TYPES are accepted, and only the
    types the extractors can build (episodic, bug).
  * Duplicates (similarity >= threshold) are rejected.
  * Reflection failures degrade to NO_MEMORY and never break the pipeline.
  * The reflection LLM is independent of the generation LLM implementation.

Usage:
    # reflect on a generation file produced by run_all_techniques.py
    python scripts/09_reflection.py \
        --task "add fibonacci_iterative" \
        --generation-file run_all_artifacts/generation_semantic.txt \
        --context-file run_all_artifacts/retrieval_semantic.json

    # programmatic
    from importlib import util; util.spec_from_file_location("r", "scripts/09_reflection.py")
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gemini_client  # noqa: E402

# ---------------------------------------------------------------------------
# Bootstrap the existing memory architecture (single source of truth for the
# store path is memory.config.DEFAULT_STORE_PATH, resolved after import).
# ---------------------------------------------------------------------------
MEMORY_PIPELINE_DIR = Path(
    os.environ.get("MEMORY_PIPELINE_DIR", str(Path.home() / "Downloads" / "memory pipeline"))
).expanduser()
if str(MEMORY_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_PIPELINE_DIR))

try:
    from memory.base import MEMORY_TYPES
    from memory.manager import MemoryManager
    from memory.extractor import BasicMemoryExtractor, LLMMemoryExtractor
    from memory.config import DEFAULT_STORE_PATH, EMBEDDING_DIMENSION, resolve_store_path

    MEMORY_AVAILABLE = True
    MEMORY_IMPORT_ERROR = ""
except Exception as error:  # pragma: no cover - environment dependent
    MEMORY_AVAILABLE = False
    MEMORY_IMPORT_ERROR = str(error)
    MEMORY_TYPES = {}

STORE = "STORE"
NO_MEMORY = "NO_MEMORY"

SUPPORTED_MEMORY_TYPES = ("episodic", "bug")

SECRET_PATTERNS = (
    re.compile(r"\b(api[_-]?key|secret[_-]?key|access[_-]?token|bearer\s+[A-Za-z0-9._-]{12,})\b", re.I),
    re.compile(r"\b(password|passwd|credential)s?\s*[:=]", re.I),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}"),          # Google API keys
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),                 # OpenAI-style keys
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),                # GitHub tokens
    re.compile(r"/Users/[^\s]+|/home/[^\s]+"),            # absolute home paths
)

logger = logging.getLogger("reflection")

REFLECTION_PROMPT = """You are the reflection layer of a retrieval-augmented coding pipeline.
You receive the original task, the retrieved repository context, the generated result, and optional test feedback.
Decide whether this episode contains reusable knowledge worth remembering for FUTURE, different tasks.

Return STRICT JSON only, no prose, using exactly these keys:
{{
  "decision": "STORE" | "NO_MEMORY",
  "result": "the final answer",
  "memory_type": "episodic" | "bug" | "",
  "memory": {{}},
  "reason": "one short sentence"
}}

Rules:
- Default to NO_MEMORY. Store only when the knowledge generalises beyond this single query.
- "episodic": a task/context/action/result episode recording a repository-specific
  convention or a reusable implementation lesson. memory = {{"task","context","action","result"}}.
- "bug": a failure with an identified cause and a fix, confirmed by test feedback or
  strong evidence in the retrieved context. memory = {{"bug","cause","fix","context"}}.
- Never store generated code as if it were verified. Never store hidden reasoning or
  chain-of-thought. Never store secrets, API keys, tokens, passwords or absolute paths.
- Do not restate the code; record the lesson.
- If test feedback reports failure and you cannot identify both cause and fix, return NO_MEMORY.
- If an equivalent memory very likely already exists, return NO_MEMORY.
- "result" is the generated result, corrected only if it clearly contradicts the retrieved
  context. Keep it code-only, no explanation.
- Each memory field must be at most 60 words.

TASK
{task}

RETRIEVED CONTEXT
{context}

GENERATED RESULT
{result}

TEST FEEDBACK
{feedback}
"""


# ============================================================
# Data structures
# ============================================================

@dataclass
class MemoryCandidate:
    """Structured memory candidate produced by the reflection stage."""

    memory_type: str
    reason: str = ""
    task: str = ""
    context: str = ""
    action: str = ""
    result: str = ""
    bug: str = ""
    cause: str = ""
    fix: str = ""

    def as_text(self) -> str:
        if self.memory_type == "bug":
            return (
                f"Bug: {self.bug}\nCause: {self.cause}\n"
                f"Fix: {self.fix}\nContext: {self.context}"
            )
        return (
            f"Task: {self.task}\nContext: {self.context}\n"
            f"Action: {self.action}\nResult: {self.result}"
        )

    def to_reflection(self) -> Dict[str, str]:
        """Shape expected by BasicMemoryExtractor for cause/fix extraction."""
        return {"cause": self.cause, "fix": self.fix}

    def to_remember_kwargs(self) -> Dict[str, Any]:
        """Arguments for MemoryManager.remember()."""
        return {
            "task": self.task or self.bug,
            "context": self.context,
            "action": self.action or self.fix,
            "result": self.result or f"Resolved: {self.fix}",
            "test_result": "FAIL" if self.memory_type == "bug" else "PASS",
            "reflection": self.to_reflection(),
        }

    @property
    def summary(self) -> str:
        return (self.cause or self.fix or self.action or self.result or self.bug or "").strip()


@dataclass
class ReflectionOutcome:
    """Result of the whole reflection stage (never raises)."""

    decision: str = NO_MEMORY
    result: str = ""
    reason: str = ""
    candidate: Optional[MemoryCandidate] = None
    raw: str = ""
    error: str = ""
    stored_ids: List[str] = field(default_factory=list)
    store_path: Optional[str] = None

    @property
    def stored(self) -> bool:
        return bool(self.stored_ids)

    @property
    def memory_text(self) -> str:
        return self.candidate.summary if self.candidate else ""

    def render(self, header: str = "") -> str:
        lines = []
        if header:
            lines.append(header)
        lines.append("Result:")
        lines.append(self.result.strip() or "(empty)")
        lines.append("")
        lines.append("Memory:")
        lines.append(self.memory_text or "(none - no reusable knowledge worth keeping)")
        lines.append("")
        lines.append(f"Decision : {self.decision}")
        lines.append(f"Reason   : {self.reason or '(none)'}")
        if self.candidate:
            lines.append(f"Type     : {self.candidate.memory_type}")
        if self.error:
            lines.append(f"Error    : {self.error}")
        if self.store_path:
            lines.append(f"Store    : {self.store_path}")
        lines.append(
            "Stored   : " + (", ".join(self.stored_ids) if self.stored_ids else "no")
        )
        return "\n".join(lines)


# ============================================================
# Helpers
# ============================================================

def strip_fences(text: str) -> str:
    """Remove ``` / ```json wrappers that models sometimes add."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def clean_result(text: str) -> str:
    """Normalise a generated result: drop markdown fences, keep the code."""
    cleaned = strip_fences(text)
    cleaned = re.sub(r"^\s*(?:Here(?:'s| is)|Below is)[^\n]*\n", "", cleaned, flags=re.I)
    return cleaned.strip()


def contains_secret(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in SECRET_PATTERNS)


def parse_json_payload(raw: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_fences(raw)
    if not cleaned:
        return None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def format_context(documents: Any) -> str:
    """Render retrieved context from rag_core Documents or plain dicts."""
    if not documents:
        return "(no context retrieved)"
    if isinstance(documents, str):
        return documents

    blocks: List[str] = []
    for document in documents:
        metadata = getattr(document, "metadata", None)
        if metadata is None and isinstance(document, dict):
            metadata = document.get("metadata", {})
            content = document.get("content", "")
            score = document.get("score")
        else:
            content = getattr(document, "content", "") or ""
            metadata = metadata or {}
            score = getattr(document, "score", None)
        location = f"{metadata.get('file_path', '?')}:{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}"
        header = f"--- {location}"
        if score is not None:
            header += f" (score={score})"
        blocks.append(f"{header}\n{content}")
    return "\n\n".join(blocks)


# ============================================================
# Reflection stage
# ============================================================

class Reflector:
    """Reflection + memory extraction, independent of the generation LLM."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 900,
        store_path: Optional[Path | str] = None,
        manager: Optional[Any] = None,
        similarity_threshold: float = 0.92,
        max_context_chars: int = 2500,
        max_result_chars: int = 3000,
        max_feedback_chars: int = 1500,
        min_memory_chars: int = 20,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.store_path = resolve_store_path(store_path) if MEMORY_AVAILABLE else None
        self.manager = manager
        self.similarity_threshold = similarity_threshold
        self.max_context_chars = max_context_chars
        self.max_result_chars = max_result_chars
        self.max_feedback_chars = max_feedback_chars
        self.min_memory_chars = min_memory_chars

    # -- manager -------------------------------------------------------
    def ensure_manager(self, store: bool = True) -> Optional[Any]:
        """MemoryManager bound to the shared store; None when unavailable."""
        if not MEMORY_AVAILABLE:
            return None
        if self.manager is not None:
            return self.manager
        if not store:
            return None
        try:
            self.manager = MemoryManager(use_default_store=True) if self.store_path is None \
                else MemoryManager(store_path=self.store_path)
            return self.manager
        except Exception as error:
            logger.error("Could not create MemoryManager: %s", error)
            return None

    # -- stage 1: extraction (no writes) -------------------------------
    def extract_memory(
        self,
        task: str,
        retrieved_context: Any = None,
        generated_result: str = "",
        test_feedback: Optional[str] = None,
    ) -> Optional[MemoryCandidate]:
        """Return a validated MemoryCandidate, or None for NO_MEMORY.

        Never writes to the store and never raises.
        """
        if not gemini_client.available():
            logger.warning("GEMINI_API_KEY not set; reflection skipped")
            return None

        prompt = REFLECTION_PROMPT.format(
            task=truncate(task or "", 1000),
            context=truncate(format_context(retrieved_context), self.max_context_chars),
            result=truncate(generated_result or "", self.max_result_chars),
            feedback=truncate(test_feedback or "(no test feedback)", self.max_feedback_chars),
        )

        try:
            raw = gemini_client.generate(
                prompt, model=self.model,
                temperature=self.temperature, max_tokens=self.max_tokens,
                json_mode=True,
            )
        except Exception as error:
            logger.error("Reflection LLM call failed: %s", error)
            return None

        payload = parse_json_payload(raw)
        if payload is None:
            logger.warning("Reflection returned unparsable JSON")
            return None

        decision = str(payload.get("decision", NO_MEMORY)).strip().upper()
        if decision != STORE:
            return None

        memory_type = str(payload.get("memory_type", "")).strip().lower()
        if memory_type not in MEMORY_TYPES:
            logger.warning("Rejected candidate: memory_type %r is not registered", memory_type)
            return None
        if memory_type not in SUPPORTED_MEMORY_TYPES:
            logger.warning(
                "Rejected candidate: type %r is registered but not produced by the "
                "extractors (supported: %s)", memory_type, ", ".join(SUPPORTED_MEMORY_TYPES),
            )
            return None

        memory = payload.get("memory") or {}
        if not isinstance(memory, dict):
            memory = {}

        candidate = MemoryCandidate(
            memory_type=memory_type,
            reason=str(payload.get("reason", "")),
            task=str(memory.get("task", "")),
            context=str(memory.get("context", "")),
            action=str(memory.get("action", "")),
            result=str(memory.get("result", "")),
            bug=str(memory.get("bug", "")),
            cause=str(memory.get("cause", "")),
            fix=str(memory.get("fix", "")),
        )

        required = {
            "episodic": ("task", "action"),
            "bug": ("bug", "cause", "fix"),
        }[memory_type]
        for name in required:
            if not str(getattr(candidate, name)).strip():
                logger.warning("Rejected candidate: missing required field %r", name)
                return None

        if len(candidate.as_text().strip()) < self.min_memory_chars:
            logger.warning("Rejected candidate: too short to be reusable")
            return None

        if contains_secret(candidate.as_text()):
            logger.warning("Rejected candidate: looks like it contains secrets/paths")
            return None

        return candidate

    # -- de-duplication ------------------------------------------------
    def is_duplicate(self, candidate: MemoryCandidate, manager: Any) -> Optional[float]:
        store = getattr(manager, "store", None)
        similarities = getattr(store, "similarities", None)
        if not callable(similarities):
            return None
        try:
            for score, memory in similarities(candidate.as_text()):
                if memory.memory_type == candidate.memory_type and score >= self.similarity_threshold:
                    return score
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("De-duplication check failed: %s", error)
        return None

    # -- stage 2: extract + persist ------------------------------------
    def reflect(
        self,
        task: str,
        retrieved_context: Any = None,
        generated_result: str = "",
        test_feedback: Optional[str] = None,
        store: bool = True,
    ) -> ReflectionOutcome:
        """Run reflection and optionally persist through MemoryManager."""
        fallback_result = clean_result(generated_result)
        store_path = str(self.store_path) if self.store_path else None

        if not MEMORY_AVAILABLE:
            return ReflectionOutcome(
                decision=NO_MEMORY, result=fallback_result,
                reason="memory package unavailable", error=MEMORY_IMPORT_ERROR,
                store_path=store_path,
            )

        candidate = self.extract_memory(task, retrieved_context, generated_result, test_feedback)
        if candidate is None:
            return ReflectionOutcome(
                decision=NO_MEMORY, result=fallback_result,
                reason="no reusable memory candidate", store_path=store_path,
            )

        manager = self.ensure_manager(store=store) if store else None
        if manager is not None:
            score = self.is_duplicate(candidate, manager)
            if score is not None:
                return ReflectionOutcome(
                    decision=NO_MEMORY, result=fallback_result, candidate=candidate,
                    reason=f"duplicate of an existing memory (similarity {score:.3f})",
                    store_path=store_path,
                )

        stored_ids: List[str] = []
        if manager is not None:
            try:
                memories = manager.remember(**candidate.to_remember_kwargs())
                stored_ids = [m.id for m in memories]
            except Exception as error:
                logger.error("MemoryManager.remember failed: %s", error)

        return ReflectionOutcome(
            decision=STORE if stored_ids else NO_MEMORY,
            result=fallback_result,
            reason=candidate.reason or ("stored" if stored_ids else "accepted by reflection"),
            candidate=candidate,
            stored_ids=stored_ids,
            store_path=store_path,
        )


# ============================================================
# Module-level interface (spec: extract_memory(...))
# ============================================================

_DEFAULT_REFLECTOR: Optional[Reflector] = None


def get_reflector(**kwargs: Any) -> Reflector:
    global _DEFAULT_REFLECTOR
    if kwargs:
        return Reflector(**kwargs)
    if _DEFAULT_REFLECTOR is None:
        _DEFAULT_REFLECTOR = Reflector()
    return _DEFAULT_REFLECTOR


def extract_memory(
    task: str,
    retrieved_context: Any = None,
    generated_result: str = "",
    test_feedback: Optional[str] = None,
    reflector: Optional[Reflector] = None,
) -> Optional[MemoryCandidate]:
    """Structured memory candidate, or None when there is nothing to store."""
    return (reflector or get_reflector()).extract_memory(
        task, retrieved_context, generated_result, test_feedback
    )


def run_reflection(
    task: str,
    retrieved_context: Any = None,
    generated_result: str = "",
    test_feedback: Optional[str] = None,
    store: bool = True,
    reflector: Optional[Reflector] = None,
) -> ReflectionOutcome:
    """Full stage: reflection + (optional) persistence through MemoryManager."""
    return (reflector or get_reflector()).reflect(
        task, retrieved_context, generated_result, test_feedback, store=store
    )


# ============================================================
# CLI
# ============================================================

def latest_generation_file(artifacts: Path) -> Optional[Path]:
    candidates = sorted(artifacts.glob("generation_*.txt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_context_file(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with open(path, "r") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload.get("documents", [])
    return payload if isinstance(payload, list) else []


def main(argv: Optional[Sequence[str]] = None) -> int:
    artifacts = PROJECT_ROOT / "run_all_artifacts"

    parser = argparse.ArgumentParser(description="Reflection and memory extraction")
    parser.add_argument("--task", default="", help="Original task / query")
    parser.add_argument("--generation-file", type=Path, default=None,
                        help="File with the generated result (default: newest generation_*.txt)")
    parser.add_argument("--context-file", type=Path, default=None,
                        help="Retrieval JSON file used as retrieved context")
    parser.add_argument("--test-feedback", default=None, help="Optional test feedback")
    parser.add_argument("--store", type=Path, default=None,
                        help="Override the shared memory store path")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--no-store", action="store_true",
                        help="Reflect only, do not persist")
    parser.add_argument("--similarity-threshold", type=float, default=0.92)
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the Result/Memory report to this file")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not MEMORY_AVAILABLE:
        print(f"Error: memory package not importable from {MEMORY_PIPELINE_DIR}: {MEMORY_IMPORT_ERROR}")
        return 1

    generation_file = args.generation_file or latest_generation_file(artifacts)
    if generation_file is None or not Path(generation_file).exists():
        print(f"Error: no generation file found (looked in {artifacts}).")
        print("Run run_all_techniques.py first, or pass --generation-file.")
        return 1

    generated_result = Path(generation_file).read_text()
    task = args.task or "unknown task"
    documents = load_context_file(args.context_file)

    print("=" * 70)
    print("REFLECTION & MEMORY EXTRACTION")
    print("=" * 70)
    print(f"Task          : {task[:100]}")
    print(f"Generation    : {generation_file}")
    print(f"Context       : {args.context_file} ({len(documents)} documents)")
    print(f"Test feedback : {args.test_feedback or '(none)'}")
    print(f"Store         : {resolve_store_path(args.store)}")

    reflector = Reflector(
        model=args.model,
        store_path=args.store,
        similarity_threshold=args.similarity_threshold,
    )
    outcome = reflector.reflect(
        task=task,
        retrieved_context=documents,
        generated_result=generated_result,
        test_feedback=args.test_feedback,
        store=not args.no_store,
    )

    header = f"# technique={Path(generation_file).stem.replace('generation_', '')}  model={args.model}"
    report = outcome.render(header)
    print()
    print(report)

    if outcome.error:
        print(f"\nReflection degraded gracefully: {outcome.error}")

    if args.output:
        Path(args.output).write_text(report + "\n")
        print(f"\nReport written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())