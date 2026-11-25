"""
memory/extractor.py — Memory extraction abstractions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Any

from memory.base import BaseMemory
# Import types to ensure registration
from memory.types.episodic import EpisodicMemory  # noqa: F401
from memory.types.bug import BugMemory  # noqa: F401


class MemoryExtractor(ABC):
    """Abstract extractor — decides what to remember."""

    @abstractmethod
    def extract(
        self,
        task: Any,
        context: Any,
        action: Any,
        result: Any,
        test_result: Optional[str] = None,
        reflection: Optional[Any] = None,
    ) -> List[BaseMemory]:
        pass


class BasicMemoryExtractor(MemoryExtractor):
    """
    Deterministic extractor without LLM.

    Rules:
      PASS  -> EpisodicMemory
      FAIL + reflection -> BugMemory (bug=test failure, cause/fix from reflection)
      FAIL without reflection -> BugMemory with best-effort fields
    Also creates EpisodicMemory for successful outcomes when appropriate.
    """

    def extract(
        self,
        task: Any,
        context: Any,
        action: Any,
        result: Any,
        test_result: Optional[str] = None,
        reflection: Optional[Any] = None,
    ) -> List[BaseMemory]:
        memories: List[BaseMemory] = []

        # Normalize to strings
        task_s = str(task) if task is not None else ""
        context_s = str(context) if context is not None else ""
        action_s = str(action) if action is not None else ""
        result_s = str(result) if result is not None else ""
        test_result_s = str(test_result).upper() if test_result is not None else ""
        # reflection may be dict or string
        reflection_dict = {}
        if isinstance(reflection, dict):
            reflection_dict = reflection
        elif isinstance(reflection, str):
            reflection_dict = {"cause": reflection, "fix": reflection}
        elif reflection is not None:
            reflection_dict = {"cause": str(reflection), "fix": str(reflection)}

        is_pass = test_result_s == "PASS" or (not test_result_s and "pass" in result_s.lower())

        # On PASS: create EpisodicMemory
        if is_pass:
            ep = EpisodicMemory(
                task=task_s,
                context=context_s,
                action=action_s,
                result=result_s if result_s else "Tests passed",
                metadata={
                    "test_result": "PASS",
                    "source": "BasicMemoryExtractor",
                },
            )
            memories.append(ep)
        else:
            # On FAIL: create BugMemory
            # Bug source = test failure or result
            bug_text = result_s if result_s else (str(test_result) if test_result else "Tests failed")
            # If result indicates failure details, use it
            if test_result_s == "FAIL" and result_s:
                bug_text = result_s
            # Extract cause/fix from reflection
            cause = reflection_dict.get("cause", "") or reflection_dict.get("reason", "") or reflection_dict.get("error", "")
            fix = reflection_dict.get("fix", "") or reflection_dict.get("solution", "") or reflection_dict.get("repair", "")
            # Fallback if reflection missing
            if not cause:
                cause = "Unknown cause (no reflection provided)"
            if not fix:
                fix = "No fix recorded"

            bug_mem = BugMemory(
                bug=bug_text,
                cause=str(cause),
                fix=str(fix),
                context=context_s,
                metadata={
                    "test_result": "FAIL",
                    "task": task_s,
                    "source": "BasicMemoryExtractor",
                },
            )
            memories.append(bug_mem)

            # Optionally also create EpisodicMemory for failed episode if desired?
            # Spec says "extractor should also create an EpisodicMemory for successful outcomes when appropriate."
            # For FAIL we keep only BugMemory to avoid noise, but caller can configure.

        return memories


class LLMMemoryExtractor(MemoryExtractor):
    """
    Placeholder for future LLM-based extraction.
    Allows swapping without changing Manager/Store.
    Currently delegates to BasicMemoryExtractor unless an LLM callable is provided.
    """

    def __init__(self, llm_callable=None, fallback: Optional[MemoryExtractor] = None):
        self.llm_callable = llm_callable
        self.fallback = fallback or BasicMemoryExtractor()

    def extract(self, task, context, action, result, test_result=None, reflection=None) -> List[BaseMemory]:
        if self.llm_callable is None:
            return self.fallback.extract(task, context, action, result, test_result, reflection)
        # Future: call LLM -> structured JSON -> memories
        # For now delegate
        llm_output = self.llm_callable(
            task=task, context=context, action=action, result=result,
            test_result=test_result, reflection=reflection
        )
        # Expect llm_output to be list[dict] or list[BaseMemory]
        if isinstance(llm_output, list) and llm_output and isinstance(llm_output[0], BaseMemory):
            return llm_output
        return self.fallback.extract(task, context, action, result, test_result, reflection)
