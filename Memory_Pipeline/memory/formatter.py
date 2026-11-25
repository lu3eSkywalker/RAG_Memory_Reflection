"""
memory/formatter.py — Format memories for LLM prompt.
"""
from __future__ import annotations

from typing import List

from memory.base import BaseMemory


class MemoryFormatter:
    """
    Formats memories into a string for LLM context.
    No retrieval logic — pure formatting.
    Generic over memory types.
    """

    def format(self, memories: List[BaseMemory]) -> str:
        if not memories:
            return ""
        lines: List[str] = ["Relevant Past Experience:", ""]
        for mem in memories:
            # Generic header based on memory_type
            type_label = mem.memory_type.capitalize()  # episodic -> Episodic
            # Use to_text() for body
            lines.append(f"[{type_label} Memory]")
            lines.append(mem.to_text())
            lines.append("")  # blank line between memories
        return "\n".join(lines).strip()
