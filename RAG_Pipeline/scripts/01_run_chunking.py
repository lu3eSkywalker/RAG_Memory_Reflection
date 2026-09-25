#!/usr/bin/env python3
"""
Run chunking component independently.
Usage: python scripts/01_run_chunking.py
"""

import sys
import json
sys.path.insert(0, "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3")

from rag_core.base import Document, Chunker
from dataclasses import dataclass, field
from typing import Dict, Any, List


# Chunker implementations (copied from rag_components for standalone use)
class SlidingWindowChunker(Chunker):
    """RepoCoder-style sliding window chunker for code repositories."""

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        window_size = self.config.get("window_size", 20)
        slice_size = self.config.get("slice_size", 2)
        overlap = self.config.get("overlap", True)

        lines = text.splitlines()
        stride = window_size // slice_size
        chunks = []

        for i in range(0, len(lines), stride):
            end = min(i + window_size, len(lines))
            chunk_lines = lines[i:end]

            if len(chunk_lines) < 3:
                continue

            chunk_metadata = {
                "start_line": i,
                "end_line": end,
                "window_size": window_size,
                "slice_size": slice_size,
                "repo": metadata.get("repo", "") if metadata else "",
                "file_path": metadata.get("file_path", "") if metadata else "",
            }

            chunks.append(Document(
                content="\n".join(chunk_lines),
                metadata=chunk_metadata
            ))

            if end >= len(lines):
                break

        return chunks


class FixedSizeChunker(Chunker):
    """Fixed-size chunker with overlap."""

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        chunk_size = self.config.get("chunk_size", 512)
        overlap = self.config.get("overlap", 50)

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 50:
                continue

            chunks.append(Document(
                content=" ".join(chunk_words),
                metadata={
                    "start_word": i,
                    "end_word": i + len(chunk_words),
                    **(metadata or {})
                }
            ))

        return chunks


class SemanticChunker(Chunker):
    """Semantic chunking using embedding similarity."""

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        threshold = self.config.get("breakpoint_threshold", 0.8)
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = []

        for para in paragraphs:
            current_chunk.append(para)
            if len("\n\n".join(current_chunk)) > 1000:
                chunks.append(Document(
                    content="\n\n".join(current_chunk),
                    metadata={**(metadata or {})}
                ))
                current_chunk = []

        if current_chunk:
            chunks.append(Document(
                content="\n\n".join(current_chunk),
                metadata={**(metadata or {})}
            ))

        return chunks


def read_code_file(file_path: str) -> str:
    """Read code from a file."""
    with open(file_path, 'r') as f:
        return f.read()


def save_chunks_to_json(chunks: List[Document], output_path: str):
    """Save chunks to a JSON file."""
    data = []
    for chunk in chunks:
        data.append({
            "content": chunk.content,
            "metadata": chunk.metadata
        })
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    code_file = "/home/sooraj/Downloads/embedding_and_chunking_result/code.py"
    output_file = "chunks.json"
    
    code = read_code_file(code_file)
    
    chunkers = {
        "sliding_window": SlidingWindowChunker("sliding_window", {"window_size": 20, "slice_size": 2}),
        "fixed_size": FixedSizeChunker("fixed_size", {"chunk_size": 100, "overlap": 20}),
        "semantic": SemanticChunker("semantic", {"breakpoint_threshold": 0.8}),
    }
    
    metadata = {"file_path": code_file, "repo": "embedding_and_chunking_result"}
    
    all_chunks = {}
    for name, chunker in chunkers.items():
        print(f"\n{'='*60}")
        print(f"Chunking with: {name}")
        print(f"{'='*60}")
        
        chunks = chunker.chunk(code, metadata)
        
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Metadata: {chunk.metadata}")
            print(f"Content:\n{chunk.content[:200]}...")
            print("-" * 40)
        
        all_chunks[name] = [
            {"content": chunk.content, "metadata": chunk.metadata}
            for chunk in chunks
        ]
    
    save_chunks_to_json(
        [Document(content=c["content"], metadata=c["metadata"]) for chunks in all_chunks.values() for c in chunks],
        output_file
    )
    print(f"\nAll chunks saved to {output_file}")


if __name__ == "__main__":
    main()