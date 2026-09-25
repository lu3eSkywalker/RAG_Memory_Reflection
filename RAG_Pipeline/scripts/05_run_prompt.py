#!/usr/bin/env python3
"""
Run prompt builder component independently.
Usage: python scripts/05_run_prompt.py
"""

import sys
import os
import json
import importlib.util
sys.path.insert(0, "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3")

from rag_core.base import Document, RetrievalResult, PromptBuilder
from rag_core.context import PipelineContext
from typing import Tuple


# Load dependent modules
def load_module(name, filename):
    scripts_dir = "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3/scripts"
    spec = importlib.util.spec_from_file_location(name, f"{scripts_dir}/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

embedding_mod = load_module("embedding", "02_run_embedding.py")
vector_store_mod = load_module("vector_store", "03_run_vector_store.py")
retrieval_mod = load_module("retrieval", "04_run_retrieval.py")

SentenceTransformerEmbedder = embedding_mod.SentenceTransformerEmbedder
MockFAISSVectorStore = vector_store_mod.MockFAISSVectorStore
CosineSimilarityRetriever = retrieval_mod.CosineSimilarityRetriever


class RepoCoderPromptBuilder(PromptBuilder):
    """RepoCoder-style prompt builder with file paths and commented code."""
    
    def build_prompt(self, query: str, retrieval_result: RetrievalResult, context: PipelineContext) -> str:
        max_examples = self.config.get("max_examples", 10)
        max_tokens = self.config.get("max_retrieval_tokens", 2000)
        include_paths = self.config.get("include_file_paths", True)
        comment_code = self.config.get("comment_out_code", True)
        extended_mode = self.config.get("extended_context_mode", False)
        
        separator = "# " + "-" * 50
        prepend = "# Here are some relevant code fragments from other files of the repo:\n"
        prepend += separator + "\n"
        
        current_tokens = len(prepend) // 4
        max_tokens_estimate = max_tokens
        
        blocks = []
        for doc in retrieval_result.documents[:max_examples]:
            block_str, token_len = self._make_block(doc, include_paths, comment_code, extended_mode)
            
            if current_tokens + token_len < max_tokens_estimate:
                blocks.append(block_str)
                current_tokens += token_len
            else:
                break
        
        prepend += "".join(reversed(blocks))
        
        return prepend + "\n" + query
    
    def _make_block(
        self, 
        doc: Document, 
        include_paths: bool, 
        comment_code: bool,
        extended_mode: bool
    ) -> Tuple[str, int]:
        meta = doc.metadata
        lines = doc.content.splitlines()
        
        if extended_mode and "file_path" in meta:
            pass
        
        parts = []
        
        if include_paths and "file_path" in meta:
            parts.append(f"# the below code fragment can be found in:")
            parts.append(f"# {meta['file_path']}")
        
        parts.append("# " + "-" * 50)
        
        if comment_code:
            parts.extend([f"# {line}" for line in lines])
        else:
            parts.extend(lines)
        
        parts.append("# " + "-" * 50)
        
        block = "\n".join(parts) + "\n"
        token_estimate = len(block) // 4
        
        return block, token_estimate


class StandardRAGPromptBuilder(PromptBuilder):
    """Standard RAG prompt builder."""
    
    def build_prompt(self, query: str, retrieval_result: RetrievalResult, context: PipelineContext) -> str:
        max_tokens = self.config.get("max_context_tokens", 3000)
        template = self.config.get("template", "default")
        
        if template == "default":
            context_str = "\n\n".join([
                f"[Document {i+1}]\n{doc.content}" 
                for i, doc in enumerate(retrieval_result.documents)
            ])
            
            prompt = f"""Based on the following context, answer the query.

Context:
{context_str}

Query: {query}

Answer:"""
            
        if len(prompt) > max_tokens * 4:
            prompt = prompt[:max_tokens * 4]
        
        return prompt


class ChainOfThoughtPromptBuilder(PromptBuilder):
    """Chain-of-thought style prompt builder."""
    
    def build_prompt(self, query: str, retrieval_result: RetrievalResult, context: PipelineContext) -> str:
        max_tokens = self.config.get("max_context_tokens", 3000)
        include_reasoning = self.config.get("include_reasoning", True)
        
        context_str = "\n\n".join([
            f"[Reference {i+1}]\n{doc.content}" 
            for i, doc in enumerate(retrieval_result.documents)
        ])
        
        reasoning_prompt = """
Let's solve this step by step:
1. First, analyze the relevant code fragments
2. Identify the key patterns and APIs
3. Then provide the completion
""" if include_reasoning else ""
        
        prompt = f"""Given the following code context, complete the code.

Context:
{context_str}

{reasoning_prompt}
Query:
{query}

Completion:"""
        
        if len(prompt) > max_tokens * 4:
            prompt = prompt[:max_tokens * 4]
        
        return prompt


def load_retrieval_data(file_path: str) -> RetrievalResult:
    """Load retrieval result from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    documents = []
    for item in data["documents"]:
        documents.append(Document(
            content=item["content"],
            metadata=item["metadata"],
            score=item.get("score")
        ))
    
    return RetrievalResult(
        query=data["query"],
        documents=documents,
        retriever_id=data.get("retriever_id", "unknown")
    )


def save_final_prompt(prompt: str, output_path: str):
    """Save final prompt to file."""
    with open(output_path, 'w') as f:
        f.write(prompt)


def main():
    retrieval_data_file = "retrieval_data.json"
    output_file = "final_prompt.txt"
    
    if not os.path.exists(retrieval_data_file):
        print(f"Error: {retrieval_data_file} not found. Run 04_run_retrieval.py first.")
        return
    
    print(f"Loading retrieval data from {retrieval_data_file}...")
    retrieval_result = load_retrieval_data(retrieval_data_file)
    print(f"Loaded {len(retrieval_result.documents)} retrieved documents")
    
    query = retrieval_result.query
    
    context = PipelineContext()
    context.original_query = query
    
    prompt_builders = {
        "repo_coder": RepoCoderPromptBuilder("repo_coder_style", {
            "max_examples": 10,
            "max_retrieval_tokens": 2000,
            "include_file_paths": True,
            "comment_out_code": True,
            "extended_context_mode": False
        }),
        "standard_rag": StandardRAGPromptBuilder("standard_rag", {
            "max_context_tokens": 3000,
            "template": "default"
        }),
        "cot_rag": ChainOfThoughtPromptBuilder("cot_rag", {
            "max_context_tokens": 3000,
            "include_reasoning": True
        }),
    }
    
    for name, builder in prompt_builders.items():
        print(f"\n{'='*60}")
        print(f"Prompt Builder: {name}")
        print(f"{'='*60}")
        
        prompt = builder.build_prompt(query, retrieval_result, context)
        
        print(f"Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
        print(f"\n--- Prompt Preview (first 1500 chars) ---")
        print(prompt[:1500])
        print("...\n")
        
        # Save the repo_coder prompt as final_prompt
        if name == "repo_coder":
            save_final_prompt(prompt, output_file)
            print(f"Final prompt saved to {output_file}")


if __name__ == "__main__":
    main()