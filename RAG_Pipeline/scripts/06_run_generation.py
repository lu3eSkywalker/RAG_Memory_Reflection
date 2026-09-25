#!/usr/bin/env python3
"""
Run generator component independently.
Usage: python scripts/06_run_generation.py
"""

import sys
import os
import json
import requests
sys.path.insert(0, "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3")

from rag_core.base import Generator, GenerationResult
from rag_core.context import PipelineContext
from typing import Dict, Any

GEMINI_API_KEY = "LLM_API_Key"


class MockGenerator(Generator):
    """Mock generator for testing without API key."""
    
    def __init__(self, component_id: str, config: Dict[str, Any]):
        super().__init__(component_id, config)
        self.model_name = self.config.get("model", "mock-model")
    
    def generate(self, prompt: str, context: PipelineContext) -> GenerationResult:
        # Generate a mock completion based on the prompt
        if "fibonacci" in prompt.lower():
            completion = "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"
        elif "factorial" in prompt.lower():
            completion = "    if n <= 1: return 1\n    return n * factorial(n - 1)"
        elif "binary_search" in prompt.lower():
            completion = "    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
        else:
            completion = "    pass  # Mock completion"
        
        return GenerationResult(
            prompt=prompt,
            completions=[completion],
            generator_id=self.component_id,
            metadata={"model": self.model_name}
        )


class GeminiGenerator(Generator):
    """Google Gemini generator for code completion."""
    
    def __init__(self, component_id: str, config: Dict[str, Any]):
        super().__init__(component_id, config)
        self.api_key = self.config.get("api_key") or GEMINI_API_KEY
        self.model_name = self.config.get("model", "gemini-2.5-flash")
        self.temperature = self.config.get("temperature", 0.2)
        self.top_p = self.config.get("top_p", 0.95)
        self.max_tokens = self.config.get("max_tokens", 2048)
    
    def generate(self, prompt: str, context: PipelineContext) -> GenerationResult:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
        }
        params = {
            "key": self.api_key
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxOutputTokens": self.max_tokens,
            }
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    completion = parts[0].get("text", "")
                else:
                    completion = ""
            else:
                completion = ""
                
        except Exception as e:
            print(f"Gemini generation failed: {e}")
            completion = "# Generation failed\n    pass\n"
        
        return GenerationResult(
            prompt=prompt,
            completions=[completion],
            generator_id=self.component_id,
            metadata={"model": self.model_name}
        )


def load_final_prompt(file_path: str) -> str:
    """Load final prompt from file."""
    with open(file_path, 'r') as f:
        return f.read()


def save_llm_response(response: str, output_path: str):
    """Save LLM response to file."""
    with open(output_path, 'w') as f:
        f.write(response)


def main():
    final_prompt_file = "final_prompt.txt"
    output_file = "llm_response.txt"
    
    if not os.path.exists(final_prompt_file):
        print(f"Error: {final_prompt_file} not found. Run 05_run_prompt.py first.")
        return
    
    print(f"Loading final prompt from {final_prompt_file}...")
    prompt = load_final_prompt(final_prompt_file)
    print(f"Prompt length: {len(prompt)} chars")
    
    print("Initializing generator...")
    generator = GeminiGenerator("gemini_generator", {
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 2048,
        "api_key": GEMINI_API_KEY
    })
    
    context = PipelineContext()
    
    print(f"\n{'='*60}")
    print("Generation")
    print(f"{'='*60}")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"\nGenerating completion...")
    
    result = generator.generate(prompt, context)
    
    print(f"\n{'='*60}")
    print("Generation Result")
    print(f"{'='*60}")
    print(f"Generator: {result.generator_id}")
    print(f"Model: {result.metadata.get('model', 'unknown')}")
    print(f"Number of completions: {len(result.completions)}")
    print(f"\n--- Completion ---")
    print(result.completions[0] if result.completions else "(empty)")
    
    # Save response
    if result.completions:
        save_llm_response(result.completions[0], output_file)
        print(f"\nLLM response saved to {output_file}")


if __name__ == "__main__":
    main()