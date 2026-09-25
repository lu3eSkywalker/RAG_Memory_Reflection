#!/usr/bin/env python3
"""
Run evaluation component independently.
Usage: python scripts/07_run_evaluation.py
"""

import sys
import os
import json
import difflib
sys.path.insert(0, "/home/sooraj/Downloads/Mini_Project_RAG/RAG_Basic_Repocoder_Inspired_EXP_3")

from rag_core.base import Evaluator, EvaluationResult
from rag_core.context import PipelineContext
from typing import List, Dict, Any

GEMINI_API_KEY = "LLM_API_Key"


class ExactMatchEvaluator(Evaluator):
    """Exact match evaluation."""
    
    def evaluate(self, predictions: List[str], ground_truths: List[str], context: PipelineContext) -> EvaluationResult:
        if not predictions or not ground_truths:
            return EvaluationResult(
                metrics={"exact_match": 0.0},
                evaluator_id=self.component_id
            )
        
        pred = predictions[0].strip()
        truth = ground_truths[0].strip()
        
        exact_match = 1.0 if pred == truth else 0.0
        
        return EvaluationResult(
            metrics={"exact_match": exact_match},
            evaluator_id=self.component_id,
            details={"prediction": pred[:100], "ground_truth": truth[:100]}
        )


class EditSimilarityEvaluator(Evaluator):
    """Edit similarity (Levenshtein) evaluation."""
    
    def evaluate(self, predictions: List[str], ground_truths: List[str], context: PipelineContext) -> EvaluationResult:
        if not predictions or not ground_truths:
            return EvaluationResult(
                metrics={"edit_similarity": 0.0},
                evaluator_id=self.component_id
            )
        
        pred = predictions[0]
        truth = ground_truths[0]
        
        similarity = difflib.SequenceMatcher(None, pred, truth).ratio()
        
        return EvaluationResult(
            metrics={"edit_similarity": similarity},
            evaluator_id=self.component_id,
            details={"prediction_len": len(pred), "truth_len": len(truth)}
        )


class PassAtKEvaluator(Evaluator):
    """Pass@k evaluation for code generation."""
    
    def evaluate(self, predictions: List[str], ground_truths: List[str], context: PipelineContext) -> EvaluationResult:
        k_values = self.config.get("k_values", [1, 5, 10])
        timeout = self.config.get("timeout", 5.0)
        
        metrics = {}
        for k in k_values:
            metrics[f"pass@{k}"] = 0.0
        
        return EvaluationResult(
            metrics=metrics,
            evaluator_id=self.component_id,
            details={"num_predictions": len(predictions), "k_values": k_values}
        )


def load_final_prompt(file_path: str) -> str:
    """Load final prompt from file."""
    with open(file_path, 'r') as f:
        return f.read()


def call_gemini_api(prompt: str) -> str:
    """Send prompt to Gemini API and return response."""
    import requests
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
    }
    params = {
        "key": GEMINI_API_KEY
    }
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
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
                return parts[0].get("text", "")
        
        return "Error: No response from API"
    except Exception as e:
        return f"Error calling Gemini API: {e}"


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
    
    print("\nSending prompt to Gemini API...")
    response = call_gemini_api(prompt)
    
    print(f"\n{'='*60}")
    print("LLM Response")
    print(f"{'='*60}")
    print(response)
    
    save_llm_response(response, output_file)
    print(f"\nLLM response saved to {output_file}")
    
    # Also run evaluations if ground truth available
    print(f"\n{'='*60}")
    print("Running Evaluations")
    print(f"{'='*60}")
    
    predictions = [response]
    ground_truths = [""]
    
    context = PipelineContext()
    context.predictions = predictions
    context.ground_truth = ground_truths[0] if ground_truths else ""
    
    evaluators = {
        "exact_match": ExactMatchEvaluator("exact_match", {}),
        "edit_similarity": EditSimilarityEvaluator("edit_similarity", {}),
        "pass_at_k": PassAtKEvaluator("pass_at_k", {"k_values": [1, 3, 5], "timeout": 5.0}),
    }
    
    for name, evaluator in evaluators.items():
        print(f"\n--- Evaluator: {name} ---")
        result = evaluator.evaluate(predictions, ground_truths, context)
        print(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()