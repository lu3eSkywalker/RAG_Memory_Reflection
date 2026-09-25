"""
Reference implementations of RAG components matching RepoCoder architecture.
"""

import os
import json
import pickle
import numpy as np
import scipy.spatial.distance
from abc import ABC
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from .base import (
    Chunker, Embedder, Retriever, PromptBuilder, 
    Generator, IterationStrategy, Evaluator, VectorStore,
    Document, RetrievalResult, GenerationResult, EvaluationResult
)
from .context import PipelineContext

logger = logging.getLogger(__name__)


# ============================================================
# CHUNKERS
# ============================================================

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
            
            if len(chunk_lines) < 3:  # Skip tiny chunks
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
        # Simplified - in practice would use sentence transformers
        threshold = self.config.get("breakpoint_threshold", 0.8)
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = []
        
        for para in paragraphs:
            current_chunk.append(para)
            # In real impl, check semantic similarity between adjacent paragraphs
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


# ============================================================
# EMBEDDERS
# ============================================================

# ============================================================
# RETRIEVERS
# ============================================================

class CosineSimilarityRetriever(Retriever):
    """Cosine similarity retriever using FAISS vector store."""
    
    def retrieve(self, query: str, index: List[Document], context: PipelineContext) -> RetrievalResult:
        top_k = self.config.get("top_k", 20)
        filter_future = self.config.get("filter_future_context", True)
        
        # Get embedder from context to embed query
        embedder = context.get("embedder_instance")
        if embedder and hasattr(embedder, 'embed_query'):
            query_embedding = embedder.embed_query(query)
        else:
            # Fallback: use first document's embedding dimension
            dim = 768
            query_embedding = np.random.randn(dim).tolist()
        
        # Use FAISS vector store if available
        vector_store = context.vector_store
        if vector_store is not None and hasattr(vector_store, 'search'):
            # FAISS index search
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            scores, indices = vector_store.search(query_array, top_k)
            
            scored_docs = []
            query_metadata = context.metadata
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(context.documents):
                    doc = context.documents[idx]
                    if filter_future and self._is_future_context(doc, query_metadata):
                        continue
                    doc.score = float(score)
                    scored_docs.append(doc)
        else:
            # Fallback: linear scan (original behavior)
            scored_docs = []
            query_metadata = context.metadata
            
            for doc in index:
                if filter_future and self._is_future_context(doc, query_metadata):
                    continue
                
                doc_embedding = doc.embedding
                if doc_embedding is None:
                    continue
                
                # Cosine similarity
                sim = 1 - scipy.spatial.distance.cosine(query_embedding, doc_embedding)
                
                scored_docs.append(Document(
                    content=doc.content,
                    metadata=doc.metadata,
                    embedding=doc.embedding,
                    score=sim
                ))
            
            scored_docs.sort(key=lambda d: d.score or 0, reverse=True)
            scored_docs = scored_docs[:top_k]
        
        return RetrievalResult(
            query=query,
            documents=scored_docs,
            retriever_id=self.component_id
        )
    
    def _is_future_context(self, doc: Document, query_metadata: Dict) -> bool:
        doc_meta = doc.metadata
        if not doc_meta or not query_metadata:
            return False
        if doc_meta.get("file_path") != query_metadata.get("file_path"):
            return False
        doc_end = doc_meta.get("end_line", 0)
        hole_start = query_metadata.get("context_start_lineno", 0)
        return doc_end > hole_start


# ============================================================
# PROMPT BUILDERS
# ============================================================

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
        
        # Estimate token count (rough: 1 token ~= 4 chars)
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
        
        # Prepend in reverse order (most relevant last = closest to query)
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
            # Extended: show more context around the retrieved window
            # In real impl, read from actual file
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
        
        # Truncate if needed
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


# ============================================================
# GENERATORS
# ============================================================

# ============================================================
# ITERATION STRATEGIES
# ============================================================

class SinglePassStrategy(IterationStrategy):
    """Standard single-pass RAG."""
    
    def run(self, initial_input: Dict, pipeline: 'RAGPipeline', context: PipelineContext) -> Dict:
        # Single pass - pipeline.run() handles this
        return {"predictions": context.predictions}


class RepoCoderIterativeStrategy(IterationStrategy):
    """RepoCoder's two-round iterative retrieval-generation."""
    
    def run(self, initial_input: Dict, pipeline: 'RAGPipeline', context: PipelineContext) -> Dict:
        num_rounds = self.config.get("num_rounds", 2)
        use_predictions = self.config.get("use_predictions_as_queries", True)
        
        all_predictions = []
        iteration_history = []
        
        # Round 1: Standard retrieval on original query
        context.iteration = 1
        logger.info(f"RepoCoder Round 1: Retrieving for original query")
        
        # Run retrieval -> prompt -> generate for round 1
        # This is simplified - real impl would re-run components
        round1_result = self._execute_round(pipeline, context, initial_input["query"], round_num=1)
        all_predictions.extend(round1_result.get("predictions", []))
        iteration_history.append({"round": 1, **round1_result})
        
        # Round 2: Use predictions as new queries
        if num_rounds >= 2 and use_predictions and round1_result.get("predictions"):
            context.iteration = 2
            logger.info(f"RepoCoder Round 2: Retrieving using predictions")
            
            # Use first prediction as new query (or aggregate)
            new_query = round1_result["predictions"][0]
            round2_result = self._execute_round(pipeline, context, new_query, round_num=2)
            all_predictions.extend(round2_result.get("predictions", []))
            iteration_history.append({"round": 2, **round2_result})
        
        return {
            "predictions": all_predictions,
            "iteration_history": iteration_history
        }
    
    def _execute_round(self, pipeline: 'RAGPipeline', context: PipelineContext, query: str, round_num: int) -> Dict:
        """Execute one retrieval-generation round."""
        # This is a simplified version - in practice you'd re-run the retriever/prompt/generator
        # with the new query. The full implementation would need to re-execute those components.
        
        # For now, just return mock
        return {
            "predictions": [f"# Round {round_num} prediction\n    pass\n"],
            "query_used": query
        }


class SelfRefineStrategy(IterationStrategy):
    """Self-refine: generate -> critique -> refine loop."""
    
    def run(self, initial_input: Dict, pipeline: 'RAGPipeline', context: PipelineContext) -> Dict:
        max_iterations = self.config.get("max_iterations", 3)
        
        all_predictions = []
        iteration_history = []
        
        current_query = initial_input["query"]
        
        for i in range(max_iterations):
            context.iteration = i + 1
            
            # Generate
            # (In real impl, execute generator component)
            prediction = f"# Iteration {i+1} prediction\n    pass\n"
            all_predictions.append(prediction)
            
            # Critique (would use a critique prompt)
            critique = f"# Critique for iteration {i+1}"
            
            # Refine (would use refine prompt with critique)
            current_query = f"{initial_input['query']}\n\nPrevious: {prediction}\nCritique: {critique}\n\nRefined:"
            
            iteration_history.append({
                "iteration": i + 1,
                "prediction": prediction,
                "critique": critique
            })
        
        return {
            "predictions": all_predictions,
            "iteration_history": iteration_history
        }


class ActiveRAGStrategy(IterationStrategy):
    """Active RAG: multi-hop retrieval with confidence stopping."""
    
    def run(self, initial_input: Dict, pipeline: 'RAGPipeline', context: PipelineContext) -> Dict:
        max_hops = self.config.get("max_hops", 3)
        confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        all_predictions = []
        iteration_history = []
        
        current_query = initial_input["query"]
        
        for hop in range(max_hops):
            context.iteration = hop + 1
            
            # Retrieve and generate
            prediction = f"# Hop {hop+1} prediction\n    pass\n"
            confidence = 0.5 + hop * 0.2  # Mock increasing confidence
            
            all_predictions.append(prediction)
            iteration_history.append({
                "hop": hop + 1,
                "prediction": prediction,
                "confidence": confidence
            })
            
            if confidence >= confidence_threshold:
                logger.info(f"ActiveRAG stopping at hop {hop+1} with confidence {confidence}")
                break
            
            # Next hop: use prediction to form new query
            current_query = f"{initial_input['query']}\n\nContext: {prediction}"
        
        return {
            "predictions": all_predictions,
            "iteration_history": iteration_history
        }


# ============================================================
# EVALUATORS
# ============================================================

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
        
        import difflib
        
        pred = predictions[0]
        truth = ground_truths[0]
        
        # SequenceMatcher ratio
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
        
        # Mock implementation - real would execute code and check tests
        metrics = {}
        for k in k_values:
            # In real impl: execute predictions[:k] against test cases
            metrics[f"pass@{k}"] = 0.0  # Placeholder
        
        return EvaluationResult(
            metrics=metrics,
            evaluator_id=self.component_id,
            details={"num_predictions": len(predictions), "k_values": k_values}
        )


# ============================================================
# VECTOR STORES
# ============================================================

class FAISSVectorStore(VectorStore):
    """FAISS vector store for efficient similarity search."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not installed. Install with: pip install faiss-cpu")
        self._index = None
        self._documents = []
        self._dimension = None
    
    def add_documents(self, documents: List[Document], context: PipelineContext) -> Any:
        if not documents:
            return self._index
        
        # Get embeddings from documents
        embeddings = []
        valid_docs = []
        for doc in documents:
            if doc.embedding is not None:
                embeddings.append(doc.embedding)
                valid_docs.append(doc)
        
        if not embeddings:
            logger.warning("No documents with embeddings to add")
            return self._index
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        if self._index is None:
            self._dimension = embeddings_array.shape[1]
            # Use IndexFlatIP for cosine similarity (with normalized vectors)
            self._index = faiss.IndexFlatIP(self._dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self._index.add(embeddings_array)
        self._documents.extend(valid_docs)
        
        # Store in context
        context.vector_store = self._index
        
        return self._index
    
    def search(self, query_embedding: List[float], top_k: int, context: PipelineContext) -> List[Tuple[Document, float]]:
        if self._index is None or len(self._documents) == 0:
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        scores, indices = self._index.search(query_array, min(top_k, len(self._documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._documents):
                doc = self._documents[idx]
                doc.score = float(score)
                results.append((doc, float(score)))
        
        return results
    
    def save(self, path: str) -> None:
        if self._index is not None:
            faiss.write_index(self._index, path)
            # Save documents metadata separately
            import pickle
            with open(path + ".docs", "wb") as f:
                pickle.dump(self._documents, f)
    
    def load(self, path: str) -> None:
        self._index = faiss.read_index(path)
        import pickle
        with open(path + ".docs", "rb") as f:
            self._documents = pickle.load(f)
        self._dimension = self._index.d


# ============================================================
# EMBEDDERS
# ============================================================

class GeminiEmbedder(Embedder):
    """Google Gemini embeddings (text-embedding-004)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not GENAI_AVAILABLE:
            raise RuntimeError("google-generativeai not installed. Install with: pip install google-generativeai")
        
        api_key = self.config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in config or environment")
        
        genai.configure(api_key=api_key)
        self.model = self.config.get("model", "models/text-embedding-004")
        self.batch_size = self.config.get("batch_size", 100)
    
    def embed(self, documents: List[Document]) -> List[Document]:
        embedded = []
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            texts = [doc.content for doc in batch]
            
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=texts,
                    task_type="retrieval_document"
                )
                
                embeddings = result["embedding"]
                
                for doc, embedding in zip(batch, embeddings):
                    embedded.append(Document(
                        content=doc.content,
                        metadata=doc.metadata,
                        embedding=embedding
                    ))
                    
            except Exception as e:
                logger.error(f"Gemini embedding failed: {e}")
                # Fallback: random embeddings
                for doc in batch:
                    mock_embedding = np.random.randn(768).tolist()
                    embedded.append(Document(
                        content=doc.content,
                        metadata=doc.metadata,
                        embedding=mock_embedding
                    ))
        
        return embedded
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query for retrieval."""
        try:
            result = genai.embed_content(
                model=self.model,
                content=query,
                task_type="retrieval_query"
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"Gemini query embedding failed: {e}")
            return np.random.randn(768).tolist()


class SentenceTransformerEmbedder(Embedder):
    """Sentence Transformers embeddings (all-MiniLM-L6-v2)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        
        self.model_name = self.config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = self.config.get("batch_size", 32)
        self.device = self.config.get("device", "cpu")
        self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def embed(self, documents: List[Document]) -> List[Document]:
        embedded = []
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            texts = [doc.content for doc in batch]
            
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                
                for doc, embedding in zip(batch, embeddings):
                    embedded.append(Document(
                        content=doc.content,
                        metadata=doc.metadata,
                        embedding=embedding.tolist()
                    ))
                    
            except Exception as e:
                logger.error(f"SentenceTransformer embedding failed: {e}")
                # Fallback: random embeddings (384 dims for MiniLM)
                for doc in batch:
                    mock_embedding = np.random.randn(384).tolist()
                    embedded.append(Document(
                        content=doc.content,
                        metadata=doc.metadata,
                        embedding=mock_embedding
                    ))
        
        return embedded
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query for retrieval."""
        try:
            embedding = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer query embedding failed: {e}")
            return np.random.randn(384).tolist()


# ============================================================
# GENERATORS
# ============================================================

class GeminiGenerator(Generator):
    """Google Gemini generator for code completion."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not GENAI_AVAILABLE:
            raise RuntimeError("google-generativeai not installed. Install with: pip install google-generativeai")
        
        api_key = self.config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in config or environment")
        
        genai.configure(api_key=api_key)
        self.model_name = self.config.get("model", "gemini-1.5-pro")
        self.temperature = self.config.get("temperature", 0.2)
        self.top_p = self.config.get("top_p", 0.95)
        self.max_tokens = self.config.get("max_tokens", 2048)
        self.model = genai.GenerativeModel(self.model_name)
    
    def generate(self, prompt: str, context: PipelineContext) -> GenerationResult:
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                max_output_tokens=self.max_tokens,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            completions = [response.text] if response.text else [""]
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            completions = ["# Generation failed\n    pass\n"]
        
        return GenerationResult(
            prompt=prompt,
            completions=completions,
            generator_id=self.component_id,
            metadata={"model": self.model_name}
        )