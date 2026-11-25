# Memory Architecture — Basic Extensible Specification

# How to run

1. Setup + Run the full spec demo (memory/__init__.py:1, example_usage.py:1)
cd "/home/sooraj/Downloads/memory pipeline"

# activate venv (uses python3.14 from .venv/pyvenv.cfg:1)
source .venv/bin/activate

python example_usage.py
Covers: EpisodicMemory -> BugMemory -> VectorMemoryStore -> MemoryRetriever -> MemoryFormatter -> RAG + Memory integration -> register_memory_type() extenbility -> Agent(rag=..., memory=...) DI (spec §18)
2. One-liner to prove each pipeline stage with jargon output
cd "/home/sooraj/Downloads/memory pipeline"
source .venv/bin/activate

# a PASS -> Episodic Memory Extraction + Embedding + Vector Store
python3 -c "from memory import MemoryManager, VectorMemoryStore, HashEmbedder, BasicMemoryExtractor; m=MemoryManager(extractor=BasicMemoryExtractor(), store=VectorMemoryStore(embedder=HashEmbedder(dim=128))); m.remember(task='Implement JWT auth', context='Express+JWT', action='Added JWT middleware', result='All tests passed', test_result='PASS'); print(m.recall('JWT auth')[0].to_text())"

# b FAIL + Reflection -> Bug Memory
python3 -c "from memory import MemoryManager, VectorMemoryStore, HashEmbedder, BasicMemoryExtractor; m=MemoryManager(extractor=BasicMemoryExtractor(), store=VectorMemoryStore(embedder=HashEmbedder(dim=128))); m.remember(task='Auth', context='Express', action='Gen middleware', result='Tests failed: AccountOwnedByWrongProgram', test_result='FAIL', reflection={'cause':'Missing token validation','fix':'Validate JWT before route'}); print(m.recall('auth bug', memory_type='bug')[0].to_text())"

# c Semantic Retrieval + top_k + memory_type filtering (spec §15)
python3 -c "from memory import MemoryManager, VectorMemoryStore, HashEmbedder, BasicMemoryExtractor; m=MemoryManager(extractor=BasicMemoryExtractor(), store=VectorMemoryStore(embedder=HashEmbedder(dim=128))); [m.remember(task=f'Task {i}', context='ctx', action='act', result='PASS', test_result='PASS') for i in range(3)]; print('count:', len(m.recall('Task', top_k=2))); print('bugs only:', len(m.recall('Task', top_k=5, memory_type='bug')))"

# d Persistence (spec §21)
python3 -c "from memory import MemoryManager, VectorMemoryStore, HashEmbedder, BasicMemoryExtractor; m=MemoryManager(extractor=BasicMemoryExtractor(), store=VectorMemoryStore(embedder=HashEmbedder(dim=128))); m.remember(task='Persist test', context='ctx', action='act', result='PASS', test_result='PASS'); m.store.save('/tmp/memory.pkl'); n=MemoryManager(store=VectorMemoryStore(embedder=HashEmbedder(dim=128))); n.store.load('/tmp/memory.pkl'); print('loaded:', n.recall('Persist')[0].to_text()[:80])"


## 1. Objective

Implement a **basic, modular Memory Architecture** that can be plugged into the existing coding-agent pipeline:

```text
Coding Task
    ↓
RAG
    ↓
Generation
    ↓
Code
    ↓
Test
    ↓
PASS / FAIL
    ↓
Memory Extraction
    ↓
Memory Store
```

The initial implementation must support:

1. **Episodic Memory**
2. **Bug Memory**

The architecture must be designed so that additional memory constructs can be added later without changing the core pipeline.

Future memory types may include:

- Semantic Memory
- Procedural Memory
- API Memory
- Convention Memory
- Preference Memory
- Project-specific Memory
- Tool/Environment Memory

Do **not** implement these future memory types now. Only create extension points for them.

---

# 2. Design Goals

The implementation should prioritize:

- Modularity
- Clear interfaces/abstract base classes
- Easy addition of new memory types
- Separation of memory extraction from memory storage
- Separation of memory retrieval from memory storage
- Reuse of the existing vector-store abstraction where possible
- Minimal implementation for the first milestone
- No hard dependency on a specific LLM, embedding model, or vector database
- Easy unit testing
- Serialization/deserialization of memory objects
- Ability to plug the memory architecture into the existing RAG pipeline

The architecture should follow the existing project's component-oriented style.

---

# 3. High-Level Architecture

```text
                         ┌────────────────────┐
                         │    Coding Task     │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │        RAG         │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │     Generation     │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │   Generated Code   │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │       Tests        │
                         └─────────┬──────────┘
                                   │
                         ┌─────────┴─────────┐
                         │                   │
                       PASS                 FAIL
                         │                   │
                         │             ┌─────▼─────┐
                         │             │ Reflection│
                         │             └─────┬─────┘
                         │                   │
                         └─────────┬─────────┘
                                   ▼
                         ┌────────────────────┐
                         │ Memory Extraction  │
                         └─────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
          ┌──────────────────┐          ┌──────────────────┐
          │ Episodic Memory  │          │   Bug Memory     │
          └────────┬─────────┘          └────────┬─────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                         ┌────────────────────┐
                         │    Memory Store    │
                         │  Vector Store      │
                         └────────────────────┘
```

---

# 4. Package Structure

Create a new memory module/package following the project's existing structure.

Recommended structure:

```text
memory/
├── __init__.py
├── base.py
├── models.py
├── extractor.py
├── manager.py
├── retriever.py
├── types/
│   ├── __init__.py
│   ├── episodic.py
│   └── bug.py
└── stores/
    ├── __init__.py
    └── vector_store.py
```

If the existing project already has equivalent abstractions, reuse them instead of duplicating them.

---

# 5. Core Memory Model

Create a generic base memory representation.

## BaseMemory

Every memory item must contain common metadata:

```python
class BaseMemory:
    id: str
    memory_type: str
    created_at: datetime
    metadata: dict
```

Required behavior:

```python
to_dict() -> dict
from_dict(data: dict) -> BaseMemory
to_text() -> str
```

`to_text()` should return a meaningful textual representation suitable for embedding.

Do not store arbitrary application state inside the memory object.

---

# 6. Episodic Memory

Episodic memory represents a **past coding experience**.

It should capture:

```text
Task
Context
Action
Result
```

## Data model

```python
class EpisodicMemory(BaseMemory):
    task: str
    context: str
    action: str
    result: str
```

Example:

```json
{
  "memory_type": "episodic",
  "task": "Implement authentication middleware",
  "context": "Node.js Express application using JWT",
  "action": "Added JWT verification middleware and attached decoded user to request",
  "result": "All authentication tests passed"
}
```

## to_text()

Produce a structured textual representation:

```text
Task: Implement authentication middleware
Context: Node.js Express application using JWT
Action: Added JWT verification middleware and attached decoded user to request
Result: All authentication tests passed
```

---

# 7. Bug Memory

Bug memory represents a **failure, its cause, and its successful fix**.

It should capture:

```text
Bug
Cause
Fix
Context
```

## Data model

```python
class BugMemory(BaseMemory):
    bug: str
    cause: str
    fix: str
    context: str
```

Example:

```json
{
  "memory_type": "bug",
  "bug": "Tests failed with AccountOwnedByWrongProgram",
  "cause": "The test environment was using an incorrectly deployed program",
  "fix": "Redeployed the program using the correct program ID and reset the local validator",
  "context": "Anchor/Solana integration test environment"
}
```

## to_text()

Produce:

```text
Bug: Tests failed with AccountOwnedByWrongProgram
Cause: The test environment was using an incorrectly deployed program
Fix: Redeployed the program using the correct program ID and reset the local validator
Context: Anchor/Solana integration test environment
```

---

# 8. Memory Type Registry

Do not hard-code memory types throughout the system.

Create a registry/factory mechanism.

Example:

```python
MEMORY_TYPES = {
    "episodic": EpisodicMemory,
    "bug": BugMemory,
}
```

The system should allow:

```python
register_memory_type(
    "semantic",
    SemanticMemory
)
```

in the future.

Adding a new memory type should NOT require modifying:

- MemoryManager
- MemoryStore
- MemoryRetriever
- Existing memory classes
- Pipeline orchestration code

---

# 9. Memory Extraction

Create a `MemoryExtractor` abstraction.

```python
class MemoryExtractor(ABC):

    @abstractmethod
    def extract(
        self,
        task,
        context,
        action,
        result,
        test_result=None,
        reflection=None
    ) -> list[BaseMemory]:
        pass
```

The extractor is responsible for deciding:

- What information is worth remembering
- Which memory type it belongs to
- What fields should be populated

---

# 10. Basic Extraction Rules

For the first implementation, do NOT require an LLM-based extractor.

Implement a simple deterministic extractor.

## On PASS

Create:

```text
EpisodicMemory
```

containing:

- Task
- Context
- Action
- Result

Example:

```text
Task → Implement function X
Context → Repository/file context
Action → Generated implementation
Result → Tests passed
```

## On FAIL + Reflection

Create:

```text
BugMemory
```

containing:

- Bug
- Cause
- Fix
- Context

The reflection output should be used as the source for:

- Cause
- Fix

The test failure should be used as the source for:

- Bug

Example:

```text
Bug:
    pytest test_auth failed

Cause:
    Missing token validation

Fix:
    Validate JWT before accessing protected route
```

The extractor should also create an `EpisodicMemory` for successful outcomes when appropriate.

---

# 11. Future LLM-Based Extraction

The architecture must allow the deterministic extractor to later be replaced by:

```python
LLMMemoryExtractor(MemoryExtractor)
```

without changing the MemoryManager or MemoryStore.

Potential future flow:

```text
Task + Context + Action + Test Result + Reflection
                         │
                         ▼
                       LLM
                         │
                         ▼
                 Structured JSON
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Episodic Memory         Bug Memory
```

Do not implement this now unless the existing project already has an LLM abstraction that makes it trivial.

---

# 12. Memory Store Interface

Create a generic storage abstraction.

```python
class MemoryStore(ABC):

    @abstractmethod
    def add(self, memory: BaseMemory) -> str:
        pass

    @abstractmethod
    def add_many(self, memories: list[BaseMemory]) -> list[str]:
        pass

    @abstractmethod
    def get(self, memory_id: str) -> BaseMemory | None:
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
```

Do NOT make `MemoryManager` directly depend on FAISS, Chroma, Pinecone, etc.

---

# 13. Vector Store Adapter

Create:

```python
class VectorMemoryStore(MemoryStore):
    ...
```

This adapter should store:

```text
Embedding
    +
Memory text
    +
Memory metadata
    +
Memory ID
```

The vector store should be replaceable.

For the first implementation, use the project's existing vector-store abstraction if one exists.

If the existing RAG pipeline already supports FAISS, reuse it.

Do not duplicate embedding/vector-store implementations unnecessarily.

---

# 14. Embedding Interface

Memory storage should depend on an embedding interface rather than a specific model.

```python
class MemoryEmbedder(ABC):

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        pass
```

If the RAG pipeline already has an `Embedder` abstraction, reuse that abstraction instead of creating `MemoryEmbedder`.

The goal is:

```text
Memory
   ↓
to_text()
   ↓
Embedder
   ↓
Vector Store
```

---

# 15. Memory Retrieval

Create:

```python
class MemoryRetriever:
```

The retriever should support semantic retrieval.

Interface:

```python
class MemoryRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str | None = None
    ) -> list[BaseMemory]:
        pass
```

Required capabilities:

### Retrieve all memory types

```python
retrieve(query="authentication bug")
```

### Retrieve only a specific type

```python
retrieve(
    query="authentication bug",
    memory_type="bug"
)
```

### Limit number of memories

```python
retrieve(
    query="authentication",
    top_k=3
)
```

---

# 16. Memory Manager

Create a high-level orchestration class.

```python
class MemoryManager:
```

It should coordinate:

```text
Extractor
    ↓
Memory Objects
    ↓
Store
```

and:

```text
Query
    ↓
Retriever
    ↓
Relevant Memories
```

Interface:

```python
class MemoryManager:

    def remember(
        self,
        task,
        context,
        action,
        result,
        test_result=None,
        reflection=None
    ) -> list[BaseMemory]:
        ...

    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str | None = None
    ) -> list[BaseMemory]:
        ...
```

The manager should NOT know how individual memory types are represented internally.

---

# 17. Memory Lifecycle

The basic lifecycle should be:

```text
1. Coding Task
       ↓
2. RAG Retrieval
       ↓
3. Code Generation
       ↓
4. Test Execution
       ↓
5. Reflection if failure
       ↓
6. Memory Extraction
       ↓
7. Memory Objects
       ↓
8. Embedding
       ↓
9. Vector Store
```

At the next task:

```text
New Coding Task
       ↓
Memory Query
       ↓
Relevant Memories
       ↓
RAG / Prompt Context
       ↓
Generation
```

Memory retrieval must therefore be usable independently of memory creation.

---

# 18. Integration With Existing RAG

Do NOT tightly couple Memory and RAG.

The integration should look like:

```python
memories = memory_manager.recall(
    query=task,
    top_k=5
)

rag_context = rag.retrieve(task)

final_context = {
    "rag": rag_context,
    "memory": memories
}
```

The existing RAG pipeline should remain functional if Memory is disabled.

Use dependency injection/configuration where practical:

```python
Agent(
    rag=rag_pipeline,
    memory=memory_manager
)
```

or:

```python
Agent(
    rag=rag_pipeline,
    memory=None
)
```

---

# 19. Memory Formatting for LLM Prompt

Create a small formatter:

```python
class MemoryFormatter:
    def format(self, memories: list[BaseMemory]) -> str:
        ...
```

Example output:

```text
Relevant Past Experience:

[Episodic Memory]
Task: Implement authentication middleware
Context: Express + JWT
Action: Added JWT verification middleware
Result: Tests passed

[Bug Memory]
Bug: JWT tests failed
Cause: Token was not validated before route access
Fix: Added JWT validation middleware
Context: Express authentication
```

The formatter must not contain retrieval logic.

---

# 20. Metadata

Every stored memory should contain useful metadata:

```python
{
    "memory_id": "...",
    "memory_type": "episodic",
    "created_at": "...",
    "project": "...",
    "language": "...",
    "repository": "...",
    "file_paths": [...],
}
```

Metadata should be optional.

Do not make project/repository metadata mandatory if the existing pipeline does not provide it.

---

# 21. Persistence

The initial implementation should support persistence through the vector store.

The architecture should NOT assume that memories live only in RAM.

Provide:

```python
save()
load()
```

where supported by the underlying store.

If FAISS is used, persist:

```text
index
metadata
memory objects
```

separately if necessary.

The exact persistence mechanism should follow the existing RAG vector-store implementation.

---

# 22. Configuration

Create a configuration object.

Example:

```python
@dataclass
class MemoryConfig:
    enabled: bool = True
    top_k: int = 5
    enabled_memory_types: list[str] = field(
        default_factory=lambda: ["episodic", "bug"]
    )
```

Future configuration may support:

```text
similarity threshold
memory compression
deduplication
memory decay
reranking
memory importance
```

Do not implement these advanced features now.

---

# 23. Deduplication

Implement only basic deduplication if it is easy using an existing store mechanism.

Do NOT build a sophisticated memory consolidation system yet.

A future version may support:

```text
Duplicate Memories
        ↓
Similarity Detection
        ↓
Merge / Update
        ↓
Consolidated Memory
```

Keep this outside the current core architecture.

---

# 24. Error Handling

Memory failure must NOT break the coding pipeline.

For example:

```text
Generation → SUCCESS
Test       → SUCCESS
Memory     → FAILURE
```

should still result in:

```text
Coding Task → SUCCESS
```

Memory operations should be isolated and logged.

Do not silently hide errors during development; log them appropriately.

---

# 25. Logging

Provide useful logs such as:

```text
[Memory] Extracted 1 episodic memory
[Memory] Extracted 1 bug memory
[Memory] Stored memory id=...
[Memory] Retrieved 3 memories for query
```

Use the project's existing logging system if one exists.

---

# 26. Testing Requirements

Create unit tests for:

## Episodic Memory

- Creation
- Serialization
- Deserialization
- `to_text()`

## Bug Memory

- Creation
- Serialization
- Deserialization
- `to_text()`

## Memory Extractor

Test:

```text
PASS → Episodic Memory
FAIL + Reflection → Bug Memory
```

## Memory Store

Test:

```text
add()
get()
delete()
clear()
add_many()
```

## Retrieval

Test:

```text
query → relevant memories
top_k respected
memory_type filtering works
```

## Memory Manager

Test end-to-end:

```text
remember()
    ↓
extract
    ↓
store

recall()
    ↓
retrieve
```

---

# 27. Example Usage

The final API should be approximately:

```python
memory = MemoryManager(
    extractor=BasicMemoryExtractor(),
    store=VectorMemoryStore(
        embedder=embedder,
        vector_store=vector_store
    )
)
```

Store a successful coding episode:

```python
memory.remember(
    task="Implement JWT authentication",
    context="Express backend",
    action="Added JWT middleware",
    result="All tests passed",
    test_result="PASS"
)
```

Store a bug:

```python
memory.remember(
    task="Implement JWT authentication",
    context="Express backend",
    action="Generated authentication middleware",
    result="Tests failed",
    test_result="FAIL",
    reflection={
        "cause": "JWT was not validated before route execution",
        "fix": "Added token validation middleware"
    }
)
```

Recall:

```python
memories = memory.recall(
    query="JWT authentication middleware",
    top_k=5
)
```

Filter:

```python
bugs = memory.recall(
    query="JWT authentication failure",
    top_k=3,
    memory_type="bug"
)
```

---

# 28. Extensibility Requirement

Adding a new memory type should require approximately:

```text
1. Create new class
       ↓
2. Inherit BaseMemory
       ↓
3. Define fields
       ↓
4. Implement to_text()
       ↓
5. Register memory type
```

For example, future:

```python
class SemanticMemory(BaseMemory):
    ...
```

or:

```python
class ProceduralMemory(BaseMemory):
    ...
```

should work without changing:

```text
MemoryManager
MemoryStore
MemoryRetriever
VectorMemoryStore
MemoryFormatter
```

unless the new memory type requires genuinely specialized behavior.

---

# 29. What NOT to Implement

Do not implement these in the first version:

- Complex memory consolidation
- Memory forgetting/decay
- Importance scoring
- Reflection-driven memory evolution
- Graph memory
- Knowledge graph
- Hierarchical memory
- Multi-agent shared memory
- Long-term/short-term memory hierarchy
- LLM-based automatic memory classification
- Sophisticated reranking
- Cross-project memory reasoning
- Memory compression/summarization
- Reinforcement learning for memory selection

These should remain future extension points.

---

# 30. Expected First-Milestone Result

The first milestone should demonstrate:

```text
Coding Task
     ↓
Existing RAG
     ↓
Generation
     ↓
Test
     ↓
PASS ───────────────┐
                    │
FAIL → Reflection ──┤
                    ▼
             Memory Extraction
                    │
             ┌──────┴──────┐
             ▼             ▼
         Episodic         Bug
         Memory          Memory
             │             │
             └──────┬──────┘
                    ▼
               Vector Store
```

Then, on a future task:

```text
New Task
   ↓
Memory Retrieval
   ↓
Relevant Episodic/Bug Memories
   ↓
RAG
   ↓
LLM
```

The key architectural principle is:

> **Memory types are plugins. Memory orchestration, retrieval, and storage must remain generic.**

The implementation should be small, clean, and extensible rather than trying to implement every possible memory technique.
