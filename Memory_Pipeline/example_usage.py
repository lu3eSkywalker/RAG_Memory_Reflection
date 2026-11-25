"""
Example usage per spec Section 27 — integration with RAG pipeline
"""
from memory import MemoryManager, VectorMemoryStore, HashEmbedder, BasicMemoryExtractor, MemoryFormatter

# Agent dependency injection example: memory is optional
def get_rag_context(task: str):
    # placeholder for existing RAG
    return f"RAG context for: {task}"

# Initialize memory
memory = MemoryManager(
    extractor=BasicMemoryExtractor(),
    store=VectorMemoryStore(embedder=HashEmbedder(dim=128)),
)

# ---- Store successful episode ----
print("=== Storing successful episode ===")
memory.remember(
    task="Implement JWT authentication",
    context="Express backend",
    action="Added JWT middleware",
    result="All tests passed",
    test_result="PASS"
)

# ---- Store bug ----
print("=== Storing bug ===")
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

# ---- Recall (integration with RAG) ----
print("\n=== Recall + RAG integration ===")
task = "Implement authentication for new route"
memories = memory.recall(query=task, top_k=5)
rag_context = get_rag_context(task)
formatter = MemoryFormatter()
final_context = {
    "rag": rag_context,
    "memory": formatter.format(memories)
}
print(final_context["memory"])
print("\nRAG:", rag_context)

# ---- Filtered recall ----
print("\n=== Filtered recall (bugs only) ===")
bugs = memory.recall(query="JWT authentication failure", top_k=3, memory_type="bug")
for b in bugs:
    print(b.to_text())
    print("---")

# ---- Extensibility demo ----
print("\n=== Extensibility: adding new memory type without changing manager/store ===")
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from memory.base import BaseMemory, register_memory_type

@dataclass
class SemanticMemory(BaseMemory):
    concept: str = ""
    description: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = field(default="semantic")
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    def to_text(self): return f"Concept: {self.concept}\nDescription: {self.description}"
    @classmethod
    def from_dict(cls, data: dict):
        if isinstance(data.get("created_at"), str):
            data=dict(data)
            try: data["created_at"]=datetime.fromisoformat(data["created_at"].replace("Z","+00:00"))
            except: pass
        return cls(id=data.get("id", str(uuid.uuid4())), memory_type=data.get("memory_type","semantic"), created_at=data.get("created_at", datetime.utcnow()), metadata=data.get("metadata",{}), concept=data.get("concept",""), description=data.get("description",""))

register_memory_type("semantic", SemanticMemory)
# Need to enable it via config
memory.config.enabled_memory_types.append("semantic")
memory.store.add(SemanticMemory(concept="JWT", description="JWT is JSON Web Token used for auth"))
print("Recalled semantic:", memory.recall("JWT", memory_type="semantic")[0].to_text())

# ---- Agent with optional memory ----
print("\n=== Agent with optional memory ===")
class Agent:
    def __init__(self, rag, memory=None):
        self.rag = rag
        self.memory = memory
    def run(self, task):
        ctx = self.rag(task) if self.rag else ""
        mem_ctx = ""
        if self.memory:
            mems = self.memory.recall(task, top_k=3)
            mem_ctx = MemoryFormatter().format(mems)
        return {"rag": ctx, "memory": mem_ctx}

agent_with = Agent(rag=get_rag_context, memory=memory)
agent_without = Agent(rag=get_rag_context, memory=None)
print("with memory:", agent_with.run("auth task")["memory"][:100])
print("without memory:", agent_without.run("auth task"))
