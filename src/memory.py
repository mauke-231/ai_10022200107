"""
PART G: INNOVATION — MEMORY-BASED RAG
Author: Maukewonge Yaw Nyarko-Tetteh | Index Number: 10022200107

Implements a session memory system that:
1. Stores past Q&A pairs in a memory buffer
2. Embeds past exchanges and retrieves relevant ones at query time
3. Injects relevant memory into the prompt context
4. Persists memory across sessions (file-based)

This goes beyond naive conversation history by treating past answers
as retrievable knowledge, enabling the RAG to answer follow-up
questions about its own prior responses.
"""

import os
import json
import time
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single Q&A exchange stored in memory."""
    entry_id: str
    query: str
    answer: str
    sources_used: List[str]       # chunk_ids used
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None   # stored as list for JSON serialisation

    def to_context_string(self) -> str:
        return f"Previous Q: {self.query}\nPrevious A: {self.answer}"


class MemoryStore:
    """
    Session memory store with semantic retrieval.

    Design:
    - Entries are embedded on write
    - On each new query, top-k relevant memories are retrieved
    - Memory is injected as additional context before fresh retrieval
    - Persisted to JSON so memory survives page refreshes (Streamlit)
    """

    def __init__(self, memory_path: str, max_entries: int = 50):
        self.memory_path = memory_path
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self.embeddings: Optional[np.ndarray] = None
        self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path) as f:
                    data = json.load(f)
                self.entries = [MemoryEntry(**e) for e in data]
                # Rebuild embedding matrix
                embs = [e.embedding for e in self.entries if e.embedding]
                if embs:
                    self.embeddings = np.array(embs, dtype=np.float32)
                logger.info(f"Memory loaded: {len(self.entries)} entries")
            except Exception as ex:
                logger.warning(f"Memory load failed: {ex}. Starting fresh.")
                self.entries = []

    def _save(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        data = []
        for e in self.entries:
            d = asdict(e)
            if d["embedding"] is not None:
                d["embedding"] = [float(x) for x in d["embedding"]]
            data.append(d)
        with open(self.memory_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Write ─────────────────────────────────────────────────────────

    def add(self, query: str, answer: str, sources: List[str],
            embedder) -> MemoryEntry:
        """Add a new Q&A pair to memory, embedding the query."""
        entry_id = f"mem_{int(time.time() * 1000)}"
        q_embedding = embedder.embed_single(query).tolist()
        entry = MemoryEntry(
            entry_id=entry_id,
            query=query,
            answer=answer,
            sources_used=sources,
            embedding=q_embedding
        )
        self.entries.append(entry)

        # Rebuild embedding matrix
        embs = [e.embedding for e in self.entries if e.embedding]
        self.embeddings = np.array(embs, dtype=np.float32)

        # Trim if over max
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
            self.embeddings = self.embeddings[-self.max_entries:]

        self._save()
        logger.info(f"Memory entry added: {entry_id}")
        return entry

    # ── Retrieve ──────────────────────────────────────────────────────

    def retrieve_relevant(self, query: str, embedder,
                           top_k: int = 3,
                           min_similarity: float = 0.5
                           ) -> List[Tuple[MemoryEntry, float]]:
        """
        Find past exchanges semantically similar to the current query.
        Returns (MemoryEntry, cosine_score) pairs.
        """
        if not self.entries or self.embeddings is None:
            return []

        q_vec = embedder.embed_single(query).reshape(1, -1)
        # Cosine similarity (embeddings are L2-normalised)
        sims = (self.embeddings @ q_vec.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score >= min_similarity:
                results.append((self.entries[idx], score))
        return results

    # ── Context string for prompt injection ───────────────────────────

    def get_memory_context(self, query: str, embedder,
                            top_k: int = 2) -> str:
        """
        Returns a formatted string of relevant past exchanges
        ready to inject into a prompt.
        """
        relevant = self.retrieve_relevant(query, embedder, top_k=top_k)
        if not relevant:
            return ""
        lines = ["RELEVANT PAST EXCHANGES (from session memory):"]
        for entry, score in relevant:
            lines.append(f"  [Memory | similarity={score:.2f}]")
            lines.append(f"  Q: {entry.query}")
            lines.append(f"  A: {entry.answer[:300]}...")
            lines.append("")
        return "\n".join(lines)

    # ── Conversation history (ordered) ───────────────────────────────

    def get_recent_history(self, n_turns: int = 6) -> List[Dict]:
        """Return last n_turns entries as role/content dicts for prompt."""
        history = []
        for entry in self.entries[-n_turns:]:
            history.append({"role": "user", "content": entry.query})
            history.append({"role": "assistant", "content": entry.answer})
        return history

    def clear(self):
        """Wipe all memory entries."""
        self.entries = []
        self.embeddings = None
        if os.path.exists(self.memory_path):
            os.remove(self.memory_path)
        logger.info("Memory cleared.")

    def __len__(self):
        return len(self.entries)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from retrieval import EmbeddingPipeline

    embedder = EmbeddingPipeline()
    store = MemoryStore("/tmp/test_memory.json")

    store.add("What is Ghana's GDP?", "Ghana's GDP growth is 4.0% in 2025.", [], embedder)
    store.add("Who won the 2024 election?", "John Dramani Mahama of NDC won.", [], embedder)

    results = store.retrieve_relevant("Tell me about Ghana economic growth", embedder)
    for entry, score in results:
        print(f"[{score:.3f}] {entry.query} → {entry.answer[:80]}")
