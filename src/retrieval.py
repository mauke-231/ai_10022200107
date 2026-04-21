"""
PART B: CUSTOM RETRIEVAL SYSTEM
Author: Maukewonge Yaw Nyarko-Tetteh | Index Number: 10022200107

Implements:
- Manual embedding pipeline (sentence-transformers)
- FAISS vector store (custom wrapper — no LangChain/LlamaIndex)
- Top-k retrieval with similarity scoring
- Hybrid search: keyword (BM25-style TF-IDF) + vector (extension)
- Failure case detection and fix
"""

import os
import json
import math
import logging
import pickle
import re
import time
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import faiss
import torch

# Workaround: torch 2.11+ loads weights as meta tensors then calls .to(device),
# which raises NotImplementedError when the target is CPU. Safe to skip here.
_orig_module_to = torch.nn.Module.to
def _safe_module_to(self, *args, **kwargs):
    try:
        return _orig_module_to(self, *args, **kwargs)
    except NotImplementedError:
        return self
torch.nn.Module.to = _safe_module_to

from sentence_transformers import SentenceTransformer

from data_engineering import Chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"   # fast, 384-dim, good quality


# ─────────────────────────────────────────────
# EMBEDDING PIPELINE
# ─────────────────────────────────────────────

class EmbeddingPipeline:
    """
    Manual embedding pipeline using sentence-transformers.
    No LangChain/LlamaIndex — raw model inference only.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dim: {self.dim}")

    def embed(self, texts: List[str], batch_size: int = 64,
              show_progress: bool = True) -> np.ndarray:
        """Return float32 numpy array shape (N, dim)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2-normalise for cosine via dot product
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text], show_progress=False)[0]


# ─────────────────────────────────────────────
# FAISS VECTOR STORE
# ─────────────────────────────────────────────

class VectorStore:
    """
    Custom FAISS wrapper.
    Stores chunk metadata alongside the index for full round-trip retrieval.
    """

    def __init__(self, dim: int):
        self.dim = dim
        # Inner-product index (dot product on L2-normalised = cosine similarity)
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []
        logger.info(f"VectorStore initialised (dim={dim})")

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Add chunks and their embeddings to the store."""
        assert len(chunks) == embeddings.shape[0]
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} vectors. Total: {self.index.ntotal}")

    def search(self, query_vec: np.ndarray, top_k: int = 5
               ) -> List[Tuple[Chunk, float]]:
        """
        Top-k vector search.
        Returns list of (Chunk, cosine_score) sorted descending.
        """
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"VectorStore saved to {path}")

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"VectorStore loaded. Total vectors: {self.index.ntotal}")


# ─────────────────────────────────────────────
# KEYWORD RETRIEVER (BM25-style TF-IDF)
# ─────────────────────────────────────────────

class KeywordRetriever:
    """
    Lightweight inverted-index BM25-style scorer.
    No external libraries — pure Python.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.tf: List[Dict[str, float]] = []   # term freq per doc
        self.df: Dict[str, int] = defaultdict(int)  # doc freq
        self.avg_dl: float = 0.0
        self.N: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.N = len(chunks)
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        dls = []
        for i, chunk in enumerate(chunks):
            tokens = self._tokenize(chunk.text)
            dls.append(len(tokens))
            tf = Counter(tokens)
            self.tf.append(tf)
            for term in set(tokens):
                self.df[term] += 1
                self.inverted_index[term].append(i)
        self.avg_dl = sum(dls) / max(len(dls), 1)
        logger.info(f"KeywordRetriever built over {self.N} docs, vocab={len(self.df)}")

    def score(self, query: str, doc_idx: int) -> float:
        terms = self._tokenize(query)
        tf = self.tf[doc_idx]
        dl = sum(tf.values())
        score = 0.0
        for term in terms:
            if term not in self.df:
                continue
            idf = math.log((self.N - self.df[term] + 0.5) /
                           (self.df[term] + 0.5) + 1)
            tf_val = tf.get(term, 0)
            numerator = tf_val * (self.k1 + 1)
            denominator = tf_val + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        terms = self._tokenize(query)
        candidate_ids: set = set()
        for term in terms:
            candidate_ids.update(self.inverted_index.get(term, []))
        if not candidate_ids:
            return []
        scores = [(i, self.score(query, i)) for i in candidate_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[i], s) for i, s in scores[:top_k] if s > 0]


# ─────────────────────────────────────────────
# HYBRID RETRIEVER (vector + keyword)
# ─────────────────────────────────────────────

class HybridRetriever:
    """
    Extension: Hybrid search combining dense (vector) + sparse (keyword) scores.
    Reciprocal Rank Fusion used to merge ranked lists without score scale issues.
    """

    def __init__(self, vector_store: VectorStore,
                 keyword_retriever: KeywordRetriever,
                 embedder: EmbeddingPipeline,
                 alpha: float = 0.6):
        """
        alpha: weight for vector score (1-alpha for keyword).
        0.6 vector + 0.4 keyword works well for policy documents.
        """
        self.vs = vector_store
        self.kr = keyword_retriever
        self.embedder = embedder
        self.alpha = alpha

    def _rrf(self, ranked_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
        """Reciprocal Rank Fusion over multiple ranked lists of chunk IDs."""
        scores: Dict[str, float] = defaultdict(float)
        for ranked in ranked_lists:
            for rank, cid in enumerate(ranked):
                scores[cid] += 1.0 / (k + rank + 1)
        return scores

    def search(self, query: str, top_k: int = 5,
               expand_query: bool = True) -> List[Tuple[Chunk, float, str]]:
        """
        Returns list of (Chunk, combined_score, method_label).
        expand_query: adds synonyms / related terms to improve recall.
        """
        search_query = self._expand_query(query) if expand_query else query

        # --- Dense retrieval ---
        q_vec = self.embedder.embed_single(search_query)
        vec_results = self.vs.search(q_vec, top_k=top_k * 2)
        vec_ranked = [c.chunk_id for c, _ in vec_results]
        vec_map = {c.chunk_id: s for c, s in vec_results}

        # --- Sparse retrieval ---
        kw_results = self.kr.search(search_query, top_k=top_k * 2)
        kw_ranked = [c.chunk_id for c, _ in kw_results]
        kw_map = {c.chunk_id: s for c, s in kw_results}

        # --- RRF fusion ---
        fused = self._rrf([vec_ranked, kw_ranked])
        all_chunks = {c.chunk_id: c for c, _ in vec_results + kw_results}

        sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)
        results = []
        for cid in sorted_ids[:top_k]:
            chunk = all_chunks[cid]
            method = "hybrid"
            results.append((chunk, fused[cid], method))

        return results

    def _expand_query(self, query: str) -> str:
        """
        Simple rule-based query expansion for Ghana budget/election domain.
        Adds related terms to improve recall on sparse queries.
        """
        expansions = {
            "gdp": "GDP gross domestic product economic growth",
            "inflation": "inflation consumer price index CPI",
            "revenue": "revenue tax collection fiscal income",
            "debt": "debt borrowing loans IMF fiscal deficit",
            "election": "election votes presidential results constituency",
            "ndc": "NDC National Democratic Congress Mahama John Dramani Mahama",
            "npp": "NPP New Patriotic Party Akufo-Addo Nana Akufo-Addo",
            "budget": "budget expenditure fiscal policy 2025",
            "education": "education schools GETFUND capitation",
            "health": "health NHIS hospitals nhif",
            "ivor greenstreet": "Ivor Kobina Greenstreet CPP",
            "akua donkor": "Akua Donkor GFP",
            "john mahama": "John Dramani Mahama NDC",
            "ashanti": "Ashanti Region",
            "ahafo": "Ahafo Region",
        }
        lower = query.lower()
        extras = []
        for key, expansion in expansions.items():
            if key in lower:
                extras.append(expansion)
        if extras:
            return query + " " + " ".join(extras)
        return query


# ─────────────────────────────────────────────
# FAILURE CASE DETECTION & FIX
# ─────────────────────────────────────────────

def detect_retrieval_failure(results: List[Tuple], query: str,
                              score_threshold: float = 0.15) -> Tuple[bool, str]:
    """
    Detects when retrieval has likely failed.
    Failure conditions:
      1. Top score below threshold (all results are poor matches)
      2. All top results are from the same page (source clustering)
      3. No results returned
    Returns (is_failure, reason).
    """
    if not results:
        return True, "NO_RESULTS: Vector store returned nothing."

    top_score = results[0][1] if len(results[0]) >= 2 else 0.0
    if top_score < score_threshold:
        return True, (f"LOW_CONFIDENCE: Top similarity score {top_score:.3f} "
                      f"is below threshold {score_threshold}. "
                      f"Query may be out-of-domain.")

    # Source clustering check
    pages = [r[0].page for r in results[:3]]
    if len(set(pages)) == 1 and pages[0] != 0:
        return True, (f"SOURCE_CLUSTER: All top-3 results from page {pages[0]}. "
                      f"Retrieval may be anchored to a single section.")

    return False, "OK"


def fallback_retrieval(query: str, vs: VectorStore,
                       embedder: EmbeddingPipeline, top_k: int = 5
                       ) -> List[Tuple[Chunk, float]]:
    """
    Fix for retrieval failure: broaden search using only nouns/keywords
    from the query and increase top_k.
    """
    # Strip stop words, keep content words
    stop = {"is","are","was","were","the","a","an","of","in","for",
            "to","and","or","what","how","why","when","who","does","did"}
    keywords = [w for w in query.lower().split() if w not in stop and len(w) > 2]
    broad_query = " ".join(keywords) if keywords else query
    logger.warning(f"Fallback retrieval with broad query: '{broad_query}'")
    q_vec = embedder.embed_single(broad_query)
    return vs.search(q_vec, top_k=top_k * 2)


# ─────────────────────────────────────────────
# INDEX BUILDER
# ─────────────────────────────────────────────

def build_index(chunks: List[Chunk], embedder: EmbeddingPipeline,
                index_path: str) -> Tuple[VectorStore, KeywordRetriever]:
    """Embed all chunks, build FAISS index + keyword index, persist."""
    logger.info(f"Building index over {len(chunks)} chunks...")
    t0 = time.time()

    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)

    vs = VectorStore(dim=embedder.dim)
    vs.add(chunks, embeddings)
    vs.save(index_path)

    kr = KeywordRetriever()
    kr.build(chunks)

    elapsed = time.time() - t0
    logger.info(f"Index built in {elapsed:.1f}s")
    return vs, kr


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_engineering import load_and_chunk

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    budget_json = os.path.join(BASE, "data", "budget_raw.json")
    csv_path = os.path.join(BASE, "data", "Ghana_Election_Result.csv")
    index_path = os.path.join(BASE, "data", "index")

    b_chunks, e_chunks = load_and_chunk(budget_json, csv_path, "semantic")
    all_chunks = b_chunks + e_chunks

    embedder = EmbeddingPipeline()
    vs, kr = build_index(all_chunks, embedder, index_path)

    retriever = HybridRetriever(vs, kr, embedder)

    print("\n=== TEST QUERY ===")
    query = "What is Ghana's GDP growth rate in 2025?"
    results = retriever.search(query, top_k=3)
    for chunk, score, method in results:
        print(f"\n[{method}] score={score:.4f} | src={chunk.source} p{chunk.page}")
        print(chunk.text[:200])

    print("\n=== FAILURE CASE TEST ===")
    bad_query = "What is the meaning of extraterrestrial life on Jupiter?"
    q_vec = embedder.embed_single(bad_query)
    bad_results = vs.search(q_vec, top_k=3)
    failed, reason = detect_retrieval_failure(
        [(c, s) for c, s in bad_results], bad_query
    )
    print(f"Failure detected: {failed} | Reason: {reason}")
