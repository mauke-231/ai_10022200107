"""
PART D: FULL RAG PIPELINE WITH LOGGING
Author: Maukewonge Yaw Nyarko-Tetteh | Index Number: 10022200107

Complete pipeline:
  User Query → Memory Check → Retrieval → Context Selection
             → Prompt Construction → LLM → Response → Memory Store

Implements per-stage logging, pure LLM comparison, and adversarial testing.
"""

import os
import json
import time
import logging
import requests
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from data_engineering import Chunk
from retrieval import (HybridRetriever, VectorStore, KeywordRetriever,
                       EmbeddingPipeline, detect_retrieval_failure,
                       fallback_retrieval, build_index)
from data_engineering import load_and_chunk
from prompt_engineering import PromptBuilder, rank_and_filter_chunks
from memory import MemoryStore

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)

# ─────────────────────────────────────────────
# LLM CLIENT (provider-agnostic)
# ─────────────────────────────────────────────

class LLMClient:
    """
    Generic LLM client supporting multiple providers.
    Reads API key and provider from environment variables.
    Supported: anthropic, openai, google, groq
    """

    def __init__(self, provider: str = None, api_key: str = None,
                 model: str = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or self._default_model()

    def _default_model(self) -> str:
        defaults = {
            "anthropic": "claude-3-haiku-20240307",
            "openai":    "gpt-4o-mini",
            "google":    "gemini-2.0-flash",
            "groq":      "llama-3.1-8b-instant",
        }
        return defaults.get(self.provider, "claude-3-haiku-20240307")

    def complete(self, prompt: str, system: str = "",
                 max_tokens: int = 800) -> Tuple[str, Dict]:
        """
        Send prompt to LLM. Returns (response_text, metadata).
        """
        if not self.api_key:
            return ("[No API key set. Set LLM_API_KEY in your environment "
                    "to get live responses.]", {})

        if self.provider == "anthropic":
            return self._call_anthropic(prompt, system, max_tokens)
        elif self.provider == "openai":
            return self._call_openai(prompt, system, max_tokens)
        elif self.provider == "google":
            return self._call_google(prompt, max_tokens)
        elif self.provider == "groq":
            return self._call_groq(prompt, system, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_anthropic(self, prompt: str, system: str,
                         max_tokens: int) -> Tuple[str, Dict]:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        meta = {"model": data["model"], "usage": data.get("usage", {})}
        return text, meta

    def _call_openai(self, prompt: str, system: str,
                      max_tokens: int) -> Tuple[str, Dict]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens}
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        meta = {"model": data["model"], "usage": data.get("usage", {})}
        return text, meta

    def _call_groq(self, prompt: str, system: str,
                    max_tokens: int) -> Tuple[str, Dict]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if system:
            prompt = system + "\n\n" + prompt
        messages = [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens}
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        meta = {"model": data["model"], "usage": data.get("usage", {})}
        return text, meta

    def _call_google(self, prompt: str, max_tokens: int) -> Tuple[str, Dict]:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.api_key}")
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens}
        }
        resp = requests.post(url, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text, {"model": self.model}


# ─────────────────────────────────────────────
# PIPELINE STAGE LOG
# ─────────────────────────────────────────────

@dataclass
class PipelineLog:
    """Captures all intermediate data for one RAG query."""
    query: str
    expanded_query: str = ""
    memory_context: str = ""
    retrieved_chunks: List[Dict] = field(default_factory=list)
    failure_detected: bool = False
    failure_reason: str = ""
    selected_chunks: List[Dict] = field(default_factory=list)
    final_prompt: str = ""
    prompt_tokens: int = 0
    llm_response: str = ""
    llm_metadata: Dict = field(default_factory=dict)
    latency_ms: float = 0.0
    template_version: str = "v2"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "expanded_query": self.expanded_query,
            "memory_context_snippet": self.memory_context[:200],
            "retrieved_chunks": self.retrieved_chunks,
            "failure_detected": self.failure_detected,
            "failure_reason": self.failure_reason,
            "selected_chunks": self.selected_chunks,
            "prompt_tokens": self.prompt_tokens,
            "llm_response_snippet": self.llm_response[:300],
            "latency_ms": round(self.latency_ms, 1),
            "template_version": self.template_version,
        }


# ─────────────────────────────────────────────
# MAIN RAG PIPELINE
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline.
    User Query → Memory Check → Hybrid Retrieval → Context Selection
              → Prompt Construction → LLM → Response → Memory Store
    """

    def __init__(self,
                 retriever: HybridRetriever,
                 embedder: EmbeddingPipeline,
                 llm_client: LLMClient,
                 memory_store: MemoryStore,
                 prompt_builder: PromptBuilder,
                 log_dir: str = "logs"):
        self.retriever = retriever
        self.embedder = embedder
        self.llm = llm_client
        self.memory = memory_store
        self.prompt_builder = prompt_builder
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._log_buffer: List[Dict] = []

    def query(self, user_query: str,
              top_k: int = 5,
              template_version: str = "v2",
              pure_llm_mode: bool = False) -> Tuple[str, PipelineLog]:
        """
        Run the full RAG pipeline.
        pure_llm_mode: bypass retrieval (for RAG vs LLM comparison in Part E).
        """
        t_start = time.time()
        log = PipelineLog(query=user_query, template_version=template_version)
        self.prompt_builder.set_template(template_version)

        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE START | query='{user_query}'")

        # ── STAGE 1: Memory retrieval ─────────────────────────────────
        logger.info("[STAGE 1] Retrieving relevant memory...")
        memory_ctx = self.memory.get_memory_context(user_query, self.embedder)
        log.memory_context = memory_ctx
        history = self.memory.get_recent_history()
        if memory_ctx:
            logger.info(f"  Memory context: {memory_ctx[:100]}...")
        else:
            logger.info("  No relevant memory found.")

        if pure_llm_mode:
            # ── PURE LLM MODE (no retrieval) ─────────────────────────
            logger.info("[STAGE 2-3] SKIPPED (pure LLM mode)")
            log.failure_reason = "pure_llm_mode"
            prompt = (f"{user_query}\n\n"
                      f"(Answer from your general knowledge about Ghana.)")
            try:
                response, meta = self.llm.complete(prompt, max_tokens=600)
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 429:
                    response = ("You've reached the rate limit for the AI service. Please wait a few minutes before asking more questions.")
                    meta = {"error": "rate_limit"}
                else:
                    response = ("An error occurred while generating the response. Please try again later.")
                    meta = {"error": str(e)}
                log.llm_response = response
                log.llm_metadata = meta
                log.latency_ms = (time.time() - t_start) * 1000
                logger.warning(f"  LLM call failed: {e}")
                self._log_to_file(log)
                return response, log
            except Exception as e:
                response = ("An unexpected error occurred. Please try again later.")
                meta = {"error": str(e)}
                log.llm_response = response
                log.llm_metadata = meta
                log.latency_ms = (time.time() - t_start) * 1000
                logger.error(f"  Unexpected error: {e}")
                self._log_to_file(log)
                return response, log
            log.llm_response = response
            log.llm_metadata = meta
            log.latency_ms = (time.time() - t_start) * 1000
            self._log_to_file(log)
            return response, log

        # ── STAGE 2: Hybrid retrieval ─────────────────────────────────
        logger.info(f"[STAGE 2] Hybrid retrieval | top_k={top_k}")
        results = self.retriever.search(user_query, top_k=top_k,
                                         expand_query=True)
        log.expanded_query = self.retriever._expand_query(user_query)  # type: ignore
        log.retrieved_chunks = [
            {"chunk_id": c.chunk_id, "source": c.source,
             "page": c.page, "score": round(s, 4),
             "text_snippet": c.text[:150]}
            for c, s, _ in results
        ]
        logger.info(f"  Retrieved {len(results)} chunks")
        for c, s, m in results[:3]:
            logger.info(f"    [{m}] score={s:.4f} | {c.source} p{c.page} | "
                        f"{c.text[:80]}...")

        # ── STAGE 3: Failure detection ────────────────────────────────
        logger.info("[STAGE 3] Checking retrieval quality...")
        raw_pairs = [(c, s) for c, s, _ in results]
        failed, reason = detect_retrieval_failure(raw_pairs, user_query)
        log.failure_detected = failed
        log.failure_reason = reason

        if failed:
            logger.warning(f"  RETRIEVAL FAILURE: {reason}")
            logger.info("  Applying fallback retrieval...")
            fallback = fallback_retrieval(user_query, self.retriever.vs,
                                          self.embedder, top_k=top_k)
            raw_pairs = fallback
        else:
            logger.info(f"  Retrieval OK | top score={raw_pairs[0][1]:.4f}")

        # ── STAGE 4: Context selection ────────────────────────────────
        logger.info("[STAGE 4] Context selection and token management...")
        selected = rank_and_filter_chunks(raw_pairs, user_query)
        log.selected_chunks = [
            {"chunk_id": c.chunk_id, "source": c.source,
             "score": round(s, 4), "text": c.text[:200]}
            for c, s in selected
        ]
        logger.info(f"  Selected {len(selected)}/{len(raw_pairs)} chunks "
                    f"after dedup + token budget")

        # ── STAGE 5: Prompt construction ──────────────────────────────
        logger.info(f"[STAGE 5] Building prompt (template={template_version})...")
        prompt_result = self.prompt_builder.build(
            user_query, selected, history
        )
        # Inject memory context into prompt if available
        final_prompt = prompt_result.final_prompt
        if memory_ctx:
            final_prompt = memory_ctx + "\n\n" + final_prompt

        log.final_prompt = final_prompt
        log.prompt_tokens = prompt_result.token_estimate
        logger.info(f"  Prompt built | ~{prompt_result.token_estimate} tokens")
        logger.info(f"  [FINAL PROMPT PREVIEW]\n{final_prompt[:400]}...")

        # ── STAGE 6: LLM generation ───────────────────────────────────
        logger.info("[STAGE 6] Calling LLM...")
        try:
            response, meta = self.llm.complete(final_prompt, max_tokens=800)
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 'unknown'
            if e.response and e.response.status_code == 429:
                response = ("You've reached the rate limit for the AI service. Please wait a few minutes before asking more questions.")
                meta = {"error": "rate_limit", "status_code": status_code}
            else:
                response = ("An error occurred while generating the response. Please try again later.")
                meta = {"error": str(e), "status_code": status_code}
            log.llm_response = response
            log.llm_metadata = meta
            log.latency_ms = (time.time() - t_start) * 1000
            logger.warning(f"  LLM call failed: {e} (status: {status_code})")
            # Skip memory storage on error
            self._log_to_file(log)
            return response, log
        except Exception as e:
            response = ("An unexpected error occurred. Please try again later.")
            meta = {"error": str(e)}
            log.llm_response = response
            log.llm_metadata = meta
            log.latency_ms = (time.time() - t_start) * 1000
            logger.error(f"  Unexpected error: {e}")
            # Skip memory storage on error
            self._log_to_file(log)
            return response, log

        log.llm_response = response
        log.llm_metadata = meta
        log.latency_ms = (time.time() - t_start) * 1000
        logger.info(f"  LLM response received | latency={log.latency_ms:.0f}ms")
        logger.info(f"  Response preview: {response[:150]}...")

        # ── STAGE 7: Memory storage ───────────────────────────────────
        logger.info("[STAGE 7] Storing to memory...")
        used_sources = [c.chunk_id for c, _ in selected]
        self.memory.add(user_query, response, used_sources, self.embedder)

        # ── Persist log ───────────────────────────────────────────────
        self._log_to_file(log)
        logger.info(f"PIPELINE COMPLETE | latency={log.latency_ms:.0f}ms")

        return response, log

    def _log_to_file(self, log: PipelineLog):
        """Append pipeline log to JSONL file."""
        log_path = os.path.join(self.log_dir, "pipeline_logs.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(log.to_dict()) + "\n")

    # ── Part E: Adversarial Testing ──────────────────────────────────

    def adversarial_test(self) -> List[Dict]:
        """
        Run predefined adversarial queries and collect results.
        Returns list of test result dicts.
        """
        adversarial_queries = [
            {
                "type": "ambiguous",
                "query": "How much did they spend on it last year?",
                "description": "No subject or referent — tests disambiguation"
            },
            {
                "type": "misleading",
                "query": "Ghana's 2025 budget shows a surplus of 10 billion cedis, correct?",
                "description": "Contains a false premise — tests hallucination resistance"
            },
            {
                "type": "out_of_domain",
                "query": "What is the population of China in 2025?",
                "description": "Completely out of domain — tests graceful degradation"
            },
            {
                "type": "incomplete",
                "query": "Results?",
                "description": "Extremely vague single-word query"
            },
        ]

        test_results = []
        for test in adversarial_queries:
            logger.info(f"\nADVERSARIAL TEST: {test['type']} | '{test['query']}'")
            # RAG response
            rag_response, log = self.query(test["query"], template_version="v2")
            # Pure LLM response for comparison
            llm_response, _ = self.query(test["query"], pure_llm_mode=True)

            test_results.append({
                "type": test["type"],
                "query": test["query"],
                "description": test["description"],
                "rag_response": rag_response[:400],
                "llm_response": llm_response[:400],
                "failure_detected": log.failure_detected,
                "failure_reason": log.failure_reason,
                "retrieved_count": len(log.retrieved_chunks),
                "top_similarity": (log.retrieved_chunks[0]["score"]
                                   if log.retrieved_chunks else 0),
            })

        # Save adversarial results
        out_path = os.path.join(self.log_dir, "adversarial_tests.json")
        with open(out_path, "w") as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Adversarial test results saved to {out_path}")
        return test_results


# ─────────────────────────────────────────────
# PIPELINE FACTORY
# ─────────────────────────────────────────────

def build_pipeline(base_dir: str,
                   api_key: str = "",
                   provider: str = "anthropic") -> RAGPipeline:
    """
    Build and return a fully initialised RAGPipeline.
    Loads or builds index as needed.
    """
    budget_json = os.path.join(base_dir, "data", "budget_raw.json")
    csv_path    = os.path.join(base_dir, "data", "Ghana_Election_Result.csv")
    index_path  = os.path.join(base_dir, "data", "index")
    memory_path = os.path.join(base_dir, "data", "memory.json")
    log_dir     = os.path.join(base_dir, "logs")

    embedder = EmbeddingPipeline()

    # Load or build index
    vs = VectorStore(dim=embedder.dim)
    if os.path.exists(os.path.join(index_path, "faiss.index")):
        logger.info("Loading existing index...")
        vs.load(index_path)
        from data_engineering import csv_to_chunks
        all_chunks = vs.chunks
        kr = KeywordRetriever()
        kr.build(all_chunks)
    else:
        logger.info("Building index from scratch...")
        b_chunks, e_chunks = load_and_chunk(budget_json, csv_path, "semantic")
        all_chunks = b_chunks + e_chunks
        vs, kr = build_index(all_chunks, embedder, index_path)

    retriever = HybridRetriever(vs, kr, embedder)
    llm_client = LLMClient(provider=provider, api_key=api_key)
    memory_store = MemoryStore(memory_path)
    prompt_builder = PromptBuilder(template_version="v2")

    return RAGPipeline(
        retriever=retriever,
        embedder=embedder,
        llm_client=llm_client,
        memory_store=memory_store,
        prompt_builder=prompt_builder,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline = build_pipeline(BASE)

    # Test query
    response, log = pipeline.query(
        "What is Ghana's inflation target for 2025?",
        template_version="v2"
    )
    print("\n=== RESPONSE ===")
    print(response)
    print("\n=== RETRIEVED CHUNKS ===")
    for chunk in log.retrieved_chunks[:3]:
        print(f"  [{chunk['score']}] {chunk['source']} p{chunk['page']}: "
              f"{chunk['text_snippet']}")
