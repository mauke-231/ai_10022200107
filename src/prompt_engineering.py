"""
PART C: PROMPT ENGINEERING & GENERATION
Author: Maukewonge Yaw Nyarko-Tetteh | Index Number: 10022200107

Implements:
- Prompt templates with context injection + hallucination control
- Context window management (token-aware truncation and ranking)
- Prompt experiment framework
"""

import re
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from data_engineering import Chunk

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI assistant for Academic City University, Ghana.
You answer questions about Ghana's 2025 Budget Statement and Ghana Election Results.

RULES:
1. Answer ONLY based on the provided context. Do NOT use prior knowledge.
2. If the context does not contain the answer, say: "I don't have enough information in the provided documents to answer this."
3. Always cite which source you used: [Budget] or [Election].
4. Be concise and factual. Do not speculate.
5. For numerical data (percentages, amounts), quote them exactly as they appear in the context.
"""

# Template v1 — Basic context injection
TEMPLATE_V1 = """{system}

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {query}

ANSWER:"""

# Template v2 — Structured with explicit hallucination guard + memory
TEMPLATE_V2 = """{system}

RETRIEVED CONTEXT (ranked by relevance):
{context}

CONVERSATION HISTORY:
{history}

CURRENT QUESTION: {query}

INSTRUCTIONS:
- Base your answer strictly on the RETRIEVED CONTEXT above.
- If you cannot find the answer, respond with: "Based on the available documents, I cannot find specific information about this."
- Cite your source as [Budget p.X] or [Election] after each fact.
- If the question is a follow-up, refer to CONVERSATION HISTORY.

YOUR ANSWER:"""

# Template v3 — Chain-of-thought reasoning with hallucination control
TEMPLATE_V3 = """{system}

RETRIEVED CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

QUESTION: {query}

Think step by step:
1. What key facts in the CONTEXT are relevant to this question?
2. What is the direct answer based only on those facts?
3. Is there anything in the question that the context does NOT address?

FINAL ANSWER (based only on the context):"""


@dataclass
class PromptResult:
    template_version: str
    query: str
    context_used: str
    final_prompt: str
    token_estimate: int


# ─────────────────────────────────────────────
# CONTEXT WINDOW MANAGEMENT
# ─────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT-4 approximation)."""
    return len(text) // 4


def truncate_to_token_budget(chunks_with_scores: List[Tuple[Chunk, float]],
                              max_tokens: int = 2000) -> List[Tuple[Chunk, float]]:
    """
    Keep highest-scoring chunks that fit within token budget.
    Strategy: greedy from top score, stop when budget exceeded.
    """
    selected = []
    used = 0
    for chunk, score in sorted(chunks_with_scores, key=lambda x: x[1], reverse=True):
        chunk_tokens = estimate_tokens(chunk.text)
        if used + chunk_tokens > max_tokens:
            logger.debug(f"Token budget hit at {used}/{max_tokens}, "
                         f"dropping {chunk.chunk_id}")
            continue
        selected.append((chunk, score))
        used += chunk_tokens
    return selected


def rank_and_filter_chunks(chunks_with_scores: List[Tuple[Chunk, float]],
                            query: str,
                            min_score: float = 0.10,
                            max_tokens: int = 2000,
                            deduplicate: bool = True) -> List[Tuple[Chunk, float]]:
    """
    Full context management pipeline:
    1. Filter by minimum score
    2. Deduplicate near-identical chunks
    3. Truncate to token budget
    """
    # Step 1: Score filter
    filtered = [(c, s) for c, s in chunks_with_scores if s >= min_score]

    # Step 2: Deduplication (remove if >80% word overlap with a higher-scored chunk)
    if deduplicate:
        seen_words: List[set] = []
        deduped = []
        for chunk, score in filtered:
            words = set(chunk.text.lower().split())
            is_dup = any(
                len(words & prev) / max(len(words), 1) > 0.8
                for prev in seen_words
            )
            if not is_dup:
                deduped.append((chunk, score))
                seen_words.append(words)
        filtered = deduped

    # Step 3: Token budget
    return truncate_to_token_budget(filtered, max_tokens)


def format_context(chunks_with_scores: List[Tuple[Chunk, float]]) -> str:
    """Format selected chunks into a numbered context string."""
    parts = []
    for i, (chunk, score) in enumerate(chunks_with_scores, 1):
        source_label = (
            f"[Budget p.{chunk.page}]" if chunk.source == "budget"
            else "[Election Data]"
        )
        parts.append(
            f"[{i}] {source_label} (relevance: {score:.3f})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

class PromptBuilder:
    """
    Builds prompts for different template versions.
    Supports conversation history injection for memory-based RAG.
    """

    def __init__(self, template_version: str = "v2", max_context_tokens: int = 2000):
        self.template_version = template_version
        self.max_context_tokens = max_context_tokens
        self.templates = {
            "v1": TEMPLATE_V1,
            "v2": TEMPLATE_V2,
            "v3": TEMPLATE_V3,
        }

    def build(self, query: str,
              chunks_with_scores: List[Tuple[Chunk, float]],
              history: Optional[List[Dict]] = None) -> PromptResult:
        """
        Build the final prompt to send to the LLM.
        chunks_with_scores: output of rank_and_filter_chunks
        history: list of {"role": "user"|"assistant", "content": str}
        """
        # Chunks are already ranked/deduped by the pipeline (Stage 4).
        # Just apply the token budget here to avoid truncating the prompt.
        managed_chunks = truncate_to_token_budget(
            chunks_with_scores, self.max_context_tokens
        )
        context_str = format_context(managed_chunks)

        # Format history
        history_str = self._format_history(history)

        # Build prompt from template
        template = self.templates.get(self.template_version, TEMPLATE_V2)
        prompt = template.format(
            system=SYSTEM_PROMPT,
            context=context_str if context_str else "No relevant context retrieved.",
            query=query,
            history=history_str if history_str else "No previous conversation."
        )

        return PromptResult(
            template_version=self.template_version,
            query=query,
            context_used=context_str,
            final_prompt=prompt,
            token_estimate=estimate_tokens(prompt)
        )

    def _format_history(self, history: Optional[List[Dict]]) -> str:
        if not history:
            return ""
        lines = []
        for turn in history[-6:]:   # last 3 exchanges (6 turns)
            role = turn.get("role", "user").capitalize()
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def set_template(self, version: str):
        if version not in self.templates:
            raise ValueError(f"Unknown template version: {version}. "
                             f"Choose from {list(self.templates.keys())}")
        self.template_version = version


# ─────────────────────────────────────────────
# PROMPT EXPERIMENT RUNNER
# ─────────────────────────────────────────────

def run_prompt_experiment(query: str,
                          chunks_with_scores: List[Tuple[Chunk, float]],
                          history: Optional[List[Dict]] = None) -> Dict[str, PromptResult]:
    """
    Run all 3 templates on the same query + context.
    Returns dict of version → PromptResult for side-by-side comparison.
    """
    results = {}
    for version in ["v1", "v2", "v3"]:
        builder = PromptBuilder(template_version=version)
        result = builder.build(query, chunks_with_scores, history)
        results[version] = result
        logger.info(f"Template {version}: ~{result.token_estimate} tokens in prompt")
    return results


if __name__ == "__main__":
    # Demonstration with mock chunk
    mock_chunk = Chunk(
        chunk_id="budget_p45_sem_0",
        text=("Ghana's GDP growth rate for 2025 is projected at 4.0 percent, "
              "up from 2.9 percent in 2024. This growth is driven by oil production, "
              "agriculture, and the services sector."),
        source="budget",
        page=45,
        strategy="semantic"
    )

    query = "What is Ghana's projected GDP growth for 2025?"
    results = run_prompt_experiment(query, [(mock_chunk, 0.85)])

    for version, result in results.items():
        print(f"\n{'='*50}")
        print(f"TEMPLATE {version.upper()} | ~{result.token_estimate} tokens")
        print(f"{'='*50}")
        print(result.final_prompt[:600])
        print("...")
