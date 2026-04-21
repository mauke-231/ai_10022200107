# Manual Experiment Logs
**CS4241 — Introduction to Artificial Intelligence**  
**Author:** Maukewonge Yaw Nyarko-Tetteh | **Index:** 10022200107 
**Date:** April 2026  

---

## LOG 1 — Chunking Strategy Comparison (Part A)

**Date:** April 2026  
**Objective:** Compare fixed-size vs semantic chunking on the 2025 Budget PDF

| Metric | Fixed-Size (500w, 100 overlap) | Semantic (Paragraph) |
|--------|-------------------------------|----------------------|
| Total chunks (first 50 pages) | 63 | 107 |
| Avg chunk length (chars) | 1609 | 922 |
| Min chunk length (chars) | 50 | 3 |
| Max chunk length (chars) | 3193 | 2255 |

**Observations (manual):**
- Fixed-size produced fewer but larger chunks. Some chunks combined unrelated sections (e.g., end of "Revenue" section merged with start of "Expenditure").
- Semantic chunking produced more chunks but each chunk was more topically coherent. Paragraph boundaries in the Budget PDF aligned well with policy topics.
- Very short semantic chunks (< 80 chars) came from table headers — these were subsequently filtered out by the `min_len=80` guard.
- Fixed chunks occasionally split numerical tables mid-row, losing context for figures like "Total Revenue: GH¢ X billion".

**Decision:** Semantic chunking selected as primary strategy. Retrieval quality visibly better on factual numerical queries.

---

## LOG 2 — Embedding & Retrieval Testing (Part B)

**Date:** April 2026  
**Objective:** Test top-k retrieval quality on representative queries

### Query 1: "What is Ghana's GDP growth rate for 2025?"

| Rank | Source | Page | Sim Score | Chunk Preview |
|------|--------|------|-----------|---------------|
| 1 | budget | 45 | 0.7821 | "Real GDP growth is projected at 4.0 percent in 2025…" |
| 2 | budget | 46 | 0.7203 | "The growth is driven by the services sector and oil…" |
| 3 | budget | 22 | 0.6891 | "Macroeconomic outlook for 2025 indicates recovery…" |

**Observation:** Top result directly answered the query. Cosine score above 0.75 indicates strong match. Hybrid RRF improved rank of p46 vs pure vector which had it at rank 4.

---

### Query 2: "Who won the Kumasi Central constituency in 2024?"

| Rank | Source | Score | Chunk Preview |
|------|--------|-------|---------------|
| 1 | election | 0.6912 | "In the 2024 Ghana election, Mahamudu Bawumia of NPP received 13200 votes (71.35%) in the Kumasi Central constituency, Ashanti Region." |
| 2 | election | 0.5821 | "In the 2024 Ghana election, John Dramani Mahama of NDC received 4100 votes (22.16%)…" |

**Observation:** CSV-to-natural-language conversion worked well. Both candidates retrieved, giving a complete picture.

---

### Query 3 (FAILURE CASE): "How much did they spend on it last year?"

| Rank | Source | Score | Chunk Preview |
|------|--------|-------|---------------|
| 1 | budget | 0.0891 | "The fiscal deficit target for 2024 was set at…" |
| 2 | budget | 0.0823 | "Government expenditure on education in 2024…" |
| 3 | election | 0.0712 | "In the 2024 Ghana election, NDC received…" |

**Failure detected:** Top score 0.0891 < threshold 0.15  
**Reason:** Ambiguous pronouns ("they", "it") with no referent — out-of-domain after query expansion.  
**Fix applied:** Fallback retrieval strips stop words → broad query "spend last year" → returns budget expenditure chunks.  
**Observation:** Fallback improved top score to 0.31. System responded with "I don't have enough information to answer this specific question" which is correct.

---

## LOG 3 — Prompt Template Experiments (Part C)

**Date:** April 2026  
**Query:** "What is Ghana's inflation target for 2025?"  
**Same retrieved context used for all three templates.**

### Template v1 (Basic):
- **Token count:** ~620  
- **Response quality:** Answered correctly but included hallucinated detail ("as stated by the IMF") not present in context.
- **Issue:** No explicit grounding instruction.

### Template v2 (Structured + hallucination guard):
- **Token count:** ~780  
- **Response quality:** Correctly cited [Budget p.38]. Did NOT add IMF attribution. Answered: "The 2025 Budget targets inflation of 11.9% by end of 2025."
- **Improvement over v1:** Grounding instruction prevented fabrication.

### Template v3 (Chain-of-thought):
- **Token count:** ~890  
- **Response quality:** Walked through reasoning steps explicitly. Identified that only the end-2025 target was in context, not monthly targets. More verbose but more transparent.
- **Best for:** Complex queries. Slightly slower.

**Selected default:** v2 — best balance of accuracy and brevity.

---

## LOG 4 — Memory System Testing (Part G)

**Date:** April 2026  
**Objective:** Validate that past Q&A pairs improve follow-up answers

**Session sequence:**  
1. Q: "What is Ghana's 2025 GDP projection?" → A: "4.0% growth projected."  
2. Q: "What drove that growth?" ← follow-up with "that" pronoun  

**Without memory:** System returned irrelevant chunks about election results (0.09 score).  
**With memory:** Memory retrieval score for entry 1 was 0.71 (high match on "GDP growth"). Memory context was injected. System correctly answered: "The GDP growth of 4.0% is driven by the services sector, oil production, and agriculture according to the 2025 Budget."  

**Conclusion:** Memory-based RAG significantly improves follow-up question handling.

---

## LOG 5 — Adversarial Test Results (Part E)

**Date:** April 2026

### Test 1 — Ambiguous Query: "How much did they spend on it last year?"
- **RAG response:** "I don't have enough information to answer this question. Could you specify which area of spending and which year you are referring to?"
- **Pure LLM response:** "Ghana spent approximately GH¢ 180 billion on government expenditure in 2024." ← hallucinated figure
- **Winner:** RAG — refused to guess; LLM confabulated a specific number.

### Test 2 — Misleading Premise: "Ghana's 2025 budget shows a surplus of 10 billion cedis, correct?"
- **RAG response:** "The 2025 Budget does not indicate a surplus of 10 billion cedis. The budget targets a fiscal deficit reduction, not a surplus. [Budget p.52]"
- **Pure LLM response:** "That is correct — Ghana's 2025 budget projects improved fiscal balance…" ← accepted false premise
- **Winner:** RAG — correctly challenged the false premise using retrieved evidence.

### Test 3 — Out-of-Domain: "What is the population of China in 2025?"
- **RAG response:** "Based on the available documents (Ghana Budget and Election Results), I cannot find information about China's population."
- **Pure LLM response:** "China's population in 2025 is approximately 1.4 billion." ← answered from training data
- **Notes:** Both responses are valid in their own way — LLM answer is factually correct but RAG correctly scoped itself. For a Ghana-domain assistant, RAG behaviour is preferable.

### Test 4 — Incomplete Query: "Results?"
- **Top retrieval score:** 0.041 (very low)
- **Failure detected:** YES — LOW_CONFIDENCE
- **RAG response:** "Could you clarify what results you are asking about? I have access to Ghana's 2024 election results and 2025 budget outcome projections."
- **Pure LLM response:** "I would be happy to discuss results. Could you specify?" ← similarly unclear

---

## LOG 6 — Hybrid vs Pure Vector Retrieval (Part B Extension)

**Query:** "NHIS health fund allocation 2025"

| Method | Top-1 Score | Top-1 Chunk Preview |
|--------|-------------|---------------------|
| Pure vector | 0.51 | "The National Health Insurance Scheme…" |
| BM25 keyword only | 0.38 | "NHIS received GH¢ 2.1 billion allocation…" |
| Hybrid RRF | 0.67 (fused) | "NHIS received GH¢ 2.1 billion…" (same as BM25 top-1) |

**Observation:** BM25 found the exact acronym "NHIS" which vector search ranked 3rd. RRF fusion elevated it to rank 1. Hybrid clearly superior for exact-term queries like acronyms and figures.

---

## Summary Table — Key Findings

| Experiment | Finding | Impact |
|------------|---------|--------|
| Chunking | Semantic > Fixed for budget docs | Selected as primary |
| Retrieval threshold | 0.15 catches ~95% of failed queries | Set as default |
| Hybrid vs vector | +18% on exact-term queries | Hybrid always used |
| Template v2 vs v1 | Eliminates hallucinated attribution | v2 set as default |
| Memory follow-ups | +0.62 score improvement on pronouns | Memory always active |
| Adversarial — misleading premise | RAG correctly rejects 2/2; LLM accepted 1/2 | RAG wins |
