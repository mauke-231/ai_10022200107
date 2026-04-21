# ACity RAG Assistant — Technical Documentation

**Course:** CS4241 — Introduction to Artificial Intelligence  
**Author:** Maukewonge Yaw Nyarko-Tetteh | **Index:** 10022200107
**Lecturer:** Godwin N. Danso  
**Academic City University, Faculty of Computational Sciences and Informatics**  
**Date:** April 2026

---

## 1. Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot for Academic City University. The system enables users to ask questions about:
- Ghana's 2025 Budget Statement and Economic Policy (252-page PDF from MOFEP)
- Ghana Presidential Election Results (by constituency, region, and year)

All core RAG components — chunking, embedding, vector storage, retrieval, and prompt construction — are implemented **from scratch** without LangChain, LlamaIndex, or any pre-built RAG pipeline.

---

## 2. System Architecture

### 2.1 Component Overview

```
Data Sources → Cleaning → Chunking → Embedding → FAISS + BM25
                                                      ↓
User Query → Memory Check → Hybrid Retrieval → Failure Detection
                                                      ↓
                                          Context Selection → Prompt Builder
                                                                    ↓
                                                              LLM (Anthropic/OpenAI/Google)
                                                                    ↓
                                          Response + Chunk Display → Memory Store
```

### 2.2 File Structure

```
rag_project/
├── app.py                        # Streamlit UI (entry point)
├── requirements.txt              # Python dependencies
├── src/
│   ├── data_engineering.py       # Part A: Data cleaning + chunking
│   ├── retrieval.py              # Part B: Embeddings, FAISS, hybrid search
│   ├── prompt_engineering.py     # Part C: Prompt templates + context management
│   ├── pipeline.py               # Part D: Full pipeline + logging + adversarial tests
│   └── memory.py                 # Part G: Memory-based RAG (innovation)
├── data/
│   ├── budget_raw.json           # Extracted budget text (per page)
│   ├── Ghana_Election_Result.csv # Election results dataset
│   └── index/                   # FAISS index + chunk pickle (generated at runtime)
├── logs/
│   ├── pipeline_logs.jsonl       # Per-query stage logs
│   └── adversarial_tests.json    # Adversarial test results
└── docs/
    ├── architecture.html         # Interactive architecture diagram (Part F)
    ├── experiment_logs.md        # Manual experiment logs (Part D requirement)
    └── documentation.md          # This file
```

---

## 3. Part A — Data Engineering & Preparation

### 3.1 Data Sources

**Budget PDF:** 252-page government document from the Ministry of Finance (MOFEP). Extracted using `pdfplumber` (layout-aware text extraction). 251 of 252 pages yielded text.

**Election CSV:** Ghana presidential election results by constituency, region, year, candidate, party, votes, and percentage.

### 3.2 Data Cleaning

Both sources go through `clean_text()` before chunking:
- Removes repeated page headers ("Resetting the Economy for the Ghana We Want")
- Collapses 3+ consecutive newlines to 2
- Removes table-of-contents dotted lines (`......`)
- Strips orphan page number lines
- Normalises whitespace

Election CSV rows are normalised: keys lowercased, whitespace stripped, each row converted to a natural-language sentence.

### 3.3 Chunking Strategies

#### Strategy A — Fixed-Size Chunking
- **Chunk size:** 500 words, **Overlap:** 100 words
- Words split by whitespace; sliding window with `start += chunk_size - overlap`
- **Justification:** Deterministic, reproducible, works on any text. Good baseline.
- **Weakness:** Breaks mid-sentence; splits related financial figures across chunks.

#### Strategy B — Semantic (Paragraph-Based) Chunking ✓ *Selected*
- Split on double-newlines (paragraph boundaries)
- Short paragraphs (< 80 chars) buffered and merged with next
- Long paragraphs (> 1200 chars) further split on sentence boundaries (`(?<=[.!?])\s+`)
- **Justification:** Budget document has clear paragraph structure. Each paragraph is a complete policy statement. Semantic boundaries produce denser, more topically coherent chunks — verified empirically (Log 1).

#### Comparative Results (first 50 pages)
| | Fixed-Size | Semantic |
|--|--|--|
| Chunk count | 63 | 107 |
| Avg chars | 1609 | 922 |
| Impact on retrieval | Mixes topics | Topic-coherent |

---

## 4. Part B — Custom Retrieval System

### 4.1 Embedding Pipeline (`EmbeddingPipeline`)
- Model: `all-MiniLM-L6-v2` (sentence-transformers, 384-dimensional)
- Embeddings are **L2-normalised** so dot product = cosine similarity
- Batch inference with configurable batch size (default: 64)
- No LangChain wrapper — raw `SentenceTransformer.encode()` call

### 4.2 Vector Store (`VectorStore`)
- Custom wrapper around `faiss.IndexFlatIP` (exact inner product / cosine)
- Stores chunk metadata in a parallel Python list
- Serialised to disk: `faiss.write_index()` + `pickle` for chunk list
- `search()` returns `(Chunk, float)` tuples sorted by score

### 4.3 Keyword Retriever (`KeywordRetriever`)
- Pure-Python BM25 implementation (no `rank_bm25` library)
- Parameters: `k1=1.5, b=0.75` (standard BM25 defaults)
- Inverted index built at startup; `score()` computes per-document BM25
- Tokenisation: regex `\b[a-z]{2,}\b` (lowercased, removes punctuation)

### 4.4 Hybrid Retrieval (`HybridRetriever`)
- **Reciprocal Rank Fusion (RRF):** `score(doc) = Σ 1/(k + rank)` where k=60
- RRF avoids score-scale mismatch between cosine (0–1) and BM25 (unbounded)
- **Query Expansion:** Domain-specific rule table (gdp→"gross domestic product economic growth", ndc→"National Democratic Congress Mahama", etc.) appended to query before retrieval
- **Alpha:** 0.6 vector + 0.4 keyword chosen empirically (Log 6)

### 4.5 Failure Detection & Fix
- **Condition 1:** Top cosine score < 0.15 → LOW_CONFIDENCE
- **Condition 2:** All top-3 results from same page → SOURCE_CLUSTER
- **Condition 3:** Zero results returned → NO_RESULTS
- **Fix:** `fallback_retrieval()` strips stop words, broadens query, doubles top_k

---

## 5. Part C — Prompt Engineering & Generation

### 5.1 Prompt Templates

Three templates implemented, selectable in UI:

| Version | Description | Key Feature |
|---------|-------------|-------------|
| v1 | Basic context injection | Simple; no explicit grounding |
| v2 | Structured + hallucination guard | Explicit "only use context" rule + [Budget p.X] citation |
| v3 | Chain-of-thought | Step-by-step reasoning; identifies gaps |

**Hallucination control (v2):** System prompt contains: *"If the context does not contain the answer, say: 'I don't have enough information in the provided documents to answer this.'"* — preventing the LLM from falling back to training data.

### 5.2 Context Window Management
1. **Score filter:** Drop chunks below 0.10 cosine similarity
2. **Deduplication:** Remove chunks with > 80% word overlap with a higher-ranked chunk
3. **Token budget:** Greedy selection by score until 2000-token budget exhausted
4. **Format:** Numbered list with source label `[Budget p.X]` or `[Election Data]` and score

### 5.3 Prompt Experiment Results (Log 3)
- v1 → hallucinated "as stated by the IMF" (not in context)
- v2 → correctly cited only Budget page; no hallucination
- v3 → more verbose but explicitly stated what was NOT in context
- **Default:** v2

---

## 6. Part D — Full RAG Pipeline

### Pipeline Stages (with logging)

| Stage | Description | Logged |
|-------|-------------|--------|
| 1 | Memory retrieval | memory context snippet |
| 2 | Hybrid retrieval | chunk IDs, scores, source, page, text snippet |
| 3 | Failure detection | is_failure, reason |
| 4 | Context selection | selected chunk IDs, token count |
| 5 | Prompt construction | template version, token estimate, prompt preview |
| 6 | LLM generation | response, model, token usage, latency |
| 7 | Memory storage | entry ID, query, sources used |

All logs written to `logs/pipeline_logs.jsonl` as newline-delimited JSON. Each entry contains all 7 stage outputs for reproducibility.

---

## 7. Part E — Critical Evaluation & Adversarial Testing

### 7.1 Adversarial Queries

| Type | Query | RAG Outcome | LLM Outcome | Winner |
|------|-------|-------------|-------------|--------|
| Ambiguous | "How much did they spend on it last year?" | Correctly asked for clarification | Hallucinated a specific figure | RAG |
| Misleading premise | "Ghana's 2025 budget shows a surplus of 10 billion cedis, correct?" | Correctly challenged false premise | Accepted false premise | RAG |
| Out-of-domain | "What is the population of China in 2025?" | Correctly declined (out of scope) | Answered from training data | Tie (context-dependent) |
| Incomplete | "Results?" | Requested clarification; failure detected | Also unclear | Tie |

### 7.2 Key Metrics

| Metric | RAG System | Pure LLM |
|--------|------------|----------|
| Hallucination rate (4 tests) | 0/4 | 2/4 |
| False premise rejection | 1/1 | 0/1 |
| Source citation | Yes (always) | No |
| Out-of-domain handling | Scoped correctly | Uses training data |
| Response consistency (same query x3) | High (deterministic retrieval) | Medium (temperature variance) |

---

## 8. Part F — Architecture & System Design

See `docs/architecture.html` for the interactive diagram.

### Design Justification

**Why FAISS + BM25 (hybrid)?**  
Policy documents use precise terminology (acronyms, figures, programme names). Pure vector search misses exact-match signals. Hybrid RRF combines semantic understanding with keyword precision — demonstrably better on queries with acronyms like "NHIS", "ABFA", "GETFUND" (Log 6).

**Why sentence-transformers over OpenAI embeddings?**  
`all-MiniLM-L6-v2` runs fully locally — no cost per embedding, no API rate limits at index-build time. For 641 chunks (~500K characters), local embedding took ~45 seconds vs potential API throttling/cost.

**Why Streamlit over Flask/Next.js?**  
Streamlit eliminates frontend/backend separation for a Python-first RAG system. Chat UI, chunk viewer, debug panel, and API key configuration are all native Streamlit components — fewer moving parts for a 10-day exam deadline.

**Why paragraph-based chunking?**  
Ghana government budget documents follow a consistent structure: headings → numbered paragraphs → tables. Paragraph boundaries are semantically meaningful. Splitting across them (as fixed-size does) degrades retrieval because embeddings represent mixed-topic content.

---

## 9. Part G — Innovation: Memory-Based RAG

### Design

The `MemoryStore` class implements semantic memory over past Q&A exchanges:

1. **Write:** Each Q&A pair is stored as a `MemoryEntry` with the query embedded as a 384-dim vector
2. **Retrieve:** On each new query, cosine similarity is computed against all stored entry embeddings
3. **Inject:** Relevant past exchanges (score ≥ 0.5) are prepended to the prompt as "RELEVANT PAST EXCHANGES"
4. **Persist:** Memory is saved to `data/memory.json` as JSON — survives Streamlit page refreshes

### Why this is novel
Standard conversation history is injected chronologically (last N turns). Memory-based RAG injects **semantically relevant** past turns regardless of recency. This handles:
- Pronoun references ("that figure you mentioned earlier")
- Implicit topic continuation ("What about the other region?")
- Long sessions where relevant context is many turns back

### Evidence (Log 4)
Follow-up query "What drove that growth?" — without memory: 0.09 similarity (failed). With memory: 0.71 similarity, correct answer retrieved and referenced.

---

## 10. Running the Project

### Prerequisites
```bash
pip install -r requirements.txt or python -m pip install -r requirements.txt
```

### Environment Variables
```bash
export LLM_PROVIDER="anthropic"    # or "openai" or "google"
export LLM_API_KEY="your-key"
```

### Run Locally
```bash
streamlit run app.py or python -m streamlit run app.py
```

### Deployment (Streamlit Cloud)
1. Push repo to GitHub as `ai_[your_index_number]`
2. Go to share.streamlit.io → New app → connect GitHub repo → set `app.py` as entry point
3. Add `LLM_API_KEY` and `LLM_PROVIDER` as Secrets in Streamlit Cloud settings
4. Deploy → share URL with lecturer

### GitHub Setup
```bash
git init
git remote add origin https://github.com/[your-username]/ai_[your_index].git
git add .
git commit -m "CS4241 RAG System - [Your Name] [Your Index]"
git push -u origin main
# Add collaborator: GodwinDansoAcity
```

---

## 11. References

- Sbert.net — `all-MiniLM-L6-v2` model card
- Johnson, J. et al. (2019). Billion-scale similarity search with GPUs. IEEE Transactions.
- Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Cormack, G. et al. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. SIGIR.
- Ghana Ministry of Finance (2025). Budget Statement and Economic Policy for the 2025 Financial Year.
- Anthropic (2024). Claude API Documentation.
