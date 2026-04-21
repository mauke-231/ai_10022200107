# ACity RAG Assistant
### CS4241 — Introduction to Artificial Intelligence | End of Semester Examination 2026

**Author:** Maukewonge Yaw Nyarko-Tetteh
**Index Number:** 10022200107
**Lecturer:** Godwin Danso  
**Academic City University — Faculty of Computational Sciences and Informatics**

---

## Project Description

A fully custom Retrieval-Augmented Generation (RAG) chatbot for Academic City University. Allows users to chat with:
- Ghana's **2025 Budget Statement and Economic Policy** (MOFEP)
- **Ghana Presidential Election Results** (by constituency, region, year)

> ⚠️ Built **without** LangChain, LlamaIndex, or any pre-built RAG pipeline. All components — chunking, embedding, vector storage, retrieval, and prompt construction — implemented manually.

## Live Demo
https://ai10022200107-mauke.streamlit.app/

## Features
- 🔍 Hybrid semantic + keyword retrieval (FAISS + BM25)
- 🧠 Memory-based RAG — remembers past exchanges semantically
- 📄 Retrieved chunk display with similarity scores
- 🛡️ Hallucination control via prompt templates
- 🔬 Pipeline debug view (all 7 stages logged)
- 🤖 Side-by-side RAG vs Pure-LLM comparison
- 🌐 Provider-agnostic LLM (Anthropic / OpenAI / Google)

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/mauke-231/ai_10022200107.git
cd ai_10022200107

# 2. Install dependencies
pip install -r requirements.txt or python -m pip install -r requirements.txt

# 3. Set your LLM API key
export LLM_PROVIDER="anthropic"   # or "openai" or "google"
export LLM_API_KEY="your-api-key"

# 4. Run the app
streamlit run app.py
```

## Project Structure
```
├── app.py                    → Streamlit UI
├── requirements.txt
├── src/
│   ├── data_engineering.py   → Part A: Cleaning + chunking
│   ├── retrieval.py          → Part B: FAISS + BM25 + hybrid
│   ├── prompt_engineering.py → Part C: Prompt templates
│   ├── pipeline.py           → Part D: Full pipeline + logging
│   └── memory.py             → Part G: Memory-based RAG
├── data/                     → Processed datasets
├── logs/                     → Pipeline + adversarial logs
└── docs/
    ├── architecture.html     → System architecture diagram
    ├── experiment_logs.md    → Manual experiment logs
    └── documentation.md      → Full technical documentation
```

## Exam Parts Coverage
| Part | Description | Marks |
|------|-------------|-------|
| A | Data Engineering & Chunking (2 strategies) | 4 |
| B | Custom FAISS + BM25 Hybrid Retrieval | 6 |
| C | 3 Prompt Templates + Hallucination Control | 4 |
| D | Full 7-Stage Pipeline with Logging | 10 |
| E | Adversarial Testing + RAG vs LLM Comparison | 6 |
| F | Architecture Diagram + Justification | 8 |
| G | Memory-Based RAG (Innovation) | 6 |
| Final | Streamlit UI + Video + Logs + Docs | 16 |

## Architecture
See `docs/architecture.html` for the full interactive diagram.

Data Flow:  
`PDF/CSV → Clean → Chunk → Embed → FAISS+BM25 → Hybrid Retrieval → Memory Check → Context Selection → Prompt → LLM → Response`
