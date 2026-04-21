"""
PART A: DATA ENGINEERING & PREPARATION
Author: Maukewonge Yaw Nyarko-Tetteh | Index Number: 10022200107

Handles:
- Loading and cleaning budget PDF + election CSV
- Two chunking strategies with justification
- Comparative analysis support
"""

import json
import re
import csv
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str          # "budget" or "election"
    page: int = 0
    strategy: str = ""   # "fixed" or "semantic"
    metadata: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove noise while preserving meaningful content."""
    # Remove page headers / footers repeated in budget PDF
    text = re.sub(r"Resetting the Economy for the Ghana We Want\s+2025 Budget", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove page number artifacts like ". 12" at line start
    text = re.sub(r"^\s*\.\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove dotted lines (table of contents artifacts)
    text = re.sub(r"\.{5,}", "", text)
    # Strip leading/trailing whitespace per line
    lines = [l.strip() for l in text.split("\n")]
    text = "\n".join(l for l in lines if l)
    return text.strip()


def clean_csv_row(row: Dict) -> Dict:
    """Normalise a single election CSV row."""
    cleaned = {}
    for k, v in row.items():
        key = k.strip().lower().replace(" ", "_")
        cleaned[key] = str(v).strip()
    return cleaned


# ─────────────────────────────────────────────
# STRATEGY 1 — FIXED-SIZE CHUNKING
# ─────────────────────────────────────────────
# Justification:
#   chunk_size=500 tokens (~400 words) keeps each chunk small enough to
#   fit comfortably inside a retrieval context window while being large
#   enough to carry a complete idea.  overlap=100 tokens ensures sentence
#   continuity across boundaries — critical when a key figure spans two
#   chunks (e.g. "Revenue was GH¢ X billion … representing Y% of GDP").
#   Fixed chunking is fast, deterministic, and easy to reproduce.

def fixed_size_chunking(text: str, source: str, page: int,
                        chunk_size: int = 500, overlap: int = 100) -> List[Chunk]:
    """Split text into fixed-size word chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(Chunk(
            chunk_id=f"{source}_p{page}_fixed_{idx}",
            text=chunk_text,
            source=source,
            page=page,
            strategy="fixed",
            metadata={"chunk_size": chunk_size, "overlap": overlap}
        ))
        idx += 1
        start += chunk_size - overlap  # slide forward with overlap
    return chunks


# ─────────────────────────────────────────────
# STRATEGY 2 — SEMANTIC / PARAGRAPH CHUNKING
# ─────────────────────────────────────────────
# Justification:
#   Budget documents are structured with clear headings and paragraphs.
#   Splitting on double-newlines preserves semantic units (e.g. a full
#   policy paragraph or a table row block). Min-length filter of 80 chars
#   removes orphan lines (headings, page numbers).  Max-length cap of
#   1200 chars prevents oversized chunks that dilute embedding signal.
#   Semantic chunking generally outperforms fixed-size on structured
#   government documents because topics don't straddle paragraph
#   boundaries as often.

def semantic_chunking(text: str, source: str, page: int,
                      min_len: int = 80, max_len: int = 1200) -> List[Chunk]:
    """Split text by paragraphs, merge short ones, cap long ones."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    buffer = ""
    idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Merge short paragraphs into buffer
        if len(para) < min_len:
            buffer = (buffer + " " + para).strip()
            continue
        # Flush buffer before long paragraph
        if buffer:
            combined = (buffer + " " + para).strip()
            buffer = ""
            para = combined

        # Cap at max_len by splitting on sentence boundaries
        if len(para) > max_len:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            temp = ""
            for sent in sentences:
                if len(temp) + len(sent) > max_len and temp:
                    chunks.append(Chunk(
                        chunk_id=f"{source}_p{page}_sem_{idx}",
                        text=temp.strip(),
                        source=source, page=page, strategy="semantic",
                        metadata={"min_len": min_len, "max_len": max_len}
                    ))
                    idx += 1
                    temp = sent
                else:
                    temp = (temp + " " + sent).strip()
            if temp:
                chunks.append(Chunk(
                    chunk_id=f"{source}_p{page}_sem_{idx}",
                    text=temp.strip(),
                    source=source, page=page, strategy="semantic",
                    metadata={"min_len": min_len, "max_len": max_len}
                ))
                idx += 1
        else:
            chunks.append(Chunk(
                chunk_id=f"{source}_p{page}_sem_{idx}",
                text=para,
                source=source, page=page, strategy="semantic",
                metadata={"min_len": min_len, "max_len": max_len}
            ))
            idx += 1

    if buffer:
        chunks.append(Chunk(
            chunk_id=f"{source}_p{page}_sem_{idx}",
            text=buffer,
            source=source, page=page, strategy="semantic",
            metadata={"min_len": min_len, "max_len": max_len}
        ))
    return chunks


# ─────────────────────────────────────────────
# CSV → CHUNKS
# ─────────────────────────────────────────────

def csv_to_chunks(csv_path: str) -> List[Chunk]:
    """Convert election CSV rows into natural-language chunks."""
    chunks = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, raw_row in enumerate(reader):
            row = clean_csv_row(raw_row)
            # Build a human-readable sentence for each row
            text = (
                f"In the {row.get('year', 'N/A')} Ghana election, "
                f"{row.get('presidential_candidate', 'N/A')} of the "
                f"{row.get('party', 'N/A')} party received "
                f"{row.get('votes', 'N/A')} votes "
                f"({row.get('percentage', 'N/A')}%) in the "
                f"{row.get('constituency', 'N/A')} constituency, "
                f"{row.get('region', 'N/A')} Region. "
                f"Total valid votes cast: {row.get('total_valid_votes', 'N/A')}."
            )
            chunks.append(Chunk(
                chunk_id=f"election_row_{i}",
                text=text,
                source="election",
                page=0,
                strategy="csv_row",
                metadata=row
            ))
    logger.info(f"CSV → {len(chunks)} election chunks")
    return chunks


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def load_and_chunk(budget_json_path: str, csv_path: str,
                   strategy: str = "semantic") -> Tuple[List[Chunk], List[Chunk]]:
    """
    Load both data sources and return (budget_chunks, election_chunks).
    strategy: "fixed" | "semantic"
    """
    logger.info(f"Loading budget data from {budget_json_path}")
    with open(budget_json_path) as f:
        pages = json.load(f)

    budget_chunks: List[Chunk] = []
    for page_data in pages:
        page_num = page_data["page"]
        raw_text = page_data["text"]
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        if strategy == "fixed":
            chunks = fixed_size_chunking(cleaned, "budget", page_num)
        else:
            chunks = semantic_chunking(cleaned, "budget", page_num)

        budget_chunks.extend(chunks)

    logger.info(f"Budget → {len(budget_chunks)} {strategy} chunks")

    election_chunks = csv_to_chunks(csv_path)
    return budget_chunks, election_chunks


def compare_chunking_strategies(budget_json_path: str) -> Dict:
    """
    Run both strategies on same data and compare metrics.
    Returns dict for experiment logs.
    """
    with open(budget_json_path) as f:
        pages = json.load(f)

    fixed_all, sem_all = [], []
    for page_data in pages[:50]:   # sample first 50 pages
        cleaned = clean_text(page_data["text"])
        fixed_all.extend(fixed_size_chunking(cleaned, "budget", page_data["page"]))
        sem_all.extend(semantic_chunking(cleaned, "budget", page_data["page"]))

    def stats(chunks):
        lengths = [len(c.text) for c in chunks]
        return {
            "count": len(chunks),
            "avg_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "min_chars": min(lengths) if lengths else 0,
            "max_chars": max(lengths) if lengths else 0,
        }

    comparison = {
        "fixed_size (500w, 100 overlap)": stats(fixed_all),
        "semantic (paragraph-based)": stats(sem_all),
    }
    return comparison


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    budget_json = os.path.join(BASE, "data", "budget_raw.json")
    csv_path = os.path.join(BASE, "data", "Ghana_Election_Result.csv")

    print("\n=== CHUNKING COMPARISON ===")
    comp = compare_chunking_strategies(budget_json)
    for strategy, stats in comp.items():
        print(f"\n{strategy}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    print("\n=== LOADING WITH SEMANTIC STRATEGY ===")
    b_chunks, e_chunks = load_and_chunk(budget_json, csv_path, "semantic")
    print(f"Budget chunks: {len(b_chunks)}")
    print(f"Election chunks: {len(e_chunks)}")
    print(f"\nSample budget chunk:\n{b_chunks[10].text[:300]}")
