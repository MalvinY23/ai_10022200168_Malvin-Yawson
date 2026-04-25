# AcityBot — RAG Chatbot for Academic City University

| | |
|---|---|
| **Student Name** | Malvin Yawson |
| **Index Number** | 10022200168 |
| **Course** | CS4241 - Introduction to Artificial Intelligence |
| **Institution** | Academic City University College, Ghana |
| **Year** | 2026 |
| **Repository** | `ai_10022200168` |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
set ANTHROPIC_API_KEY=sk-ant-your-key-here       # Windows
export ANTHROPIC_API_KEY=sk-ant-your-key-here    # Mac/Linux

# 3. Run the app
python -m streamlit run app.py
```

On first run, the app automatically downloads the datasets, builds embeddings, and saves the FAISS index (~3–5 minutes). Every run after that loads from cache and starts in ~5 seconds.

---

## Project Structure

```
ai_10022200168/
├── app.py                          # Streamlit UI — 5 tabs
├── requirements.txt
├── README.md
├── .gitignore
├── rag/
│   ├── __init__.py
│   ├── data_loader.py              # Part A: CSV + PDF ingestion & cleaning
│   ├── chunker.py                  # Part A: 3 chunking strategies
│   ├── embedder.py                 # Part B: MiniLM-L6-v2 embedding pipeline
│   ├── vector_store.py             # Part B: FAISS IndexFlatIP vector store
│   ├── retriever.py                # Part B: top-k, query expansion, hybrid BM25
│   ├── prompt_builder.py           # Part C: V1/V2/V3 prompt templates
│   ├── pipeline.py                 # Part D: full pipeline + Part G feedback loop
│   └── logger.py                   # Part D: structured per-stage logging
├── tests/
│   └── test_smoke.py               # 6 unit tests
└── experiment_logs/
    └── MANUAL_EXPERIMENT_LOG.md    # Hand-written experiment records
```

---

## Architecture

```
User Query
    │
    ▼
[Query Expansion]     Domain synonym table (NPP, NDC, GDP, budget, cedi …)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
[Dense Retrieval]               [BM25 Retrieval]
 FAISS IndexFlatIP               rank_bm25, exact keyword match
 cosine similarity
    │                                  │
    └──────────┬───────────────────────┘
               ▼
    [Hybrid Fusion — RRF]
     α=0.7 dense + 0.3 BM25
               │
               ▼
    [Failure Detection]
     flag if top score < 0.25
               │
               ▼
    [Feedback Adjustment]    ← Part G Innovation
     ±0.05 per chunk from 👍/👎
               │
               ▼
    [Context Window Management]
     filter score < 0.15, truncate to 6000 chars
               │
               ▼
    [Prompt Builder]
     V1 Naive / V2 Guarded / V3 Chain-of-Thought
               │
               ▼
    [Claude claude-sonnet-4-20250514 — Anthropic]
     max_tokens=1024
               │
               ▼
    [Response + Logger]
     timestamps, scores, prompts, latencies
               │
               ▼
    [Streamlit UI]
     Chat · Chunks · Logs · Adversarial · Architecture
               │
               ▼
    [Feedback Loop]
     👍/👎 → FeedbackStore → re-rank next query
```

---

## Part A: Data Engineering

### Datasets

| Source | Format | Description |
|--------|--------|-------------|
| Ghana Election Results | CSV | Constituency-level 2024 presidential results |
| 2025 Ghana Budget Statement | PDF | ~300 pages of fiscal and economic policy |

### Cleaning

**CSV:** strip whitespace, drop empty/duplicate rows, normalise column names, fill NaN values, serialise each row to natural language.

**PDF:** remove hyphenation, collapse whitespace, strip non-printable characters, skip near-empty pages.

### Chunking Strategy Comparison

| Strategy | Chunk Size | Overlap | Avg Length | Precision@5 |
|---|---|---|---|---|
| Fixed 256/50 | 256 chars | 50 | ~210 | 0.67 |
| **Fixed 512/100** ✅ | **512 chars** | **100** | **~420** | **1.00** |
| Fixed 1024/200 | 1024 chars | 200 | ~850 | 0.67 |
| Paragraph-Aware | Variable | N/A | ~400 | 0.80 |
| CSV Row ×1 ✅ | 1 row | N/A | ~120 | 1.00 |
| CSV Row ×5 | 5 rows | N/A | ~600 | 0.60 |

**Selected:** Fixed 512/100 for PDF, Row×1 for CSV — best Precision@5 in manual experiments.

**Justification for 512-char / 100-char overlap:**
The 2025 Budget PDF contains dense policy prose where a fiscal figure and its explanation typically span 2–3 sentences (~400–500 characters). A 512-char window captures this semantic unit completely. The 100-char overlap (~20%) prevents context loss at chunk boundaries — without overlap, a figure like "4.0 percent" could appear at the end of one chunk while its explanatory clause begins the next, breaking retrieval coherence.

---

## Part B: Custom Retrieval System

### Embedding Pipeline
- **Model:** `all-MiniLM-L6-v2` (22M params, 384-dim)
- **Why:** Fast on CPU (~60ms/query), strong semantic quality, fits within 256-token limit for our 512-char chunks
- **Normalisation:** L2-normalised → cosine similarity = dot product
- **Asymmetric encoding:** `"query: ..."` vs `"passage: ..."` prefixes

### Vector Store
- **Index:** `faiss.IndexFlatIP` — exact cosine search, no approximation error
- **Metadata:** parallel Python list (no SQL dependency)
- **Persistence:** `faiss.write_index` / `faiss.read_index`

### Extension: Hybrid Search (BM25 + Dense)
Reciprocal Rank Fusion merges dense and keyword ranked lists:
```
score(chunk) = 0.7 × 1/(60 + dense_rank) + 0.3 × 1/(60 + bm25_rank)
```
Fixes acronym and proper noun retrieval failures (NDC, NPP, constituency names).

### Failure Cases & Fixes

| Failure | Root Cause | Fix Applied |
|---|---|---|
| Acronym query "NDC" → wrong chunks | Embedding generalises; sparse acronyms have low density | BM25 exact match rescue |
| Short query "NPP 2024" → low score | Insufficient context for dense retrieval | Query expansion adds full party name |
| Out-of-domain query | No relevant context in corpus | Failure flag (score < 0.25) + polite decline |

---

## Part C: Prompt Engineering

### Prompt Versions

**V1 — Naive (baseline):**
Simple context injection. No hallucination guard. LLM fabricated statistics in 2/3 test cases.

**V2 — Structured + Guard (recommended):**
Explicit rules: cite sources, refuse if unsure, no fabrication. Hallucination rate: 0/3.

**V3 — Chain-of-Thought:**
Scaffolded 4-step reasoning. Best quality for complex queries. 40% slower than V2.

### Context Window Management
1. Drop chunks with score < 0.15 (noise filter)
2. Sort by score descending
3. Greedy fill to 6,000-char budget
4. Truncate last chunk rather than drop it

---

## Part D: Full Pipeline Logging

Every query logs at each stage:
- **Retrieval:** expanded query, chunk IDs, similarity scores, failure flag, latency
- **Prompt:** version used, context chars, chunks selected, full prompt preview
- **LLM:** model, response text, latency
- **Session:** written to `experiment_logs/session_*.json` + `full_history.jsonl`

---

## Part E: Adversarial Testing

| Test | Type | RAG Result | Pure LLM Result |
|---|---|---|---|
| "Who won?" | Ambiguous | Refused, asked clarification ✅ | Guessed Ghana 2024 election |
| "Budget 50B USD to education" | Misleading premise | Corrected false figure ✅ | Partially agreed |
| "2020 US election" | Out-of-domain | Scoped to Ghana corpus ✅ | Full US election summary |
| "How many votes did candidate get in region?" | Incomplete | Flagged ambiguity ✅ | Guessed likely meaning |

RAG hallucination rate: **0/4 (0%)** vs Pure LLM: **2/4 (50%)**

---

## Part G: Innovation — Feedback-Driven Retrieval

Users rate each response 👍 or 👎. Rating applies a ±0.05 score delta to retrieved chunks, stored in `experiment_logs/feedback_store.json`. Applied on every subsequent query — no model retraining required. After 3 negative signals, a noisy chunk is demoted past better alternatives.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (for LLM) | Paste in sidebar if not set as env variable |

Without an API key, the full pipeline still runs (retrieval, chunking, logging) — only the LLM response step returns a placeholder.

---

## Datasets

- **Election CSV:** https://github.com/GodwinDansoAcity/acitydataset/blob/main/Ghana_Election_Result.csv
- **Budget PDF:** https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf

Both downloaded automatically on first run and cached in `data/`.

---

*Malvin Yawson | 10022200168 | CS4241 — Academic City University College, Ghana, 2026*
