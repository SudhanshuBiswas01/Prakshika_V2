# StudyAssistant v2.0 — Project Progress Log

> This is NOT the README. This is a raw dev journal — what actually happened, what broke, what decisions were made, and why. Written step by step as the project was built.

---

## Day 1 — Project Setup & Stage 1 Start
**Date:** 2026-05-03

### What I started with
- Blank folder: `C:\Sudhanshu OG\IIT GN\SSC LLM V2\`
- One PDF already there: `Science_1_SSC_Testbook.pdf` (~8.4 MB)
- This is a Maharashtra SSC Board Science 1 (Class 9) textbook — NOT NCERT
- Scope for this build: **Chapters 1 and 2 only**

### First thing I did — scaffolding
Created the project skeleton before writing a single line of ML code:
- `requirements.txt` — pinned all versions as specified
- `.env.example` — empty keys template so the real `.env` never gets committed
- `.gitignore` — blocked `.env`, `*.pdf`, `chroma_wk10/`, and all Python cache

### Hit a wall immediately — ChromaDB won't build
Tried to install `chromadb==0.5.*` → failed.  
Tried `chromadb==0.4.24` → also failed.

**Root cause:** Both versions pull in `chroma-hnswlib`, which is a C extension that needs Microsoft Visual C++ 14.0+ to compile from source. This machine doesn't have MSVC installed.

**Fix attempted:** Install `hnswlib` pre-built binary wheel first, then install chromadb with `--no-deps` to skip the conflicting build.

> This is a common Windows-only pain point. On Linux/Mac it compiles fine. The workaround is to install the pre-compiled `hnswlib` wheel separately so chromadb doesn't try to build from source.

### Packages that installed cleanly (first pass)
- `pymupdf` ✅ (already installed, v1.27.2)
- `rank_bm25` ✅ (already installed)
- `python-dotenv` ✅
- `pandas` ✅
- `openai` ✅ (v1.68.2)
- `langchain`, `langchain-community`, `langchain-openai`, `langchain-anthropic` ✅
- `sentence-transformers` ✅
- `jupyter`, `ipykernel` ✅
- `tiktoken` ✅ (downloading)

### Still pending
- [ ] chromadb — blocked on hnswlib C++ build issue, trying workaround
- [ ] notebook.ipynb — Stage 1 code being written

---

## Next up
Once chromadb is sorted -> write Stage 1 notebook cells

---

## Stage 1 Completed — Ingestion & Chunking
**Date:** 2026-05-03

### What went wrong first
Chapter detection regex matched bare page numbers printed in the PDF footer (`1`, `2`, `3`...) as chapter headings. So chapter boundaries came out completely wrong — only 7 pages were pulled, all from Chapter 1.

**Fix:** Used `stage1_inspect.py` to manually read the first line of every page and map real chapter starts. Hardcoded the verified ranges:
- Chapter 1 (Gravitation): pages 11–25
- Chapter 2 (Periodic Classification of Elements): pages 26–39

> Lesson: SSC textbook PDFs have page footers with bare numbers that fool naive regex chapter detectors. Always inspect the raw text before writing any parser.

### Chunking issues and fix
First run: avg token count was 538 (limit is 250), max was 865. The PDF barely uses blank lines between paragraphs, so the blank-line splitter left whole pages as one block.

**Fix:** Added a secondary split on `(?<=[.!?])\n` (sentence-ending punctuation + newline) inside any block that exceeds the token limit. This brought avg down to 168 tokens.

### Classifier fix
The `question_or_exercise` regex was firing on prose sections because it matched `Exercise` anywhere and numbered lists like `1. Gravitation`. Tightened it to only match:
- Lines starting with `Q.N` pattern
- Lines starting with `Exercise` as a standalone word
- Lines with `N. <10+ chars>?` pattern (actual questions)

### Final Stage 1 numbers
- Total chunks: **120**
- Content types: prose=98, question_or_exercise=17, worked_example=5
- Token stats: min=4, max=606, avg=168
- BM25 on 5 queries: topically correct hits on both chapters
- Output: `wk10_chunks.json` saved

### Remaining edge case
Max token is 606 — a few dense table/equation blocks have no punctuation newlines so the secondary splitter can't break them. Acceptable for now; flagged in `chunking_diff.md` later.

---

## Pivot: Groq instead of OpenAI/Anthropic
No OpenAI or Anthropic keys available. Switched to:
- **Embeddings**: `sentence-transformers` locally (`all-MiniLM-L6-v2`, 384-dim) — free, no API needed
- **LLM**: Groq `llama-3.1-8b-instant` — fast, free tier
- Installed `groq` + `langchain-groq`, verified with `test_groq.py`
- Had to upgrade `protobuf` (5.29 -> 7.34) because TensorFlow gencode conflicted with sentence-transformers

---

## Stage 2 Completed — Embeddings & Vector Store
**Date:** 2026-05-03

### What happened
- Embedded all 120 chunks with `all-MiniLM-L6-v2` (local, ~2 seconds for 4 batches)
- Stored in ChromaDB `PersistentClient` at `./chroma_wk10`, collection `ssc_science_v2`
- Gate check works: `collection.count() == 0` prevents re-embedding on re-run

### 10-question retrieval eval
- **6/10 YES** — top-1 chunk contains the answer
- **1/10 PARTIAL** — chunk lists topics but doesn't explain
- **3/10 NO** — wrong chunk retrieved

### 3 misses diagnosed (see `retrieval_misses.md`)
1. "Value of G" — retrieved a question chunk about G instead of the definition chunk. Question chunks are keyword-dense but answer-poor.
2. "g vs height" — retrieved a velocity/stone chunk. The real answer is in formula + table format which embeds poorly.
3. "Merits of modern periodic table" — retrieved Mendeleev's table merits. Embedding can't distinguish "modern" vs "Mendeleev" when both say "periodic table".

---

## Next up — Stage 3: Grounded Generation (Groq + llama-3.1-8b-instant)

