import fitz, json, re, tiktoken
from rank_bm25 import BM25Okapi
from collections import Counter

PDF_PATH = "Science_1_SSC_Testbook.pdf"
CHUNKS_FILE = "wk10_chunks.json"
ENC = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 250

# ── Manually verified chapter page ranges (1-indexed PDF pages) ──────────────
# Inspected via stage1_inspect.py:
#   Ch1 Gravitation        : pages 11-25
#   Ch2 Periodic Table     : pages 26-39
CHAPTER_RANGES = {
    1: (11, 25),   # inclusive
    2: (26, 39),
}


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def load_chapter_pages(pdf_path: str, chapter_ranges: dict) -> list[dict]:
    """Load only the pages belonging to the requested chapters."""
    doc = fitz.open(pdf_path)
    pages = []
    for ch, (start, end) in chapter_ranges.items():
        for page_num in range(start, end + 1):          # 1-indexed
            page = doc[page_num - 1]                    # 0-indexed in fitz
            text = page.get_text("text")
            pages.append({"page": page_num, "chapter": ch, "text": text})
    doc.close()
    print(f"Loaded {len(pages)} pages across chapters {list(chapter_ranges.keys())}")
    return pages


def classify_content_type(text: str) -> str:
    t = text.strip()
    if re.search(r"\bExample\s*\d+\s*:", t, re.IGNORECASE):
        return "worked_example"
    # Only match numbered questions / explicit exercise blocks
    if re.match(r"^Q\.?\s*\d+\.?", t):
        return "question_or_exercise"
    if re.match(r"^Exercise\s*\d*\b", t, re.IGNORECASE):
        return "question_or_exercise"
    if re.search(r"^\d+\.\s+.{10,}\?", t, re.MULTILINE):
        return "question_or_exercise"
    return "prose"


def extract_section(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        # Skip bare page numbers (single/double digit lines)
        if re.fullmatch(r"\d{1,3}", line):
            continue
        if 4 < len(line) < 80 and line[0].isupper():
            return line
    return "Unknown"


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split on blank lines first; if blocks are still large,
    also split on sentence-ending periods followed by newline.
    """
    blocks = re.split(r"\n{2,}", text)
    result = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # If block is already within limit, keep it
        if count_tokens(block) <= MAX_TOKENS:
            result.append(block)
        else:
            # Secondary split: period/newline boundaries
            sub_blocks = re.split(r"(?<=[.!?])\n", block)
            sub_blocks = [s.strip() for s in sub_blocks if s.strip()]
            result.extend(sub_blocks)
    return result


def chunk_pages(pages: list[dict], source: str) -> list[dict]:
    chunks = []
    ctr = 0

    for page_info in pages:
        page_num = page_info["page"]
        chapter = page_info["chapter"]
        paragraphs = split_into_paragraphs(page_info["text"])
        buffer = ""
        buffer_type = "prose"

        def emit(content: str, ctype: str):
            nonlocal ctr
            if not content.strip():
                return
            chunks.append({
                "chunk_id": f"ch{chapter}_p{page_num}_{ctr:04d}",
                "source": source,
                "section": extract_section(content),
                "content_type": ctype,
                "page": page_num,
                "chapter": chapter,
                "token_count": count_tokens(content),
                "content": content.strip(),
            })
            ctr += 1

        def flush_prose():
            nonlocal buffer, buffer_type
            emit(buffer, buffer_type)
            buffer = ""
            buffer_type = "prose"

        for para in paragraphs:
            if not para:
                continue
            p_type = classify_content_type(para)

            if p_type in ("worked_example", "question_or_exercise"):
                flush_prose()
                p_tokens = count_tokens(para)
                if p_tokens > MAX_TOKENS:
                    # hard-split by words
                    words = para.split()
                    sub = ""
                    for word in words:
                        candidate = (sub + " " + word).strip()
                        if count_tokens(candidate) > MAX_TOKENS:
                            emit(sub, p_type)
                            sub = word
                        else:
                            sub = candidate
                    if sub:
                        emit(sub, p_type)
                else:
                    emit(para, p_type)
                continue

            # prose — accumulate
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if count_tokens(candidate) > MAX_TOKENS:
                flush_prose()
                buffer = para
            else:
                buffer = candidate

        flush_prose()  # end of page

    return chunks


# ── Run Stage 1 ───────────────────────────────────────────────────────────────
pages = load_chapter_pages(PDF_PATH, CHAPTER_RANGES)
chunks = chunk_pages(pages, PDF_PATH)

print(f"Total chunks : {len(chunks)}")
dist = Counter(c["content_type"] for c in chunks)
print("Content types:", dict(dist))
toks = [c["token_count"] for c in chunks]
print(f"Tokens  min={min(toks)}  max={max(toks)}  avg={sum(toks)/len(toks):.1f}")

# Sample each type
print("\n--- Chunk samples ---")
for ctype in ["prose", "worked_example", "question_or_exercise"]:
    s = next((c for c in chunks if c["content_type"] == ctype), None)
    if s:
        print(f"\n[{ctype}] id={s['chunk_id']}  page={s['page']}  tokens={s['token_count']}")
        print(s["content"][:350])
    else:
        print(f"\n[{ctype}] NONE FOUND")

# ── BM25 on 5 test queries ────────────────────────────────────────────────────
tokenized = [c["content"].lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized)

test_queries = [
    "law of universal gravitation",
    "periodic table Mendeleev",
    "acceleration due to gravity",
    "physical and chemical change",
    "valency of elements",
]

print("\n\n--- BM25 Top-3 Results ---")
for q in test_queries:
    scores = bm25.get_scores(q.lower().split())
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    print(f"\nQ: {q}")
    for rank, idx in enumerate(top):
        snippet = chunks[idx]["content"][:140].replace("\n", " ")
        print(f"  [{rank+1}] {chunks[idx]['chunk_id']}  p{chunks[idx]['page']}  score={scores[idx]:.3f}")
        print(f"       {snippet}")

# ── Persist ───────────────────────────────────────────────────────────────────
with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(chunks)} chunks to {CHUNKS_FILE}")
