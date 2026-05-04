"""
Stage 4 — Evaluation
- 12-question eval set: 6 direct + 3 paraphrased + 3 out-of-scope
- Runs all through ask(), saves eval_raw.csv and eval_scored.csv
"""
import os, json, csv, time
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PATH = "./chroma_wk10"
COLLECTION  = "ssc_science_v2"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"

STRICT_SYSTEM = """Answer only using the provided context. After every factual claim, cite the source as [Source: chunk_id]. If the answer is not present in the context, reply exactly: 'I don't have that in my study materials.' Do not infer or extrapolate beyond the context."""

# ── Setup ─────────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder    = SentenceTransformer(EMBED_MODEL)
chroma      = chromadb.PersistentClient(path=CHROMA_PATH)
collection  = chroma.get_or_create_collection(
    name=COLLECTION, metadata={"hnsw:space": "cosine"},
)
print(f"Collection: {collection.count()} chunks")


def retrieve(query, k=5):
    q_emb = embedder.encode([query]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k,
                           include=["documents", "metadatas", "distances"])
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "chunk_id": res["ids"][0][i],
            "content":  res["documents"][0][i],
            "score":    round(1 - res["distances"][0][i], 4),
            "metadata": res["metadatas"][0][i],
        })
    return hits


def build_context(chunks):
    parts = []
    for c in chunks:
        parts.append(f"[chunk_id: {c['chunk_id']} | page: {c['metadata']['page']}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def ask(question, k=5, debug=False):
    chunks = retrieve(question, k=k)
    if debug:
        print(f"\nQ: {question}")
        for c in chunks:
            print(f"  {c['chunk_id']}  score={c['score']}  p{c['metadata']['page']}")
    context = build_context(chunks)
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": STRICT_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=1024,
    )
    answer = resp.choices[0].message.content.strip()
    return {
        "answer":    answer,
        "sources":   [f"page {c['metadata']['page']}" for c in chunks],
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }


# ── 12-question eval set ─────────────────────────────────────────────────────
eval_set = [
    # 6 direct
    {"q": "What is Newton's universal law of gravitation?",           "type": "direct"},
    {"q": "What is the value of the gravitational constant G?",       "type": "direct"},
    {"q": "State Kepler's three laws of planetary motion.",           "type": "direct"},
    {"q": "What is the difference between mass and weight?",          "type": "direct"},
    {"q": "What are the merits of Mendeleev's periodic table?",      "type": "direct"},
    {"q": "What is the modern periodic law?",                         "type": "direct"},
    # 3 paraphrased
    {"q": "How does the gravitational pull between two objects depend on their masses and distance?", "type": "paraphrased"},
    {"q": "Why do elements in the same group of the periodic table show similar properties?",        "type": "paraphrased"},
    {"q": "If I go to the top of a mountain, will my weight change? Explain.",                       "type": "paraphrased"},
    # 3 out-of-scope (1 plausibly answerable)
    {"q": "What is the boiling point of water at different altitudes?",              "type": "oos"},
    {"q": "What is the atomic mass of the element with atomic number 118?",         "type": "oos"},         # plausibly in periodic table but not in SSC Ch2
    {"q": "Explain the process of photosynthesis in detail.",                        "type": "oos"},
]

print(f"\nRunning {len(eval_set)} questions through ask()...\n")

raw_rows = []
for i, item in enumerate(eval_set):
    q = item["q"]
    qtype = item["type"]
    print(f"[{i+1:2d}/{len(eval_set)}] ({qtype:12s}) {q[:70]}...")
    # Rate limit: Groq free tier = 6000 TPM, wait between calls
    if i > 0:
        time.sleep(12)
    result = ask(q)
    raw_rows.append({
        "id":        i + 1,
        "type":      qtype,
        "question":  q,
        "answer":    result["answer"],
        "chunk_ids": "; ".join(result["chunk_ids"]),
        "sources":   "; ".join(result["sources"]),
    })
    print(f"         -> {result['answer'][:120].replace(chr(10), ' ')}...")

# ── Save eval_raw.csv ────────────────────────────────────────────────────────
with open("data/eval_raw.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "type", "question", "answer", "chunk_ids", "sources"])
    w.writeheader()
    w.writerows(raw_rows)
print(f"\nSaved eval_raw.csv ({len(raw_rows)} rows)")

# ── Auto-score (heuristic, then manual review) ──────────────────────────────
scored_rows = []
for row in raw_rows:
    ans = row["answer"].lower()
    qtype = row["type"]

    # Heuristic scoring — will need manual review
    refused = "i don't have that in my study materials" in ans
    has_citation = "[source:" in ans.lower()

    if qtype == "oos":
        correct = "NA"
        grounded = "NA"
        refused_oos = "Y" if refused else "N"
    else:
        correct = "Y" if (not refused and len(ans) > 50) else "N"
        grounded = "Y" if has_citation else "N"
        refused_oos = "NA"

    scored_rows.append({
        "id":               row["id"],
        "type":             row["type"],
        "question":         row["question"],
        "answer_preview":   row["answer"][:200].replace("\n", " "),
        "correct":          correct,
        "grounded":         grounded,
        "refused_when_oos": refused_oos,
    })

with open("data/eval_scored.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "type", "question", "answer_preview",
                                       "correct", "grounded", "refused_when_oos"])
    w.writeheader()
    w.writerows(scored_rows)
print(f"Saved eval_scored.csv ({len(scored_rows)} rows)")

# ── Print summary ────────────────────────────────────────────────────────────
print("\n--- Eval Summary ---")
in_scope = [r for r in scored_rows if r["type"] != "oos"]
oos      = [r for r in scored_rows if r["type"] == "oos"]
correct_y   = sum(1 for r in in_scope if r["correct"] == "Y")
grounded_y  = sum(1 for r in in_scope if r["grounded"] == "Y")
refused_y   = sum(1 for r in oos if r["refused_when_oos"] == "Y")
print(f"In-scope correct:   {correct_y}/{len(in_scope)}")
print(f"In-scope grounded:  {grounded_y}/{len(in_scope)}")
print(f"OOS refused:        {refused_y}/{len(oos)}")
