"""
Stage 5 — One Targeted Fix
- Worst failure: Q2 "What is the value of G?" -> falsely refused
- Diagnosis: The value of G IS in the chunks (ch1_p14, ch1_p16) but retriever
  returned question/exercise chunks instead of the definition chunk
- Classification: synonym_mismatch — "gravitational constant G" vs "value of G"
  combined with question chunks drowning out the definition chunk
- Fix: metadata filter to deprioritize question_or_exercise chunks in retrieval
- Re-run full 12-Q eval as eval_v2_scored.csv
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

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder    = SentenceTransformer(EMBED_MODEL)
chroma      = chromadb.PersistentClient(path=CHROMA_PATH)
collection  = chroma.get_or_create_collection(
    name=COLLECTION, metadata={"hnsw:space": "cosine"},
)
print(f"Collection: {collection.count()} chunks")


# ── IMPROVED retriever: fetch more, then filter out question chunks ───────────
def retrieve_v2(query, k=5):
    """
    Fix: retrieve k+5 results, then deprioritize question_or_exercise chunks.
    This ensures definition/prose chunks rise to the top instead of
    keyword-dense question chunks that match the query but lack answers.
    """
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=k + 5,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score = round(1 - distance, 4)
        ctype = results["metadatas"][0][i].get("content_type", "prose")
        # Penalize question/exercise chunks — they match keywords but lack answers
        adjusted_score = score * 0.7 if ctype == "question_or_exercise" else score
        hits.append({
            "chunk_id":       results["ids"][0][i],
            "content":        results["documents"][0][i],
            "score":          score,
            "adjusted_score": round(adjusted_score, 4),
            "metadata":       results["metadatas"][0][i],
        })
    # Re-rank by adjusted score, take top k
    hits.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return hits[:k]


def build_context(chunks):
    parts = []
    for c in chunks:
        parts.append(f"[chunk_id: {c['chunk_id']} | page: {c['metadata']['page']}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def ask_v2(question, k=5, debug=False):
    chunks = retrieve_v2(question, k=k)
    if debug:
        print(f"\nQ: {question}")
        for c in chunks:
            print(f"  {c['chunk_id']}  score={c['score']}  adj={c['adjusted_score']}  type={c['metadata']['content_type']}")
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


# ── Quick debug: test the fix on Q2 ──────────────────────────────────────────
print("\n--- Debug: Q2 with v2 retriever ---")
ask_v2("What is the value of the gravitational constant G?", debug=True)


# ── Re-run full 12-Q eval ────────────────────────────────────────────────────
eval_set = [
    {"q": "What is Newton's universal law of gravitation?",           "type": "direct"},
    {"q": "What is the value of the gravitational constant G?",       "type": "direct"},
    {"q": "State Kepler's three laws of planetary motion.",           "type": "direct"},
    {"q": "What is the difference between mass and weight?",          "type": "direct"},
    {"q": "What are the merits of Mendeleev's periodic table?",      "type": "direct"},
    {"q": "What is the modern periodic law?",                         "type": "direct"},
    {"q": "How does the gravitational pull between two objects depend on their masses and distance?", "type": "paraphrased"},
    {"q": "Why do elements in the same group of the periodic table show similar properties?",        "type": "paraphrased"},
    {"q": "If I go to the top of a mountain, will my weight change? Explain.",                       "type": "paraphrased"},
    {"q": "What is the boiling point of water at different altitudes?",              "type": "oos"},
    {"q": "What is the atomic mass of the element with atomic number 118?",         "type": "oos"},
    {"q": "Explain the process of photosynthesis in detail.",                        "type": "oos"},
]

print(f"\n\nRe-running {len(eval_set)} questions with v2 retriever...\n")

raw_rows = []
for i, item in enumerate(eval_set):
    q = item["q"]
    qtype = item["type"]
    print(f"[{i+1:2d}/{len(eval_set)}] ({qtype:12s}) {q[:70]}...")
    if i > 0:
        time.sleep(12)
    result = ask_v2(q)
    raw_rows.append({
        "id":        i + 1,
        "type":      qtype,
        "question":  q,
        "answer":    result["answer"],
        "chunk_ids": "; ".join(result["chunk_ids"]),
        "sources":   "; ".join(result["sources"]),
    })
    print(f"         -> {result['answer'][:120].replace(chr(10), ' ')}...")

# ── Score ─────────────────────────────────────────────────────────────────────
scored_rows = []
for row in raw_rows:
    ans = row["answer"].lower()
    qtype = row["type"]
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

with open("eval_v2_scored.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "type", "question", "answer_preview",
                                       "correct", "grounded", "refused_when_oos"])
    w.writeheader()
    w.writerows(scored_rows)
print(f"\nSaved eval_v2_scored.csv ({len(scored_rows)} rows)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n--- Eval v2 Summary ---")
in_scope = [r for r in scored_rows if r["type"] != "oos"]
oos      = [r for r in scored_rows if r["type"] == "oos"]
correct_y   = sum(1 for r in in_scope if r["correct"] == "Y")
grounded_y  = sum(1 for r in in_scope if r["grounded"] == "Y")
refused_y   = sum(1 for r in oos if r["refused_when_oos"] == "Y")
print(f"In-scope correct:   {correct_y}/{len(in_scope)}")
print(f"In-scope grounded:  {grounded_y}/{len(in_scope)}")
print(f"OOS refused:        {refused_y}/{len(oos)}")
print(f"\nCompare with v1: correct 7/9, grounded 8/9, OOS 3/3")
