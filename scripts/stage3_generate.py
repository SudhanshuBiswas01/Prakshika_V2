"""
Stage 3 — Grounded Generation
- Wires retriever to Groq llama-3.1-8b-instant at temperature=0
- Runs PERMISSIVE prompt on 3 queries (1 out-of-scope)
- Then STRICT prompt on same 3 queries
- Saves both outputs to prompt_diff.md
- Builds ask(question, debug=False) -> {answer, sources, chunk_ids}
"""
import os, json
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PATH  = "./chroma_wk10"
COLLECTION   = "ssc_science_v2"
EMBED_MODEL  = "all-MiniLM-L6-v2"
GROQ_MODEL   = "llama-3.1-8b-instant"

# ── Setup ─────────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder    = SentenceTransformer(EMBED_MODEL)
chroma      = chromadb.PersistentClient(path=CHROMA_PATH)
collection  = chroma.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},
)
print(f"ChromaDB collection: {collection.count()} chunks loaded")


# ── Retriever ─────────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = 5) -> list[dict]:
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score = round(1 - distance, 4)
        hits.append({
            "chunk_id": results["ids"][0][i],
            "content":  results["documents"][0][i],
            "score":    score,
            "metadata": results["metadatas"][0][i],
        })
    return hits


# ── Prompts ───────────────────────────────────────────────────────────────────
PERMISSIVE_SYSTEM = """You are a helpful science tutor. Use the provided context to answer the student's question. If the context doesn't fully cover the answer, you may use your general knowledge to help."""

STRICT_SYSTEM = """Answer only using the provided context. After every factual claim, cite the source as [Source: chunk_id]. If the answer is not present in the context, reply exactly: 'I don't have that in my study materials.' Do not infer or extrapolate beyond the context."""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[chunk_id: {c['chunk_id']} | page: {c['metadata']['page']}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def generate(question: str, system_prompt: str, context: str, debug: bool = False) -> str:
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"
    if debug:
        print(f"\n[DEBUG] System prompt: {system_prompt[:80]}...")
        print(f"[DEBUG] Context length: {len(context)} chars")
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


def ask(question: str, k: int = 5, debug: bool = False) -> dict:
    """Main entry point: retrieve + generate with STRICT prompt."""
    chunks = retrieve(question, k=k)
    if debug:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print(f"Top-{k} retrieved chunks:")
        for c in chunks:
            print(f"  {c['chunk_id']}  score={c['score']}  page={c['metadata']['page']}")
            print(f"    {c['content'][:120].replace(chr(10), ' ')}")
        print(f"{'='*50}")

    context = build_context(chunks)
    answer  = generate(question, STRICT_SYSTEM, context, debug=debug)
    sources = [f"page {c['metadata']['page']}" for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    return {
        "answer":    answer,
        "sources":   sources,
        "chunk_ids": chunk_ids,
    }


# ── 3 test queries (1 out-of-scope) ──────────────────────────────────────────
test_queries = [
    "What is Newton's law of gravitation and what is the formula?",
    "Explain Mendeleev's periodic table and its limitations.",
    "What is the boiling point of water at different altitudes?",   # OOS
]

print("\n\n" + "="*60)
print("PERMISSIVE PROMPT — 3 queries")
print("="*60)

permissive_outputs = []
for q in test_queries:
    chunks = retrieve(q, k=5)
    context = build_context(chunks)
    answer = generate(q, PERMISSIVE_SYSTEM, context)
    permissive_outputs.append({"question": q, "answer": answer})
    print(f"\nQ: {q}")
    print(f"A: {answer[:500]}")

print("\n\n" + "="*60)
print("STRICT PROMPT — same 3 queries")
print("="*60)

strict_outputs = []
for q in test_queries:
    chunks = retrieve(q, k=5)
    context = build_context(chunks)
    answer = generate(q, STRICT_SYSTEM, context)
    chunk_ids = [c["chunk_id"] for c in chunks]
    strict_outputs.append({"question": q, "answer": answer, "chunk_ids": chunk_ids})
    print(f"\nQ: {q}")
    print(f"A: {answer[:500]}")


# ── Save prompt_diff.md ──────────────────────────────────────────────────────
with open("docs/prompt_diff.md", "w", encoding="utf-8") as f:
    f.write("# Prompt Diff: Permissive vs Strict\n\n")
    f.write("Comparison of responses using permissive vs strict system prompts.\n")
    f.write(f"Model: `{GROQ_MODEL}` | temperature=0\n\n---\n\n")
    for i, q in enumerate(test_queries):
        oos_tag = " **(OUT OF SCOPE)**" if i == 2 else ""
        f.write(f"## Query {i+1}: {q}{oos_tag}\n\n")
        f.write(f"### Permissive\n```\n{permissive_outputs[i]['answer']}\n```\n\n")
        f.write(f"### Strict\n```\n{strict_outputs[i]['answer']}\n```\n\n")
        if i == 2:
            f.write("> **Observation**: The permissive prompt likely answers using general knowledge. The strict prompt should refuse.\n\n")
        f.write("---\n\n")

print("\nSaved prompt_diff.md")
