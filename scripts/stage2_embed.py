"""
Stage 2 — Embeddings & Vector Store
- Embeds all chunks using sentence-transformers (local, no API cost)
- Persists to ChromaDB at ./chroma_wk10, collection "ssc_science_v2"
- Gated: skips embedding if collection already populated
- retrieve(query, k=5) returns ranked results with scores
- Runs 10-question retrieval eval, logs top-1 chunk per question
"""
import os, json
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

CHUNKS_FILE   = "data/wk10_chunks.json"
CHROMA_PATH   = "./chroma_wk10"
COLLECTION    = "ssc_science_v2"
EMBED_MODEL   = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
RETRIEVAL_LOG = "data/retrieval_log.json"

# ── Load chunks ───────────────────────────────────────────────────────────────
with open(CHUNKS_FILE, encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

# ── Embedder (local sentence-transformers) ────────────────────────────────────
print(f"Loading embedding model: {EMBED_MODEL} ...")
embedder = SentenceTransformer(EMBED_MODEL)
print("Embedding model ready.")

# ── ChromaDB persistent client ────────────────────────────────────────────────
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

# ── Gate: only embed if collection is empty ───────────────────────────────────
if collection.count() == 0:
    print(f"Collection empty — embedding {len(chunks)} chunks ...")
    texts = [c["content"] for c in chunks]
    ids   = [c["chunk_id"] for c in chunks]
    metas = [
        {
            "source":       c["source"],
            "section":      c["section"],
            "content_type": c["content_type"],
            "page":         c["page"],
            "chapter":      c["chapter"],
            "token_count":  c["token_count"],
        }
        for c in chunks
    ]

    # Embed in one batch (120 chunks is small enough)
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metas,
    )
    print(f"Embedded and stored {collection.count()} chunks in ChromaDB.")
else:
    print(f"Collection already has {collection.count()} chunks — skipping embedding.")


# ── retrieve() ────────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return top-k chunks by cosine similarity."""
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score    = round(1 - distance, 4)   # cosine similarity
        hits.append({
            "chunk_id": results["ids"][0][i],
            "content":  results["documents"][0][i],
            "score":    score,
            "metadata": results["metadatas"][0][i],
        })
    return hits


# ── 10-question retrieval eval ────────────────────────────────────────────────
eval_questions = [
    # Chapter 1 — Gravitation
    "What is Newton's universal law of gravitation?",
    "What is the value of the universal gravitational constant G?",
    "How does acceleration due to gravity change with height?",
    "What is the difference between mass and weight?",
    "Explain Kepler's laws of planetary motion",
    # Chapter 2 — Periodic Classification
    "What is Mendeleev's periodic table based on?",
    "What are the merits of the modern periodic table?",
    "How does atomic radius change across a period?",
    "What is the electronic configuration of elements in group 1?",
    "What is the law of octaves given by Newlands?",
]

print("\n\n" + "="*60)
print("Retrieval Eval — Top-1 chunk per question")
print("="*60)

log = []
for q in eval_questions:
    hits = retrieve(q, k=5)
    top  = hits[0]
    print(f"\nQ : {q}")
    print(f"#1: {top['chunk_id']}  score={top['score']}  page={top['metadata']['page']}")
    print(f"    {top['content'][:180].replace(chr(10), ' ')}")
    log.append({
        "question":     q,
        "top1_chunk_id": top["chunk_id"],
        "top1_score":   top["score"],
        "top1_page":    top["metadata"]["page"],
        "top1_snippet": top["content"][:300],
        "top5":         [{"chunk_id": h["chunk_id"], "score": h["score"]} for h in hits],
    })

with open(RETRIEVAL_LOG, "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2, ensure_ascii=False)
print(f"\nRetrieval log saved to {RETRIEVAL_LOG}")
print("\nNOTE: Manually mark YES/NO in retrieval_log.json for each top-1 result.")
