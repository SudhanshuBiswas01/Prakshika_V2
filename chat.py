"""
Interactive Chat Assistant — StudyAssistant v2.0
- Uses the Stage 5 optimized retriever (ChromaDB client)
- Safely penalizes question chunks so actual facts bubble up
- Provides an interactive CLI loop for the user to ask questions
"""
import os, sys
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# Suppress warnings for a clean CLI
import warnings
warnings.filterwarnings("ignore")

# ── 1. Setup ──────────────────────────────────────────────────────────────────
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not found in .env")
    sys.exit(1)

CHROMA_PATH = "./chroma_wk10"
COLLECTION  = "ssc_science_v2"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"

print("Initializing StudyAssistant v2.0...")
print("Loading embeddings and vector database... (this might take a few seconds)")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder    = SentenceTransformer(EMBED_MODEL)
chroma      = chromadb.PersistentClient(path=CHROMA_PATH)
collection  = chroma.get_or_create_collection(
    name=COLLECTION, metadata={"hnsw:space": "cosine"}
)

STRICT_SYSTEM = """You are a precise study assistant. You must answer the user's question using ONLY the provided context.
If the context does not contain the answer to the user's question, you MUST reply exactly: 'I don't have that in my study materials.'
Do not just summarize the context if it doesn't answer the question.
After every factual claim, cite the source as [Source: chunk_id]. Do not infer or extrapolate beyond the context."""

# ── 2. The Optimized Retriever from Stage 5 ───────────────────────────────────
def retrieve_v2(query, k=5):
    """Retrieves k+5 chunks and deprioritizes question_or_exercise chunks."""
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
        
        # Penalize question/exercise chunks so real answers rise to the top
        adjusted_score = score * 0.7 if ctype == "question_or_exercise" else score
        hits.append({
            "chunk_id":       results["ids"][0][i],
            "content":        results["documents"][0][i],
            "adjusted_score": round(adjusted_score, 4),
            "metadata":       results["metadatas"][0][i],
        })
        
    hits.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return hits[:k]

def build_context(chunks):
    parts = []
    for c in chunks:
        parts.append(f"[chunk_id: {c['chunk_id']} | page: {c['metadata']['page']}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)

def ask(question: str):
    chunks = retrieve_v2(question, k=5)
    
    print("\n[Retrieved Context]:")
    for c in chunks:
        ctype = c['metadata'].get('content_type', 'prose')
        print(f" - {c['chunk_id']} (Page {c['metadata']['page']}) [{ctype}]")
    print("-" * 40 + "\n")
    
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
    return resp.choices[0].message.content.strip()

# ── 3. Interactive Loop ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("Welcome to Prakshika: SSC Science 1 Study Assistant")
print("Ask questions about Chapter 1 (Gravitation) and Chapter 2 (Periodic Classification).")
print("Type 'exit' or 'quit' to close the app.")
print("="*60 + "\n")

while True:
    try:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
            
        print("\nPrakshika is thinking...")
        answer = ask(user_input)
        print(f"Prakshika:\n{answer}\n")
        print("=" * 60 + "\n")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"\n[Error]: {e}\n")
