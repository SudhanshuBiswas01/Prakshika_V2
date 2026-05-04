"""
Interactive Chat Assistant — StudyAssistant v2.0
- Loads the ChromaDB vectorstore and HuggingFace Embeddings
- Initializes Groq LLM and the CRAG LCEL pipeline
- Provides an interactive CLI loop for the user to ask questions
"""
import os, sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import warnings
# Suppress expected deprecation/symlink warnings for a cleaner CLI
warnings.filterwarnings("ignore")

# ── 1. Setup ──────────────────────────────────────────────────────────────────
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not found in .env")
    sys.exit(1)

# We use the Chroma DB path created by our earlier embedding stage
CHROMA_PATH = "./chroma_wk10"
COLLECTION  = "ssc_science_v2"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"

print("Initializing StudyAssistant v2.0...")
print("Loading embeddings and vector database... (this might take a few seconds)")

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_PATH, 
    embedding_function=embeddings, 
    collection_name=COLLECTION
)
# Note: If chroma_lcel doesn't exist, it means the notebook cell #3 wasn't run.
# The user can still run it, it'll just be empty until they run notebook or stage2
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print("Loading LLM and CRAG pipeline...")
llm = ChatGroq(model=GROQ_MODEL, temperature=0)

# ── 2. Prompts & CRAG Grader ──────────────────────────────────────────────────
template = """Answer only using the provided context. After every factual claim, cite the source as [Source: chunk_id]. If the answer is not present in the context, reply exactly: 'I don't have that in my study materials.' Do not infer or extrapolate beyond the context.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"[chunk_id: {d.metadata.get('chunk_id', 'Unknown')} | page: {d.metadata.get('page', 'Unknown')}]\n{d.page_content}" for d in docs)

class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a retrieved document to a user question. If the document contains factual information related to the question, grade it as 'yes'. If the document is just a textbook exercise asking the same question but providing no answer, grade it as 'no'."),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
])
structured_llm = llm.with_structured_output(Grade)
grader_chain = grader_prompt | structured_llm

def crag_retrieve_and_generate(question: str):
    docs = retriever.invoke(question)
    if not docs:
         return "I don't have that in my study materials. (No documents found)"
         
    relevant_docs = []
    for d in docs:
        grade = grader_chain.invoke({"document": d.page_content, "question": question})
        if grade.binary_score == "yes" and d.metadata.get("content_type") != "question_or_exercise":
            relevant_docs.append(d)
            
    if not relevant_docs:
        return "I don't have that in my study materials. (Filtered by CRAG Grader)"
        
    context = format_docs(relevant_docs)
    final_prompt = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(final_prompt)
    return response.content

# ── 3. Interactive Loop ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("📚 Welcome to Prakshika: SSC Science 1 Study Assistant")
print("Ask questions about Chapter 1 (Gravitation) and Chapter 2 (Periodic Classification).")
print("Type 'exit' or 'quit' to close the app.")
print("="*60 + "\n")

while True:
    try:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
            
        print("\nPrakshika is thinking...\n")
        answer = crag_retrieve_and_generate(user_input)
        print(f"Prakshika:\n{answer}\n")
        print("-" * 60)
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"\n[Error]: {e}\n")
