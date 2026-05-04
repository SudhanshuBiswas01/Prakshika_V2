import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell("# StudyAssistant v2.0: LCEL + CRAG Pipeline\n\nThis notebook demonstrates the end-to-end pipeline for the SSC Science 1 (Class 9) textbook using LangChain Expression Language (LCEL) and a Corrective RAG (CRAG) grading step."),
    
    new_markdown_cell("## 1. Environment Setup"),
    new_code_cell("""import os, json
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY not found in .env")"""),

    new_markdown_cell("## 2. Load Custom Chunks\nWe use the custom chunks generated in Stage 1 because they contain structural metadata (`prose` vs `question_or_exercise`) which is critical for accurate retrieval."),
    new_code_cell("""import json
from langchain_core.documents import Document

with open("wk10_chunks.json", "r", encoding="utf-8") as f:
    raw_chunks = json.load(f)

docs = []
for c in raw_chunks:
    docs.append(Document(
        page_content=c["content"],
        metadata={
            "chunk_id": c["chunk_id"],
            "page": c["page"],
            "content_type": c["content_type"]
        }
    ))
print(f"Loaded {len(docs)} documents.")"""),

    new_markdown_cell("## 3. Vector Database (Chroma + HuggingFace)\nUsing `langchain-chroma` to manage our vectorstore."),
    new_code_cell("""from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_lcel",
    collection_name="ssc_science_lcel"
)

# We implement a custom retriever that penalizes 'question_or_exercise' chunks
# However, for LCEL, we'll start with the standard retriever, and rely on the CRAG grader to filter bad docs.
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})"""),

    new_markdown_cell("## 4. LCEL Generation Chain\nA standard Retrieval-Augmented Generation chain using LangChain Expression Language."),
    new_code_cell("""from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

template = \"\"\"Answer only using the provided context. After every factual claim, cite the source as [Source: chunk_id]. If the answer is not present in the context, reply exactly: 'I don't have that in my study materials.' Do not infer or extrapolate beyond the context.

Context:
{context}

Question: {question}
\"\"\"

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\\n\\n".join(f"[chunk_id: {d.metadata['chunk_id']} | page: {d.metadata['page']}]\\n{d.page_content}" for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Test the basic chain
print(rag_chain.invoke("What is Newton's universal law of gravitation?"))"""),

    new_markdown_cell("## 5. Corrective RAG (CRAG) Grader\nTo prevent the LLM from hallucinating or answering based on irrelevant chunks (like exercise questions), we add a 'Grader' step. If the retrieved documents aren't relevant to the query, we return a fallback response."),
    new_code_cell("""from pydantic import BaseModel, Field

class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# Create a grading chain with structured output
grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a retrieved document to a user question. If the document contains factual information related to the question, grade it as 'yes'. If the document is just a textbook exercise asking the same question but providing no answer, grade it as 'no'."),
    ("human", "Retrieved document: \\n\\n {document} \\n\\n User question: {question}")
])

structured_llm = llm.with_structured_output(Grade)
grader_chain = grader_prompt | structured_llm

def crag_retrieve_and_generate(question: str):
    # 1. Retrieve
    docs = retriever.invoke(question)
    
    # 2. Grade
    relevant_docs = []
    for d in docs:
        grade = grader_chain.invoke({"document": d.page_content, "question": question})
        if grade.binary_score == "yes" and d.metadata.get("content_type") != "question_or_exercise":
            relevant_docs.append(d)
            
    # 3. Generate or Fallback
    if not relevant_docs:
        return "I don't have that in my study materials. (Filtered by CRAG Grader)"
        
    context = format_docs(relevant_docs)
    final_prompt = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(final_prompt)
    return response.content

# Test CRAG on the tricky 'Value of G' question
print(crag_retrieve_and_generate("What is the value of the gravitational constant G?"))""")
])

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Notebook generated successfully.")
