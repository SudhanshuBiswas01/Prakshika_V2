# 📚 Prakshika: StudyAssistant v2.0

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![Groq](https://img.shields.io/badge/Groq-Llama%203.1-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-red)
![SentenceTransformers](https://img.shields.io/badge/Embeddings-MiniLM-yellow)

**Prakshika** is a production-grade, retrieval-augmented generation (RAG) study assistant built specifically on the Maharashtra SSC Board Science 1 (Class 10) textbook. It provides students with highly accurate, grounded answers, complete with exact source chunk and page citations.

---

## ✨ Key Features

*   **Custom Content-Aware Ingestion:** Instead of blindly chunking text by character count, the pipeline parses PDFs using `PyMuPDF` and uses RegEx to classify chunks as `prose`, `worked_example`, or `question_or_exercise`. It maintains strict token limits (250 tokens) without tearing contexts.
*   **Targeted Metadata Filtering:** Uses a custom retrieval algorithm that penalizes keyword-dense but factual-poor textbook exercise blocks, ensuring real definitions surface to the LLM.
*   **Local Embeddings:** Free, offline local embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`) mapped to a persistent `ChromaDB` vector store.
*   **Grounded Generation (Groq):** Powered by Groq's lightning-fast inference (`llama-3.1-8b-instant`), guided by a rigorous system prompt that refuses out-of-scope questions and strictly enforces citations.
*   **Interactive CLI & Jupyter Notebook:** Includes both a live terminal chat app (`chat.py`) and an advanced LCEL (LangChain Expression Language) notebook with a Corrective RAG (CRAG) Grader.

---

## 📂 Repository Structure

*   `chat.py`: **[Start Here]** The live, interactive CLI application for chatting with Prakshika.
*   `notebook.ipynb`: The end-to-end pipeline rewritten using LangChain's LCEL and CRAG methodologies.
*   `stage1_run.py` to `stage5_fix.py`: The individual debugging scripts isolating every step of the ML engineering process.
*   `project_progress.md`: A live developer journal logging every bug, pivot, and decision made throughout the build.
*   `eval_*.csv`: Output from the automated evaluation suite.
*   `fix_memo.md` & `chunking_diff.md`: Memos explaining architectural decisions.

---

## ⚙️ Step-by-Step Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SudhanshuBiswas01/Prakshika_V2.git
cd Prakshika_V2
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Ensure you have Python 3.11+ installed.
```bash
pip install -r requirements.txt
pip install langchain-chroma langchain-huggingface langgraph nbformat
```

### 4. Add the PDF Source Data
Place your textbook PDF in the root directory and name it exactly:
`Science_1_SSC_Testbook.pdf`
*(Note: Due to copyright, the PDF is excluded from this repository via `.gitignore`.)*

### 5. Setup your API Keys
Create a `.env` file in the root directory and add your free Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

---

## 🚀 How to Run

There are three ways to use Prakshika, depending on what you want to do:

### Option A: The Interactive CLI Chat (Recommended)
Want to just start asking questions? Run the chat app. It automatically handles the optimal retrieval and generation loop.
```bash
python chat.py
```

### Option B: The LCEL / CRAG Jupyter Notebook
Want to see the LangChain integration with a Corrective RAG Grader? Open the notebook.
```bash
jupyter notebook notebook.ipynb
```
*(Run the cells from top to bottom. It will initialize the database, grade the chunks, and generate the answers).*

### Option C: The Step-by-Step Engineering Scripts
Want to reconstruct the entire vector database from scratch or run the automated evaluation suite? Run the stage scripts in order:

1. **Ingest & Chunk:** Parses the PDF and creates `wk10_chunks.json`.
   ```bash
   python stage1_run.py
   ```
2. **Embed & Store:** Converts text to vectors and stores them in `./chroma_wk10`.
   ```bash
   python stage2_embed.py
   ```
3. **Prompt Testing:** Tests strict vs permissive prompts.
   ```bash
   python stage3_generate.py
   ```
4. **Evaluation Suite:** Runs a 12-question benchmark and saves to CSV.
   ```bash
   python stage4_eval.py
   ```
5. **Metadata Filter Fix:** Re-runs the eval with the `question_or_exercise` penalty filter applied.
   ```bash
   python stage5_fix.py
   ```

---

## 📊 Evaluation Results

On the 12-question benchmark set, the V2 pipeline scored:
*   **In-scope Correctness:** 8/9
*   **In-scope Grounding (Citations):** 9/9
*   **Out-of-scope Refusals:** 3/3 (Successfully refused hallucination)
