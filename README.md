# StudyAssistant v2.0 - Prakshika

A production-grade, retrieval-augmented generation (RAG) study assistant built on the Maharashtra SSC Board Science 1 (Class 9) textbook. It provides accurate, grounded answers to students with exact page citations.

## Features
*   **Custom Content-Aware Ingestion:** Parses PDFs using `PyMuPDF` and segments text into structural chunks (`prose`, `worked_example`, `question_or_exercise`), maintaining hard token limits (250 tokens) without tearing contexts.
*   **Local Embeddings:** Free, offline local embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`) and persistent storage using `ChromaDB`.
*   **Grounded Generation (Groq):** Powered by Groq's fast inference (`llama-3.1-8b-instant`), guided by a strict system prompt that refuses out-of-scope questions and enforces citations.
*   **Targeted Retrieval Fixes:** Implements runtime metadata filtering to penalize keyword-dense but factual-poor textbook exercise blocks.
*   **LCEL & CRAG Notebook:** The pipeline is encapsulated in an end-to-end Jupyter Notebook using LangChain Expression Language (LCEL), including a Corrective RAG (CRAG) Grader step to autonomously filter retrieved documents before generation.

## Repository Structure
- `notebook.ipynb`: The end-to-end pipeline rewritten using LangChain's LCEL and CRAG methodologies.
- `stage1_run.py` to `stage5_fix.py`: The individual debugging scripts we used to isolate, build, and test each step of the RAG pipeline prior to LCEL integration.
- `project_progress.md`: A live developer journal logging every bug, pivot, and decision made throughout the build process.
- `eval_*.csv`: Output from the automated evaluation suite against the 12-question benchmark.

## Setup Instructions

1.  **Clone the repo**
    ```bash
    git clone https://github.com/SudhanshuBiswas01/Prakshika_V2.git
    cd Prakshika_V2
    ```

2.  **Environment Setup**
    Ensure you have Python 3.13+ installed.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `langchain-chroma` and `langgraph` are required to run the advanced LCEL/CRAG notebook.*

3.  **PDF Data Requirement**
    Place your textbook PDF in the root directory and name it `Science_1_SSC_Testbook.pdf`. *(Note: Due to copyright, the PDF is excluded via `.gitignore`.)*

4.  **API Keys**
    Create a `.env` file in the root directory:
    ```
    GROQ_API_KEY=gsk_your_groq_api_key_here
    ```

5.  **Run the Pipeline**
    To experience the LCEL chain with CRAG grading:
    ```bash
    jupyter notebook notebook.ipynb
    ```
    Or run the standalone evaluation scripts in sequence:
    ```bash
    python stage1_run.py
    python stage2_embed.py
    # etc...
    ```

## Evaluation Results
On the benchmark set, the V2 pipeline scored:
*   **In-scope Correctness:** 8/9
*   **In-scope Grounding (Citations):** 9/9
*   **Out-of-scope Refusal:** 3/3
