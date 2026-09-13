# FinSight AI 🏦✨
> An Intelligent, Privacy-First Generative AI Banking Assistant with BM25 + Pinecone Hybrid RAG

**FinSight AI** is a full-stack, state-of-the-art conversational AI platform tailor-made for the **Banking, Financial Services, and Insurance (BFSI)** sector. It provides an intelligent chat interface powered by a Small Language Model (SLM) running entirely locally, ensuring strict data privacy and regulatory compliance.

---

## 🎯 The Problem FinSight Solves

In the heavily regulated BFSI sector, banks cannot blindly send customer data to cloud APIs (like OpenAI) due to privacy laws (e.g., GDPR, CCPA). Furthermore, customers require **millisecond-fast, deterministic answers** to their banking queries, rather than slow, hallucinated generative text. 

**FinSight AI solves this by introducing a 4-Tier Hybrid Architecture:**
It combines the lightning speed of BM25 lexical search, the contextual intelligence of semantic Vector Search (Pinecone RAG), deterministic banking function calling, and the conversational capabilities of a local SLM (Llama-3)—all without sending sensitive prompts to third-party language models.

---

## 🏗️ System Architecture (The 4-Tier Brain)

FinSight AI intelligently routes user queries through four distinct processing layers located in `backend/main.py`:

*   **Tier 1: Speed & Compliance (Exact & Lexical Matching via BM25 / Dataset Matcher)**
    *   **How it works**: Uses **Okapi BM25** matched against a curated BFSI Q&A dataset with term-frequency saturation and length normalization.
    *   **Result**: Instant (sub-millisecond), deterministic answers to exact queries (e.g., "Lost Credit Card"). No SLM overhead needed.
*   **Tier 1.5: Smart Fuzzy Search (Cloud Vector Match)**
    *   **How it works**: If direct lexical matching fails, the query is passed to **Pinecone vector database** (`all-MiniLM-L6-v2` dense embeddings) to search for semantic similarities in the official FAQ dataset.
    *   **Threshold**: Forgiving (`0.55`).
    *   **Result**: Instant answers to vaguely worded queries (e.g., "I misplaced my plastic money card" matches "Lost Credit Card").
*   **Tier 2: Action Agents (Banking Tool Execution)**
    *   **How it works**: Deterministic function calling integrated within the backend (`banking_tools.py`).
    *   **Result**: Real-time integration with banking systems. Can immediately check user balances or calculate loan eligibility.
*   **Tier 3: Complex Policy RAG (Retrieval-Augmented Generation)**
    *   **How it works**: Uses **BM25 + Pinecone Hybrid Retrieval** to extract verified clauses from banking policy files (`data/docs/loan_policy.txt`).
    *   **Result**: Injects high-precision verified policy chunks into the SLM prompt as grounded context.
*   **Tier 4: The SLM (Local Generation)**
    *   **How it works**: A quantized **Llama-3-8B** model running completely locally via `llama-cpp-python` streams the final reasoned answer back to the frontend UI.
    *   **Fine-Tuning**: This SLM was explicitly fine-tuned via LoRA on our specialized `bfsi_dataset.json` (150+ Banking Q&A) to ensure it naturally understands banking jargon, loan processes, and policy constraints out of the box.

---

## 🔬 Benchmark: Why BM25 Replaced Standard TF-IDF

To optimize lexical chunk and Q&A retrieval, we ran empirical benchmarks comparing **TF-IDF (Cosine)** against **BM25 (Okapi)** on our 762-sample banking knowledge dataset (`bfsi_dataset.json`) and policy corpus.

### 📊 Benchmark Results (762 Banking Samples)

| Metric | TF-IDF (Cosine) | BM25 (Okapi) | Improvement (BM25) |
| :--- | :---: | :---: | :---: |
| **Top-1 Retrieval Accuracy** | 60.53% | **66.45%** | **+5.92% 🏆** |
| **Top-3 Retrieval Accuracy** | 78.29% | **83.55%** | **+5.26% 🏆** |
| **Mean Reciprocal Rank (MRR)** | 0.7216 | **0.7563** | **+0.0347 🏆** |
| **Average Query Latency** | 0.334 ms | **0.199 ms** | **~40% Faster ⚡** |

### 💡 Why BM25 Performs Better for Banking RAG:
1. **Term Frequency Saturation ($k_1 \approx 1.2–1.5$):** TF-IDF increases score linearly with repeated keywords. BM25 puts an asymptotic bound on repeated terms, preventing repetitive policy clauses from dominating concise, accurate answers.
2. **Document Length Normalization ($b = 0.75$):** Chunks in banking guidelines vary in length. BM25 balances long policy descriptions and short FAQs without bias.
3. **Sub-millisecond Speed:** Inverted index lookups in BM25 avoid dense vector/matrix cosine multiplication overhead on the CPU.

To reproduce this benchmark on your machine:
```bash
python backend/benchmark_retrievers.py
```

---

## 💻 Tech Stack & Dependencies

### Frontend (React UI)
*   **Framework**: React (Vite)
*   **Node.js**: v16.0.0 or higher
*   **Key Libraries**:
    *   `axios` (API requests)
    *   Web Speech API (Native voice-to-text integration)

### Backend (Python Server)
*   **Framework**: FastAPI
*   **Python**: v3.8.0 or higher
*   **Core Libraries**:
    *   `fastapi` & `uvicorn` (Server orchestration)
    *   `rank-bm25` (Okapi BM25 Lexical Retrieval Engine)
    *   `llama-cpp-python` (Local SLM Execution)
    *   `scikit-learn` (ML utilities & baseline benchmarks)
    *   `sentence-transformers` (Local Embedding Generation)
    *   `pinecone` (v3.1.0+ for vector database)

---

## 🚀 Getting Started

Follow these instructions to run the project locally on your machine.

### 1. Prerequisites
You must download the localized LLM model file before starting.
1. Create a directory: `backend/models/`
2. Download a GGUF format model (e.g., `Llama-3-8B-Instruct.Q4_K_M.gguf`), rename it to `llama-3-8b.Q4_K_M.gguf`, and place it in the `backend/models/` folder.
3. Obtain a **Pinecone API Key** from [pinecone.io](https://pinecone.io).

### 2. Configure Backend Secrets
Create a `.env` file inside `backend/`:
```env
PINECONE_API_KEY=your_actual_api_key_here
```

### 3. Run the Backend
Open a terminal inside the backend directory:
```bash
cd backend

# Create and activate a Virtual Environment
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI Server
python main.py
```
*(You should see "Application startup complete" and logs indicating BM25 index initialization and Pinecone connection.)*

### 4. Run the Frontend
Open a **new** terminal inside the frontend directory:
```bash
cd frontend

# Install dependencies
npm install

# Start the Vite Dev Server
npm run dev
```

### 5. Start Chatting
Open your browser and navigate to `http://localhost:5173`. You can type or use the **microphone button** to speak to FinSight AI!

---

## 🛠️ Project Structure

```text
FinSight-AI/
├── backend/
│   ├── data/                 # Data Layer
│   │   ├── bfsi_dataset.json # Curated Banking FAQs
│   │   └── docs/             # Knowledge base for RAG (e.g., loan_policy.txt)
│   ├── models/               # Local Models Directory (.gguf)
│   ├── banking_tools.py      # Tier 2 Logic (Balance Check, Loans)
│   ├── dataset_matcher.py    # Tier 1 Exact / Lexical Matcher
│   ├── rag_engine.py         # BM25 Lexical Policy Chunk Retrieval Engine
│   ├── benchmark_retrievers.py # Automated TF-IDF vs BM25 Evaluation Suite
│   ├── pinecone_engine.py    # Tier 1.5 & Tier 3 Dense Vector Architecture 
│   ├── main.py               # The Orchestrator FastAPI Server
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main React App containing UI Chat logic & Web Speech API
│   │   ├── App.css           # Blue/White Banking styling
│   └── package.json          # Node dependencies
└── README.md                 # Project Documentation & Benchmarks
```
