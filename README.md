# FinSight AI 🏦✨
> An Intelligent, Privacy-First Generative AI Banking Assistant

**FinSight AI** is a full-stack, state-of-the-art conversational AI platform tailor-made for the **Banking, Financial Services, and Insurance (BFSI)** sector. It provides an intelligent chat interface powered by a Small Language Model (SLM) running entirely locally, ensuring strict data privacy and regulatory compliance.

## 🎯 The Problem FinSight Solves

In the heavily regulated BFSI sector, banks cannot blindly send customer data to cloud APIs (like OpenAI) due to privacy laws (e.g., GDPR, CCPA). Furthermore, customers require **millisecond-fast, deterministic answers** to their banking queries, rather than slow, hallucinated generative text. 

**FinSight AI solves this by introducing a 4-Tier Hybrid Architecture:**
It combines the lightning speed of traditional Keyword Search (TF-IDF), the contextual intelligence of semantic Vector Search (Pinecone RAG), deterministic banking function calling, and the conversational capabilities of a local SLM (Llama-3)—all without sending sensitive prompts to third-party language models.

---

## 🏗️ System Architecture (The 4-Tier Brain)

FinSight AI intelligently routes user queries through four distinct processing layers located in `backend/main.py`:

*   **Tier 1: Speed & Compliance (Exact Keyword Match)**
    *   **How it works**: Uses `scikit-learn` TF-IDF Vectorization matched against a curated BFSI Q&A dataset.
    *   **Threshold**: Very strict (`0.65`). 
    *   **Result**: Instant (0.1s), deterministic answers to exact queries (e.g., "Lost Credit Card"). No SLM used.
*   **Tier 1.5: Smart Fuzzy Search (Cloud Vector Match)**
    *   **How it works**: If TF-IDF fails, the query is passed to **Pinecone vector database** to search for semantic similarities in the official FAQ dataset.
    *   **Threshold**: Forgiving (`0.55`).
    *   **Result**: Instant answers to vaguely worded queries (e.g., "I misplaced my plastic money card" matches "Lost Credit Card").
*   **Tier 2: Action Agents (Banking Tool Execution)**
    *   **How it works**: Deterministic function calling integrated within the backend.
    *   **Result**: Real-time integration with mock systems. Can immediately check user balances or calculate loan eligibility.
*   **Tier 3: Complex Policy RAG (Retrieval-Augmented Generation)**
    *   **How it works**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to query Pinecone for relevant documents inside `loan_policy.txt`.
    *   **Result**: Extracts the verified policy text and feeds it to the SLM as hard context.
*   **Tier 4: The SLM (Local Generation)**
    *   **How it works**: A quantized **Llama-3-8B** model running completely locally via `llama-cpp-python` streams the final reasoned answer back to the frontend UI.

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
    *   `llama-cpp-python` (Local SLM Execution)
    *   `scikit-learn` (TF-IDF keyword matching)
    *   `sentence-transformers` (Local Embedding Generation)
    *   `pinecone` (v3.1.0+ for hybrid vector DB)

---

## 🚀 Getting Started

Follow these instructions to run the project locally on your machine.

### 1. Prerequisites
You must download the localized LLM model file before starting.
1. Create a directory: `FinSight_Project/backend/models/`
2. Download a GGUF format model (e.g., Llama-3-8B-Instruct.Q4_K_M.gguf), rename it to `llama-3-8b.Q4_K_M.gguf`, and place it in the `models` folder.
3. Obtain a **Pinecone API Key** from [pinecone.io](https://pinecone.io).

### 2. Configure Backend Secrets
Create a `.env` file inside `FinSight_Project/backend/`:
```env
PINECONE_API_KEY=your_actual_api_key_here
```

### 3. Run the Backend
Open a terminal inside the backend directory:
```bash
cd FinSight_Project/backend

# Create and activate a Virtual Environment
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI Server
python main.py
```
*(You should see "Application startup complete" and logs indicating Pinecone dataset upsertion.)*

### 4. Run the Frontend
Open a **new** terminal inside the frontend directory:
```bash
cd FinSight_Project/frontend

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
FinSight_Project/
├── backend/
│   ├── data/                 # Data Layer
│   │   ├── bfsi_dataset.json # 150+ Curated Banking FAQs
│   │   └── docs/             # Knowledge base for RAG (e.g., loan_policy.txt)
│   ├── models/               # Local Models Directory (.gguf)
│   ├── banking_tools.py      # Tier 2 Logic (Balance Check, Loans)
│   ├── dataset_matcher.py    # Tier 1 Logic (TF-IDF Scikit-Learn)
│   ├── pinecone_engine.py    # Tier 1.5 & Tier 3 Architecture 
│   ├── main.py               # The Orchestrator FastAPI Server
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main React App containing UI Chat logic & Web Speech API
│   │   ├── App.css           # Blue/White Banking styling
│   └── package.json          # Node dependencies
└── README.md                 # Project Documentation
```
