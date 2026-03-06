from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import os
from banking_tools import get_balance, check_loan_eligibility

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

from fastapi.responses import StreamingResponse

# Model Configuration
MODEL_PATH = "models/llama-3-8b.Q4_K_M.gguf"
N_CTX = 2048
N_THREADS = 6
N_GPU_LAYERS = 0 # GPU disabled (CPU-only)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"WARNING: Model not found at {MODEL_PATH}. Please place the .gguf file in the models directory.")

try:
    # Initialize the model
    # verbose=False to reduce log noise
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=True
    )
except Exception as e:
    print(f"Error loading model: {e}")
    llm = None

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model not loaded")

    user_message = request.message

    # 1. Tier 1: Dataset Exact Match (Behavior Cloning)
    # Highest priority: If the question is in our dataset, return the answer directly.
    from dataset_matcher import matcher
    dataset_response = matcher.find_match(user_message)
    
    if dataset_response:
        print(f"✅ USING TIER 1: Local Exact Match (Score: High)")
        async def response_generator():
            yield dataset_response
        return StreamingResponse(response_generator(), media_type="text/plain")
    
    # 1.1 Tier 1 Fallback: Pinecone Semantic Search (Hybrid)
    # If partial match failed, try asking Pinecone for an FAQ match
    from pinecone_engine import pinecone_engine
    faq_response = pinecone_engine.query_faq(user_message, threshold=0.55)
    if faq_response:
        print(f"✅ USING TIER 1.5: Cloud Fuzzy Match (Pinecone)")
        async def response_generator():
            yield faq_response
        return StreamingResponse(response_generator(), media_type="text/plain")

    # Tier 2 & 3: Context Retrieval & SLM Generation
    
    # Check for Banking Tools (Tier 2) - keyword based for now
    tool_context = ""
    system_instruction = "You are FinSight AI, a helpful banking assistant." # Initialize system_instruction here
    
    message_lower = user_message.lower()

    if "balance" in message_lower:
        print(f"✅ USING TIER 2: Banking Tools (Get Balance)")
        balance_info = get_balance()
        tool_context = f"\nUser Account Info: The user's current account balance is {balance_info}."
        system_instruction += " The user is asking about their balance. Use the provided context to answer."
    elif "loan" in message_lower and "apply" in message_lower:
        print(f"✅ USING TIER 2: Banking Tools (Check Eligibility)")
        import re
        amount_match = re.search(r'\d+', user_message)
        amount = int(amount_match.group()) if amount_match else 10000 
        eligibility = check_loan_eligibility(amount)
        tool_context = f"\nLoan Eligibility Info: {eligibility}"
        system_instruction += " The user is inquiring about a loan. Use the eligibility result to answer."

    # Retrieve Policy Documents (Tier 3)
    # We always fetch context for difficult questions to ground the SLM
    rag_context = ""
    context = tool_context # Initialize context with any tool output
    # Heuristic: If query length > 10 chars and no specific banking tool used, check docs
    # Or strict keyword matching for "policy", "terms", "interest", "penalty", "fees"
    rag_keywords = ["policy", "terms", "interest", "rate", "penalty", "fee", "document", "eligib", "late"]
    if any(k in message_lower for k in rag_keywords) or len(user_message.split()) > 4:
        from pinecone_engine import pinecone_engine
        retrieved_docs = pinecone_engine.retrieve_context(user_message)
        if retrieved_docs:
            context += f"\n\nRelevant Bank Policy/Information (Verified):\n{retrieved_docs}"
            system_instruction += " Use the provided 'Relevant Bank Policy/Information' to answer the user's question accurately."

    # Strict Alpaca Formatting with Context
    prompt = f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{system_instruction}
User Query: {user_message}

### Input:
{context}

### Response:
"""

    def generate():
        token_generator = llm(
            prompt,
            max_tokens=512,  # Adjust as needed
            stop=["### Instruction:", "### Input:"], # Stop tokens appropriate for Alpaca
            stream=True,
            echo=False
        )
        for output in token_generator:
            yield output['choices'][0]['text']

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
