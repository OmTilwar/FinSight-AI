import os
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
print(f"API Key: {API_KEY[:4]}...")

pc = pinecone.Pinecone(api_key=API_KEY)
index = pc.Index("lendkraft-docs")

# 1. Check Stats
print("\n--- Index Stats ---")
stats = index.describe_index_stats()
print(stats)

# 2. Check Raw Query
print("\n--- Raw Query Test 'lost card' ---")
model = SentenceTransformer('all-MiniLM-L6-v2')
vec = model.encode("lost card").tolist()

results = index.query(
    vector=vec,
    top_k=3,
    include_metadata=True
)

for match in results.matches:
    print(f"Score: {match.score:.3f}")
    print(f"Metadata: {match.metadata}")

# 3. Check Filter Query
print("\n--- Filter Query Test 'type=faq' ---")
results_filter = index.query(
    vector=vec,
    top_k=3,
    include_metadata=True,
    filter={"type": "faq"}
)
for match in results_filter.matches:
    print(f"Score: {match.score:.3f}")
    print(f"Metadata: {match.metadata}")
