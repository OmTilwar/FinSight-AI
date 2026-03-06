from pinecone_engine import pinecone_engine
from dataset_matcher import matcher

# This query has NO common keywords with "Lost Credit Card" (which is the actual instruction)
# TF-IDF should fail. Pinecone should succeed.
query = "I misplaced my plastic money card"

print(f"Testing Query: '{query}'")

# 1. Test TF-IDF (Should Fail)
tfidf_result = matcher.find_match(query)
print(f"TF-IDF Result: {tfidf_result if tfidf_result else 'NONE (Correct)'}")

# 2. Test Pinecone (Should Succeed)
print("\nQuerying Pinecone...")
pinecone_result = pinecone_engine.query_faq(query, threshold=0.7) # Lower threshold for test script to be safe

if pinecone_result:
    print(f"✅ PINECONE MATCH FOUND!")
    print(f"Answer: {pinecone_result[:100]}...")
else:
    print("❌ PINECONE MATCH FAILED")
