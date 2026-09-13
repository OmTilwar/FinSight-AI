import os
import re
from rank_bm25 import BM25Okapi

# Path to documents
DOCS_PATH = os.path.join(os.path.dirname(__file__), "data", "docs")

def tokenize(text: str):
    """Clean and tokenize text into lowercase word tokens."""
    return re.findall(r'\b\w+\b', text.lower())

class RAGEngine:
    def __init__(self):
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.load_documents()

    def load_documents(self):
        """
        Loads all .txt files from the docs directory, splits into paragraphs/sections,
        and initializes BM25Okapi index.
        """
        if not os.path.exists(DOCS_PATH):
            print(f"Docs directory not found at {DOCS_PATH}")
            return

        chunks = []
        for filename in os.listdir(DOCS_PATH):
            if filename.endswith(".txt"):
                filepath = os.path.join(DOCS_PATH, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Splitting by double newline to get distinct policy sections
                        file_chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
                        chunks.extend(file_chunks)
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")

        self.documents = chunks
        
        if self.documents:
            self.tokenized_corpus = [tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            print(f"RAG Engine (BM25): Indexed {len(self.documents)} chunks.")
        else:
            print("RAG Engine: No documents found.")

    def retrieve_context(self, query: str, top_k: int = 1, threshold: float = 1.0) -> str:
        """
        Retrieves top_k most relevant chunks for the query using BM25.
        Returns a single string of combined context.
        """
        if not self.documents or self.bm25 is None:
            return ""

        try:
            tokenized_query = tokenize(query)
            if not tokenized_query:
                return ""

            scores = self.bm25.get_scores(tokenized_query)
            
            # Sort indices in descending order of BM25 score
            ranked_indices = scores.argsort()[::-1]
            
            results = []
            for idx in ranked_indices[:top_k]:
                score = scores[idx]
                if score >= threshold:
                    print(f"RAG BM25 Match Found! (Chunk {idx}, Score: {score:.2f})")
                    results.append(self.documents[idx])

            return "\n\n".join(results)

        except Exception as e:
            print(f"Error retrieving context via BM25: {e}")
            return ""

# Singleton instance
rag_engine = RAGEngine()
