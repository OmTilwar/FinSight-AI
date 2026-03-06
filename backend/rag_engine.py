import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Path to documents
DOCS_PATH = os.path.join(os.path.dirname(__file__), "data", "docs")

class RAGEngine:
    def __init__(self):
        self.documents = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load_documents()

    def load_documents(self):
        """
        Loads all .txt files from the docs directory, splits into chunks (paragraphs).
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
                        # Simple splitting by double newline to get paragraphs/sections
                        file_chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
                        chunks.extend(file_chunks)
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")

        self.documents = chunks
        
        if self.documents:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            print(f"RAG Engine: Indexed {len(self.documents)} chunks.")
        else:
            print("RAG Engine: No documents found.")

    def retrieve_context(self, query, top_k=1, threshold=0.1):
        """
        Retrieves top_k most relevant chunks for the query.
        Returns a single string of combined context.
        """
        if not self.documents or self.vectorizer is None:
            return ""

        try:
            query_vec = self.vectorizer.transform([query])
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get indices of top_k matches
            # sort in descending order
            related_docs_indices = cosine_similarities.argsort()[:-top_k-1:-1]
            
            results = []
            for idx in related_docs_indices:
                score = cosine_similarities[idx]
                if score >= threshold:
                    print(f"RAG Match Found! Score: {score:.2f}")
                    results.append(self.documents[idx])

            return "\n\n".join(results)

        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""

# Singleton instance
rag_engine = RAGEngine()
