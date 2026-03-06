import os
import time
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import json

load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "finsight-docs"
DOCS_PATH = os.path.join(os.path.dirname(__file__), "data", "docs")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "bfsi_dataset.json")

class PineconeEngine:
    def __init__(self):
        self.pc = None
        self.index = None
        self.model = None
        
        if PINECONE_API_KEY:
            # Initialize Pinecone
            print(f"Connecting to Pinecone with key: {PINECONE_API_KEY[:4]}...")
            self.pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
            
            # Initialize Embedding Model
            print("Loading embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Check/Create Index
            self._setup_index()
            
            # Upsert Docs (Simple check: if index checks empty, upsert)
            self._upsert_documents()
            self._upsert_dataset()
        else:
            print("Pinecone API Key not found!")

    def _setup_index(self):
        print("Checking existing indexes...")
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        print(f"Existing indexes: {existing_indexes}")
        
        if INDEX_NAME not in existing_indexes:
            print(f"Creating index: {INDEX_NAME}")
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=384, # Dimension for all-MiniLM-L6-v2
                metric='cosine',
                spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1') # Adjust region if needed
            )
            time.sleep(10) # Wait for initialization
        
        self.index = self.pc.Index(INDEX_NAME)

    def _upsert_documents(self):
        stats = self.index.describe_index_stats()
        if stats.total_vector_count > 0:
            print("Index already populated. Skipping upsert.")
            return

        print("Upserting documents to Pinecone...")
        chunks = []
        if os.path.exists(DOCS_PATH):
            for filename in os.listdir(DOCS_PATH):
                if filename.endswith(".txt"):
                    filepath = os.path.join(DOCS_PATH, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Split by double newline
                        file_chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
                        for i, chunk in enumerate(file_chunks):
                             chunks.append({
                                 "id": f"{filename}_{i}",
                                 "text": chunk
                             })

        if chunks:
            vectors = []
            for chunk in chunks:
                embedding = self.model.encode(chunk['text']).tolist()
                vectors.append({
                    "id": chunk['id'],
                    "values": embedding,
                    "metadata": {"text": chunk['text']}
                })
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(vectors=batch)
            
            print(f"Upserted {len(vectors)} chunks.")

    def _upsert_dataset(self):
        # We can perform a smart check here too, but for now let's just do it.
        # Ideally check if 'faq_0' exists or metadata query.
        print("Upserting dataset to Pinecone...")
        
        if not os.path.exists(DATASET_PATH):
            print("Dataset not found.")
            return

        with open(DATASET_PATH, 'r') as f:
            dataset = json.load(f)

        vectors = []
        for i, item in enumerate(dataset):
            # Combine instruction and input for the embedding content
            text_content = f"{item['instruction']} {item['input']}".strip()
            
            embedding = self.model.encode(text_content).tolist()
            vectors.append({
                "id": f"faq_{i}",
                "values": embedding,
                "metadata": {
                    "text": text_content,
                    "answer": item['output'],
                    "type": "faq"
                }
            })

        if vectors:
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(vectors=batch)
            print(f"Upserted {len(vectors)} FAQ items.")

    def retrieve_context(self, query, top_k=2, threshold=0.3):
        if not self.index or not self.model:
            return ""

        try:
            query_embedding = self.model.encode(query).tolist()
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            contexts = []
            for match in results.matches:
                if match.score >= threshold:
                    print(f"Pinecone Match: {match.score:.2f}")
                    contexts.append(match.metadata['text'])
            
            return "\n\n".join(contexts)

        except Exception as e:
            print(f"Retrieval Error: {e}")
            return ""

    def query_faq(self, query, threshold=0.80):
        """
        Specific query for FAQ items with high threshold.
        Returns the exact ANSWER string if found.
        """
        if not self.index or not self.model:
            return None

        try:
            query_embedding = self.model.encode(query).tolist()
            results = self.index.query(
                vector=query_embedding,
                top_k=1,
                include_metadata=True,
                filter={"type": "faq"}
            )
            
            if results.matches:
                match = results.matches[0]
                if match.score >= threshold:
                    print(f"Pinecone FAQ Match: {match.score:.2f}")
                    return match.metadata['answer']
            
            return None

        except Exception as e:
            print(f"FAQ Query Error: {e}")
            return None

# Singleton
pinecone_engine = PineconeEngine()
