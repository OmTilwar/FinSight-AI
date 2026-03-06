import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Path to the dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "bfsi_dataset.json")

class DatasetMatcher:
    def __init__(self):
        self.dataset = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load_dataset()

    def load_dataset(self):
        """Loads the dataset and prepares the TF-IDF matrix."""
        if not os.path.exists(DATASET_PATH):
            print(f"Dataset not found at {DATASET_PATH}")
            return

        try:
            with open(DATASET_PATH, 'r') as f:
                self.dataset = json.load(f)
            
            # Prepare corpus for vectorization (using 'instruction' field)
            corpus = [item['instruction'] for item in self.dataset]
            
            if corpus:
                # Use n-grams (1-3 words) to catch phrases like "operating hours" better
                self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
                self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
                print(f"Loaded {len(self.dataset)} samples from dataset.")
            else:
                print("Dataset is empty.")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def find_match(self, query, threshold=0.75):
        """
        Finds the best matching response from the dataset.
        Returns the response string if a match is found with score > threshold.
        Otherwise returns None.
        """
        if not self.dataset or self.vectorizer is None:
            return None

        try:
            query_vec = self.vectorizer.transform([query])
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            best_match_idx = np.argmax(cosine_similarities)
            best_score = cosine_similarities[best_match_idx]

            if best_score >= threshold:
                print(f"Dataset Match Found! Score: {best_score:.2f}")
                return self.dataset[best_match_idx]['output']
            else:
                return None
        except Exception as e:
            print(f"Error finding match: {e}")
            return None

# Singleton instance
matcher = DatasetMatcher()
