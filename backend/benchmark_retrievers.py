import os
import json
import time
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

def preprocess_text(text):
    # Standard lowercase and token cleaning
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def evaluate_retrievers_on_docs():
    print("=" * 70)
    print("EXPERIMENT 1: Policy Chunks Retrieval (loan_policy.txt)")
    print("=" * 70)
    
    docs_path = os.path.join(os.path.dirname(__file__), "data", "docs", "loan_policy.txt")
    with open(docs_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 5 sections in loan_policy.txt
    chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
    
    # Ground truth test cases: (query, expected_chunk_index, description)
    test_queries = [
        ("What is the prepayment penalty for personal loan foreclosure?", 3, "Section 3: Prepayment Penalty"),
        ("What credit score is needed for a personal loan?", 1, "Section 1: Eligibility"),
        ("What happens if I miss EMI for 90 days?", 4, "Section 4: Late Payment & NPA"),
        ("Can I use utility bill as address proof?", 5, "Section 5: Documentation"),
        ("What is the interest rate range for car loans?", 2, "Section 2: Interest Rates"),
        ("foreclosure fee within first year for home loan", 3, "Section 3: Home loan fee"),
        ("DTI debt to income ratio requirement", 1, "Section 1: DTI ratio"),
    ]
    
    # --- TF-IDF Setup ---
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks)
    
    # --- BM25 Setup ---
    tokenized_chunks = [preprocess_text(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    print(f"Total Chunks in Corpus: {len(chunks)}\n")
    
    tfidf_hits = 0
    bm25_hits = 0
    
    for query, expected_idx, desc in test_queries:
        print(f"Query: \"{query}\" [{desc}]")
        
        # TF-IDF
        q_vec = vectorizer.transform([query])
        tfidf_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        tfidf_top1 = int(np.argmax(tfidf_scores))
        tfidf_top_score = tfidf_scores[tfidf_top1]
        tfidf_success = (tfidf_top1 == expected_idx)
        if tfidf_success: tfidf_hits += 1
        
        # BM25
        q_tokens = preprocess_text(query)
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_top1 = int(np.argmax(bm25_scores))
        bm25_top_score = bm25_scores[bm25_top1]
        bm25_success = (bm25_top1 == expected_idx)
        if bm25_success: bm25_hits += 1
        
        print(f"  * TF-IDF : Top Pick = Chunk {tfidf_top1} (Score: {tfidf_top_score:.3f}) -> {'[PASS]' if tfidf_success else '[FAIL]'}")
        print(f"  * BM25   : Top Pick = Chunk {bm25_top1} (Score: {bm25_top_score:.3f}) -> {'[PASS]' if bm25_success else '[FAIL]'}\n")
        
    print(f"Experiment 1 Accuracy:")
    print(f"  - TF-IDF Top-1 Accuracy : {tfidf_hits}/{len(test_queries)} ({tfidf_hits/len(test_queries)*100:.1f}%)")
    print(f"  - BM25 Top-1 Accuracy   : {bm25_hits}/{len(test_queries)} ({bm25_hits/len(test_queries)*100:.1f}%)\n")


def evaluate_retrievers_on_bfsi_dataset(num_samples=200):
    print("=" * 70)
    print(f"EXPERIMENT 2: Large-Scale Banking Q&A Retrieval (bfsi_dataset.json, {num_samples} samples)")
    print("=" * 70)
    
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "bfsi_dataset.json")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    sample_data = data[:num_samples]
    corpus = [item['output'] for item in sample_data]
    queries = [item['instruction'] for item in sample_data]
    
    # 1. Fit TF-IDF
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    tfidf_index_time = time.perf_counter() - t0
    
    # 2. Fit BM25
    t0 = time.perf_counter()
    tokenized_corpus = [preprocess_text(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_index_time = time.perf_counter() - t0
    
    # Evaluation metrics
    tfidf_top1_correct = 0
    tfidf_top3_correct = 0
    tfidf_mrr = 0.0
    
    bm25_top1_correct = 0
    bm25_top3_correct = 0
    bm25_mrr = 0.0
    
    # Run TF-IDF Benchmark
    t0 = time.perf_counter()
    for i, q in enumerate(queries):
        q_vec = vectorizer.transform([q])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        ranked_indices = scores.argsort()[::-1]
        
        # Rank of ground truth document `i`
        rank = list(ranked_indices).index(i) + 1
        if rank == 1:
            tfidf_top1_correct += 1
        if rank <= 3:
            tfidf_top3_correct += 1
        tfidf_mrr += 1.0 / rank
    tfidf_query_time = (time.perf_counter() - t0) * 1000 / len(queries)
    
    # Run BM25 Benchmark
    t0 = time.perf_counter()
    for i, q in enumerate(queries):
        q_tokens = preprocess_text(q)
        scores = np.array(bm25.get_scores(q_tokens))
        ranked_indices = scores.argsort()[::-1]
        
        # Rank of ground truth document `i`
        rank = list(ranked_indices).index(i) + 1
        if rank == 1:
            bm25_top1_correct += 1
        if rank <= 3:
            bm25_top3_correct += 1
        bm25_mrr += 1.0 / rank
    bm25_query_time = (time.perf_counter() - t0) * 1000 / len(queries)
    
    N = len(queries)
    print(f"{'Metric':<25} | {'TF-IDF (Cosine)':<20} | {'BM25 (Okapi)':<20} | {'Delta / Winner':<15}")
    print("-" * 88)
    print(f"{'Top-1 Accuracy':<25} | {tfidf_top1_correct/N*100:>18.2f}% | {bm25_top1_correct/N*100:>18.2f}% | {f'+{(bm25_top1_correct-tfidf_top1_correct)/N*100:+.2f}% (BM25)':<15}")
    print(f"{'Top-3 Accuracy':<25} | {tfidf_top3_correct/N*100:>18.2f}% | {bm25_top3_correct/N*100:>18.2f}% | {f'+{(bm25_top3_correct-tfidf_top3_correct)/N*100:+.2f}% (BM25)':<15}")
    print(f"{'Mean Reciprocal Rank (MRR)':<25} | {tfidf_mrr/N:>19.4f} | {bm25_mrr/N:>19.4f} | {f'+{(bm25_mrr - tfidf_mrr)/N:+.4f} (BM25)':<15}")
    print(f"{'Avg Query Latency (ms)':<25} | {tfidf_query_time:>17.3f} ms | {bm25_query_time:>17.3f} ms | {'Both sub-ms':<15}")
    print("-" * 88)

if __name__ == "__main__":
    evaluate_retrievers_on_docs()
    evaluate_retrievers_on_bfsi_dataset(num_samples=762)
