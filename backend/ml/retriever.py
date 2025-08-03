import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json

class TravelRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        
    def build_index(self, data_path: str):
        """Build FAISS index from travel data"""
        with open(data_path, 'r') as f:
            self.documents = [json.loads(line) for line in f]
        
        texts = [f"{doc['query']} {doc['response']}" for doc in self.documents]
        embeddings = self.encoder.encode(texts)
        
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant documents"""
        query_embedding = self.encoder.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'document': self.documents[idx],
                'score': float(score)
            })
        return results