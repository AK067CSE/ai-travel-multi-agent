from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Any

class TravelReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder"""
        pairs = []
        for doc in documents:
            text = f"{doc['document']['query']} {doc['document']['response']}"
            pairs.append([query, text])
        
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze()
        
        # Sort by scores
        scored_docs = [(score.item(), doc) for score, doc in zip(scores, documents)]
        scored_docs.sort(reverse=True)
        
        return [doc for _, doc in scored_docs[:top_k]]