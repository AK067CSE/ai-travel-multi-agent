from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class QueryRewriter:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
    def rewrite_query(self, query: str) -> str:
        """Enhance query for better retrieval"""
        prompt = f"Rewrite this travel query to be more specific and detailed: {query}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                temperature=0.7,
                do_sample=True
            )
        
        rewritten = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return rewritten if rewritten else query