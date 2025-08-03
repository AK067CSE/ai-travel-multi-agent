"""
Simple RAG System for Travel Multi-Agent AI
Using basic embeddings and keyword search (no BGE dependency issues)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# LangChain integration
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class SimpleTravelRAG:
    """
    Simple RAG system for travel queries using TF-IDF embeddings
    No complex dependencies - works reliably
    """
    
    def __init__(self, 
                 data_path: str = "../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl",
                 top_k: int = 5):
        
        self.data_path = data_path
        self.top_k = top_k
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
            temperature=0.3,
            max_tokens=2000
        )
        
        # Data storage
        self.documents = []
        self.metadata = []
        self.vectorizer = None
        self.doc_vectors = None
        
        # Conversation memory
        self.conversation_memory = {}
        
        logger.info("Simple Travel RAG system initialized")
    
    def load_and_index_data(self) -> None:
        """Load travel dataset and create TF-IDF index"""
        logger.info("Loading and indexing travel dataset...")
        
        documents = []
        metadata = []
        
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        
                        # Create document text
                        doc_text = self._create_document_text(data)
                        documents.append(doc_text)
                        
                        # Store metadata
                        metadata.append({
                            "id": str(i),
                            "prompt": data.get("prompt", ""),
                            "response": data.get("response", ""),
                            "source": data.get("metadata", {}).get("source", "unknown"),
                            "destination": data.get("metadata", {}).get("destination", ""),
                            "type": data.get("metadata", {}).get("type", "general")
                        })
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line {i}: {e}")
                        continue
        
        self.documents = documents
        self.metadata = metadata
        
        logger.info(f"Loaded {len(documents)} travel documents")
        
        # Create TF-IDF vectors
        self._create_tfidf_index()
        
        logger.info("Indexing completed successfully")
    
    def _create_document_text(self, data: Dict[str, Any]) -> str:
        """Create comprehensive document text for better retrieval"""
        parts = []
        
        # Add prompt/query
        if data.get("prompt"):
            parts.append(f"Query: {data['prompt']}")
        
        # Add response
        if data.get("response"):
            parts.append(f"Response: {data['response']}")
        
        # Add metadata fields
        metadata = data.get("metadata", {})
        for field in ["destination", "type"]:
            if metadata.get(field):
                parts.append(f"{field.title()}: {metadata[field]}")
        
        return "\n".join(parts)
    
    def _create_tfidf_index(self) -> None:
        """Create TF-IDF index for document retrieval"""
        try:
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            # Fit and transform documents
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)
            
            logger.info(f"TF-IDF index created with {self.doc_vectors.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Error creating TF-IDF index: {e}")
            raise
    
    def retrieve_similar_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve similar documents using TF-IDF cosine similarity"""
        try:
            if not self.vectorizer or self.doc_vectors is None:
                logger.error("TF-IDF index not initialized")
                return []
            
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Get top-k most similar documents
            top_indices = np.argsort(similarities)[::-1][:self.top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append({
                        "id": str(idx),
                        "document": self.documents[idx],
                        "metadata": self.metadata[idx],
                        "similarity_score": float(similarities[idx])
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            return []
    
    def generate_rag_response(self, query: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """Generate response using simple RAG"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve_similar_documents(query)
            
            # Get conversation history
            conversation_history = self.conversation_memory.get(user_id, [])
            
            # Create context from retrieved documents
            context = self._create_context(retrieved_docs)
            
            # Generate response
            response = self._generate_response(query, context, conversation_history)
            
            # Update conversation memory
            self._update_conversation_memory(user_id, query, response)
            
            return {
                "response": response,
                "retrieved_docs": len(retrieved_docs),
                "sources": [doc["metadata"] for doc in retrieved_docs],
                "context_used": len(context) > 0,
                "similarity_scores": [doc["similarity_score"] for doc in retrieved_docs]
            }
            
        except Exception as e:
            logger.error(f"RAG generation error: {e}")
            return {
                "response": "I apologize, but I'm having trouble accessing my travel knowledge base right now. Please try again.",
                "retrieved_docs": 0,
                "sources": [],
                "context_used": False,
                "similarity_scores": []
            }
    
    def _create_context(self, retrieved_docs: List[Dict]) -> str:
        """Create context string from retrieved documents"""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc["metadata"]
            
            context_entry = f"Example {i} (similarity: {doc['similarity_score']:.2f}):\n"
            
            if metadata.get("prompt"):
                context_entry += f"Query: {metadata['prompt']}\n"
            
            if metadata.get("response"):
                # Limit response length for context
                response = metadata["response"][:400] + "..." if len(metadata["response"]) > 400 else metadata["response"]
                context_entry += f"Response: {response}\n"
            
            context_parts.append(context_entry)
        
        return "\n---\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str, conversation_history: List[Dict]) -> str:
        """Generate response using LLM with RAG context"""
        # Create conversation history string
        history_str = ""
        if conversation_history:
            recent_history = conversation_history[-2:]  # Last 2 exchanges
            for exchange in recent_history:
                history_str += f"User: {exchange['query']}\nAssistant: {exchange['response'][:200]}...\n\n"
        
        # Create RAG prompt
        rag_prompt = ChatPromptTemplate.from_template(
            """You are an expert travel advisor with access to a comprehensive travel knowledge base. Use the provided examples to give detailed, personalized travel advice.

CONVERSATION HISTORY:
{conversation_history}

RELEVANT TRAVEL EXAMPLES:
{context}

CURRENT QUERY: {query}

Instructions:
1. Use the provided examples as inspiration and reference
2. Provide specific, actionable travel advice
3. Include relevant details like prices, locations, and timing when available
4. If examples show similar trips, adapt them to the current query
5. Be conversational and helpful
6. Mention that your recommendations are based on similar successful trips

RESPONSE:"""
        )
        
        try:
            chain = rag_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "query": query,
                "context": context if context else "No specific examples found in knowledge base.",
                "conversation_history": history_str if history_str else "No previous conversation."
            })
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def _update_conversation_memory(self, user_id: str, query: str, response: str) -> None:
        """Update conversation memory for user"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        self.conversation_memory[user_id].append({
            "query": query,
            "response": response,
            "timestamp": self._get_timestamp()
        })
        
        # Keep only last 5 exchanges
        if len(self.conversation_memory[user_id]) > 5:
            self.conversation_memory[user_id] = self.conversation_memory[user_id][-5:]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "total_documents": len(self.documents),
            "tfidf_features": self.doc_vectors.shape[1] if self.doc_vectors is not None else 0,
            "active_conversations": len(self.conversation_memory),
            "embedding_method": "TF-IDF",
            "llm_model": "llama3-70b-8192"
        }
