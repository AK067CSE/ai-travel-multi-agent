from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import json
from pathlib import Path
import re
import random
import time

# Initialize FastAPI app
app = FastAPI(
    title="Travel AI Assistant API",
    description="AI-powered travel planning and recommendation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TravelQuery(BaseModel):
    query: str
    rewrite_query: bool = True
    max_results: int = 5

class TravelResponse(BaseModel):
    original_query: str
    rewritten_query: Optional[str] = None
    entities: Dict[str, List[str]]
    intent: str
    results: List[Dict[str, Any]]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    message: str

# Simple ML components without heavy dependencies
class SimpleTravelRetriever:
    def __init__(self):
        self.documents = []

    def build_index(self, data_path: str):
        """Build simple index from travel data"""
        self.documents = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        doc = json.loads(line)
                        self.documents.append(doc)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                        continue
        print(f"Loaded {len(self.documents)} documents")

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval"""
        query_words = set(query.lower().split())
        scored_docs = []

        for doc in self.documents:
            # Simple scoring based on keyword overlap
            doc_text = f"{doc.get('prompt', '')} {doc.get('response', '')}".lower()
            doc_words = set(doc_text.split())
            overlap = len(query_words.intersection(doc_words))

            if overlap > 0:
                scored_docs.append({
                    'document': doc,
                    'score': overlap / len(query_words)
                })

        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:k]

class SimpleNLPProcessor:
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Simple entity extraction using regex patterns"""
        entities = {
            'locations': [],
            'dates': [],
            'activities': [],
            'budget': []
        }

        # Simple patterns for common travel entities
        location_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized words
        ]

        budget_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Dollar amounts
            r'\d+\s*(?:dollars?|USD|usd)',
        ]

        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            entities['locations'].extend([m for m in matches if len(m) > 2])

        for pattern in budget_patterns:
            matches = re.findall(pattern, text)
            entities['budget'].extend(matches)

        return entities

    def detect_intent(self, query: str) -> str:
        """Simple intent classification"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['hotel', 'stay', 'accommodation']):
            return 'accommodation'
        elif any(word in query_lower for word in ['flight', 'airline', 'fly']):
            return 'transportation'
        elif any(word in query_lower for word in ['restaurant', 'food', 'eat']):
            return 'dining'
        elif any(word in query_lower for word in ['activity', 'tour', 'visit', 'see']):
            return 'activities'
        else:
            return 'general'

# Global variables for ML components
retriever = None
nlp_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize ML components on startup"""
    global retriever, nlp_processor

    print("🚀 Initializing Travel AI Assistant...")

    try:
        # Initialize NLP processor
        print("📝 Loading NLP processor...")
        nlp_processor = SimpleNLPProcessor()

        # Initialize retriever and build index
        print("🔍 Loading retriever and building index...")
        retriever = SimpleTravelRetriever()

        # Check if data file exists
        data_path = "../data.jsonl"
        if os.path.exists(data_path):
            retriever.build_index(data_path)
            print(f"✅ Index built successfully from {data_path}")
        else:
            print(f"⚠️ Warning: Data file {data_path} not found. Retriever will not work properly.")

        print("🎉 All components initialized successfully!")

    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
        raise e

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint for health check"""
    return HealthResponse(
        status="healthy",
        message="Travel AI Assistant API is running!"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )

@app.post("/search", response_model=TravelResponse)
async def search_travel(query_request: TravelQuery):
    """
    Main search endpoint for travel queries

    This endpoint:
    1. Processes the natural language query
    2. Extracts entities and intent
    3. Retrieves relevant travel recommendations using simple keyword matching
    """
    start_time = time.time()

    try:
        if not all([retriever, nlp_processor]):
            raise HTTPException(
                status_code=503,
                detail="ML components not properly initialized"
            )

        original_query = query_request.query

        # Extract entities and intent
        entities = nlp_processor.extract_entities(original_query)
        intent = nlp_processor.detect_intent(original_query)

        # Simple query enhancement (no complex rewriting)
        search_query = original_query
        rewritten_query = None

        if query_request.rewrite_query:
            # Simple query enhancement by adding related terms
            enhanced_terms = []
            if intent == 'accommodation':
                enhanced_terms.extend(['hotel', 'stay', 'accommodation'])
            elif intent == 'transportation':
                enhanced_terms.extend(['flight', 'travel', 'transport'])
            elif intent == 'dining':
                enhanced_terms.extend(['restaurant', 'food', 'dining'])
            elif intent == 'activities':
                enhanced_terms.extend(['activity', 'tour', 'attraction'])

            if enhanced_terms:
                rewritten_query = f"{original_query} {' '.join(enhanced_terms)}"
                search_query = rewritten_query

        # Retrieve relevant documents
        retrieved_docs = retriever.retrieve(search_query, k=query_request.max_results)

        processing_time = time.time() - start_time

        return TravelResponse(
            original_query=original_query,
            rewritten_query=rewritten_query,
            entities=entities,
            intent=intent,
            results=retrieved_docs,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = {
            "status": "operational",
            "components": {
                "retriever": retriever is not None,
                "nlp_processor": nlp_processor is not None
            }
        }

        # Add index stats if retriever is available
        if retriever and retriever.documents:
            stats["document_count"] = len(retriever.documents)

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

if __name__ == "__main__":
    print("🌟 Starting Travel AI Assistant API...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
