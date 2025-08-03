"""
Advanced RAG System for Travel Multi-Agent AI
Using BGE embeddings, hybrid retrieval, re-ranking, and Groq LLM
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# Core RAG components
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# LangChain integration
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class AdvancedTravelRAG:
    """
    Advanced RAG system optimized for travel queries
    Features:
    - BGE embeddings for semantic search
    - Hybrid retrieval (dense + sparse)
    - Re-ranking for precision
    - Query expansion
    - Conversational memory
    """
    
    def __init__(self,
                 data_path: str = "../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl",
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 vector_db_path: str = "./rag_vectordb",
                 top_k: int = 10,
                 rerank_top_k: int = 5):
        
        self.data_path = data_path
        self.vector_db_path = vector_db_path
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        
        # Initialize BGE embeddings (best for semantic search)
        logger.info(f"Loading BGE embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",  # Use larger model for RAG
            temperature=0.3,  # Lower temperature for factual responses
            max_tokens=2000
        )
        
        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = None
        
        # BM25 for sparse retrieval
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
        
        # Conversation memory
        self.conversation_memory = {}
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        logger.info("Advanced Travel RAG system initialized")
    
    def load_and_index_data(self) -> None:
        """Load travel dataset and create vector index"""
        logger.info("Loading and indexing travel dataset...")
        
        # Load dataset
        documents = []
        metadata = []
        
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        
                        # Create comprehensive document text
                        doc_text = self._create_document_text(data)
                        documents.append(doc_text)
                        
                        # Store metadata
                        metadata.append({
                            "id": str(i),
                            "query": data.get("query", ""),
                            "response": data.get("response", ""),
                            "source": data.get("source", "unknown"),
                            "destination": data.get("destination", ""),
                            "budget": data.get("budget", ""),
                            "duration": data.get("duration", ""),
                            "type": data.get("type", "general")
                        })
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line {i}: {e}")
                        continue
        
        self.documents = documents
        self.doc_metadata = metadata
        
        logger.info(f"Loaded {len(documents)} travel documents")
        
        # Create vector index
        self._create_vector_index(documents, metadata)
        
        # Create BM25 index for sparse retrieval
        self._create_bm25_index(documents)
        
        logger.info("Indexing completed successfully")
    
    def _create_document_text(self, data: Dict[str, Any]) -> str:
        """Create comprehensive document text for better retrieval"""
        parts = []
        
        # Add query
        if data.get("query"):
            parts.append(f"Query: {data['query']}")
        
        # Add response
        if data.get("response"):
            parts.append(f"Response: {data['response']}")
        
        # Add structured fields for better search
        for field in ["destination", "budget", "duration", "type"]:
            if data.get(field):
                parts.append(f"{field.title()}: {data[field]}")
        
        return "\n".join(parts)
    
    def _create_vector_index(self, documents: List[str], metadata: List[Dict]) -> None:
        """Create ChromaDB vector index with BGE embeddings"""
        try:
            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection("travel_docs")
            except:
                pass
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name="travel_docs",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Generate embeddings in batches
            batch_size = 32
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metadata = metadata[i:i+batch_size]
                batch_ids = [str(j) for j in range(i, min(i+batch_size, len(documents)))]
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(batch_docs, convert_to_tensor=False)
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                logger.info(f"Indexed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            raise
    
    def _create_bm25_index(self, documents: List[str]) -> None:
        """Create BM25 index for sparse retrieval"""
        try:
            # Tokenize documents
            tokenized_docs = []
            for doc in documents:
                tokens = word_tokenize(doc.lower())
                # Remove stopwords and short tokens
                tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
                tokenized_docs.append(tokens)
            
            # Create BM25 index
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info("BM25 index created successfully")
            
        except Exception as e:
            logger.error(f"Error creating BM25 index: {e}")
            raise

    def expand_query(self, query: str) -> str:
        """Expand query for better retrieval using LLM"""
        try:
            expansion_prompt = ChatPromptTemplate.from_template(
                """You are a travel query expansion expert. Given a travel query, expand it with related terms, synonyms, and travel-specific keywords that would help find relevant travel information.

Original query: {query}

Expanded query (include original + related travel terms):"""
            )

            chain = expansion_prompt | self.llm | StrOutputParser()
            expanded = chain.invoke({"query": query})

            # Combine original and expanded
            return f"{query} {expanded}"

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query

    def hybrid_retrieve(self, query: str, expand_query: bool = True) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining dense (BGE) and sparse (BM25) search
        """
        if expand_query:
            expanded_query = self.expand_query(query)
        else:
            expanded_query = query

        # Dense retrieval with BGE embeddings
        dense_results = self._dense_retrieve(expanded_query)

        # Sparse retrieval with BM25
        sparse_results = self._sparse_retrieve(query)  # Use original query for BM25

        # Combine and re-rank results
        combined_results = self._combine_results(dense_results, sparse_results)

        # Re-rank for final precision
        reranked_results = self._rerank_results(query, combined_results)

        return reranked_results[:self.rerank_top_k]

    def _dense_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Dense retrieval using BGE embeddings"""
        try:
            if not self.collection:
                logger.error("Vector collection not initialized")
                return []

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=self.top_k
            )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": 1 - results['distances'][0][i],  # Convert distance to similarity
                    "source": "dense"
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Dense retrieval error: {e}")
            return []

    def _sparse_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Sparse retrieval using BM25"""
        try:
            if not self.bm25:
                logger.error("BM25 index not initialized")
                return []

            # Tokenize query
            query_tokens = word_tokenize(query.lower())
            query_tokens = [token for token in query_tokens if token not in self.stop_words and len(token) > 2]

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Get top results
            top_indices = np.argsort(scores)[::-1][:self.top_k]

            formatted_results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include relevant results
                    formatted_results.append({
                        "id": str(idx),
                        "document": self.documents[idx],
                        "metadata": self.doc_metadata[idx],
                        "score": scores[idx],
                        "source": "sparse"
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Sparse retrieval error: {e}")
            return []

    def _combine_results(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """Combine dense and sparse results using Reciprocal Rank Fusion (RRF)"""
        # Create combined ranking using RRF
        combined_scores = {}
        k = 60  # RRF parameter

        # Add dense results
        for rank, result in enumerate(dense_results):
            doc_id = result["id"]
            rrf_score = 1 / (k + rank + 1)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score

        # Add sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = result["id"]
            rrf_score = 1 / (k + rank + 1)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score

        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        # Create result list
        id_to_result = {}
        for result in dense_results + sparse_results:
            id_to_result[result["id"]] = result

        combined_results = []
        for doc_id in sorted_ids:
            if doc_id in id_to_result:
                result = id_to_result[doc_id].copy()
                result["combined_score"] = combined_scores[doc_id]
                combined_results.append(result)

        return combined_results

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-rank results using cross-encoder (simplified LLM-based reranking)"""
        try:
            if len(results) <= self.rerank_top_k:
                return results

            # Use LLM for relevance scoring (simplified cross-encoder)
            rerank_prompt = ChatPromptTemplate.from_template(
                """Rate the relevance of this travel document to the query on a scale of 1-10.

Query: {query}

Document: {document}

Relevance score (1-10):"""
            )

            scored_results = []
            for result in results[:self.top_k]:  # Only rerank top results
                try:
                    chain = rerank_prompt | self.llm | StrOutputParser()
                    score_text = chain.invoke({
                        "query": query,
                        "document": result["document"][:1000]  # Limit document length
                    })

                    # Extract numeric score
                    try:
                        score = float(score_text.strip().split()[0])
                    except:
                        score = result.get("combined_score", 0.5)

                    result["rerank_score"] = score
                    scored_results.append(result)

                except Exception as e:
                    logger.warning(f"Reranking failed for result: {e}")
                    result["rerank_score"] = result.get("combined_score", 0.5)
                    scored_results.append(result)

            # Sort by rerank score
            scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_results

        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return results

    def generate_rag_response(self, query: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """Generate response using RAG with conversational memory"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.hybrid_retrieve(query)

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
                "context_used": len(context) > 0
            }

        except Exception as e:
            logger.error(f"RAG generation error: {e}")
            return {
                "response": "I apologize, but I'm having trouble accessing my travel knowledge base right now. Please try again.",
                "retrieved_docs": 0,
                "sources": [],
                "context_used": False
            }

    def _create_context(self, retrieved_docs: List[Dict]) -> str:
        """Create context string from retrieved documents"""
        if not retrieved_docs:
            return ""

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc["metadata"]

            # Create a structured context entry
            context_entry = f"Example {i}:\n"

            if metadata.get("query"):
                context_entry += f"Similar Query: {metadata['query']}\n"

            if metadata.get("response"):
                # Limit response length for context
                response = metadata["response"][:500] + "..." if len(metadata["response"]) > 500 else metadata["response"]
                context_entry += f"Expert Response: {response}\n"

            # Add structured fields
            for field in ["destination", "budget", "duration", "type"]:
                if metadata.get(field):
                    context_entry += f"{field.title()}: {metadata[field]}\n"

            context_parts.append(context_entry)

        return "\n---\n".join(context_parts)

    def _generate_response(self, query: str, context: str, conversation_history: List[Dict]) -> str:
        """Generate response using LLM with RAG context"""
        # Create conversation history string
        history_str = ""
        if conversation_history:
            recent_history = conversation_history[-3:]  # Last 3 exchanges
            for exchange in recent_history:
                history_str += f"User: {exchange['query']}\nAssistant: {exchange['response']}\n\n"

        # Create RAG prompt
        rag_prompt = ChatPromptTemplate.from_template(
            """You are an expert travel advisor with access to a comprehensive travel knowledge base. Use the provided examples and context to give detailed, personalized travel advice.

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
6. If no relevant examples are found, use your general travel knowledge

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

        # Keep only last 10 exchanges
        if len(self.conversation_memory[user_id]) > 10:
            self.conversation_memory[user_id] = self.conversation_memory[user_id][-10:]

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def search_similar_trips(self, destination: str, budget: str = "", duration: str = "") -> List[Dict]:
        """Search for similar trips with specific filters"""
        # Create search query
        query_parts = [f"trip to {destination}"]
        if budget:
            query_parts.append(f"budget {budget}")
        if duration:
            query_parts.append(f"{duration} days")

        search_query = " ".join(query_parts)

        # Retrieve similar trips
        results = self.hybrid_retrieve(search_query, expand_query=False)

        # Filter and format results
        similar_trips = []
        for result in results:
            metadata = result["metadata"]
            if destination.lower() in metadata.get("destination", "").lower():
                similar_trips.append({
                    "query": metadata.get("query", ""),
                    "response": metadata.get("response", ""),
                    "destination": metadata.get("destination", ""),
                    "budget": metadata.get("budget", ""),
                    "duration": metadata.get("duration", ""),
                    "similarity_score": result.get("rerank_score", 0)
                })

        return similar_trips

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "total_documents": len(self.documents),
            "vector_index_ready": self.collection is not None,
            "bm25_index_ready": self.bm25 is not None,
            "active_conversations": len(self.conversation_memory),
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "llm_model": "llama3-70b-8192"
        }
