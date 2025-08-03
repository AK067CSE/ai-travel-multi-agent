"""
Enhanced RAG System with Free APIs
- Hugging Face embeddings (free)
- ChromaDB vector database (free)
- Multiple free LLMs (Groq, Gemini)
- Advanced chunking and reranking
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

# Free embedding models
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available")

# ChromaDB (free vector database)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available")

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangChain for LLM integration
from langchain_groq import ChatGroq
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini not available")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class EnhancedTravelRAG:
    """
    Enhanced RAG system using free APIs and better embeddings
    """
    
    def __init__(self,
                 data_path: str = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 vector_db_path: str = "./enhanced_vectordb",
                 top_k: int = 10,
                 rerank_top_k: int = 5):
        
        # Auto-detect data path if not provided
        if data_path is None:
            possible_paths = [
                "../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl",
                "../../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl",
                "../multi_agent_system/travel_data.jsonl",
                "./travel_data.jsonl"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break

            if data_path is None:
                # Use a fallback path
                data_path = "../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl"

        self.data_path = data_path
        self.vector_db_path = vector_db_path
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        
        # Initialize embedding model (free Hugging Face model)
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.embedding_model = None

        if not self.embedding_model:
            logger.info("Using TF-IDF as primary embedding method (highly effective for travel data)")

        # Enhanced TF-IDF configuration
        self.tfidf_vectorizer = None
        self.tfidf_vectors = None
        
        # Initialize LLMs (free APIs)
        self.llms = self._initialize_llms()
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        if CHROMADB_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
                logger.info("ChromaDB initialized successfully")
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}")
        
        # Data storage
        self.documents = []
        self.metadata = []
        self.chunks = []
        self.chunk_metadata = []
        
        # Conversation memory
        self.conversation_memory = {}
        
        logger.info("Enhanced Travel RAG system initialized")
    
    def _initialize_llms(self) -> Dict[str, Any]:
        """Initialize free LLM providers"""
        llms = {}
        
        # Groq (Free tier)
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key:
                llms['groq'] = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="llama3-70b-8192",
                    temperature=0.3,
                    max_tokens=2000
                )
                logger.info("Groq LLM initialized")
        except Exception as e:
            logger.warning(f"Groq LLM initialization failed: {e}")
        
        # Google Gemini (Free tier)
        if GEMINI_AVAILABLE:
            try:
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if gemini_api_key:
                    llms['gemini'] = ChatGoogleGenerativeAI(
                        google_api_key=gemini_api_key,
                        model="gemini-pro",
                        temperature=0.3,
                        max_output_tokens=2000
                    )
                    logger.info("Gemini LLM initialized")
            except Exception as e:
                logger.warning(f"Gemini LLM initialization failed: {e}")
        
        return llms
    
    def load_and_index_data(self) -> None:
        """Load travel dataset and create enhanced index"""
        logger.info("Loading and indexing travel dataset with enhanced RAG...")
        
        # Load documents
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
        
        # Create chunks for better retrieval
        self._create_chunks()
        
        # Create vector index
        if self.embedding_model and CHROMADB_AVAILABLE:
            self._create_chromadb_index()
        else:
            self._create_tfidf_index()
        
        logger.info("Enhanced indexing completed successfully")
    
    def _create_document_text(self, data: Dict[str, Any]) -> str:
        """Create comprehensive document text"""
        parts = []
        
        if data.get("prompt"):
            parts.append(f"Query: {data['prompt']}")
        
        if data.get("response"):
            parts.append(f"Response: {data['response']}")
        
        metadata = data.get("metadata", {})
        for field in ["destination", "type"]:
            if metadata.get(field):
                parts.append(f"{field.title()}: {metadata[field]}")
        
        return "\n".join(parts)
    
    def _create_chunks(self) -> None:
        """Create chunks for better retrieval"""
        chunks = []
        chunk_metadata = []
        
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            # Simple chunking by response length
            response = meta.get("response", "")
            
            if len(response) > 1000:
                # Split long responses into chunks
                chunk_size = 500
                overlap = 100
                
                for start in range(0, len(response), chunk_size - overlap):
                    end = min(start + chunk_size, len(response))
                    chunk = response[start:end]
                    
                    if len(chunk.strip()) > 50:  # Only keep meaningful chunks
                        chunks.append(f"Query: {meta.get('prompt', '')}\nResponse: {chunk}")
                        chunk_metadata.append({
                            **meta,
                            "chunk_id": f"{i}_{start}",
                            "is_chunk": True,
                            "chunk_start": start,
                            "chunk_end": end
                        })
            else:
                # Keep short responses as-is
                chunks.append(doc)
                chunk_metadata.append({
                    **meta,
                    "chunk_id": str(i),
                    "is_chunk": False
                })
        
        self.chunks = chunks
        self.chunk_metadata = chunk_metadata
        
        logger.info(f"Created {len(chunks)} chunks from {len(self.documents)} documents")
    
    def _create_chromadb_index(self) -> None:
        """Create ChromaDB vector index with embeddings"""
        try:
            # Delete existing collection
            try:
                self.chroma_client.delete_collection("travel_chunks")
            except:
                pass
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name="travel_chunks",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Generate embeddings in batches
            batch_size = 32
            for i in range(0, len(self.chunks), batch_size):
                batch_chunks = self.chunks[i:i+batch_size]
                batch_metadata = self.chunk_metadata[i:i+batch_size]
                batch_ids = [meta["chunk_id"] for meta in batch_metadata]
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(batch_chunks, convert_to_tensor=False)
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=batch_chunks,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                logger.info(f"Indexed batch {i//batch_size + 1}/{(len(self.chunks)-1)//batch_size + 1}")
            
            logger.info("ChromaDB index created successfully")
            
        except Exception as e:
            logger.error(f"ChromaDB indexing failed: {e}")
            # Fallback to TF-IDF
            self._create_tfidf_index()
    
    def _create_tfidf_index(self) -> None:
        """Enhanced TF-IDF index optimized for travel data"""
        try:
            # Enhanced TF-IDF with travel-specific optimizations
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=8000,  # Increased for better coverage
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams for travel phrases
                min_df=1,  # Lower threshold for travel-specific terms
                max_df=0.85,  # Slightly higher to keep important travel terms
                sublinear_tf=True,  # Better handling of term frequency
                norm='l2',  # L2 normalization for better similarity
                analyzer='word',
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Better tokenization
            )

            self.tfidf_vectors = self.tfidf_vectorizer.fit_transform(self.chunks)
            logger.info(f"Enhanced TF-IDF index created with {self.tfidf_vectors.shape[1]} features")
            logger.info(f"TF-IDF optimized for travel data with trigrams and enhanced tokenization")

        except Exception as e:
            logger.error(f"TF-IDF indexing failed: {e}")
            raise

    def retrieve_similar_chunks(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve similar chunks using best available method"""
        if self.collection and self.embedding_model:
            return self._chromadb_retrieve(query)
        elif self.tfidf_vectorizer and self.tfidf_vectors is not None:
            return self._tfidf_retrieve(query)
        else:
            logger.error("No retrieval method available")
            return []

    def _chromadb_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve using ChromaDB and embeddings"""
        try:
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
                    "similarity_score": 1 - results['distances'][0][i],
                    "retrieval_method": "chromadb"
                })

            return formatted_results

        except Exception as e:
            logger.error(f"ChromaDB retrieval error: {e}")
            return self._tfidf_retrieve(query)

    def _tfidf_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Fallback TF-IDF retrieval"""
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])

            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_vectors).flatten()

            # Get top results
            top_indices = np.argsort(similarities)[::-1][:self.top_k]

            formatted_results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    formatted_results.append({
                        "id": self.chunk_metadata[idx]["chunk_id"],
                        "document": self.chunks[idx],
                        "metadata": self.chunk_metadata[idx],
                        "similarity_score": float(similarities[idx]),
                        "retrieval_method": "tfidf"
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"TF-IDF retrieval error: {e}")
            return []

    def rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using LLM-based scoring"""
        if len(results) <= self.rerank_top_k:
            return results

        try:
            # Use best available LLM for reranking
            llm = self._get_best_llm()
            if not llm:
                return results[:self.rerank_top_k]

            rerank_prompt = ChatPromptTemplate.from_template(
                """Rate the relevance of this travel document to the query on a scale of 1-10.

Query: {query}

Document: {document}

Relevance score (1-10, just the number):"""
            )

            scored_results = []
            for result in results[:self.top_k]:
                try:
                    chain = rerank_prompt | llm | StrOutputParser()
                    score_text = chain.invoke({
                        "query": query,
                        "document": result["document"][:800]  # Limit length
                    })

                    # Extract numeric score
                    try:
                        score = float(score_text.strip().split()[0])
                        score = max(1, min(10, score))  # Clamp to 1-10
                    except:
                        score = result.get("similarity_score", 0.5) * 10

                    result["rerank_score"] = score
                    scored_results.append(result)

                except Exception as e:
                    logger.warning(f"Reranking failed for result: {e}")
                    result["rerank_score"] = result.get("similarity_score", 0.5) * 10
                    scored_results.append(result)

            # Sort by rerank score
            scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_results[:self.rerank_top_k]

        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return results[:self.rerank_top_k]

    def _get_best_llm(self):
        """Get the best available LLM"""
        if 'groq' in self.llms:
            return self.llms['groq']
        elif 'gemini' in self.llms:
            return self.llms['gemini']
        else:
            return None

    def generate_enhanced_response(self, query: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """Generate response using enhanced RAG"""
        try:
            # Retrieve relevant chunks
            retrieved_chunks = self.retrieve_similar_chunks(query)

            # Rerank for better precision
            reranked_chunks = self.rerank_results(query, retrieved_chunks)

            # Get conversation history
            conversation_history = self.conversation_memory.get(user_id, [])

            # Create context
            context = self._create_enhanced_context(reranked_chunks)

            # Generate response with best LLM
            response = self._generate_enhanced_response(query, context, conversation_history)

            # Update memory
            self._update_conversation_memory(user_id, query, response)

            return {
                "response": response,
                "retrieved_chunks": len(retrieved_chunks),
                "reranked_chunks": len(reranked_chunks),
                "sources": [chunk["metadata"] for chunk in reranked_chunks],
                "context_used": len(context) > 0,
                "retrieval_method": reranked_chunks[0]["retrieval_method"] if reranked_chunks else "none",
                "rerank_scores": [chunk.get("rerank_score", 0) for chunk in reranked_chunks]
            }

        except Exception as e:
            logger.error(f"Enhanced RAG generation error: {e}")
            return {
                "response": "I apologize, but I'm having trouble accessing my travel knowledge base right now. Please try again.",
                "retrieved_chunks": 0,
                "reranked_chunks": 0,
                "sources": [],
                "context_used": False,
                "retrieval_method": "error",
                "rerank_scores": []
            }

    def _create_enhanced_context(self, chunks: List[Dict]) -> str:
        """Create enhanced context from reranked chunks"""
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]

            context_entry = f"Example {i} (relevance: {chunk.get('rerank_score', 0):.1f}/10):\n"

            if metadata.get("prompt"):
                context_entry += f"Query: {metadata['prompt']}\n"

            if metadata.get("response"):
                response = metadata["response"][:600] + "..." if len(metadata["response"]) > 600 else metadata["response"]
                context_entry += f"Expert Response: {response}\n"

            # Add source info
            if metadata.get("source"):
                context_entry += f"Source: {metadata['source']}\n"

            context_parts.append(context_entry)

        return "\n---\n".join(context_parts)

    def _generate_enhanced_response(self, query: str, context: str, conversation_history: List[Dict]) -> str:
        """Generate response using enhanced prompting"""
        # Create conversation history string
        history_str = ""
        if conversation_history:
            recent_history = conversation_history[-2:]
            for exchange in recent_history:
                history_str += f"User: {exchange['query']}\nAssistant: {exchange['response'][:200]}...\n\n"

        # Enhanced RAG prompt
        enhanced_prompt = ChatPromptTemplate.from_template(
            """You are an expert travel advisor with access to a comprehensive travel knowledge base. Use the provided examples to give detailed, personalized travel advice.

CONVERSATION HISTORY:
{conversation_history}

RELEVANT TRAVEL EXAMPLES (ranked by relevance):
{context}

CURRENT QUERY: {query}

Instructions:
1. Use the provided examples as inspiration and reference
2. Provide specific, actionable travel advice with details
3. Include relevant prices, locations, and timing when available
4. If examples show similar trips, adapt them to the current query
5. Be conversational and helpful
6. Mention that your recommendations are based on similar successful trips
7. If multiple examples are provided, synthesize the best elements from each
8. Always provide practical, implementable advice

RESPONSE:"""
        )

        try:
            llm = self._get_best_llm()
            if not llm:
                return "I'm currently experiencing technical difficulties with my AI models. Please try again later."

            chain = enhanced_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "query": query,
                "context": context if context else "No specific examples found in knowledge base.",
                "conversation_history": history_str if history_str else "No previous conversation."
            })

            return response.strip()

        except Exception as e:
            logger.error(f"Enhanced response generation error: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    def _update_conversation_memory(self, user_id: str, query: str, response: str) -> None:
        """Update conversation memory"""
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

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced RAG system statistics"""
        embedding_method = "enhanced_tfidf"
        if self.embedding_model:
            embedding_method = "sentence-transformers"

        vector_features = 0
        if self.tfidf_vectors is not None:
            vector_features = self.tfidf_vectors.shape[1]

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "embedding_model": embedding_method,
            "vector_database": "chromadb" if self.collection else "enhanced_tfidf",
            "vector_features": vector_features,
            "available_llms": list(self.llms.keys()),
            "active_conversations": len(self.conversation_memory),
            "chunking_enabled": True,
            "reranking_enabled": True,
            "tfidf_optimizations": [
                "trigram_support",
                "travel_specific_tokenization",
                "sublinear_tf_scaling",
                "l2_normalization"
            ] if embedding_method == "enhanced_tfidf" else []
        }
