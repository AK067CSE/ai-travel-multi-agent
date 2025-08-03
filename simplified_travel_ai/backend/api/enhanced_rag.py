"""
Enhanced RAG System for Production Travel AI
Combines the best features from the original complex system
- BGE embeddings for semantic search
- Hybrid retrieval (dense + sparse)
- Re-ranking for precision
- Query expansion
- Conversational memory
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# Core RAG components
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("BM25 not available")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    
    # Download required NLTK data (handle both old and new versions)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except:
            try:
                nltk.download('punkt_tab')
            except:
                pass

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords')
        except:
            pass
        
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available")

# LangChain integration
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available")

from django.conf import settings

logger = logging.getLogger(__name__)

class EnhancedTravelRAG:
    """
    Production-level RAG system for travel queries
    Features:
    - BGE embeddings for semantic search
    - Hybrid retrieval (dense + sparse)
    - Re-ranking for precision
    - Query expansion
    - Conversational memory
    """
    
    def __init__(self,
                 data_path: str = None,
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 vector_db_path: str = "./vectordb",
                 top_k: int = 10,
                 rerank_top_k: int = 5):
        
        self.data_path = data_path or self._find_data_path()
        self.vector_db_path = vector_db_path
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        
        # Initialize components based on availability
        self.embedding_model = None
        self.llm = None
        self.chroma_client = None
        self.collection = None
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
        self.conversation_memory = {}
        self.stop_words = set()
        
        self._initialize_components(embedding_model)
        
        logger.info("Enhanced Travel RAG system initialized")
    
    def _find_data_path(self) -> str:
        """Find available data path from your comprehensive datasets"""
        possible_paths = [
            # Your comprehensive datasets (copied to local data folder)
            "./data/travel_complete_dataset.jsonl",
            "./data/travel_planning_dataset.jsonl",
            "./data/travel_chat_dataset.jsonl",

            # Original dataset locations
            "../data_expansion/final_datasets/travel_complete_dataset_20250719_224702.jsonl",
            "../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl",
            "../data_expansion/final_datasets/travel_chat_dataset_20250719_224702.jsonl",
            "../../data_expansion/final_datasets/travel_complete_dataset_20250719_224702.jsonl",
            "../../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl",

            # Fallback paths
            "./data/travel_dataset.jsonl"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Using dataset: {path}")
                return path

        # Return default path even if it doesn't exist
        logger.warning("No dataset found, will use sample data")
        return "./data/travel_dataset.jsonl"
    
    def _initialize_components(self, embedding_model: str):
        """Initialize available components"""
        try:
            # Initialize BGE embeddings if available
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info(f"Loading BGE embedding model: {embedding_model}")
                self.embedding_model = SentenceTransformer(embedding_model)
                # Import cosine similarity for semantic search
                from sklearn.metrics.pairwise import cosine_similarity
                self.cosine_similarity = cosine_similarity
                logger.info("Semantic search with cosine similarity enabled")
            
            # Initialize LLM if available
            if LANGCHAIN_AVAILABLE:
                groq_api_key = getattr(settings, 'AI_CONFIG', {}).get('GROQ_API_KEY') or os.getenv("GROQ_API_KEY")
                if groq_api_key:
                    self.llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                        temperature=0.3,
                        max_tokens=2000
                    )
                    logger.info("Groq LLM initialized for RAG with Llama-4-Scout")
            
            # Initialize vector database if available
            if CHROMADB_AVAILABLE:
                self.chroma_client = chromadb.PersistentClient(path=self.vector_db_path)
            
            # Initialize stopwords if available
            if NLTK_AVAILABLE:
                self.stop_words = set(stopwords.words('english'))
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def load_and_index_data(self) -> bool:
        """Load comprehensive travel datasets and create vector index"""
        try:
            logger.info("Loading and indexing comprehensive travel datasets...")

            # Load multiple datasets
            documents = []
            metadata = []

            # Dataset paths to try (in order of preference)
            dataset_paths = [
                "./data/travel_complete_dataset.jsonl",
                "./data/travel_planning_dataset.jsonl",
                "./data/travel_chat_dataset.jsonl",
                "../data_expansion/final_datasets/travel_complete_dataset_20250719_224702.jsonl",
                "../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl",
                "../data_expansion/final_datasets/travel_chat_dataset_20250719_224702.jsonl",
                self.data_path  # Fallback to configured path
            ]

            datasets_loaded = 0
            total_entries = 0

            for dataset_path in dataset_paths:
                if os.path.exists(dataset_path):
                    logger.info(f"Loading dataset: {dataset_path}")
                    entries_loaded = self._load_single_dataset(dataset_path, documents, metadata, total_entries)
                    if entries_loaded > 0:
                        datasets_loaded += 1
                        total_entries += entries_loaded
                        logger.info(f"Loaded {entries_loaded} entries from {os.path.basename(dataset_path)}")

                    # If we have enough data, we can stop
                    if total_entries >= 1000:  # Reasonable limit for performance
                        logger.info(f"Loaded sufficient data ({total_entries} entries), stopping")
                        break

            # If no datasets found, create sample data
            if total_entries == 0:
                logger.warning("No datasets found, using sample data")
                documents, metadata = self._create_sample_data()
                total_entries = len(documents)

            self.documents = documents
            self.doc_metadata = metadata

            logger.info(f"Successfully loaded {total_entries} travel documents from {datasets_loaded} datasets")

            # Create vector index if possible
            if self.embedding_model and self.chroma_client:
                self._create_vector_index(documents, metadata)

            # Create BM25 index if possible
            if BM25_AVAILABLE and NLTK_AVAILABLE:
                self._create_bm25_index(documents)

            logger.info("Indexing completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading and indexing data: {e}")
            return False

    def _load_single_dataset(self, dataset_path: str, documents: List[str], metadata: List[Dict], start_id: int) -> int:
        """Load a single dataset file and append to documents/metadata"""
        entries_loaded = 0

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())

                        # Create comprehensive document text
                        doc_text = self._create_document_text(data)
                        documents.append(doc_text)

                        # Store metadata with enhanced fields
                        metadata_entry = {
                            "id": str(start_id + entries_loaded),
                            "query": data.get("query", ""),
                            "response": data.get("response", ""),
                            "source": data.get("source", os.path.basename(dataset_path)),
                            "destination": data.get("destination", ""),
                            "budget": data.get("budget", ""),
                            "duration": data.get("duration", ""),
                            "type": data.get("type", "general"),
                            "dataset": os.path.basename(dataset_path)
                        }

                        # Handle different dataset formats
                        if "travel_planning" in dataset_path:
                            metadata_entry["category"] = "planning"
                        elif "travel_chat" in dataset_path:
                            metadata_entry["category"] = "chat"
                        elif "travel_complete" in dataset_path:
                            metadata_entry["category"] = "complete"
                        else:
                            metadata_entry["category"] = "general"

                        # Add any additional fields from your datasets
                        for field in ["country", "city", "activities", "season", "group_size", "interests"]:
                            if field in data:
                                metadata_entry[field] = data[field]

                        metadata.append(metadata_entry)
                        entries_loaded += 1

                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line {i} in {dataset_path}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {i} in {dataset_path}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_path}: {e}")

        return entries_loaded
    
    def _create_sample_data(self) -> Tuple[List[str], List[Dict]]:
        """Create sample travel data for demonstration"""
        sample_data = [
            {
                "query": "Plan a romantic trip to Paris for 5 days",
                "response": "For a romantic 5-day Paris trip, I recommend staying in the Marais district. Visit the Eiffel Tower at sunset, take a Seine river cruise, explore Montmartre, and dine at intimate bistros. Budget around $2000-3000 for accommodation, meals, and activities.",
                "destination": "Paris",
                "budget": "mid-range",
                "duration": "5 days",
                "type": "romantic"
            },
            {
                "query": "Budget backpacking in Southeast Asia",
                "response": "Southeast Asia is perfect for budget backpacking. Start in Thailand (Bangkok, Chiang Mai), then Vietnam (Ho Chi Minh, Hanoi), and Cambodia (Siem Reap). Budget $30-50/day including hostels, street food, and local transport. Best time: November-March.",
                "destination": "Southeast Asia",
                "budget": "budget",
                "duration": "2-4 weeks",
                "type": "backpacking"
            },
            {
                "query": "Family vacation to Japan with kids",
                "response": "Japan is very family-friendly. Visit Tokyo (Disneyland, teamLab), Kyoto (temples, bamboo forest), and Osaka (Universal Studios, Osaka Castle). Use JR Pass for transport. Stay in family rooms or apartments. Budget $4000-6000 for a family of 4 for 10 days.",
                "destination": "Japan",
                "budget": "mid-range",
                "duration": "10 days",
                "type": "family"
            }
        ]
        
        documents = []
        metadata = []
        
        for i, data in enumerate(sample_data):
            doc_text = self._create_document_text(data)
            documents.append(doc_text)
            
            metadata.append({
                "id": str(i),
                "query": data["query"],
                "response": data["response"],
                "source": "sample",
                "destination": data["destination"],
                "budget": data["budget"],
                "duration": data["duration"],
                "type": data["type"]
            })
        
        return documents, metadata

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
                try:
                    # Try NLTK tokenizer first
                    tokens = word_tokenize(doc.lower())
                except:
                    # Fallback to simple tokenization
                    tokens = doc.lower().split()

                # Remove stopwords and short tokens
                tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
                tokenized_docs.append(tokens)

            # Create BM25 index
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info("BM25 index created successfully")

        except Exception as e:
            logger.error(f"Error creating BM25 index: {e}")
            # Don't raise - continue without BM25

    def generate_rag_response(self, query: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """Generate response using RAG with conversational memory"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.hybrid_retrieve(query) if self.embedding_model else self.simple_retrieve(query)

            # Get conversation history
            conversation_history = self.conversation_memory.get(user_id, [])

            # Create context from retrieved documents
            context = self._create_context(retrieved_docs)

            # Generate response
            if self.llm:
                response = self._generate_llm_response(query, context, conversation_history)
            else:
                response = self._generate_fallback_response(query, retrieved_docs)

            # Update conversation memory
            self._update_conversation_memory(user_id, query, response)

            return {
                "success": True,
                "response": response,
                "retrieved_docs": len(retrieved_docs),
                "sources": [doc.get("metadata", {}) for doc in retrieved_docs],
                "context_used": len(context) > 0,
                "system_used": "Enhanced_RAG" if self.llm else "Fallback_RAG"
            }

        except Exception as e:
            logger.error(f"RAG generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I'm having trouble accessing my travel knowledge base right now. Please try again.",
                "retrieved_docs": 0,
                "sources": [],
                "context_used": False,
                "system_used": "Error_Handler"
            }

    def hybrid_retrieve(self, query: str, expand_query: bool = True) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining dense (BGE) and sparse (BM25) search"""
        if not self.embedding_model:
            return self.simple_retrieve(query)

        try:
            # Expand query if LLM is available
            if expand_query and self.llm:
                expanded_query = self.expand_query(query)
            else:
                expanded_query = query

            # Enhanced semantic retrieval with cosine similarity
            semantic_results = self._semantic_retrieve_with_cosine(expanded_query)

            # Sparse retrieval with BM25 if available for keyword matching
            if self.bm25:
                sparse_results = self._sparse_retrieve(query)
                # Combine with semantic scoring
                combined_results = self._combine_semantic_results(semantic_results, sparse_results, query)
            else:
                combined_results = semantic_results

            # Filter results by relevance threshold
            relevant_results = [r for r in combined_results if r.get('relevance_score', 0) > 0.3]

            logger.info(f"Found {len(relevant_results)} highly relevant documents for query: {query[:50]}...")
            return relevant_results[:self.rerank_top_k]

        except Exception as e:
            logger.error(f"Hybrid retrieval error: {e}")
            return self.simple_retrieve(query)

    def _semantic_retrieve_with_cosine(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced semantic retrieval using cosine similarity"""
        if not self.embedding_model or not self.documents:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])

            # Generate document embeddings if not cached
            if not hasattr(self, '_doc_embeddings') or self._doc_embeddings is None:
                logger.info("Generating document embeddings for semantic search...")
                self._doc_embeddings = self.embedding_model.encode(self.documents)

            # Calculate cosine similarities
            similarities = self.cosine_similarity(query_embedding, self._doc_embeddings)[0]

            # Get top results with similarity scores
            top_indices = np.argsort(similarities)[::-1][:self.top_k]

            results = []
            for idx in top_indices:
                similarity_score = similarities[idx]
                if similarity_score > 0.2:  # Relevance threshold
                    results.append({
                        'content': self.documents[idx],
                        'metadata': self.doc_metadata[idx] if idx < len(self.doc_metadata) else {},
                        'relevance_score': float(similarity_score),
                        'retrieval_method': 'semantic_cosine'
                    })

            logger.info(f"Semantic search found {len(results)} relevant documents")
            return results

        except Exception as e:
            logger.error(f"Semantic retrieval error: {e}")
            return []

    def _combine_semantic_results(self, semantic_results: List[Dict], sparse_results: List[Dict], query: str) -> List[Dict]:
        """Combine semantic and sparse results with intelligent scoring"""
        combined = {}

        # Add semantic results with high weight
        for result in semantic_results:
            content = result['content']
            combined[content] = {
                **result,
                'final_score': result['relevance_score'] * 0.7,  # High weight for semantic
                'methods': ['semantic']
            }

        # Add sparse results with keyword matching bonus
        for result in sparse_results:
            content = result['content']
            if content in combined:
                # Boost score if found by both methods
                combined[content]['final_score'] += result.get('relevance_score', 0.3) * 0.3
                combined[content]['methods'].append('sparse')
            else:
                combined[content] = {
                    **result,
                    'final_score': result.get('relevance_score', 0.3) * 0.3,
                    'methods': ['sparse']
                }

        # Sort by final score and return
        sorted_results = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)
        return sorted_results

    def simple_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval as fallback"""
        results = []
        query_lower = query.lower()

        for i, (doc, metadata) in enumerate(zip(self.documents, self.doc_metadata)):
            doc_lower = doc.lower()

            # Simple keyword matching
            score = 0
            query_words = query_lower.split()
            for word in query_words:
                if word in doc_lower:
                    score += doc_lower.count(word)

            if score > 0:
                results.append({
                    "id": str(i),
                    "document": doc,
                    "metadata": metadata,
                    "score": score,
                    "source": "simple"
                })

        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:self.rerank_top_k]

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
                return []

            # Tokenize query with fallback
            try:
                query_tokens = word_tokenize(query.lower())
            except:
                query_tokens = query.lower().split()

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

    def _create_context(self, retrieved_docs: List[Dict]) -> str:
        """Create context string from retrieved documents"""
        if not retrieved_docs:
            return ""

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})

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

    def _generate_llm_response(self, query: str, context: str, conversation_history: List[Dict]) -> str:
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
            logger.error(f"LLM response generation error: {e}")
            return self._generate_fallback_response(query, [])

    def _generate_fallback_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate fallback response when LLM is not available"""
        if retrieved_docs:
            # Use the best retrieved document
            best_doc = retrieved_docs[0]
            metadata = best_doc.get("metadata", {})

            if metadata.get("response"):
                return f"Based on similar travel queries, here's what I found:\n\n{metadata['response']}"

        # Basic keyword-based response
        query_lower = query.lower()

        if any(word in query_lower for word in ['paris', 'france']):
            return "Paris is a wonderful destination! Consider visiting the Eiffel Tower, Louvre Museum, and enjoying French cuisine. Best time to visit is April-June or September-October."
        elif any(word in query_lower for word in ['japan', 'tokyo', 'kyoto']):
            return "Japan offers amazing experiences! Tokyo has modern attractions, while Kyoto showcases traditional culture. Spring (cherry blossoms) and fall are ideal seasons."
        elif any(word in query_lower for word in ['budget', 'cheap', 'backpack']):
            return "For budget travel, consider Southeast Asia, Eastern Europe, or Central America. Hostels, street food, and local transport can keep costs low."
        else:
            return "I'd be happy to help you plan your trip! Could you tell me more about your preferred destinations, travel style, or what kind of experiences you're looking for?"

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

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "total_documents": len(self.documents),
            "vector_index_ready": self.collection is not None,
            "bm25_index_ready": self.bm25 is not None,
            "active_conversations": len(self.conversation_memory),
            "embedding_model": "BAAI/bge-base-en-v1.5" if self.embedding_model else "Not available",
            "llm_model": "llama3-70b-8192" if self.llm else "Not available",
            "components_available": {
                "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
                "chromadb": CHROMADB_AVAILABLE,
                "bm25": BM25_AVAILABLE,
                "nltk": NLTK_AVAILABLE,
                "langchain": LANGCHAIN_AVAILABLE
            }
        }
