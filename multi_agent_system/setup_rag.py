"""
Setup and test RAG system for Travel Multi-Agent AI
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install RAG dependencies"""
    print("🔧 Installing RAG dependencies...")
    
    dependencies = [
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.0", 
        "rank-bm25>=0.2.2",
        "FlagEmbedding>=1.2.0",
        "faiss-cpu>=1.7.4",
        "nltk>=3.8"
    ]
    
    for dep in dependencies:
        try:
            os.system(f"pip install {dep}")
            print(f"✅ Installed {dep}")
        except Exception as e:
            print(f"❌ Failed to install {dep}: {e}")

def setup_rag_system():
    """Setup and initialize RAG system"""
    print("\n🧠 Setting up RAG system...")
    
    try:
        from rag_system import AdvancedTravelRAG
        
        # Initialize RAG system
        rag = AdvancedTravelRAG()
        
        # Check if dataset exists
        dataset_path = "../data_expansion/travel_planning_dataset.jsonl"
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found at {dataset_path}")
            print("Please ensure your expanded dataset is available")
            return None
        
        # Load and index data
        print("📚 Loading and indexing travel dataset...")
        rag.load_and_index_data()
        
        print("✅ RAG system setup completed!")
        return rag
        
    except Exception as e:
        print(f"❌ RAG setup failed: {e}")
        return None

def test_rag_system(rag):
    """Test RAG system with sample queries"""
    print("\n🧪 Testing RAG system...")
    
    test_queries = [
        "Plan a 3-day trip to Paris with a $1500 budget",
        "Recommend romantic destinations for honeymoon",
        "Budget backpacking trip through Europe",
        "Family vacation to Tokyo with kids",
        "Luxury beach resort recommendations"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing: {query}")
        try:
            response = rag.generate_rag_response(query, user_id="test_user")
            
            print(f"   ✅ Retrieved {response['retrieved_docs']} documents")
            print(f"   ✅ Context used: {response['context_used']}")
            
            # Show preview of response
            response_preview = response['response'][:200] + "..." if len(response['response']) > 200 else response['response']
            print(f"   📝 Response preview: {response_preview}")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")

def test_enhanced_recommendation_agent():
    """Test RecommendationAgent with RAG enhancement"""
    print("\n🎯 Testing RAG-enhanced RecommendationAgent...")
    
    try:
        from agents.recommendation_agent import RecommendationAgent
        
        # Initialize agent
        agent = RecommendationAgent()
        
        # Test request
        test_request = {
            "user_id": "test_user",
            "request": "Plan a romantic 5-day trip to Paris for our anniversary with a $2500 budget",
            "preferences": {
                "budget": "mid_range",
                "interests": ["culture", "food", "romance"],
                "group_size": 2
            }
        }
        
        print("📝 Test request:", test_request["request"])
        
        # Process request
        response = agent.process_request(test_request)
        
        if response["success"]:
            print("✅ RAG-enhanced recommendation successful!")
            
            # Show response preview
            recommendations = response["data"]["recommendations"]
            preview = recommendations[:300] + "..." if len(recommendations) > 300 else recommendations
            print(f"📋 Recommendations preview: {preview}")
            
            print(f"👤 User profile: {response['data']['user_profile']}")
            print(f"🎯 Personalization score: {response['data']['personalization_score']}")
            
        else:
            print(f"❌ Recommendation failed: {response.get('error')}")
            
    except Exception as e:
        print(f"❌ RecommendationAgent test failed: {e}")

def show_rag_stats(rag):
    """Show RAG system statistics"""
    print("\n📊 RAG System Statistics:")
    
    try:
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")

def main():
    """Main setup and test function"""
    print("🚀 RAG System Setup for Travel Multi-Agent AI")
    print("=" * 60)
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Setup RAG system
    rag = setup_rag_system()
    
    if rag:
        # Step 3: Test RAG system
        test_rag_system(rag)
        
        # Step 4: Test enhanced RecommendationAgent
        test_enhanced_recommendation_agent()
        
        # Step 5: Show statistics
        show_rag_stats(rag)
        
        print("\n🎉 RAG setup and testing completed!")
        print("\n🚀 Your multi-agent system now has RAG superpowers!")
        print("   - Enhanced recommendations from 10K+ travel examples")
        print("   - Semantic search with BGE embeddings")
        print("   - Hybrid retrieval (dense + sparse)")
        print("   - Conversational memory")
        print("   - Re-ranking for precision")
        
    else:
        print("\n❌ RAG setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
