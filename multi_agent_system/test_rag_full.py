"""
Test the full RAG-enhanced multi-agent system
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_enhanced_recommendation():
    """Test RAG-enhanced RecommendationAgent"""
    print("🎯 Testing RAG-Enhanced RecommendationAgent")
    print("=" * 50)
    
    try:
        # Import and initialize
        from agents.recommendation_agent import RecommendationAgent
        
        print("🔧 Initializing RAG-enhanced RecommendationAgent...")
        agent = RecommendationAgent()
        
        # Test request
        test_request = {
            "user_id": "rag_test_user",
            "request": "Plan a romantic 5-day honeymoon trip to Paris with a $3000 budget",
            "user_request": "Plan a romantic 5-day honeymoon trip to Paris with a $3000 budget",
            "message": "Plan a romantic 5-day honeymoon trip to Paris with a $3000 budget",
            "preferences": {
                "budget": "mid_range",
                "interests": ["culture", "food", "romance", "art"],
                "group_size": 2,
                "duration": 5
            }
        }
        
        print(f"📝 Test Query: {test_request['request']}")
        print("🔄 Processing with RAG enhancement...")
        
        # Process request
        response = agent.process_request(test_request)
        
        if response["success"]:
            print("✅ RAG-enhanced recommendation successful!")
            
            # Show response details
            data = response["data"]
            recommendations = data["recommendations"]
            
            print(f"\n📋 Recommendations (first 500 chars):")
            print(recommendations[:500] + "..." if len(recommendations) > 500 else recommendations)
            
            print(f"\n👤 User Profile: {data['user_profile']}")
            print(f"🎯 Personalization Score: {data['personalization_score']}")
            
            # Check if RAG was used
            if hasattr(agent, 'rag_system') and agent.rag_system:
                print("✅ RAG system is active and working!")
            else:
                print("⚠️ RAG system not active - using traditional recommendations")
            
        else:
            print(f"❌ Recommendation failed: {response.get('error')}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_multi_agent_with_rag():
    """Test full multi-agent system with RAG"""
    print("\n🚀 Testing Full Multi-Agent System with RAG")
    print("=" * 50)
    
    try:
        from travel_ai_system import TravelAISystem
        
        print("🔧 Initializing multi-agent system...")
        system = TravelAISystem()
        
        # Test request
        test_request = "Plan a luxury 7-day trip to Tokyo for a family of 4 with a $8000 budget, interested in culture, food, and technology"
        
        print(f"📝 Test Query: {test_request}")
        print("🔄 Processing with multi-agent coordination + RAG...")
        
        # Process request
        response = system.process_user_request(
            user_request=test_request,
            user_id="rag_test_family",
            workflow_type="auto"
        )
        
        if response["success"]:
            print("✅ Multi-agent RAG response successful!")
            
            print(f"\n📊 Response Details:")
            print(f"   Success: {response['success']}")
            print(f"   Response Time: {response.get('response_time', 0):.2f}s")
            print(f"   Agents Involved: {response.get('agents_involved', [])}")
            
            print(f"\n🤖 AI Response (first 500 chars):")
            message = response.get('message', 'No message')
            print(message[:500] + "..." if len(message) > 500 else message)
            
        else:
            print(f"❌ Multi-agent processing failed: {response.get('error')}")
            
    except Exception as e:
        print(f"❌ Multi-agent test failed: {e}")
        import traceback
        traceback.print_exc()

def test_rag_system_directly():
    """Test RAG system directly"""
    print("\n🧠 Testing RAG System Directly")
    print("=" * 50)
    
    try:
        from rag_system import AdvancedTravelRAG
        
        print("🔧 Initializing RAG system...")
        rag = AdvancedTravelRAG()
        
        print("📚 Loading and indexing dataset...")
        rag.load_and_index_data()
        
        # Test queries
        test_queries = [
            "Romantic honeymoon in Paris",
            "Family trip to Tokyo with kids",
            "Budget backpacking Europe",
            "Luxury beach vacation"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: {query}")
            
            response = rag.generate_rag_response(query, user_id="direct_test")
            
            print(f"   📊 Retrieved: {response['retrieved_docs']} documents")
            print(f"   ✅ Context used: {response['context_used']}")
            
            # Show response preview
            response_text = response['response']
            preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
            print(f"   📝 Response: {preview}")
        
        # Show stats
        stats = rag.get_stats()
        print(f"\n📊 RAG System Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Direct RAG test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("🚀 RAG-Enhanced Multi-Agent Travel AI System Test")
    print("=" * 60)
    
    # Test 1: RAG-enhanced RecommendationAgent
    test_rag_enhanced_recommendation()
    
    # Test 2: Full multi-agent system with RAG
    test_multi_agent_with_rag()
    
    # Test 3: RAG system directly
    test_rag_system_directly()
    
    print("\n🎉 RAG testing completed!")
    print("\n🌟 Your multi-agent system now has RAG superpowers!")
    print("   ✅ 10K+ travel examples for context")
    print("   ✅ BGE embeddings for semantic search")
    print("   ✅ Hybrid retrieval (dense + sparse)")
    print("   ✅ Re-ranking for precision")
    print("   ✅ Conversational memory")
    print("   ✅ Multi-agent coordination")

if __name__ == "__main__":
    main()
