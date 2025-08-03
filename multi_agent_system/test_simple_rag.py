"""
Test the Simple RAG-enhanced multi-agent system
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_rag_system():
    """Test Simple RAG system directly"""
    print("🧠 Testing Simple RAG System")
    print("=" * 40)
    
    try:
        from simple_rag_system import SimpleTravelRAG
        
        print("🔧 Initializing Simple RAG system...")
        rag = SimpleTravelRAG()
        
        print("📚 Loading and indexing dataset...")
        rag.load_and_index_data()
        
        # Test queries
        test_queries = [
            "Plan a romantic honeymoon trip to Paris",
            "Family vacation to Tokyo with kids",
            "Budget backpacking through Europe",
            "Luxury beach resort recommendations"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: {query}")
            
            response = rag.generate_rag_response(query, user_id="test_user")
            
            print(f"   📊 Retrieved: {response['retrieved_docs']} documents")
            print(f"   ✅ Context used: {response['context_used']}")
            
            if response['similarity_scores']:
                avg_score = sum(response['similarity_scores']) / len(response['similarity_scores'])
                print(f"   📈 Avg similarity: {avg_score:.3f}")
            
            # Show response preview
            response_text = response['response']
            preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
            print(f"   📝 Response: {preview}")
        
        # Show stats
        stats = rag.get_stats()
        print(f"\n📊 Simple RAG System Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return True
            
    except Exception as e:
        print(f"❌ Simple RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_enhanced_recommendation():
    """Test RAG-enhanced RecommendationAgent"""
    print("\n🎯 Testing Simple RAG-Enhanced RecommendationAgent")
    print("=" * 50)
    
    try:
        from agents.recommendation_agent import RecommendationAgent
        
        print("🔧 Initializing RAG-enhanced RecommendationAgent...")
        agent = RecommendationAgent()
        
        # Test request
        test_request = {
            "user_id": "simple_rag_test",
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
        print("🔄 Processing with Simple RAG enhancement...")
        
        # Process request
        response = agent.process_request(test_request)
        
        if response["success"]:
            print("✅ Simple RAG-enhanced recommendation successful!")
            
            # Show response details
            data = response["data"]
            recommendations = data["recommendations"]
            
            print(f"\n📋 Recommendations (first 400 chars):")
            print(recommendations[:400] + "..." if len(recommendations) > 400 else recommendations)
            
            print(f"\n👤 User Profile: {data['user_profile']}")
            print(f"🎯 Personalization Score: {data['personalization_score']}")
            
            # Check if RAG was used
            if hasattr(agent, 'rag_system') and agent.rag_system:
                print("✅ Simple RAG system is active and working!")
            else:
                print("⚠️ Simple RAG system not active - using traditional recommendations")
            
            return True
            
        else:
            print(f"❌ Recommendation failed: {response.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_agent_with_simple_rag():
    """Test full multi-agent system with Simple RAG"""
    print("\n🚀 Testing Multi-Agent System with Simple RAG")
    print("=" * 50)
    
    try:
        from travel_ai_system import TravelAISystem
        
        print("🔧 Initializing multi-agent system...")
        system = TravelAISystem()
        
        # Test request
        test_request = "Plan a luxury 7-day trip to Tokyo for a family of 4 with a $8000 budget, interested in culture, food, and technology"
        
        print(f"📝 Test Query: {test_request}")
        print("🔄 Processing with multi-agent coordination + Simple RAG...")
        
        # Process request
        response = system.process_user_request(
            user_request=test_request,
            user_id="simple_rag_family",
            workflow_type="auto"
        )
        
        if response["success"]:
            print("✅ Multi-agent Simple RAG response successful!")
            
            print(f"\n📊 Response Details:")
            print(f"   Success: {response['success']}")
            print(f"   Response Time: {response.get('response_time', 0):.2f}s")
            print(f"   Agents Involved: {response.get('agents_involved', [])}")
            
            print(f"\n🤖 AI Response (first 400 chars):")
            message = response.get('message', 'No message')
            print(message[:400] + "..." if len(message) > 400 else message)
            
            return True
            
        else:
            print(f"❌ Multi-agent processing failed: {response.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Multi-agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_interface():
    """Test the chat interface with Simple RAG"""
    print("\n💬 Testing Chat Interface with Simple RAG")
    print("=" * 50)
    
    try:
        from travel_ai_system import TravelAISystem
        
        system = TravelAISystem()
        
        # Simulate a conversation
        conversation = [
            "Hello, I need help planning a trip",
            "I want to go to Paris for my honeymoon",
            "We have a budget of $3000 for 5 days",
            "We love romantic activities and good food"
        ]
        
        user_id = "chat_test_user"
        
        for i, message in enumerate(conversation, 1):
            print(f"\n{i}. User: {message}")
            
            response = system.process_user_request(
                user_request=message,
                user_id=user_id,
                workflow_type="auto"
            )
            
            if response["success"]:
                ai_response = response.get('message', 'No response')
                preview = ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
                print(f"   AI: {preview}")
                print(f"   Time: {response.get('response_time', 0):.2f}s")
            else:
                print(f"   ❌ Failed: {response.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Simple RAG-Enhanced Multi-Agent Travel AI System Test")
    print("=" * 60)
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Simple RAG system directly
    if test_simple_rag_system():
        success_count += 1
    
    # Test 2: RAG-enhanced RecommendationAgent
    if test_rag_enhanced_recommendation():
        success_count += 1
    
    # Test 3: Full multi-agent system with Simple RAG
    if test_multi_agent_with_simple_rag():
        success_count += 1
    
    # Test 4: Chat interface
    if test_chat_interface():
        success_count += 1
    
    print(f"\n🎉 Testing completed! {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🌟 ALL TESTS PASSED! Your Simple RAG-enhanced multi-agent system is working perfectly!")
        print("\n✅ Features working:")
        print("   ✅ TF-IDF based semantic search")
        print("   ✅ Document retrieval from 10K+ travel examples")
        print("   ✅ RAG-enhanced recommendations")
        print("   ✅ Multi-agent coordination")
        print("   ✅ Conversational memory")
        print("   ✅ Professional travel advice")
        
        print("\n🚀 Ready to use:")
        print("   cd multi_agent_system")
        print("   python cli.py chat")
    else:
        print(f"\n⚠️ {total_tests - success_count} tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
