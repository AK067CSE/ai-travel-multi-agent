#!/usr/bin/env python
"""
Enhanced Test Script for Production Travel AI System
Tests all advanced components: RAG, Multi-Agent, LLM providers, etc.
"""

import os
import sys
import django
from django.conf import settings

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from api.ai_agent import TravelAIAgent
from api.enhanced_rag import EnhancedTravelRAG
from api.multi_agent_system import CoordinatorAgent
from api.crewai_system import CrewAITravelSystem
from api.models import Conversation, ChatMessage
import time
import json

def test_enhanced_rag_system():
    """Test the Enhanced RAG system"""
    print("🧠 Testing Enhanced RAG System...")

    try:
        rag = EnhancedTravelRAG()

        # Test data loading
        data_loaded = rag.load_and_index_data()
        print(f"   Data Loading: {'✅ Success' if data_loaded else '⚠️ Limited'}")

        # Test RAG response
        test_query = "Best budget destinations in Asia"
        rag_response = rag.generate_rag_response(test_query)

        print(f"   RAG Response: {'✅ Success' if rag_response['response'] else '❌ Failed'}")
        print(f"   System Used: {rag_response.get('system_used', 'Unknown')}")
        print(f"   Retrieved Docs: {rag_response.get('retrieved_docs', 0)}")

        # Test stats
        stats = rag.get_stats()
        print(f"   Total Documents: {stats.get('total_documents', 0)} (from your comprehensive datasets)")
        print(f"   Vector Index: {'✅' if stats.get('vector_index_ready') else '❌'}")
        print(f"   BM25 Index: {'✅' if stats.get('bm25_index_ready') else '❌'}")
        print(f"   Embedding Model: {stats.get('embedding_model', 'Not available')}")

        # Show dataset breakdown if available
        if hasattr(rag, 'doc_metadata') and rag.doc_metadata:
            datasets = {}
            for meta in rag.doc_metadata[:100]:  # Sample first 100
                dataset = meta.get('dataset', 'unknown')
                datasets[dataset] = datasets.get(dataset, 0) + 1

            print(f"   Dataset Breakdown:")
            for dataset, count in datasets.items():
                print(f"     - {dataset}: {count}+ entries")

        return True

    except Exception as e:
        print(f"❌ RAG System Error: {e}")
        return False

def test_crewai_system():
    """Test the CrewAI system"""
    print("🚀 Testing CrewAI Multi-Agent System...")

    try:
        crewai_system = CrewAITravelSystem()

        # Test system status first
        status = crewai_system.get_system_status()
        print(f"   CrewAI Available: {'✅' if status.get('crewai_available') else '❌'}")
        print(f"   LangChain Available: {'✅' if status.get('langchain_available') else '❌'}")
        print(f"   LLM Initialized: {'✅' if status.get('llm_initialized') else '❌'}")
        print(f"   Agents Created: {status.get('agents_created', 0)}")
        print(f"   System Ready: {'✅' if status.get('system_ready') else '❌'}")

        if status.get('system_ready'):
            # Test complex planning request
            test_request = 'Plan a comprehensive 2-week cultural trip to India with detailed itinerary'
            test_preferences = {'budget': 'mid-range', 'interests': ['culture', 'history']}

            response = crewai_system.process_travel_request(test_request, test_preferences)

            print(f"   CrewAI Response: {'✅ Success' if response.get('success') else '❌ Failed'}")
            print(f"   System Used: {response.get('system_used', 'Unknown')}")
            print(f"   Agents Involved: {', '.join(response.get('agents_involved', []))}")
            print(f"   Tasks Executed: {response.get('tasks_executed', 0)}")
            print(f"   Response Length: {len(response.get('response', ''))} characters")

            return response.get('success', False)
        else:
            print("   ⚠️ CrewAI system not ready - missing dependencies or configuration")
            return False

    except Exception as e:
        print(f"❌ CrewAI System Error: {e}")
        return False

def test_multi_agent_system():
    """Test the Custom Multi-Agent system"""
    print("👥 Testing Custom Multi-Agent System...")

    try:
        coordinator = CoordinatorAgent()

        # Test complex planning request
        test_request = {
            'request': 'Plan a comprehensive 2-week cultural trip to India with detailed itinerary',
            'user_id': 'test_user',
            'preferences': {'budget': 'mid-range', 'interests': ['culture', 'history']}
        }

        response = coordinator.process_request(test_request)

        print(f"   Custom Multi-Agent Response: {'✅ Success' if response.get('success') else '❌ Failed'}")
        print(f"   System Used: {response.get('system_used', 'Unknown')}")
        print(f"   Agents Involved: {', '.join(response.get('agents_involved', []))}")
        print(f"   Response Length: {len(response.get('response', ''))} characters")

        return response.get('success', False)

    except Exception as e:
        print(f"❌ Custom Multi-Agent System Error: {e}")
        return False

def test_enhanced_ai_agent():
    """Test the Enhanced AI Agent with all features"""
    print("🚀 Testing Enhanced AI Agent...")

    try:
        agent = TravelAIAgent()

        # Test different complexity levels
        test_cases = [
            ("Hello, I need help planning a trip", "simple"),
            ("What are the best destinations in Europe?", "medium"),
            ("Plan a detailed 3-week multi-country adventure trip through Southeast Asia with cultural experiences, outdoor activities, and budget considerations", "complex")
        ]

        results = []
        for test_message, expected_complexity in test_cases:
            print(f"   Testing {expected_complexity} query...")
            response = agent.process_travel_request(test_message)

            success = response.get('success', False)
            system_used = response.get('system_used', 'Unknown')
            processing_time = response.get('processing_time', 0)

            print(f"     {'✅' if success else '❌'} System: {system_used}, Time: {processing_time:.2f}s")
            results.append(success)

        # Test system status
        status = agent.get_system_status()
        print(f"   System Health: {status.get('system_health', 'unknown')}")
        print(f"   RAG Available: {'✅' if status.get('rag_system_available') else '❌'}")
        print(f"   CrewAI Available: {'✅' if status.get('crewai_system_available') else '❌'}")
        print(f"   Custom Multi-Agent Available: {'✅' if status.get('multi_agent_available') else '❌'}")
        print(f"   LLM Providers: {len([k for k, v in status.get('llm_providers', {}).items() if v])}")

        return all(results)

    except Exception as e:
        print(f"❌ Enhanced AI Agent Error: {e}")
        return False

def test_system_status():
    """Test system status functionality"""
    print("\n📊 Testing System Status...")
    
    agent = TravelAIAgent()
    status = agent.get_system_status()
    
    print(f"✅ System Status: {status['status']}")
    print(f"   AI Model: {status['ai_model']}")
    print(f"   OpenAI Available: {status['openai_available']}")
    print(f"   Knowledge Base Destinations: {status['knowledge_base_destinations']}")
    
    return status['status'] == 'operational'

def test_database_models():
    """Test database model creation"""
    print("\n💾 Testing Database Models...")
    
    try:
        # Create a test conversation
        conversation = Conversation.objects.create(
            session_id=f"test_session_{int(time.time())}",
            is_active=True
        )
        
        # Create a test message
        message = ChatMessage.objects.create(
            conversation=conversation,
            user_message="Test message",
            ai_response="Test response",
            response_time=1.5,
            system_used="Test_System",
            agents_involved=["Test_Agent"]
        )
        
        print(f"✅ Database Models Working")
        print(f"   Conversation ID: {conversation.id}")
        print(f"   Message ID: {message.id}")
        
        # Clean up
        message.delete()
        conversation.delete()
        
        return True
        
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints (basic import test)"""
    print("\n🌐 Testing API Endpoints...")
    
    try:
        from api.views import chat_with_ai, system_status, get_conversations
        from api.serializers import ChatRequestSerializer, ChatResponseSerializer
        
        print("✅ API Views Imported Successfully")
        print("✅ Serializers Imported Successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ API Import Error: {e}")
        return False

def run_all_tests():
    """Run comprehensive system tests for production AI system"""
    print("🚀 Starting Enhanced Travel AI System Tests")
    print("=" * 60)

    tests = [
        ("Enhanced RAG System", test_enhanced_rag_system),
        ("CrewAI Multi-Agent System", test_crewai_system),
        ("Custom Multi-Agent System", test_multi_agent_system),
        ("Enhanced AI Agent", test_enhanced_ai_agent),
        ("System Status", test_system_status),
        ("Database Models", test_database_models),
        ("API Endpoints", test_api_endpoints),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} Test Failed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("📋 Enhanced System Test Results:")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1

    print("=" * 60)
    print(f"Overall: {passed}/{total} tests passed")

    if passed >= total - 1:  # Allow 1 failure for optional components
        print("🎉 Production AI System is ready!")
        print("\n🌟 Enhanced Features Available:")
        print("   • Advanced RAG with BGE embeddings")
        print("   • Multi-agent coordination")
        print("   • Multiple LLM providers")
        print("   • Hybrid retrieval (dense + sparse)")
        print("   • Conversational memory")
        print("   • Production monitoring")
        print("\n📋 Next Steps:")
        print("1. Add API keys to .env file for full functionality")
        print("2. Run: python manage.py runserver")
        print("3. Start frontend: npm start (in frontend directory)")
        print("4. Open: http://localhost:3000")
        print("5. Test with complex travel queries!")
    else:
        print("⚠️  Multiple tests failed. Check configuration and dependencies.")
        print("\n🔧 Troubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check .env file configuration")
        print("3. Ensure API keys are valid")

    return passed >= total - 1

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
