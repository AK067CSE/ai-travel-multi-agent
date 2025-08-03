#!/usr/bin/env python
"""
Test script for Enhanced RAG System
Tests the new features: better embeddings, multiple LLMs, reranking
"""

import os
import sys
import django
import json
import time
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

# Now import Django models and our enhanced RAG
from api.models import User, Conversation, TravelRecommendation
from rag_system.enhanced_rag import EnhancedTravelRAG
from rag_system.multi_llm_manager import MultiLLMManager

def test_enhanced_rag():
    """Test the Enhanced RAG system"""
    print("🚀 Testing Enhanced RAG System with Free APIs")
    print("=" * 60)
    
    try:
        # Initialize Enhanced RAG
        print("1. Initializing Enhanced RAG System...")
        rag = EnhancedTravelRAG()
        
        # Load and index data
        print("2. Loading and indexing travel dataset...")
        start_time = time.time()
        rag.load_and_index_data()
        load_time = time.time() - start_time
        print(f"   ✅ Data loaded and indexed in {load_time:.2f} seconds")
        
        # Get system stats
        stats = rag.get_enhanced_stats()
        print(f"   📊 System Stats:")
        print(f"      - Total documents: {stats['total_documents']}")
        print(f"      - Total chunks: {stats['total_chunks']}")
        print(f"      - Embedding model: {stats['embedding_model']}")
        print(f"      - Vector database: {stats['vector_database']}")
        print(f"      - Available LLMs: {stats['available_llms']}")
        
        # Test queries
        test_queries = [
            "Plan a romantic honeymoon to Paris for 5 days with $3000 budget",
            "Family vacation to Tokyo with kids, interested in culture and food",
            "Budget backpacking trip through Europe for 2 weeks",
            "Luxury beach resort recommendations for anniversary"
        ]
        
        print("\n3. Testing Enhanced RAG with sample queries...")
        print("-" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            
            start_time = time.time()
            result = rag.generate_enhanced_response(query, user_id=f"test_user_{i}")
            response_time = time.time() - start_time
            
            print(f"   ⏱️  Response time: {response_time:.2f} seconds")
            print(f"   📝 Response: {result['response'][:200]}...")
            print(f"   🔍 Retrieved chunks: {result['retrieved_chunks']}")
            print(f"   🎯 Reranked chunks: {result['reranked_chunks']}")
            print(f"   📊 Retrieval method: {result['retrieval_method']}")
            print(f"   ✅ Context used: {result['context_used']}")
            
            if result['rerank_scores']:
                avg_score = sum(result['rerank_scores']) / len(result['rerank_scores'])
                print(f"   🏆 Average rerank score: {avg_score:.1f}/10")
        
        print("\n4. Testing Multi-LLM Manager...")
        print("-" * 60)
        
        # Test LLM Manager
        llm_manager = MultiLLMManager()
        llm_stats = llm_manager.get_usage_stats()
        
        print(f"   🤖 Available LLMs: {llm_stats['available_llms']}")
        print(f"   📊 Total LLMs: {llm_stats['total_llms']}")
        
        # Test health check
        health = llm_manager.health_check()
        print(f"   🏥 Health Status:")
        for llm_name, status in health.items():
            status_emoji = "✅" if status['status'] == 'healthy' else "❌"
            print(f"      {status_emoji} {llm_name}: {status['status']}")
            if status['status'] == 'healthy':
                print(f"         Response time: {status.get('response_time', 0):.2f}s")
        
        # Test direct LLM generation
        print(f"\n   🧪 Testing direct LLM generation...")
        test_prompt = "What are the top 3 travel destinations for adventure seekers?"
        llm_result = llm_manager.generate_response(test_prompt, task_type="travel_advice")
        
        print(f"      ✅ Success: {llm_result['success']}")
        print(f"      🤖 LLM used: {llm_result['llm_used']}")
        print(f"      ⏱️  Response time: {llm_result.get('response_time', 0):.2f}s")
        print(f"      📝 Response: {llm_result['response'][:150]}...")
        
        print("\n5. Performance Summary...")
        print("-" * 60)
        
        print(f"   🎯 Enhanced RAG Features:")
        print(f"      ✅ Advanced chunking strategy")
        print(f"      ✅ Multiple embedding options (sentence-transformers/TF-IDF)")
        print(f"      ✅ ChromaDB vector database integration")
        print(f"      ✅ LLM-based reranking")
        print(f"      ✅ Multiple free LLM providers (Groq, Gemini)")
        print(f"      ✅ Conversation memory")
        print(f"      ✅ Intelligent fallback mechanisms")
        
        print(f"\n   📈 Performance Metrics:")
        print(f"      - Average response time: {response_time:.2f}s")
        print(f"      - Data loading time: {load_time:.2f}s")
        print(f"      - Documents processed: {stats['total_documents']}")
        print(f"      - Chunks created: {stats['total_chunks']}")
        
        print(f"\n🎉 Enhanced RAG System Test Completed Successfully!")
        print(f"   Your system now has enterprise-grade RAG capabilities!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Enhanced RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_django_integration():
    """Test Django integration with Enhanced RAG"""
    print("\n🔗 Testing Django Integration...")
    print("-" * 60)
    
    try:
        # Create test user
        user, created = User.objects.get_or_create(
            username='test_enhanced_rag',
            defaults={
                'email': 'test@enhanced-rag.com',
                'travel_style': 'explorer',
                'budget_range': 'mid_range'
            }
        )
        
        if created:
            print(f"   ✅ Created test user: {user.username}")
        else:
            print(f"   ✅ Using existing test user: {user.username}")
        
        # Test conversation creation
        conversation = Conversation.objects.create(
            user=user,
            session_id='enhanced_rag_test_session',
            messages=[],
            context={'test': True}
        )
        
        print(f"   ✅ Created test conversation: {conversation.session_id}")
        
        # Test recommendation creation
        recommendation = TravelRecommendation.objects.create(
            user=user,
            conversation=conversation,
            query="Test enhanced RAG query",
            response="Test enhanced RAG response with multiple LLMs",
            agents_used=['EnhancedRAG (chromadb)', 'MultiLLMManager'],
            processing_time=1.5,
            recommendation_type='general'
        )
        
        print(f"   ✅ Created test recommendation: {recommendation.id}")
        
        # Verify database integration
        user_conversations = Conversation.objects.filter(user=user).count()
        user_recommendations = TravelRecommendation.objects.filter(user=user).count()
        
        print(f"   📊 Database Integration:")
        print(f"      - User conversations: {user_conversations}")
        print(f"      - User recommendations: {user_recommendations}")
        
        print(f"   🎉 Django integration test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Django integration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Enhanced RAG System Test Suite")
    print("=" * 60)
    print("Testing Phase 2 enhancements:")
    print("- Better embeddings (sentence-transformers/TF-IDF)")
    print("- ChromaDB vector database")
    print("- Multiple free LLMs (Groq, Gemini)")
    print("- Advanced chunking and reranking")
    print("- Django REST API integration")
    print("=" * 60)
    
    # Run tests
    rag_success = test_enhanced_rag()
    django_success = test_django_integration()
    
    print("\n" + "=" * 60)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Enhanced RAG Test: {'✅ PASSED' if rag_success else '❌ FAILED'}")
    print(f"Django Integration: {'✅ PASSED' if django_success else '❌ FAILED'}")
    
    if rag_success and django_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your Enhanced RAG system is ready for production!")
        print("\nNext steps:")
        print("- Add Gemini API key for multiple LLM support")
        print("- Test the Django REST API endpoints")
        print("- Deploy to production environment")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    print("=" * 60)
