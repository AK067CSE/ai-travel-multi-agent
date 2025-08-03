#!/usr/bin/env python
"""
Quick test of enhanced RAG system
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

print("🚀 Quick Enhanced RAG Test")
print("=" * 40)

try:
    from rag_system.enhanced_rag import EnhancedTravelRAG
    print("✅ Enhanced RAG imported successfully")
    
    # Initialize
    print("📥 Initializing Enhanced RAG...")
    rag = EnhancedTravelRAG()
    
    # Load data
    print("📊 Loading travel dataset...")
    rag.load_and_index_data()
    
    # Get stats
    stats = rag.get_enhanced_stats()
    print(f"✅ System ready!")
    print(f"   📄 Documents: {stats['total_documents']}")
    print(f"   🧩 Chunks: {stats['total_chunks']}")
    print(f"   🔍 Embedding: {stats['embedding_model']}")
    print(f"   🎯 Features: {stats.get('vector_features', 'N/A')}")
    print(f"   🤖 LLMs: {len(stats['available_llms'])}")
    
    # Test query
    print("\n🔍 Testing query...")
    result = rag.generate_enhanced_response("Plan a romantic Paris trip for 3 days")
    
    print(f"✅ Query successful!")
    print(f"   📝 Response length: {len(result['response'])} chars")
    print(f"   🔍 Retrieved: {result['retrieved_chunks']} chunks")
    print(f"   🎯 Reranked: {result['reranked_chunks']} chunks")
    print(f"   📊 Method: {result['retrieval_method']}")
    
    print(f"\n📝 Sample response:")
    print(f"   {result['response'][:200]}...")
    
    print(f"\n🎉 Enhanced RAG is working perfectly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 40)
