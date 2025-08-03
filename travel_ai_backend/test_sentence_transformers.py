#!/usr/bin/env python
"""
Test sentence-transformers installation and functionality
"""

print("🧪 Testing sentence-transformers...")

# Test 1: Import
try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers imported successfully")
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"❌ sentence-transformers import failed: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Test 2: Model Loading
if SENTENCE_TRANSFORMERS_AVAILABLE:
    try:
        print("📥 Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded successfully")
        
        # Test 3: Encoding
        test_text = "This is a test sentence for embeddings"
        print("🔄 Testing encoding...")
        embeddings = model.encode([test_text])
        print(f"✅ Encoding successful! Shape: {embeddings.shape}")
        
        print("🎉 sentence-transformers is working perfectly!")
        
    except Exception as e:
        print(f"❌ Model loading/encoding failed: {e}")
        print("💡 This might be due to network issues or model download problems")
        
else:
    print("❌ Cannot test model loading - import failed")

print("\n" + "="*50)
print("DIAGNOSIS:")
if SENTENCE_TRANSFORMERS_AVAILABLE:
    print("✅ sentence-transformers is properly installed and working")
else:
    print("❌ sentence-transformers needs to be fixed")
    print("💡 Try: pip install --upgrade sentence-transformers")
