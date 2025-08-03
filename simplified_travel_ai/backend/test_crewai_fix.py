#!/usr/bin/env python
"""
Test CrewAI fix with proper LLM initialization
"""

import os
import sys
import django
from django.conf import settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

def test_crewai_llm_direct():
    """Test LLM initialization directly"""
    print("🧪 Testing LLM Initialization Methods...")
    
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("   ❌ No Groq API key")
        return False
    
    # Test different initialization methods
    methods = [
        ("ChatGroq with model parameter", lambda: test_chatgroq_model(groq_api_key)),
        ("ChatGroq with model_name parameter", lambda: test_chatgroq_model_name(groq_api_key)),
        ("LiteLLM direct", lambda: test_litellm_direct(groq_api_key)),
    ]
    
    for method_name, test_func in methods:
        try:
            print(f"   🔧 Testing {method_name}...")
            result = test_func()
            if result:
                print(f"   ✅ {method_name} works!")
                return True
            else:
                print(f"   ❌ {method_name} failed")
        except Exception as e:
            print(f"   ❌ {method_name} error: {e}")
    
    return False

def test_chatgroq_model(api_key):
    """Test ChatGroq with model parameter"""
    from langchain_groq import ChatGroq
    
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.7,
        max_tokens=100,
        groq_api_key=api_key
    )
    
    result = llm.invoke("Say hello in 5 words")
    print(f"      Response: {result.content}")
    return True

def test_chatgroq_model_name(api_key):
    """Test ChatGroq with model_name parameter"""
    from langchain_groq import ChatGroq
    
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.7,
        max_tokens=100,
        groq_api_key=api_key
    )
    
    result = llm.invoke("Say hello in 5 words")
    print(f"      Response: {result.content}")
    return True

def test_litellm_direct(api_key):
    """Test LiteLLM directly"""
    import litellm
    
    response = litellm.completion(
        model="groq/llama3-70b-8192",
        messages=[{"role": "user", "content": "Say hello in 5 words"}],
        api_key=api_key,
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"      Response: {response.choices[0].message.content}")
    return True

def test_crewai_fixed():
    """Test CrewAI with fixed LLM"""
    print("\n🤖 Testing Fixed CrewAI System...")
    try:
        from api.crewai_system import CrewAITravelSystem
        
        # Create new instance to test fixes
        system = CrewAITravelSystem()
        status = system.get_system_status()
        
        print(f"   📊 CrewAI Available: {'✅' if status.get('crewai_available') else '❌'}")
        print(f"   👥 Agents Created: {status.get('agents_created', 0)}")
        print(f"   🔧 Crew Ready: {'✅' if status.get('crew_ready') else '❌'}")
        print(f"   🧠 LLM Ready: {'✅' if system.llm else '❌'}")
        
        if status.get('crew_ready') and system.llm:
            # Test simple execution
            print("   🚀 Testing CrewAI execution...")
            request = {
                'request': 'Suggest one romantic place in Paris',
                'user_id': 'test_user',
                'preferences': {'budget': 'medium'}
            }
            
            result = system.process_travel_request(request)
            
            if result.get('success'):
                print("   ✅ CrewAI execution successful!")
                print(f"   ⏱️ Processing time: {result.get('processing_time', 0):.2f}s")
                print(f"   📝 Response: {result.get('response', '')[:100]}...")
                return True
            else:
                print(f"   ❌ CrewAI execution failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print("   ⚠️ CrewAI not ready for testing")
            return False
            
    except Exception as e:
        print(f"   ❌ CrewAI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_setup():
    """Test environment setup for CrewAI"""
    print("\n🔧 Testing Environment Setup...")
    
    groq_api_key = os.getenv('GROQ_API_KEY')
    print(f"   🔑 GROQ_API_KEY: {'✅ Set' if groq_api_key else '❌ Missing'}")
    
    # Set environment variable for LiteLLM
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        print("   ✅ Environment variable set for LiteLLM")
    
    # Test imports
    try:
        import crewai
        print(f"   📦 CrewAI: ✅ v{crewai.__version__}")
    except ImportError:
        print("   📦 CrewAI: ❌ Not available")
        return False
    
    try:
        import litellm
        print("   📦 LiteLLM: ✅ Available")
    except ImportError:
        print("   📦 LiteLLM: ❌ Not available")
        return False
    
    try:
        from langchain_groq import ChatGroq
        print("   📦 LangChain Groq: ✅ Available")
    except ImportError:
        print("   📦 LangChain Groq: ❌ Not available")
        return False
    
    return True

def main():
    """Run CrewAI fix tests"""
    print("🚀 CrewAI Fix Test")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("LLM Initialization", test_crewai_llm_direct),
        ("Fixed CrewAI System", test_crewai_fixed),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} Test Error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 CrewAI Fix Test Results:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 CrewAI is now working perfectly!")
        print("\n🌟 Fixed Issues:")
        print("   • LLM initialization with proper parameters")
        print("   • Environment variables configured")
        print("   • CrewAI execution working")
    elif passed >= 2:
        print("🔧 CrewAI mostly working - minor issues remain")
    else:
        print("⚠️ CrewAI still has issues - using fallback system")

if __name__ == "__main__":
    main()
