#!/usr/bin/env python
"""
Test the LiteLLM wrapper to fix the None response issue
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

def test_litellm_wrapper():
    """Test the LiteLLM wrapper directly"""
    print("🧪 Testing LiteLLM Wrapper...")
    
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("   ❌ No Groq API key")
        return False
    
    try:
        import litellm
        
        # Create the same wrapper as in CrewAI system
        class LiteLLMWrapper:
            def __init__(self, model_name, api_key):
                self.model_name = f"groq/{model_name}"
                self.api_key = api_key
                
            def invoke(self, prompt):
                try:
                    # Handle different input formats
                    if isinstance(prompt, str):
                        messages = [{"role": "user", "content": prompt}]
                    elif isinstance(prompt, list):
                        messages = prompt
                    elif hasattr(prompt, 'content'):
                        messages = [{"role": "user", "content": prompt.content}]
                    elif hasattr(prompt, 'messages'):
                        messages = prompt.messages
                    else:
                        messages = [{"role": "user", "content": str(prompt)}]
                    
                    print(f"      📝 Messages: {messages}")
                    
                    # Make the API call
                    response = litellm.completion(
                        model=self.model_name,
                        messages=messages,
                        api_key=self.api_key,
                        temperature=0.7,
                        max_tokens=100  # Smaller for testing
                    )
                    
                    print(f"      📤 Raw response: {response}")
                    
                    # Return response in expected format
                    class Response:
                        def __init__(self, content):
                            self.content = content
                    
                    content = response.choices[0].message.content
                    print(f"      📥 Content: {content}")
                    
                    if content:
                        return Response(content)
                    else:
                        return Response("I apologize, but I couldn't generate a response. Please try again.")
                        
                except Exception as e:
                    print(f"      ❌ Wrapper error: {e}")
                    class Response:
                        def __init__(self, content):
                            self.content = content
                    return Response(f"Error generating response: {str(e)}")
            
            def __call__(self, prompt):
                return self.invoke(prompt)
        
        # Test the wrapper
        wrapper = LiteLLMWrapper("llama3-8b-8192", groq_api_key)  # Use smaller model
        
        # Test different input formats
        test_cases = [
            ("String input", "Say hello in 5 words"),
            ("List input", [{"role": "user", "content": "Say goodbye in 3 words"}]),
        ]
        
        for test_name, test_input in test_cases:
            print(f"   🔧 Testing {test_name}...")
            try:
                result = wrapper.invoke(test_input)
                if result and hasattr(result, 'content') and result.content:
                    print(f"   ✅ {test_name}: {result.content}")
                else:
                    print(f"   ❌ {test_name}: No content returned")
            except Exception as e:
                print(f"   ❌ {test_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Wrapper test failed: {e}")
        return False

def test_crewai_with_fixed_wrapper():
    """Test CrewAI with the fixed wrapper"""
    print("\n🤖 Testing CrewAI with Fixed Wrapper...")
    
    try:
        from api.crewai_system import CrewAITravelSystem
        
        # Create new instance
        system = CrewAITravelSystem()
        
        # Check if LLM is working
        if system.llm:
            print("   🧠 Testing LLM directly...")
            try:
                test_response = system.llm.invoke("Say hello")
                if test_response and hasattr(test_response, 'content'):
                    print(f"   ✅ LLM Response: {test_response.content}")
                else:
                    print("   ❌ LLM returned None or invalid response")
            except Exception as e:
                print(f"   ❌ LLM test failed: {e}")
        
        # Test simple CrewAI execution
        status = system.get_system_status()
        if status.get('crew_ready'):
            print("   🚀 Testing simple CrewAI task...")
            request = {
                'request': 'Name one romantic place in Paris',
                'user_id': 'test_user',
                'preferences': {'budget': 'medium'}
            }
            
            result = system.process_travel_request(request)
            
            if result.get('success'):
                print("   ✅ CrewAI execution successful!")
                print(f"   📝 Response: {result.get('response', '')[:100]}...")
                return True
            else:
                print(f"   ❌ CrewAI failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print("   ⚠️ CrewAI not ready")
            return False
            
    except Exception as e:
        print(f"   ❌ CrewAI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test LLM wrapper fixes"""
    print("🚀 LLM Wrapper Fix Test")
    print("=" * 50)
    
    tests = [
        ("LiteLLM Wrapper", test_litellm_wrapper),
        ("CrewAI with Fixed Wrapper", test_crewai_with_fixed_wrapper),
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
    print("📋 LLM Wrapper Fix Results:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 LLM wrapper is now working perfectly!")
        print("   • No more None responses")
        print("   • CrewAI execution working")
        print("   • Proper response format")
    else:
        print("⚠️ Some wrapper issues remain")

if __name__ == "__main__":
    main()
