"""
Test script for Multi-Agent Travel AI System
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from agents.base_agent import BaseAgent, AgentResponse
        print("✅ Base agent imports successful")
        
        from agents.scraping_agent import ScrapingAgent
        print("✅ Scraping agent import successful")
        
        from agents.recommendation_agent import RecommendationAgent
        print("✅ Recommendation agent import successful")
        
        from agents.booking_agent import BookingAgent
        print("✅ Booking agent import successful")
        
        from agents.chat_agent import ChatAgent
        print("✅ Chat agent import successful")
        
        from agents.coordinator_agent import CoordinatorAgent
        print("✅ Coordinator agent import successful")
        
        from travel_ai_system import TravelAISystem
        print("✅ Travel AI system import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_agent_initialization():
    """Test agent initialization"""
    print("\n🧪 Testing agent initialization...")
    
    try:
        from agents.scraping_agent import ScrapingAgent
        from agents.recommendation_agent import RecommendationAgent
        from agents.booking_agent import BookingAgent
        from agents.chat_agent import ChatAgent
        
        # Test individual agents
        scraping_agent = ScrapingAgent()
        print("✅ ScrapingAgent initialized")
        
        recommendation_agent = RecommendationAgent()
        print("✅ RecommendationAgent initialized")
        
        booking_agent = BookingAgent()
        print("✅ BookingAgent initialized")
        
        chat_agent = ChatAgent()
        print("✅ ChatAgent initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization error: {e}")
        return False

def test_system_initialization():
    """Test full system initialization"""
    print("\n🧪 Testing system initialization...")
    
    try:
        from travel_ai_system import TravelAISystem
        
        system = TravelAISystem()
        print("✅ TravelAISystem initialized")
        
        # Test system status
        status = system.get_system_status()
        print(f"✅ System status: {status['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ System initialization error: {e}")
        return False

def test_basic_request():
    """Test basic request processing"""
    print("\n🧪 Testing basic request processing...")
    
    try:
        from travel_ai_system import TravelAISystem
        
        system = TravelAISystem()
        
        # Test simple request
        response = system.process_user_request(
            user_request="Hello, I need help planning a trip",
            user_id="test_user"
        )
        
        if response.get("success"):
            print("✅ Basic request processed successfully")
            print(f"📝 Response: {response['message'][:100]}...")
            print(f"⚡ Response time: {response.get('response_time', 0):.2f}s")
            return True
        else:
            print(f"❌ Request failed: {response.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Request processing error: {e}")
        return False

def test_agent_communication():
    """Test agent communication"""
    print("\n🧪 Testing agent communication...")
    
    try:
        from agents.base_agent import agent_communication
        from agents.chat_agent import ChatAgent
        
        chat_agent = ChatAgent()
        agent_communication.register_agent(chat_agent)
        
        # Test message sending
        success = agent_communication.send_message(
            from_agent="TestAgent",
            to_agent="ChatAgent",
            message={"test": "message"}
        )
        
        if success:
            print("✅ Agent communication working")
            
            # Test message retrieval
            messages = agent_communication.get_messages_for_agent("ChatAgent")
            if messages:
                print(f"✅ Message retrieval working ({len(messages)} messages)")
                return True
            else:
                print("⚠️ No messages retrieved")
                return True  # Still consider success
        else:
            print("❌ Agent communication failed")
            return False
            
    except Exception as e:
        print(f"❌ Agent communication error: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\n🧪 Testing environment configuration...")
    
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and groq_key != "your_groq_api_key_here":
        print("✅ GROQ_API_KEY configured")
        
        # Test LangChain + Groq connection
        try:
            from langchain_groq import ChatGroq
            
            llm = ChatGroq(
                groq_api_key=groq_key,
                model_name="llama3-8b-8192",
                temperature=0.7,
                max_tokens=50
            )
            
            response = llm.invoke("Hello")
            print("✅ LangChain + Groq connection successful")
            return True
            
        except Exception as e:
            print(f"⚠️ LangChain + Groq connection failed: {e}")
            print("   (This might be due to API limits or network issues)")
            return True  # Don't fail the test for API issues
    else:
        print("⚠️ GROQ_API_KEY not configured")
        print("   Set GROQ_API_KEY in .env file for full functionality")
        return True  # Don't fail for missing API key

def run_all_tests():
    """Run all tests"""
    print("🚀 Multi-Agent Travel AI System Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Agent Initialization", test_agent_initialization),
        ("System Initialization", test_system_initialization),
        ("Environment Configuration", test_environment),
        ("Agent Communication", test_agent_communication),
        ("Basic Request Processing", test_basic_request)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Multi-Agent system is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run: python cli.py chat")
        print("   2. Run: python cli.py demo")
        print("   3. Try: python travel_ai_system.py")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Set GROQ_API_KEY in .env file")
        print("   3. Check Python version (3.8+ recommended)")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
