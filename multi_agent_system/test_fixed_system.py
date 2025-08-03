"""
Test the fixed multi-agent system
"""

from travel_ai_system import TravelAISystem

def test_fixed_system():
    print("🧪 Testing Fixed Multi-Agent System")
    print("=" * 50)
    
    # Initialize system
    system = TravelAISystem()
    
    # Test request
    test_request = "Plan a 3-day trip to Paris with a $1500 budget"
    
    print(f"📝 Test Request: {test_request}")
    print("\n🔄 Processing...")
    
    # Process request
    response = system.process_user_request(
        user_request=test_request,
        user_id="test_user",
        workflow_type="auto"
    )
    
    print(f"\n📊 Response:")
    print(f"Success: {response.get('success')}")
    print(f"Response Time: {response.get('response_time', 0):.2f}s")
    print(f"Agents Involved: {response.get('agents_involved', [])}")
    
    print(f"\n🤖 Assistant Response:")
    print(response.get('message', 'No message'))
    
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    test_fixed_system()
