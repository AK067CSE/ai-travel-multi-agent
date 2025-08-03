"""
Debug the coordinator workflow
"""

from agents.coordinator_agent import CoordinatorAgent

def test_coordinator():
    print("🧪 Testing Coordinator Workflow")
    print("=" * 50)
    
    coordinator = CoordinatorAgent()
    
    # Test request
    request = {
        "request": "Plan a 3-day trip to Paris with a $1500 budget",
        "user_id": "test_user",
        "session_id": "test_session",
        "workflow_type": "auto",
        "context": {}
    }
    
    print(f"📝 Original Request: {request['request']}")
    
    # Process through coordinator
    response = coordinator.process_request(request)
    
    print(f"\n📊 Coordinator Response:")
    print(f"Success: {response.get('success')}")
    
    if response.get('success'):
        data = response.get('data', {})
        result = data.get('result', {})

        print(f"\n🔍 Raw Response Data:")
        print(f"Data keys: {list(data.keys())}")
        print(f"Result keys: {list(result.keys())}")

        print(f"\n🔍 Workflow Results:")
        results = result.get('results', [])
        print(f"Number of results: {len(results)}")

        for i, agent_result in enumerate(results, 1):
            print(f"\n{i}. Agent: {agent_result.get('agent')}")
            print(f"   Task: {agent_result.get('task')}")
            print(f"   Success: {agent_result.get('success')}")

            if agent_result.get('result'):
                agent_data = agent_result['result'].get('data', {})
                print(f"   Response Keys: {list(agent_data.keys())}")

                # Show actual response content
                for key in ['message', 'recommendations', 'response']:
                    if key in agent_data:
                        content = agent_data[key]
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"   {key}: {preview}")
    else:
        print(f"❌ Coordinator failed: {response.get('error')}")
    
    print(f"\n✅ Debug completed!")

if __name__ == "__main__":
    test_coordinator()
