"""
Debug script to test individual agents
"""

import json
from agents.chat_agent import ChatAgent
from agents.recommendation_agent import RecommendationAgent

def test_chat_agent():
    print("🧪 Testing ChatAgent...")
    
    chat_agent = ChatAgent()
    
    request = {
        "user_id": "test_user",
        "message": "Plan a 3-day trip to Paris with a $1500 budget",
        "session_id": "test_session"
    }
    
    response = chat_agent.process_request(request)
    print("📋 ChatAgent Response:")
    print(json.dumps(response, indent=2))
    
    return response

def test_recommendation_agent():
    print("\n🧪 Testing RecommendationAgent...")
    
    rec_agent = RecommendationAgent()
    
    request = {
        "user_id": "test_user",
        "request": "Plan a 3-day trip to Paris with a $1500 budget",
        "preferences": {
            "budget": "mid_range",
            "interests": ["culture", "food", "sightseeing"],
            "duration": 3
        }
    }
    
    response = rec_agent.process_request(request)
    print("📋 RecommendationAgent Response:")
    print(json.dumps(response, indent=2))
    
    return response

if __name__ == "__main__":
    print("🔍 Debugging Agent Responses")
    print("=" * 50)
    
    chat_response = test_chat_agent()
    rec_response = test_recommendation_agent()
    
    print("\n✅ Debug complete!")
