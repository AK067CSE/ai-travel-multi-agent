#!/usr/bin/env python
"""
Test CrewAI Travel System
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

print("🚀 Testing CrewAI Travel System")
print("=" * 50)

try:
    from agents.crew_ai_system import CrewAITravelSystem
    print("✅ CrewAI system imported successfully")
    
    # Initialize CrewAI system
    print("📥 Initializing CrewAI Travel System...")
    crew_system = CrewAITravelSystem()
    
    # Get system status
    status = crew_system.get_system_status()
    print(f"✅ CrewAI System Status:")
    print(f"   🤖 Agents: {status['agents_count']}")
    print(f"   🛠️  Tools: {status['tools_count']}")
    print(f"   🧠 LLM: {'✅' if status['llm_configured'] else '❌'}")
    print(f"   📊 Status: {status['status']}")
    print(f"   👥 Available Agents: {', '.join(status['agents'])}")
    
    # Test travel planning
    print(f"\n🎯 Testing CrewAI Travel Planning...")
    test_request = "Plan a romantic 3-day honeymoon to Paris with a $2500 budget"
    user_preferences = {
        "budget": "$2500",
        "duration": "3 days",
        "style": "romantic",
        "interests": ["culture", "dining", "sightseeing"]
    }
    
    print(f"   📝 Request: {test_request}")
    print(f"   👤 Preferences: {user_preferences}")
    
    # Create travel plan
    print(f"\n🔄 Creating travel plan with CrewAI agents...")
    result = crew_system.create_travel_plan(test_request, user_preferences)
    
    if result['success']:
        print(f"✅ CrewAI Travel Plan Created Successfully!")
        print(f"   🎯 Agents Involved: {', '.join(result['agents_involved'])}")
        print(f"   📊 Process: {result['process']}")
        print(f"   📝 Plan Preview: {result['travel_plan'][:300]}...")
    else:
        print(f"❌ CrewAI Travel Planning Failed:")
        print(f"   Error: {result.get('error', 'Unknown error')}")
        print(f"   Fallback: {result.get('fallback_message', 'No fallback')}")
    
    print(f"\n🎉 CrewAI Test Completed!")
    
except ImportError as e:
    print(f"❌ CrewAI import failed: {e}")
    print("💡 Make sure CrewAI is installed: pip install crewai")
    
except Exception as e:
    print(f"❌ CrewAI test error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 50)
