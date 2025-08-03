"""
Multi-Agent Travel AI System
Main interface for the travel assistant system
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.coordinator_agent import CoordinatorAgent
from agents.base_agent import agent_communication
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TravelAISystem:
    """Main Travel AI System with Multi-Agent Architecture"""
    
    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.system_status = "initialized"
        self.active_sessions = {}
        self.system_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
            "agents_performance": {}
        }
        
        logger.info("Travel AI System initialized with multi-agent architecture")
    
    def process_user_request(self, user_request: str, user_id: str = "anonymous", 
                           session_id: Optional[str] = None, 
                           workflow_type: str = "auto") -> Dict[str, Any]:
        """
        Process user request through the multi-agent system
        
        Args:
            user_request: User's travel request
            user_id: Unique user identifier
            session_id: Session identifier for conversation continuity
            workflow_type: Type of workflow (auto, simple, complex)
        
        Returns:
            System response with agent results
        """
        start_time = datetime.now()
        
        try:
            # Update metrics
            self.system_metrics["total_requests"] += 1
            
            # Create session if needed
            if not session_id:
                session_id = f"session_{user_id}_{len(self.active_sessions)}"
            
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    "user_id": user_id,
                    "created_at": start_time.isoformat(),
                    "request_count": 0,
                    "context": {}
                }
            
            session = self.active_sessions[session_id]
            session["request_count"] += 1
            session["last_activity"] = start_time.isoformat()
            
            # Prepare request for coordinator
            coordinator_request = {
                "request": user_request,
                "user_id": user_id,
                "session_id": session_id,
                "workflow_type": workflow_type,
                "context": session["context"]
            }
            
            # Process through coordinator
            result = self.coordinator.process_request(coordinator_request)
            
            # Update session context with results
            if result.get("success"):
                session["context"]["last_result"] = result["data"]
                self.system_metrics["successful_requests"] += 1
            else:
                self.system_metrics["failed_requests"] += 1
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_response_time_metric(response_time)
            
            # Format response for user
            user_response = self._format_user_response(result, response_time)
            
            logger.info(f"Processed request for user {user_id} in {response_time:.2f}s")
            
            return user_response
            
        except Exception as e:
            logger.error(f"System error processing request: {e}")
            self.system_metrics["failed_requests"] += 1
            
            return {
                "success": False,
                "message": "I apologize, but I encountered an error processing your request. Please try again.",
                "error": str(e),
                "session_id": session_id,
                "response_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _format_user_response(self, coordinator_result: Dict[str, Any], 
                            response_time: float) -> Dict[str, Any]:
        """Format coordinator response for end user"""
        
        if not coordinator_result.get("success"):
            return {
                "success": False,
                "message": "I'm sorry, I couldn't process your request right now. Please try again.",
                "error": coordinator_result.get("error"),
                "response_time": response_time
            }
        
        data = coordinator_result.get("data", {})
        result = data.get("result", {})
        
        # Extract the main response message
        main_message = self._extract_main_message(result)
        
        # Get additional information
        agents_involved = data.get("agents_involved", [])
        workflow_id = data.get("workflow_id")
        
        return {
            "success": True,
            "message": main_message,
            "workflow_id": workflow_id,
            "agents_involved": agents_involved,
            "response_time": response_time,
            "additional_data": self._extract_additional_data(result),
            "suggested_actions": self._extract_suggested_actions(result)
        }
    
    def _extract_main_message(self, result: Dict[str, Any]) -> str:
        """Extract main response message from agent results"""
        
        # Look for chat agent response first
        for agent_result in result.get("results", []):
            if agent_result.get("agent") == "ChatAgent":
                agent_data = agent_result.get("result", {}).get("data", {})
                if agent_data.get("message"):
                    return agent_data["message"]

        # Look for recommendation agent response
        for agent_result in result.get("results", []):
            if agent_result.get("agent") == "RecommendationAgent":
                agent_data = agent_result.get("result", {}).get("data", {})
                if agent_data.get("recommendations"):
                    return agent_data["recommendations"]

        # Look for any successful agent response
        for agent_result in result.get("results", []):
            if agent_result.get("success"):
                agent_data = agent_result.get("result", {}).get("data", {})
                # Try different possible response keys
                for key in ["message", "recommendations", "response", "content"]:
                    if agent_data.get(key):
                        return agent_data[key]
        
        # Look for booking agent response
        for agent_result in result.get("results", []):
            if agent_result.get("agent") == "BookingAgent":
                agent_data = agent_result.get("result", {}).get("data", {})
                if agent_data.get("confirmation_message"):
                    return agent_data["confirmation_message"]
        
        # Fallback message
        return "I've processed your request. How else can I help you with your travel plans?"
    
    def _extract_additional_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional data from agent results"""
        additional_data = {}
        
        for agent_result in result.get("results", []):
            agent_name = agent_result.get("agent")
            agent_data = agent_result.get("result", {}).get("data", {})
            
            if agent_name == "ScrapingAgent":
                additional_data["scraped_data"] = agent_data.get("scraped_data")
            elif agent_name == "RecommendationAgent":
                additional_data["personalization_score"] = agent_data.get("personalization_score")
                additional_data["user_profile"] = agent_data.get("user_profile")
            elif agent_name == "BookingAgent":
                additional_data["booking_details"] = agent_data.get("booking_details")
                additional_data["next_steps"] = agent_data.get("next_steps")
        
        return additional_data
    
    def _extract_suggested_actions(self, result: Dict[str, Any]) -> List[str]:
        """Extract suggested actions from agent results"""
        actions = []
        
        for agent_result in result.get("results", []):
            agent_data = agent_result.get("result", {}).get("data", {})
            if agent_data.get("suggested_actions"):
                actions.extend(agent_data["suggested_actions"])
        
        return list(set(actions))  # Remove duplicates
    
    def _update_response_time_metric(self, response_time: float):
        """Update average response time metric"""
        current_avg = self.system_metrics["average_response_time"]
        total_requests = self.system_metrics["total_requests"]
        
        # Calculate new average
        new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        self.system_metrics["average_response_time"] = new_avg
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics"""
        return {
            "status": self.system_status,
            "metrics": self.system_metrics,
            "active_sessions": len(self.active_sessions),
            "agents_status": {
                "coordinator": "active",
                "scraping": "active",
                "recommendation": "active", 
                "booking": "active",
                "chat": "active"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a specific session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "user_id": session["user_id"],
            "created_at": session["created_at"],
            "last_activity": session.get("last_activity"),
            "request_count": session["request_count"],
            "context_keys": list(session["context"].keys())
        }
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a user session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cleared session {session_id}")
            return True
        return False
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down Travel AI System")
        self.system_status = "shutdown"
        self.active_sessions.clear()

# Example usage functions
def create_travel_ai_system() -> TravelAISystem:
    """Create and initialize the travel AI system"""
    return TravelAISystem()

def demo_conversation():
    """Demo conversation with the travel AI system"""
    system = create_travel_ai_system()
    
    print("🤖 Travel AI System Demo")
    print("=" * 50)
    
    # Demo requests
    demo_requests = [
        "Hi! I want to plan a trip to Paris for 5 days",
        "I'm interested in art, culture, and good food. My budget is around $2000",
        "Can you recommend some hotels and restaurants?",
        "I'd like to book the hotel you recommended"
    ]
    
    user_id = "demo_user"
    session_id = None
    
    for i, request in enumerate(demo_requests, 1):
        print(f"\n👤 User: {request}")
        
        response = system.process_user_request(
            user_request=request,
            user_id=user_id,
            session_id=session_id
        )
        
        if not session_id:
            session_id = response.get("workflow_id", "demo_session")
        
        print(f"🤖 Assistant: {response['message']}")
        print(f"⚡ Response time: {response['response_time']:.2f}s")
        print(f"🔧 Agents involved: {', '.join(response.get('agents_involved', []))}")
        
        if response.get("suggested_actions"):
            print(f"💡 Suggestions: {', '.join(response['suggested_actions'])}")
    
    # Show system status
    print(f"\n📊 System Status:")
    status = system.get_system_status()
    print(f"Total requests: {status['metrics']['total_requests']}")
    print(f"Success rate: {status['metrics']['successful_requests']}/{status['metrics']['total_requests']}")
    print(f"Average response time: {status['metrics']['average_response_time']:.2f}s")

if __name__ == "__main__":
    demo_conversation()
