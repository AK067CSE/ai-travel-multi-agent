"""
Chat Agent - Handles user interactions and conversations
"""

import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain_core.memory import ConversationBufferWindowMemory  # Not needed for our implementation
from .base_agent import BaseAgent, AgentResponse, agent_communication
import logging

logger = logging.getLogger(__name__)

class ChatAgent(BaseAgent):
    """Agent responsible for handling user conversations and interactions"""
    
    def __init__(self, model_name: str = "llama3-8b-8192"):
        super().__init__("ChatAgent", model_name)
        self.conversation_memory = {}
        self.user_sessions = {}
        self.intent_keywords = {
            "booking": ["book", "reserve", "reservation", "booking", "buy", "purchase"],
            "recommendation": ["recommend", "suggest", "advice", "best", "good", "help me choose"],
            "information": ["tell me", "what is", "how", "when", "where", "info", "information"],
            "scraping": ["find", "search", "look for", "get data", "scrape", "collect"],
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
            "goodbye": ["bye", "goodbye", "see you", "farewell", "thanks", "thank you"]
        }
    
    def create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for chat interactions"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a friendly and knowledgeable travel assistant chatbot. Your role is to:
            
            1. Engage users in natural, helpful conversations about travel
            2. Understand user intents and route requests to appropriate specialists
            3. Provide quick answers to common travel questions
            4. Maintain context throughout conversations
            5. Be enthusiastic about travel while being practical and helpful
            
            Conversation Guidelines:
            - Be warm, friendly, and professional
            - Ask clarifying questions when needed
            - Provide specific, actionable information
            - Acknowledge when you need to connect users with specialists
            - Remember previous conversation context
            - Use emojis appropriately to make conversations engaging
            
            Available Specialists:
            - Scraping Agent: For finding travel data and information
            - Recommendation Agent: For personalized travel suggestions
            - Booking Agent: For reservations and bookings
            
            Always prioritize user satisfaction and provide value in every interaction."""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Conversation History: {conversation_history}
            
            User Message: {user_message}
            
            Context: {context}
            
            Please respond helpfully and determine if this requires specialist assistance."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process chat request"""
        try:
            user_id = request.get("user_id", "anonymous")
            user_message = request.get("message", "")
            session_id = request.get("session_id", user_id)
            
            self.log_activity(f"Processing chat message", {
                "user_id": user_id,
                "session_id": session_id,
                "message_length": len(user_message)
            })
            
            # Initialize or get user session
            if session_id not in self.user_sessions:
                self.user_sessions[session_id] = {
                    "user_id": user_id,
                    "conversation_history": [],
                    "context": {},
                    "created_at": self._get_timestamp()
                }
            
            session = self.user_sessions[session_id]
            
            # Detect intent
            intent = self.detect_intent(user_message)
            
            # Add user message to history
            session["conversation_history"].append({
                "role": "user",
                "message": user_message,
                "timestamp": self._get_timestamp(),
                "intent": intent
            })
            
            # Generate response
            if intent in ["booking", "recommendation", "scraping"]:
                response = self.handle_specialist_request(session, user_message, intent)
            else:
                response = self.generate_chat_response(session, user_message)
            
            # Add assistant response to history
            session["conversation_history"].append({
                "role": "assistant",
                "message": response["message"],
                "timestamp": self._get_timestamp(),
                "intent": intent,
                "specialist_involved": response.get("specialist_involved", False)
            })
            
            # Limit conversation history
            if len(session["conversation_history"]) > 20:
                session["conversation_history"] = session["conversation_history"][-20:]
            
            agent_response = AgentResponse(
                agent_name=self.agent_name,
                success=True,
                data={
                    "message": response["message"],
                    "intent": intent,
                    "session_id": session_id,
                    "specialist_involved": response.get("specialist_involved", False),
                    "suggested_actions": response.get("suggested_actions", [])
                },
                metadata={
                    "user_id": user_id,
                    "conversation_turn": len(session["conversation_history"]) // 2
                }
            )
            
            return agent_response.to_dict()
            
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            response = AgentResponse(
                agent_name=self.agent_name,
                success=False,
                error=str(e)
            )
            return response.to_dict()
    
    def detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        message_lower = message.lower()
        
        # Check each intent category
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        
        return "general"
    
    def handle_specialist_request(self, session: Dict[str, Any], 
                                user_message: str, intent: str) -> Dict[str, Any]:
        """Handle requests that need specialist agents"""
        
        if intent == "recommendation":
            # Extract preferences from conversation
            preferences = self.extract_preferences_from_conversation(session["conversation_history"])
            
            # Send request to recommendation agent
            specialist_request = {
                "user_id": session["user_id"],
                "request": user_message,
                "preferences": preferences,
                "travel_data": session.get("context", {})
            }
            
            # In a real implementation, this would call the recommendation agent
            response_message = f"""🎯 I'd love to help you with personalized recommendations! 
            
Based on our conversation, I'm connecting you with our recommendation specialist who will provide detailed suggestions tailored to your preferences.

While I'm gathering that information, could you tell me:
- What's your approximate budget?
- What type of activities do you enjoy?
- Are you traveling solo, as a couple, or with a group?

This will help me provide the best recommendations for you! ✈️"""
            
            return {
                "message": response_message,
                "specialist_involved": True,
                "suggested_actions": ["Provide budget range", "Share activity preferences", "Specify group size"]
            }
        
        elif intent == "booking":
            response_message = f"""📋 I can help you with booking! 
            
I'm connecting you with our booking specialist who can assist with:
- Hotel reservations 🏨
- Flight bookings ✈️
- Travel packages 🎒
- Booking modifications and cancellations

What would you like to book today?"""
            
            return {
                "message": response_message,
                "specialist_involved": True,
                "suggested_actions": ["Specify booking type", "Provide travel dates", "Share destination"]
            }
        
        elif intent == "scraping":
            response_message = f"""🔍 I can help you find the latest travel information!
            
Our data specialist can gather current information about:
- Hotel prices and availability
- Tourist attractions and reviews
- Restaurant recommendations
- Local events and activities

What specific information are you looking for?"""
            
            return {
                "message": response_message,
                "specialist_involved": True,
                "suggested_actions": ["Specify destination", "Choose information type", "Set date range"]
            }
        
        return {"message": "I'm here to help! Could you provide more details about what you need?"}
    
    def generate_chat_response(self, session: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """Generate general chat response using LLM"""
        try:
            prompt_template = self.create_prompt_template()
            chain = self.create_chain(prompt_template)
            
            # Prepare conversation history
            history = session["conversation_history"][-10:]  # Last 10 messages
            history_text = "\n".join([
                f"{msg['role']}: {msg['message']}" for msg in history
            ])
            
            context = {
                "conversation_history": history_text,
                "user_message": user_message,
                "context": json.dumps(session.get("context", {}), indent=2)
            }
            
            response_text = chain.invoke(context)
            
            return {
                "message": response_text,
                "specialist_involved": False
            }
            
        except Exception as e:
            logger.error(f"Chat response generation error: {e}")
            return self.generate_fallback_response(user_message)
    
    def generate_fallback_response(self, user_message: str) -> Dict[str, Any]:
        """Generate fallback response when LLM fails"""
        intent = self.detect_intent(user_message)
        
        fallback_responses = {
            "greeting": "Hello! 👋 I'm your travel assistant. How can I help you plan your next adventure?",
            "goodbye": "Thank you for chatting with me! Have a wonderful day and safe travels! ✈️",
            "general": "I'm here to help with all your travel needs! You can ask me about destinations, bookings, recommendations, or any travel-related questions.",
            "information": "I'd be happy to help you with travel information! What would you like to know more about?"
        }
        
        return {
            "message": fallback_responses.get(intent, fallback_responses["general"]),
            "specialist_involved": False
        }
    
    def extract_preferences_from_conversation(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract user preferences from conversation history"""
        preferences = {
            "interests": [],
            "budget": "mid_range",
            "travel_style": "explorer",
            "group_size": 1
        }
        
        # Simple keyword extraction (in real implementation, use NLP)
        all_messages = " ".join([msg["message"] for msg in conversation_history if msg["role"] == "user"])
        all_messages_lower = all_messages.lower()
        
        # Extract interests
        interest_keywords = {
            "culture": ["culture", "museum", "history", "art", "heritage"],
            "food": ["food", "restaurant", "cuisine", "dining", "eat"],
            "adventure": ["adventure", "hiking", "climbing", "extreme", "sports"],
            "relaxation": ["relax", "spa", "beach", "peaceful", "quiet"],
            "nightlife": ["nightlife", "bar", "club", "party", "entertainment"]
        }
        
        for interest, keywords in interest_keywords.items():
            if any(keyword in all_messages_lower for keyword in keywords):
                preferences["interests"].append(interest)
        
        # Extract budget hints
        if any(word in all_messages_lower for word in ["cheap", "budget", "affordable"]):
            preferences["budget"] = "budget"
        elif any(word in all_messages_lower for word in ["luxury", "expensive", "premium"]):
            preferences["budget"] = "luxury"
        
        # Extract group size hints
        if any(word in all_messages_lower for word in ["couple", "two", "partner"]):
            preferences["group_size"] = 2
        elif any(word in all_messages_lower for word in ["family", "group", "friends"]):
            preferences["group_size"] = 4
        
        return preferences
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary for a session"""
        if session_id not in self.user_sessions:
            return {"error": "Session not found"}
        
        session = self.user_sessions[session_id]
        history = session["conversation_history"]
        
        return {
            "session_id": session_id,
            "user_id": session["user_id"],
            "message_count": len(history),
            "created_at": session["created_at"],
            "last_activity": history[-1]["timestamp"] if history else session["created_at"],
            "intents_detected": list(set([msg.get("intent", "general") for msg in history])),
            "specialists_involved": any(msg.get("specialist_involved", False) for msg in history)
        }
    
    def clear_session(self, session_id: str):
        """Clear a user session"""
        if session_id in self.user_sessions:
            del self.user_sessions[session_id]
            self.log_activity(f"Cleared session {session_id}")
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
