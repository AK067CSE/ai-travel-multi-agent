"""
Recommendation Agent - Provides personalized travel recommendations
"""

import json
import random
import os
import sys
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from .base_agent import BaseAgent, AgentResponse
import logging

# Import RAG system
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from simple_rag_system import SimpleTravelRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG system not available - using fallback recommendations")

logger = logging.getLogger(__name__)

class RecommendationAgent(BaseAgent):
    """Agent responsible for generating personalized travel recommendations"""
    
    def __init__(self, model_name: str = "llama3-70b-8192"):  # Use larger model for better recommendations
        super().__init__("RecommendationAgent", model_name)
        self.user_profiles = {}
        self.recommendation_history = {}

        # Initialize RAG system
        self.rag_system = None
        if RAG_AVAILABLE:
            try:
                self.rag_system = SimpleTravelRAG()
                # Load and index data if not already done
                if os.path.exists("../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl"):
                    self.rag_system.load_and_index_data()
                    logger.info("Simple RAG system initialized successfully")
                else:
                    logger.warning("Travel dataset not found - RAG system disabled")
                    self.rag_system = None
            except Exception as e:
                logger.error(f"Failed to initialize Simple RAG system: {e}")
                self.rag_system = None
        self.load_travel_knowledge()
    
    def create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for travel recommendations"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert travel advisor with deep knowledge of destinations worldwide. 
            Your role is to provide personalized, detailed travel recommendations based on:
            
            1. User preferences (budget, interests, travel style)
            2. Destination data (attractions, hotels, restaurants)
            3. Seasonal considerations and weather
            4. Cultural insights and local tips
            5. Safety and practical information
            
            Provide specific, actionable recommendations with:
            - Detailed itineraries
            - Budget breakdowns
            - Specific place names and addresses
            - Local tips and cultural insights
            - Alternative options for different budgets
            
            Be enthusiastic, helpful, and provide insider knowledge that makes trips memorable."""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """User Profile: {user_profile}
            
            Available Data: {travel_data}
            
            User Request: {user_request}
            
            Please provide personalized travel recommendations based on this information."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def load_travel_knowledge(self):
        """Load travel knowledge base"""
        self.travel_knowledge = {
            "destinations": {
                "Paris": {
                    "best_months": ["April", "May", "September", "October"],
                    "must_see": ["Eiffel Tower", "Louvre", "Notre-Dame", "Champs-Élysées"],
                    "hidden_gems": ["Sainte-Chapelle", "Père Lachaise", "Marché des Enfants Rouges"],
                    "budget_tips": ["Free museums first Sunday", "Picnic in parks", "Happy hour wine bars"],
                    "local_cuisine": ["Croissants", "Coq au vin", "Macarons", "French onion soup"]
                },
                "Tokyo": {
                    "best_months": ["March", "April", "May", "October", "November"],
                    "must_see": ["Senso-ji Temple", "Tokyo Skytree", "Shibuya Crossing", "Meiji Shrine"],
                    "hidden_gems": ["Omoide Yokocho", "Todoroki Valley", "Kiyosumi Gardens"],
                    "budget_tips": ["Convenience store meals", "Free temple visits", "100-yen shops"],
                    "local_cuisine": ["Sushi", "Ramen", "Tempura", "Takoyaki"]
                },
                "New York": {
                    "best_months": ["April", "May", "June", "September", "October"],
                    "must_see": ["Central Park", "Times Square", "Statue of Liberty", "Brooklyn Bridge"],
                    "hidden_gems": ["High Line", "The Cloisters", "Smorgasburg", "Roosevelt Island Tram"],
                    "budget_tips": ["Free Staten Island Ferry", "Happy hour specials", "Food trucks"],
                    "local_cuisine": ["Pizza", "Bagels", "Cheesecake", "Hot dogs"]
                }
            },
            "travel_styles": {
                "budget": {"daily_budget": 50, "accommodation": "hostels", "transport": "public"},
                "mid_range": {"daily_budget": 150, "accommodation": "3-star hotels", "transport": "mix"},
                "luxury": {"daily_budget": 400, "accommodation": "5-star hotels", "transport": "private"}
            }
        }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process recommendation request"""
        try:
            user_id = request.get("user_id", "anonymous")
            user_request = request.get("request", "")
            travel_data = request.get("travel_data", {})
            
            self.log_activity(f"Processing recommendation request", {
                "user_id": user_id,
                "request_type": "recommendation"
            })
            
            # Get or create user profile
            user_profile = self.get_user_profile(user_id, request.get("preferences", {}))
            
            # Generate recommendations using LLM
            recommendations = self.generate_recommendations(user_profile, travel_data, user_request)
            
            # Store recommendation in history
            self.store_recommendation_history(user_id, recommendations)
            
            response = AgentResponse(
                agent_name=self.agent_name,
                success=True,
                data={
                    "recommendations": recommendations,
                    "user_profile": user_profile,
                    "personalization_score": self.calculate_personalization_score(user_profile, recommendations)
                },
                metadata={"user_id": user_id, "recommendation_type": "personalized"}
            )
            
            return response.to_dict()
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            response = AgentResponse(
                agent_name=self.agent_name,
                success=False,
                error=str(e)
            )
            return response.to_dict()
    
    def get_user_profile(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "budget_range": preferences.get("budget", "mid_range"),
                "interests": preferences.get("interests", ["culture", "food", "sightseeing"]),
                "travel_style": preferences.get("travel_style", "explorer"),
                "group_size": preferences.get("group_size", 2),
                "age_group": preferences.get("age_group", "adult"),
                "dietary_restrictions": preferences.get("dietary_restrictions", []),
                "accessibility_needs": preferences.get("accessibility_needs", []),
                "previous_destinations": [],
                "preferred_activities": preferences.get("activities", [])
            }
        else:
            # Update existing profile with new preferences
            self.user_profiles[user_id].update(preferences)
        
        return self.user_profiles[user_id]
    
    def generate_recommendations(self, user_profile: Dict[str, Any],
                               travel_data: Dict[str, Any], user_request: str) -> str:
        """Generate personalized recommendations using RAG-enhanced LLM"""
        try:
            # Try RAG-enhanced recommendations first
            if self.rag_system:
                return self.generate_rag_recommendations(user_profile, travel_data, user_request)
            else:
                return self.generate_traditional_recommendations(user_profile, travel_data, user_request)

        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return self.generate_fallback_recommendations(user_profile, travel_data)

    def generate_rag_recommendations(self, user_profile: Dict[str, Any],
                                   travel_data: Dict[str, Any], user_request: str) -> str:
        """Generate recommendations using RAG system"""
        try:
            # Create enhanced query for RAG
            rag_query = self.create_rag_query(user_profile, user_request)

            # Get RAG response
            rag_response = self.rag_system.generate_rag_response(
                query=rag_query,
                user_id=user_profile.get("user_id", "anonymous")
            )

            # If RAG found relevant examples, use them
            if rag_response["context_used"] and rag_response["retrieved_docs"] > 0:
                logger.info(f"RAG enhanced recommendation with {rag_response['retrieved_docs']} examples")
                return rag_response["response"]
            else:
                # Fallback to traditional recommendations
                logger.info("RAG found no relevant examples, using traditional recommendations")
                return self.generate_traditional_recommendations(user_profile, travel_data, user_request)

        except Exception as e:
            logger.error(f"RAG recommendation error: {e}")
            return self.generate_traditional_recommendations(user_profile, travel_data, user_request)

    def create_rag_query(self, user_profile: Dict[str, Any], user_request: str) -> str:
        """Create optimized query for RAG system"""
        query_parts = [user_request]

        # Add user profile context to query
        if user_profile.get("budget_range"):
            query_parts.append(f"budget {user_profile['budget_range']}")

        if user_profile.get("interests"):
            interests = ", ".join(user_profile["interests"])
            query_parts.append(f"interests: {interests}")

        if user_profile.get("group_size"):
            query_parts.append(f"group size {user_profile['group_size']}")

        return " ".join(query_parts)

    def generate_traditional_recommendations(self, user_profile: Dict[str, Any],
                                           travel_data: Dict[str, Any], user_request: str) -> str:
        """Generate recommendations using traditional LLM approach"""
        try:
            prompt_template = self.create_prompt_template()
            chain = self.create_chain(prompt_template)

            # Prepare context
            context = {
                "user_profile": json.dumps(user_profile, indent=2),
                "travel_data": json.dumps(travel_data, indent=2),
                "user_request": user_request
            }

            # Add knowledge base context
            destination = self.extract_destination_from_request(user_request)
            if destination in self.travel_knowledge["destinations"]:
                context["travel_data"] += f"\n\nKnowledge Base for {destination}:\n"
                context["travel_data"] += json.dumps(
                    self.travel_knowledge["destinations"][destination], indent=2
                )

            recommendations = chain.invoke(context)
            return recommendations

        except Exception as e:
            logger.error(f"Traditional LLM recommendation error: {e}")
            return self.generate_fallback_recommendations(user_profile, travel_data)
    
    def extract_destination_from_request(self, request: str) -> str:
        """Extract destination from user request"""
        request_lower = request.lower()
        for destination in self.travel_knowledge["destinations"].keys():
            if destination.lower() in request_lower:
                return destination
        return ""
    
    def generate_fallback_recommendations(self, user_profile: Dict[str, Any], 
                                        travel_data: Dict[str, Any]) -> str:
        """Generate basic recommendations when LLM fails"""
        budget = user_profile.get("budget_range", "mid_range")
        interests = user_profile.get("interests", [])
        
        recommendations = f"""
        Based on your {budget} budget and interests in {', '.join(interests)}, here are some recommendations:
        
        🏨 Accommodation:
        - Look for {self.travel_knowledge['travel_styles'][budget]['accommodation']}
        - Budget: ${self.travel_knowledge['travel_styles'][budget]['daily_budget']}/day
        
        🎯 Activities:
        - Focus on {', '.join(interests)} related attractions
        - Consider local cultural experiences
        
        🍽️ Dining:
        - Try local specialties
        - Mix of restaurant types based on your budget
        
        🚗 Transportation:
        - Use {self.travel_knowledge['travel_styles'][budget]['transport']} transport
        """
        
        return recommendations
    
    def store_recommendation_history(self, user_id: str, recommendations: str):
        """Store recommendation in user history"""
        if user_id not in self.recommendation_history:
            self.recommendation_history[user_id] = []
        
        self.recommendation_history[user_id].append({
            "timestamp": self._get_timestamp(),
            "recommendations": recommendations
        })
        
        # Keep only last 10 recommendations
        if len(self.recommendation_history[user_id]) > 10:
            self.recommendation_history[user_id] = self.recommendation_history[user_id][-10:]
    
    def calculate_personalization_score(self, user_profile: Dict[str, Any], 
                                      recommendations: str) -> float:
        """Calculate how well recommendations match user profile"""
        score = 0.5  # Base score
        
        # Check if user interests are mentioned
        interests = user_profile.get("interests", [])
        for interest in interests:
            if interest.lower() in recommendations.lower():
                score += 0.1
        
        # Check budget considerations
        budget = user_profile.get("budget_range", "mid_range")
        if budget in recommendations.lower():
            score += 0.2
        
        return min(score, 1.0)
    
    def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's recommendation history"""
        return self.recommendation_history.get(user_id, [])

    def update_user_preferences(self, user_id: str, feedback: Dict[str, Any]):
        """Update user preferences based on feedback"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]

            # Update based on feedback
            if "liked_activities" in feedback:
                profile["interests"].extend(feedback["liked_activities"])
                profile["interests"] = list(set(profile["interests"]))  # Remove duplicates

            if "budget_feedback" in feedback:
                profile["budget_range"] = feedback["budget_feedback"]

            self.log_activity(f"Updated preferences for user {user_id}")

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
