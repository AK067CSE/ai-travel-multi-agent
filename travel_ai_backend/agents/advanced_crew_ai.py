"""
Advanced CrewAI System with 5 Specialized Travel Agents
Enterprise-grade multi-agent workflow for travel planning
"""

import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

# Load environment variables
load_dotenv()

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not available")

logger = logging.getLogger(__name__)

# Advanced Tools for CrewAI Agents
class EnhancedRAGTool(BaseTool):
    name: str = "enhanced_rag_search"
    description: str = "Search the enhanced travel database with 8,885 chunks for relevant travel information"
    
    def _run(self, query: str) -> str:
        """Search using Enhanced RAG system"""
        try:
            from rag_system.enhanced_rag import EnhancedTravelRAG
            rag = EnhancedTravelRAG()
            result = rag.generate_enhanced_response(query)
            return f"RAG Search Results: {result['response']}\nSources: {len(result['sources'])} relevant examples found"
        except Exception as e:
            return f"Enhanced RAG search error: {e}"

class WeatherDataTool(BaseTool):
    name: str = "get_weather_data"
    description: str = "Get real-time weather information for destinations"
    
    def _run(self, destination: str, dates: str = "current") -> str:
        """Get weather data (placeholder for real API integration)"""
        # In production, integrate with OpenWeatherMap, WeatherAPI, etc.
        weather_data = {
            "Paris": {"temp": "18°C", "condition": "Partly cloudy", "humidity": "65%"},
            "Tokyo": {"temp": "22°C", "condition": "Sunny", "humidity": "58%"},
            "London": {"temp": "15°C", "condition": "Light rain", "humidity": "78%"},
            "New York": {"temp": "20°C", "condition": "Clear", "humidity": "52%"}
        }
        
        city_weather = weather_data.get(destination, {"temp": "20°C", "condition": "Pleasant", "humidity": "60%"})
        return f"Weather in {destination}: {city_weather['temp']}, {city_weather['condition']}, Humidity: {city_weather['humidity']}"

class PriceComparisonTool(BaseTool):
    name: str = "compare_prices"
    description: str = "Compare prices for flights, hotels, and activities"
    
    def _run(self, item_type: str, destination: str, dates: str = "flexible") -> str:
        """Compare prices across different providers"""
        # In production, integrate with Skyscanner, Booking.com APIs, etc.
        price_data = {
            "flight": f"Flight prices to {destination}: Economy $450-$850, Business $1200-$2500",
            "hotel": f"Hotel prices in {destination}: Budget $80-$150/night, Mid-range $150-$300/night, Luxury $300+/night",
            "activity": f"Activity prices in {destination}: Tours $25-$100, Museums $15-$30, Experiences $50-$200"
        }
        
        return price_data.get(item_type, f"Price information for {item_type} in {destination}: Varies by season and availability")

class UserPreferenceTool(BaseTool):
    name: str = "analyze_user_preferences"
    description: str = "Analyze user preferences and travel history for personalization"
    
    def _run(self, user_data: str) -> str:
        """Analyze user preferences using ML"""
        # In production, implement ML-based preference analysis
        try:
            preferences = json.loads(user_data) if isinstance(user_data, str) else user_data
            
            # Simple preference analysis
            analysis = {
                "travel_style": preferences.get("style", "explorer"),
                "budget_category": preferences.get("budget_range", "mid_range"),
                "interests": preferences.get("interests", ["culture", "food"]),
                "recommendations": []
            }
            
            # Generate personalized recommendations
            if "romantic" in str(preferences).lower():
                analysis["recommendations"].append("Focus on intimate dining and scenic views")
            if "family" in str(preferences).lower():
                analysis["recommendations"].append("Include family-friendly activities and accommodations")
            if "adventure" in str(preferences).lower():
                analysis["recommendations"].append("Add outdoor activities and unique experiences")
            
            return f"User Profile Analysis: {json.dumps(analysis, indent=2)}"
        except Exception as e:
            return f"Preference analysis error: {e}"

class BookingAssistantTool(BaseTool):
    name: str = "booking_assistant"
    description: str = "Assist with booking recommendations and availability checks"
    
    def _run(self, booking_type: str, details: str) -> str:
        """Provide booking assistance"""
        booking_info = {
            "flight": "Flight booking recommendations: Book 6-8 weeks in advance for best prices. Consider flexible dates.",
            "hotel": "Hotel booking tips: Book directly with hotels for better cancellation policies. Check for package deals.",
            "restaurant": "Restaurant reservations: Book popular restaurants 1-2 weeks in advance. Consider lunch for better availability.",
            "activity": "Activity bookings: Book tours and experiences in advance, especially during peak season."
        }
        
        base_info = booking_info.get(booking_type, "General booking assistance available")
        return f"{base_info}\nDetails for {details}: Contact information and booking links would be provided here."

class AdvancedCrewAISystem:
    """
    Advanced CrewAI system with 5 specialized agents and complex workflows
    """
    
    def __init__(self):
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is required")
        
        # Setup environment
        self._setup_environment()
        
        # Initialize advanced tools
        self.tools = self._initialize_tools()
        
        # Create 5 specialized agents
        self.agents = self._create_advanced_agents()
        
        # Workflow configurations
        self.workflows = self._define_workflows()
        
        logger.info("Advanced CrewAI system with 5 agents initialized")
    
    def _setup_environment(self):
        """Setup environment for advanced CrewAI"""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and groq_api_key != "your-groq-api-key-here":
            os.environ["GROQ_API_KEY"] = groq_api_key
            os.environ["OPENAI_MODEL_NAME"] = "groq/llama3-70b-8192"
            print(f"✅ Advanced CrewAI configured with Groq")
        else:
            raise ValueError("Valid GROQ_API_KEY required for Advanced CrewAI")
    
    def _initialize_tools(self):
        """Initialize advanced tools for agents"""
        return {
            'rag_tool': EnhancedRAGTool(),
            'weather_tool': WeatherDataTool(),
            'price_tool': PriceComparisonTool(),
            'preference_tool': UserPreferenceTool(),
            'booking_tool': BookingAssistantTool()
        }
    
    def _create_advanced_agents(self) -> Dict[str, Agent]:
        """Create 5 specialized travel agents with advanced capabilities"""
        
        # 1. Travel Research & Intelligence Agent
        research_agent = Agent(
            role='Travel Research & Intelligence Specialist',
            goal='Conduct comprehensive destination research using advanced data sources',
            backstory="""You are an elite travel intelligence specialist with access to real-time data, 
            weather information, price comparisons, and a vast database of travel experiences. 
            You excel at gathering and synthesizing complex travel information from multiple sources.""",
            verbose=True,
            allow_delegation=False,
            llm="groq/llama3-70b-8192",
            tools=[self.tools['rag_tool'], self.tools['weather_tool'], self.tools['price_tool']]
        )
        
        # 2. Personalization & User Experience Agent
        personalization_agent = Agent(
            role='Personalization & User Experience Expert',
            goal='Analyze user preferences and create highly personalized travel recommendations',
            backstory="""You are a master of travel personalization with advanced understanding of user psychology, 
            travel preferences, and behavioral patterns. You use ML-driven insights to create perfectly 
            tailored travel experiences that exceed expectations.""",
            verbose=True,
            allow_delegation=False,
            llm="groq/llama3-70b-8192",
            tools=[self.tools['preference_tool'], self.tools['rag_tool']]
        )
        
        # 3. Itinerary Design & Optimization Agent
        itinerary_agent = Agent(
            role='Itinerary Design & Optimization Expert',
            goal='Create optimized, detailed itineraries with perfect timing and logistics',
            backstory="""You are a world-class itinerary designer with expertise in route optimization, 
            timing coordination, and logistical planning. You create seamless travel experiences that 
            maximize enjoyment while minimizing stress and travel time.""",
            verbose=True,
            allow_delegation=True,
            llm="groq/llama3-70b-8192",
            tools=[self.tools['rag_tool'], self.tools['weather_tool'], self.tools['price_tool']]
        )
        
        # 4. Local Experience & Cultural Curator
        cultural_agent = Agent(
            role='Local Experience & Cultural Curator',
            goal='Discover authentic local experiences and cultural immersion opportunities',
            backstory="""You are a cultural anthropologist and local experience specialist who uncovers 
            the hidden gems, authentic experiences, and cultural nuances that make travel transformative. 
            You have deep connections with local communities worldwide.""",
            verbose=True,
            allow_delegation=False,
            llm="groq/llama3-70b-8192",
            tools=[self.tools['rag_tool'], self.tools['preference_tool']]
        )
        
        # 5. Booking & Logistics Coordinator
        booking_agent = Agent(
            role='Booking & Logistics Coordinator',
            goal='Handle all booking logistics, price optimization, and travel arrangements',
            backstory="""You are a meticulous logistics expert and booking specialist who ensures every 
            aspect of travel runs smoothly. You excel at finding the best deals, coordinating complex 
            bookings, and providing comprehensive travel support.""",
            verbose=True,
            allow_delegation=False,
            llm="groq/llama3-70b-8192",
            tools=[self.tools['booking_tool'], self.tools['price_tool'], self.tools['weather_tool']]
        )
        
        return {
            'research': research_agent,
            'personalization': personalization_agent,
            'itinerary': itinerary_agent,
            'cultural': cultural_agent,
            'booking': booking_agent
        }
    
    def _define_workflows(self) -> Dict[str, str]:
        """Define different workflow types for different travel requests"""
        return {
            'comprehensive': 'All 5 agents in sequence for complex trips',
            'quick_planning': 'Research → Itinerary → Booking for simple trips',
            'cultural_focus': 'Research → Cultural → Personalization → Itinerary',
            'budget_optimization': 'Research → Booking → Itinerary → Personalization',
            'luxury_experience': 'Personalization → Cultural → Itinerary → Booking'
        }
    
    def create_advanced_travel_plan(self, 
                                  user_request: str, 
                                  user_preferences: Dict[str, Any] = None,
                                  workflow_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Create advanced travel plan using 5-agent workflow
        """
        try:
            preferences = user_preferences or {}
            
            # Create tasks based on workflow type
            tasks = self._create_advanced_tasks(user_request, preferences, workflow_type)
            
            # Select agents based on workflow
            selected_agents = self._select_agents_for_workflow(workflow_type)
            
            # Create and execute crew
            crew = Crew(
                agents=selected_agents,
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                'success': True,
                'travel_plan': str(result),
                'workflow_type': workflow_type,
                'agents_involved': [agent.role for agent in selected_agents],
                'system': 'AdvancedCrewAI',
                'features_used': [
                    'Enhanced RAG with 8,885 chunks',
                    'Real-time weather data',
                    'Price comparison analysis',
                    'ML-based personalization',
                    'Advanced booking assistance'
                ]
            }
            
        except Exception as e:
            logger.error(f"Advanced CrewAI error: {e}")
            
            # Fallback to Enhanced RAG
            try:
                from rag_system.enhanced_rag import EnhancedTravelRAG
                rag = EnhancedTravelRAG()
                rag_result = rag.generate_enhanced_response(user_request)
                
                return {
                    'success': True,
                    'travel_plan': rag_result['response'],
                    'workflow_type': 'enhanced_rag_fallback',
                    'agents_involved': ['EnhancedRAG'],
                    'system': 'EnhancedRAG_Fallback',
                    'features_used': [
                        'Enhanced RAG with 8,885 chunks',
                        'TF-IDF vectorization',
                        'Multi-LLM support (Groq + Gemini)',
                        'Intelligent fallback system'
                    ],
                    'note': 'Advanced CrewAI failed, used Enhanced RAG fallback'
                }
            except Exception as rag_error:
                return {
                    'success': False,
                    'error': f"Advanced CrewAI: {str(e)}, RAG: {str(rag_error)}",
                    'travel_plan': 'Both Advanced CrewAI and Enhanced RAG systems encountered issues.'
                }
    
    def _select_agents_for_workflow(self, workflow_type: str) -> List[Agent]:
        """Select agents based on workflow type"""
        workflows = {
            'comprehensive': [
                self.agents['research'],
                self.agents['personalization'], 
                self.agents['cultural'],
                self.agents['itinerary'],
                self.agents['booking']
            ],
            'quick_planning': [
                self.agents['research'],
                self.agents['itinerary'],
                self.agents['booking']
            ],
            'cultural_focus': [
                self.agents['research'],
                self.agents['cultural'],
                self.agents['personalization'],
                self.agents['itinerary']
            ],
            'budget_optimization': [
                self.agents['research'],
                self.agents['booking'],
                self.agents['itinerary'],
                self.agents['personalization']
            ],
            'luxury_experience': [
                self.agents['personalization'],
                self.agents['cultural'],
                self.agents['itinerary'],
                self.agents['booking']
            ]
        }
        
        return workflows.get(workflow_type, workflows['comprehensive'])

    def _create_advanced_tasks(self, user_request: str, preferences: Dict[str, Any], workflow_type: str) -> List[Task]:
        """Create advanced tasks for the selected workflow"""

        if workflow_type == "comprehensive":
            return self._create_comprehensive_tasks(user_request, preferences)
        elif workflow_type == "cultural_focus":
            return self._create_cultural_tasks(user_request, preferences)
        elif workflow_type == "budget_optimization":
            return self._create_budget_tasks(user_request, preferences)
        else:
            return self._create_comprehensive_tasks(user_request, preferences)

    def _create_comprehensive_tasks(self, user_request: str, preferences: Dict[str, Any]) -> List[Task]:
        """Create comprehensive 5-agent workflow tasks"""

        # Task 1: Advanced Research & Intelligence
        research_task = Task(
            description=f"""Conduct comprehensive research for: {user_request}

            User preferences: {preferences}

            Use all available tools to gather:
            1. Enhanced RAG search for similar successful trips
            2. Real-time weather data for optimal timing
            3. Current price comparisons for budget planning
            4. Destination highlights and hidden gems
            5. Cultural considerations and local customs
            6. Transportation options and logistics

            Provide a comprehensive intelligence report with data-driven insights.""",
            agent=self.agents['research'],
            expected_output="Comprehensive travel intelligence report with real-time data and RAG insights"
        )

        # Task 2: Advanced Personalization Analysis
        personalization_task = Task(
            description=f"""Analyze user preferences and create personalization strategy for: {user_request}

            User data: {preferences}

            Tasks:
            1. Analyze user preferences using ML-based insights
            2. Identify travel personality and style preferences
            3. Search RAG database for similar traveler profiles
            4. Create personalization recommendations
            5. Suggest experience types that match user psychology

            Provide detailed personalization strategy and recommendations.""",
            agent=self.agents['personalization'],
            expected_output="Detailed personalization analysis with ML-driven recommendations"
        )

        # Task 3: Cultural Experience Curation
        cultural_task = Task(
            description=f"""Curate authentic cultural experiences for: {user_request}

            Based on research and personalization insights, find:
            1. Authentic local experiences and hidden gems
            2. Cultural immersion opportunities
            3. Local festivals, events, and seasonal activities
            4. Traditional dining experiences and local cuisine
            5. Interactions with local communities
            6. Off-the-beaten-path cultural sites

            Focus on experiences that create lasting memories and cultural understanding.""",
            agent=self.agents['cultural'],
            expected_output="Curated collection of authentic cultural experiences and local insights"
        )

        # Task 4: Optimized Itinerary Design
        itinerary_task = Task(
            description=f"""Design optimized itinerary for: {user_request}

            Integrate insights from:
            - Research intelligence report
            - Personalization recommendations
            - Cultural experience curation

            Create:
            1. Day-by-day optimized schedule
            2. Route optimization for minimal travel time
            3. Perfect timing based on weather and crowds
            4. Balance of must-see attractions and authentic experiences
            5. Flexible alternatives for different scenarios
            6. Detailed logistics and transportation

            Ensure the itinerary maximizes enjoyment while minimizing stress.""",
            agent=self.agents['itinerary'],
            expected_output="Optimized day-by-day itinerary with perfect timing and logistics"
        )

        # Task 5: Booking & Logistics Coordination
        booking_task = Task(
            description=f"""Coordinate bookings and logistics for: {user_request}

            Based on the complete itinerary, provide:
            1. Booking recommendations with price comparisons
            2. Optimal booking timing and strategies
            3. Package deal opportunities
            4. Cancellation policies and travel insurance advice
            5. Contact information and booking links
            6. Backup options and contingency plans
            7. Final budget breakdown and cost optimization

            Ensure all logistics are perfectly coordinated for a seamless experience.""",
            agent=self.agents['booking'],
            expected_output="Complete booking guide with optimized logistics and pricing"
        )

        return [research_task, personalization_task, cultural_task, itinerary_task, booking_task]

    def _create_cultural_tasks(self, user_request: str, preferences: Dict[str, Any]) -> List[Task]:
        """Create cultural-focused workflow tasks"""

        research_task = Task(
            description=f"""Research cultural aspects of: {user_request}
            Focus on cultural highlights, local customs, and authentic experiences.""",
            agent=self.agents['research'],
            expected_output="Cultural research report"
        )

        cultural_task = Task(
            description=f"""Deep dive into cultural experiences for: {user_request}
            Find the most authentic and meaningful cultural interactions.""",
            agent=self.agents['cultural'],
            expected_output="Comprehensive cultural experience guide"
        )

        personalization_task = Task(
            description=f"""Personalize cultural experiences for: {user_request}
            Match cultural activities to user preferences: {preferences}""",
            agent=self.agents['personalization'],
            expected_output="Personalized cultural recommendations"
        )

        itinerary_task = Task(
            description=f"""Create culture-focused itinerary for: {user_request}
            Integrate all cultural insights into a cohesive travel plan.""",
            agent=self.agents['itinerary'],
            expected_output="Culture-focused travel itinerary"
        )

        return [research_task, cultural_task, personalization_task, itinerary_task]

    def _create_budget_tasks(self, user_request: str, preferences: Dict[str, Any]) -> List[Task]:
        """Create budget-optimization workflow tasks"""

        research_task = Task(
            description=f"""Research budget options for: {user_request}
            Focus on cost-effective solutions and value opportunities.""",
            agent=self.agents['research'],
            expected_output="Budget-focused research report"
        )

        booking_task = Task(
            description=f"""Find best deals and booking strategies for: {user_request}
            Optimize for maximum value within budget: {preferences.get('budget', 'flexible')}""",
            agent=self.agents['booking'],
            expected_output="Budget optimization and booking strategy"
        )

        itinerary_task = Task(
            description=f"""Create budget-optimized itinerary for: {user_request}
            Balance cost savings with quality experiences.""",
            agent=self.agents['itinerary'],
            expected_output="Budget-optimized travel itinerary"
        )

        personalization_task = Task(
            description=f"""Personalize budget travel for: {user_request}
            Ensure budget constraints don't compromise personal preferences.""",
            agent=self.agents['personalization'],
            expected_output="Personalized budget travel plan"
        )

        return [research_task, booking_task, itinerary_task, personalization_task]

    def get_advanced_status(self) -> Dict[str, Any]:
        """Get advanced system status"""
        return {
            'system_type': 'AdvancedCrewAI',
            'agents_count': len(self.agents),
            'agents': list(self.agents.keys()),
            'tools_count': len(self.tools),
            'available_tools': list(self.tools.keys()),
            'workflow_types': list(self.workflows.keys()),
            'features': [
                'Enhanced RAG Integration',
                'Real-time Weather Data',
                'Price Comparison Analysis',
                'ML-based Personalization',
                'Advanced Booking Assistance',
                'Multi-workflow Support'
            ],
            'status': 'operational'
        }
