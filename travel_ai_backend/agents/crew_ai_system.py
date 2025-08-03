"""
CrewAI Integration for Travel AI System
Advanced multi-agent framework using CrewAI (mentioned in Omnibound JD)
"""

import os
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not available - install with: pip install crewai")

from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

class SearchTravelDatabaseTool(BaseTool):
    name: str = "search_travel_database"
    description: str = "Search the travel database for relevant information"

    def _run(self, query: str) -> str:
        """Search the travel database for relevant information"""
        try:
            from rag_system.enhanced_rag import EnhancedTravelRAG
            rag = EnhancedTravelRAG()
            result = rag.generate_enhanced_response(query)
            return result['response']
        except Exception as e:
            return f"Error searching travel database: {e}"

class WeatherInfoTool(BaseTool):
    name: str = "get_weather_info"
    description: str = "Get weather information for a destination"

    def _run(self, destination: str) -> str:
        """Get weather information for a destination"""
        return f"Weather information for {destination}: Generally pleasant, check local forecasts for specific dates."

class AccommodationTool(BaseTool):
    name: str = "find_accommodations"
    description: str = "Find accommodation options"

    def _run(self, destination: str, budget: str = "mid-range", dates: str = "flexible") -> str:
        """Find accommodation options"""
        return f"Accommodation options in {destination} for {budget} budget during {dates}: Various hotels and rentals available."

class ActivitiesSearchTool(BaseTool):
    name: str = "search_activities"
    description: str = "Search for activities based on interests"

    def _run(self, destination: str, interests: str = "general") -> str:
        """Search for activities based on interests"""
        return f"Activities in {destination} for interests ({interests}): Museums, tours, restaurants, and local experiences available."

class CrewAITravelSystem:
    """
    Advanced travel planning system using CrewAI framework
    """
    
    def __init__(self):
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is required. Install with: pip install crewai")
        
        # Initialize LLM
        self.llm = self._get_llm()

        # Available tools (initialize before agents)
        self.tools = [
            SearchTravelDatabaseTool(),
            WeatherInfoTool(),
            AccommodationTool(),
            ActivitiesSearchTool()
        ]

        # Create specialized agents
        self.agents = self._create_agents()
        
        logger.info("CrewAI Travel System initialized successfully")
    
    def _get_llm(self):
        """Get the best available LLM for CrewAI"""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and groq_api_key != "your-groq-api-key-here":
            # CrewAI specific environment setup
            os.environ["GROQ_API_KEY"] = groq_api_key

            # For CrewAI, we need to specify the model with provider prefix
            return "groq/llama3-70b-8192"
        else:
            raise ValueError("Valid GROQ_API_KEY required for CrewAI system")
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized travel agents"""
        
        # Travel Research Specialist
        research_agent = Agent(
            role='Travel Research Specialist',
            goal='Research destinations, attractions, and travel information',
            backstory="""You are an expert travel researcher with extensive knowledge of global destinations.
            You excel at finding detailed information about places, attractions, cultural experiences, and practical travel advice.
            You use reliable sources and provide comprehensive, accurate information.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.tools[0], self.tools[1]]  # SearchTravelDatabaseTool, WeatherInfoTool
        )
        
        # Itinerary Planning Expert
        planning_agent = Agent(
            role='Itinerary Planning Expert',
            goal='Create detailed, personalized travel itineraries',
            backstory="""You are a master itinerary planner with 15+ years of experience creating perfect travel experiences.
            You excel at balancing must-see attractions with hidden gems, optimizing travel routes, and considering practical factors
            like transportation, timing, and budget constraints.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self.tools[0], self.tools[2], self.tools[3]]  # SearchTravelDatabaseTool, AccommodationTool, ActivitiesSearchTool
        )

        # Local Experience Curator
        experience_agent = Agent(
            role='Local Experience Curator',
            goal='Find authentic local experiences and hidden gems',
            backstory="""You are a local experience specialist who knows the authentic, off-the-beaten-path experiences
            that make travel memorable. You have connections with local guides, know the best local restaurants,
            and can recommend experiences that tourists typically miss.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.tools[0], self.tools[3]]  # SearchTravelDatabaseTool, ActivitiesSearchTool
        )

        # Budget & Logistics Coordinator
        logistics_agent = Agent(
            role='Budget & Logistics Coordinator',
            goal='Handle practical travel arrangements and budget optimization',
            backstory="""You are a meticulous travel logistics expert who ensures every trip runs smoothly.
            You excel at finding the best deals, optimizing budgets, coordinating transportation,
            and handling all the practical details that make or break a trip.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.tools[2], self.tools[1]]  # AccommodationTool, WeatherInfoTool
        )

        # Travel Advisor & Quality Assurance
        advisor_agent = Agent(
            role='Senior Travel Advisor',
            goal='Provide expert travel advice and ensure high-quality recommendations',
            backstory="""You are a senior travel advisor with 20+ years of experience helping travelers create amazing experiences.
            You review all recommendations for quality, feasibility, and alignment with traveler preferences.
            You provide the final expert touch that elevates good travel plans to extraordinary ones.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=self.tools
        )
        
        return {
            'researcher': research_agent,
            'planner': planning_agent,
            'experience_curator': experience_agent,
            'logistics': logistics_agent,
            'advisor': advisor_agent
        }
    
    def create_travel_plan(self, user_request: str, user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a comprehensive travel plan using CrewAI agents
        """
        try:
            # Create tasks for the crew
            tasks = self._create_tasks(user_request, user_preferences or {})
            
            # Create the crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            return {
                'success': True,
                'travel_plan': result,
                'agents_involved': list(self.agents.keys()),
                'process': 'CrewAI Sequential'
            }
            
        except Exception as e:
            logger.error(f"CrewAI travel planning error: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_message': 'CrewAI system encountered an issue. Please try again.'
            }
    
    def _create_tasks(self, user_request: str, preferences: Dict[str, Any]) -> List[Task]:
        """Create tasks for the CrewAI crew"""
        
        # Task 1: Research
        research_task = Task(
            description=f"""Research the destination and requirements for this travel request: {user_request}
            
            User preferences: {preferences}
            
            Provide comprehensive information about:
            - Destination overview and highlights
            - Best time to visit
            - Cultural considerations
            - Weather conditions
            - Transportation options
            
            Use the travel database to find relevant examples and information.""",
            agent=self.agents['researcher'],
            expected_output="Detailed destination research report with practical information"
        )
        
        # Task 2: Experience Curation
        experience_task = Task(
            description=f"""Based on the research, find authentic local experiences for: {user_request}
            
            Focus on:
            - Unique local experiences
            - Hidden gems and off-the-beaten-path attractions
            - Cultural activities and interactions
            - Local dining recommendations
            - Authentic experiences that match user interests
            
            Consider user preferences: {preferences}""",
            agent=self.agents['experience_curator'],
            expected_output="Curated list of authentic local experiences and hidden gems"
        )
        
        # Task 3: Logistics Planning
        logistics_task = Task(
            description=f"""Handle the practical arrangements for: {user_request}
            
            Plan:
            - Accommodation recommendations within budget
            - Transportation between locations
            - Budget breakdown and optimization
            - Practical timing and logistics
            - Weather-appropriate planning
            
            User preferences and constraints: {preferences}""",
            agent=self.agents['logistics'],
            expected_output="Detailed logistics plan with budget breakdown and practical arrangements"
        )
        
        # Task 4: Itinerary Creation
        planning_task = Task(
            description=f"""Create a detailed itinerary combining research, experiences, and logistics for: {user_request}
            
            Integrate:
            - Research findings from the researcher
            - Local experiences from the curator
            - Logistics and budget from the coordinator
            
            Create a day-by-day itinerary that balances:
            - Must-see attractions with authentic experiences
            - Practical timing and transportation
            - Budget considerations
            - User preferences: {preferences}""",
            agent=self.agents['planner'],
            expected_output="Comprehensive day-by-day travel itinerary"
        )
        
        # Task 5: Final Review and Recommendations
        advisor_task = Task(
            description=f"""Review and enhance the complete travel plan for: {user_request}
            
            Provide:
            - Expert review of the itinerary
            - Additional professional recommendations
            - Quality assurance and feasibility check
            - Final expert tips and advice
            - Personalization based on user preferences: {preferences}
            
            Ensure the plan is practical, exciting, and perfectly tailored to the traveler.""",
            agent=self.agents['advisor'],
            expected_output="Final expert-reviewed travel plan with professional recommendations"
        )
        
        return [research_task, experience_task, logistics_task, planning_task, advisor_task]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get CrewAI system status"""
        return {
            'crewai_available': CREWAI_AVAILABLE,
            'agents_count': len(self.agents),
            'agents': list(self.agents.keys()),
            'tools_count': len(self.tools),
            'llm_configured': self.llm is not None,
            'status': 'operational' if CREWAI_AVAILABLE and self.llm else 'degraded'
        }
