"""
Simplified CrewAI Integration for Travel AI System
Working version with proper LLM configuration
"""

import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from crewai import Agent, Task, Crew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not available")

logger = logging.getLogger(__name__)

class SimplifiedCrewAI:
    """
    Simplified CrewAI system that works with our existing Enhanced RAG
    """
    
    def __init__(self):
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is required")
        
        # Set up environment for CrewAI
        self._setup_environment()
        
        # Create agents without complex tools for now
        self.agents = self._create_simple_agents()
        
        logger.info("Simplified CrewAI system initialized")
    
    def _setup_environment(self):
        """Setup environment variables for CrewAI"""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and groq_api_key != "your-groq-api-key-here":
            # Set all required environment variables for CrewAI + Groq
            os.environ["GROQ_API_KEY"] = groq_api_key
            os.environ["OPENAI_MODEL_NAME"] = "groq/llama3-70b-8192"
            os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

            # Debug: Print to verify
            print(f"✅ GROQ_API_KEY set: {groq_api_key[:10]}...")
            print(f"✅ Model: groq/llama3-70b-8192")
        else:
            raise ValueError("Valid GROQ_API_KEY required")
    
    def _create_simple_agents(self) -> Dict[str, Agent]:
        """Create simplified agents without complex tools"""
        
        # Travel Research Agent
        researcher = Agent(
            role='Travel Research Specialist',
            goal='Research destinations and provide comprehensive travel information',
            backstory="""You are an expert travel researcher with extensive knowledge of global destinations.
            You provide detailed information about places, attractions, cultural experiences, and practical travel advice.""",
            verbose=True,
            allow_delegation=False,
            llm="groq/llama3-70b-8192"
        )
        
        # Itinerary Planner
        planner = Agent(
            role='Itinerary Planning Expert',
            goal='Create detailed, personalized travel itineraries',
            backstory="""You are a master itinerary planner with 15+ years of experience.
            You excel at creating perfect travel experiences that balance must-see attractions with practical considerations.""",
            verbose=True,
            allow_delegation=False,
            llm="groq/llama3-70b-8192"
        )

        # Travel Advisor
        advisor = Agent(
            role='Senior Travel Advisor',
            goal='Provide expert travel advice and final recommendations',
            backstory="""You are a senior travel advisor with 20+ years of experience.
            You review travel plans for quality and provide expert recommendations.""",
            verbose=True,
            allow_delegation=False,
            llm="groq/llama3-70b-8192"
        )
        
        return {
            'researcher': researcher,
            'planner': planner,
            'advisor': advisor
        }
    
    def create_travel_plan(self, user_request: str, preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create travel plan using simplified CrewAI"""
        try:
            # Create tasks
            tasks = self._create_simple_tasks(user_request, preferences or {})
            
            # Create crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                verbose=True
            )
            
            # Execute
            result = crew.kickoff()
            
            return {
                'success': True,
                'travel_plan': str(result),
                'agents_involved': list(self.agents.keys()),
                'system': 'SimplifiedCrewAI'
            }
            
        except Exception as e:
            logger.error(f"SimplifiedCrewAI error: {e}")
            
            # Fallback to our Enhanced RAG system
            try:
                from rag_system.enhanced_rag import EnhancedTravelRAG
                rag = EnhancedTravelRAG()
                rag_result = rag.generate_enhanced_response(user_request)
                
                return {
                    'success': True,
                    'travel_plan': rag_result['response'],
                    'agents_involved': ['EnhancedRAG'],
                    'system': 'EnhancedRAG_Fallback',
                    'note': 'CrewAI failed, used Enhanced RAG fallback'
                }
            except Exception as rag_error:
                return {
                    'success': False,
                    'error': f"CrewAI: {str(e)}, RAG: {str(rag_error)}",
                    'travel_plan': 'Both CrewAI and Enhanced RAG systems encountered issues.'
                }
    
    def _create_simple_tasks(self, user_request: str, preferences: Dict[str, Any]) -> List[Task]:
        """Create simplified tasks"""
        
        # Research Task
        research_task = Task(
            description=f"""Research the travel request: {user_request}
            
            User preferences: {preferences}
            
            Provide information about:
            - Destination highlights and attractions
            - Best time to visit and weather
            - Cultural considerations
            - Transportation options
            - Budget considerations
            
            Be comprehensive and practical.""",
            agent=self.agents['researcher'],
            expected_output="Comprehensive destination research report"
        )
        
        # Planning Task
        planning_task = Task(
            description=f"""Create a detailed itinerary for: {user_request}
            
            Based on the research provided, create:
            - Day-by-day itinerary
            - Recommended accommodations
            - Must-see attractions and activities
            - Dining recommendations
            - Transportation between locations
            - Budget breakdown
            
            Consider user preferences: {preferences}""",
            agent=self.agents['planner'],
            expected_output="Detailed day-by-day travel itinerary"
        )
        
        # Advisory Task
        advisory_task = Task(
            description=f"""Review and enhance the travel plan for: {user_request}
            
            Provide:
            - Expert review of the itinerary
            - Additional professional recommendations
            - Practical tips and advice
            - Final quality check
            
            Ensure the plan is practical, exciting, and well-suited to: {preferences}""",
            agent=self.agents['advisor'],
            expected_output="Final expert-reviewed travel plan with professional recommendations"
        )
        
        return [research_task, planning_task, advisory_task]
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'crewai_available': CREWAI_AVAILABLE,
            'agents_count': len(self.agents),
            'agents': list(self.agents.keys()),
            'status': 'operational' if CREWAI_AVAILABLE else 'unavailable',
            'system_type': 'SimplifiedCrewAI'
        }

# Test function
def test_simplified_crewai():
    """Test the simplified CrewAI system"""
    print("🧪 Testing Simplified CrewAI System")
    print("=" * 50)
    
    try:
        # Initialize
        crew_system = SimplifiedCrewAI()
        
        # Get status
        status = crew_system.get_status()
        print(f"✅ System Status: {status['status']}")
        print(f"🤖 Agents: {status['agents_count']} ({', '.join(status['agents'])})")
        
        # Test travel planning
        test_request = "Plan a romantic 2-day Paris trip for $1500"
        preferences = {"budget": "$1500", "style": "romantic", "duration": "2 days"}
        
        print(f"\n🎯 Testing: {test_request}")
        result = crew_system.create_travel_plan(test_request, preferences)
        
        if result['success']:
            print(f"✅ Success! System: {result['system']}")
            print(f"🤖 Agents: {', '.join(result['agents_involved'])}")
            print(f"📝 Plan preview: {result['travel_plan'][:200]}...")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_simplified_crewai()
