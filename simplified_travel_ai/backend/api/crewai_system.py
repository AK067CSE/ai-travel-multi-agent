"""
Real CrewAI Multi-Agent System for Travel Planning
Production-level implementation using CrewAI framework
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# CrewAI imports
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
    logger.info("CrewAI successfully imported")

    # Optional tools
    try:
        from crewai_tools import SerperDevTool, WebsiteSearchTool
        CREWAI_TOOLS_AVAILABLE = True
    except ImportError:
        CREWAI_TOOLS_AVAILABLE = False
        logger.warning("CrewAI tools not available")

except ImportError as e:
    CREWAI_AVAILABLE = False
    CREWAI_TOOLS_AVAILABLE = False
    logger.warning(f"CrewAI not available: {e}")

# LangChain imports
try:
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available")

from django.conf import settings
from .enhanced_rag import EnhancedTravelRAG

class TravelDatabaseTool(BaseTool):
    """Custom tool to search travel database using RAG"""
    name: str = "travel_database_search"
    description: str = "Search comprehensive travel database for destinations, recommendations, and travel information"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize RAG system as instance variable (not field)
        self._rag_system = None
        try:
            self._rag_system = EnhancedTravelRAG()
            self._rag_system.load_and_index_data()
        except Exception as e:
            logger.error(f"Failed to initialize RAG system in tool: {e}")

    def _run(self, query: str) -> str:
        """Execute the tool"""
        try:
            if self._rag_system:
                result = self._rag_system.generate_rag_response(query)
                return result.get('response', 'No information found')
            else:
                return f"Travel database search for: {query} - Database temporarily unavailable"
        except Exception as e:
            logger.error(f"Travel database tool error: {e}")
            return f"Error searching travel database: {str(e)}"

class CrewAITravelSystem:
    """
    Production CrewAI Multi-Agent Travel System
    Features:
    - Real CrewAI agents with specialized roles
    - LangChain LLM integration
    - Custom travel database tool
    - Coordinated multi-agent workflows
    """
    
    def __init__(self):
        self.llm = None
        self.agents = {}
        self.tools = []
        self.crew = None
        
        # Initialize LLM
        self._initialize_llm()
        
        # Initialize tools
        self._initialize_tools()
        
        # Create agents
        self._create_agents()
        
        # Create crew
        self._create_crew()
        
        logger.info("CrewAI Travel System initialized successfully")
    
    def _initialize_llm(self):
        """Initialize LLM for agents"""
        try:
            # Try Groq first (fastest and available)
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key and LANGCHAIN_AVAILABLE:
                try:
                    from langchain_groq import ChatGroq
                    # Use LiteLLM directly for CrewAI compatibility
                    try:
                        import litellm

                        # Set environment variable for LiteLLM
                        os.environ["GROQ_API_KEY"] = groq_api_key

                        # Create a proper LiteLLM wrapper that CrewAI can use
                        class LiteLLMWrapper:
                            def __init__(self, model_name, api_key):
                                self.model_name = f"groq/{model_name}"
                                self.api_key = api_key

                            def invoke(self, prompt):
                                try:
                                    # Log what CrewAI is passing
                                    logger.info(f"LiteLLM wrapper received: {type(prompt)} - {str(prompt)[:200]}...")

                                    # Handle different input formats
                                    if isinstance(prompt, str):
                                        messages = [{"role": "user", "content": prompt}]
                                    elif isinstance(prompt, list):
                                        messages = prompt
                                    elif hasattr(prompt, 'content'):
                                        messages = [{"role": "user", "content": prompt.content}]
                                    elif hasattr(prompt, 'messages'):
                                        messages = prompt.messages
                                    else:
                                        messages = [{"role": "user", "content": str(prompt)}]

                                    logger.info(f"Converted to messages: {messages}")

                                    # Make the API call
                                    response = litellm.completion(
                                        model=self.model_name,
                                        messages=messages,
                                        api_key=self.api_key,
                                        temperature=0.7,
                                        max_tokens=2000
                                    )

                                    # Return response in expected format
                                    class Response:
                                        def __init__(self, content):
                                            self.content = content

                                    content = response.choices[0].message.content
                                    if content:
                                        return Response(content)
                                    else:
                                        return Response("I apologize, but I couldn't generate a response. Please try again.")

                                except Exception as e:
                                    logger.error(f"LiteLLM wrapper error: {e}")
                                    class Response:
                                        def __init__(self, content):
                                            self.content = content
                                    return Response(f"Error generating response: {str(e)}")

                            def __call__(self, prompt):
                                return self.invoke(prompt)

                        self.llm = LiteLLMWrapper("llama3-70b-8192", groq_api_key)
                        logger.info("Using LiteLLM wrapper for CrewAI compatibility")

                    except Exception as e:
                        logger.warning(f"LiteLLM wrapper failed: {e}")
                        # Fallback to ChatGroq
                        try:
                            from langchain_groq import ChatGroq
                            self.llm = ChatGroq(
                                model_name="llama3-70b-8192",
                                temperature=0.7,
                                max_tokens=2000,
                                groq_api_key=groq_api_key
                            )
                            logger.info("Using ChatGroq fallback")
                        except Exception as e2:
                            logger.error(f"All LLM methods failed: {e2}")
                            self.llm = None
                    logger.info("Using Groq LLM for CrewAI agents")
                    return
                except Exception as e:
                    logger.error(f"Failed to initialize Groq LLM: {e}")

            # Fallback to OpenAI
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if openai_api_key and LANGCHAIN_AVAILABLE:
                try:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(
                        openai_api_key=openai_api_key,
                        model_name="gpt-3.5-turbo",
                        temperature=0.7,
                        max_tokens=2000
                    )
                    logger.info("Using OpenAI LLM for CrewAI agents")
                    return
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI LLM: {e}")

            # Fallback to Anthropic
            anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
            if anthropic_api_key and LANGCHAIN_AVAILABLE:
                try:
                    from langchain_anthropic import ChatAnthropic
                    self.llm = ChatAnthropic(
                        anthropic_api_key=anthropic_api_key,
                        model="claude-3-sonnet-20240229",
                        temperature=0.7,
                        max_tokens=2000
                    )
                    logger.info("Using Anthropic LLM for CrewAI agents")
                    return
                except Exception as e:
                    logger.error(f"Failed to initialize Anthropic LLM: {e}")

            logger.warning("No LLM available for CrewAI agents")

        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
    
    def _initialize_tools(self):
        """Initialize tools for agents"""
        try:
            # Travel database tool (always available)
            self.tools.append(TravelDatabaseTool())
            
            # Web search tool (if API key and tools available)
            serper_api_key = os.getenv('SERPER_API_KEY')
            if serper_api_key and CREWAI_TOOLS_AVAILABLE:
                try:
                    self.tools.append(SerperDevTool())
                except Exception as e:
                    logger.warning(f"Could not initialize SerperDevTool: {e}")
            
            logger.info(f"Initialized {len(self.tools)} tools for CrewAI agents")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {e}")
    
    def _create_agents(self):
        """Create specialized CrewAI agents"""
        if not CREWAI_AVAILABLE or not self.llm:
            logger.warning("Cannot create CrewAI agents - missing dependencies")
            return
        
        try:
            # Travel Research Agent
            self.agents['researcher'] = Agent(
                role='Travel Research Specialist',
                goal='Research comprehensive travel information including destinations, accommodations, activities, and local insights',
                backstory="""You are an expert travel researcher with extensive knowledge of global destinations. 
                You excel at finding detailed, accurate, and up-to-date travel information from multiple sources.""",
                verbose=True,
                allow_delegation=False,
                tools=self.tools,
                llm=self.llm
            )
            
            # Travel Planning Agent
            self.agents['planner'] = Agent(
                role='Travel Planning Expert',
                goal='Create detailed, personalized travel itineraries based on research and user preferences',
                backstory="""You are a professional travel planner with years of experience creating amazing trips. 
                You specialize in crafting detailed itineraries that balance must-see attractions, local experiences, 
                budget considerations, and practical logistics.""",
                verbose=True,
                allow_delegation=False,
                tools=self.tools,
                llm=self.llm
            )
            
            # Travel Advisor Agent
            self.agents['advisor'] = Agent(
                role='Travel Advisory Consultant',
                goal='Provide expert travel advice, recommendations, and practical tips for travelers',
                backstory="""You are a seasoned travel advisor who has helped thousands of travelers plan successful trips. 
                You provide practical advice on everything from visa requirements to local customs, 
                safety tips, and money-saving strategies.""",
                verbose=True,
                allow_delegation=False,
                tools=self.tools,
                llm=self.llm
            )
            
            logger.info(f"Created {len(self.agents)} CrewAI agents")
            
        except Exception as e:
            logger.error(f"Error creating agents: {e}")
    
    def _create_crew(self):
        """Create CrewAI crew with agents"""
        if not CREWAI_AVAILABLE or not self.agents:
            logger.warning("Cannot create CrewAI crew - missing agents")
            return
        
        try:
            self.crew = Crew(
                agents=list(self.agents.values()),
                verbose=True,
                process=Process.sequential
            )
            
            logger.info("CrewAI crew created successfully")
            
        except Exception as e:
            logger.error(f"Error creating crew: {e}")
    
    def process_travel_request(self, request: str, user_preferences: Dict = None) -> Dict[str, Any]:
        """Process travel request using CrewAI multi-agent system"""
        if not CREWAI_AVAILABLE or not self.crew:
            return self._fallback_response(request)
        
        try:
            start_time = datetime.now()
            
            # Create tasks based on request complexity
            tasks = self._create_tasks(request, user_preferences or {})

            # Update crew tasks and execute (CrewAI v0.148.0 compatibility)
            self.crew.tasks = tasks
            result = self.crew.kickoff()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'response': str(result),
                'system_used': 'CrewAI_Multi_Agent',
                'agents_involved': list(self.agents.keys()),
                'processing_time': processing_time,
                'tasks_executed': len(tasks)
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"CrewAI processing error: {e}")
            
            return {
                'success': False,
                'response': f"CrewAI system encountered an error: {str(e)}. Using fallback response.",
                'system_used': 'CrewAI_Error_Fallback',
                'agents_involved': [],
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def _create_tasks(self, request: str, preferences: Dict) -> List[Task]:
        """Create tasks for CrewAI agents based on request"""
        tasks = []
        
        try:
            # Research task
            research_task = Task(
                description=f"""Research comprehensive travel information for: {request}
                
                User preferences: {preferences}
                
                Please provide detailed information about:
                - Destinations and attractions
                - Accommodation options
                - Transportation methods
                - Local activities and experiences
                - Budget considerations
                - Best time to visit
                - Cultural insights and tips
                
                Use the travel database tool to find relevant information.""",
                agent=self.agents['researcher'],
                expected_output="Comprehensive travel research report with detailed information about destinations, activities, and practical considerations."
            )
            tasks.append(research_task)
            
            # Planning task
            planning_task = Task(
                description=f"""Based on the research findings, create a detailed travel plan for: {request}
                
                User preferences: {preferences}
                
                Create a comprehensive itinerary including:
                - Day-by-day schedule
                - Recommended accommodations
                - Transportation arrangements
                - Activity bookings and timing
                - Budget breakdown
                - Packing suggestions
                - Important reminders
                
                Make the plan practical and actionable.""",
                agent=self.agents['planner'],
                expected_output="Detailed travel itinerary with day-by-day plans, budget breakdown, and practical recommendations."
            )
            tasks.append(planning_task)
            
            # Advisory task
            advisory_task = Task(
                description=f"""Provide expert travel advice and final recommendations for: {request}
                
                Based on the research and planning, offer:
                - Pro tips and insider knowledge
                - Safety and health considerations
                - Money-saving strategies
                - Cultural etiquette advice
                - Emergency preparedness
                - Final recommendations and alternatives
                
                Make the advice practical and actionable.""",
                agent=self.agents['advisor'],
                expected_output="Expert travel advice with practical tips, safety considerations, and final recommendations."
            )
            tasks.append(advisory_task)
            
        except Exception as e:
            logger.error(f"Error creating tasks: {e}")
        
        return tasks
    
    def _fallback_response(self, request: str) -> Dict[str, Any]:
        """Fallback response when CrewAI is not available"""
        return {
            'success': False,
            'response': f"CrewAI system is not available. Request: {request} - Please ensure CrewAI dependencies are installed and API keys are configured.",
            'system_used': 'CrewAI_Unavailable',
            'agents_involved': [],
            'processing_time': 0.0,
            'error': 'CrewAI not available'
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get CrewAI system status"""
        return {
            'crewai_available': CREWAI_AVAILABLE,
            'langchain_available': LANGCHAIN_AVAILABLE,
            'llm_initialized': self.llm is not None,
            'agents_created': len(self.agents),
            'tools_available': len(self.tools),
            'crew_ready': self.crew is not None,
            'system_ready': all([
                CREWAI_AVAILABLE,
                LANGCHAIN_AVAILABLE,
                self.llm is not None,
                len(self.agents) > 0,
                self.crew is not None
            ])
        }
