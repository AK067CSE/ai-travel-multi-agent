"""
Production Multi-Agent Travel System
Simplified but powerful multi-agent architecture for travel planning
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# LangChain imports
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available")

# CrewAI imports (optional)
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not available")

from django.conf import settings
from .enhanced_rag import EnhancedTravelRAG

logger = logging.getLogger(__name__)

class AgentType(Enum):
    COORDINATOR = "coordinator"
    RESEARCH = "research"
    PLANNING = "planning"
    BOOKING = "booking"
    CHAT = "chat"

@dataclass
class AgentResponse:
    success: bool
    data: Any
    agent_type: str
    processing_time: float
    error: Optional[str] = None

class BaseAgent:
    """Base class for all travel agents"""
    
    def __init__(self, agent_type: AgentType, model_name: str = "llama3-70b-8192"):
        self.agent_type = agent_type
        self.model_name = model_name
        self.llm = None
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0
        }
        
        # Initialize LLM if available
        if LANGCHAIN_AVAILABLE:
            groq_api_key = getattr(settings, 'AI_CONFIG', {}).get('GROQ_API_KEY') or os.getenv("GROQ_API_KEY")
            if groq_api_key:
                self.llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name=model_name,
                    temperature=0.7,
                    max_tokens=2000
                )
    
    def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """Process agent request - to be implemented by subclasses"""
        raise NotImplementedError
    
    def log_activity(self, activity: str, metadata: Dict = None):
        """Log agent activity"""
        logger.info(f"[{self.agent_type.value}] {activity}", extra=metadata or {})

class ResearchAgent(BaseAgent):
    """Agent responsible for travel research and information gathering"""
    
    def __init__(self):
        super().__init__(AgentType.RESEARCH)
        self.rag_system = None
        
        # Initialize RAG system
        try:
            self.rag_system = EnhancedTravelRAG()
            self.rag_system.load_and_index_data()
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
    
    def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """Research travel information using RAG"""
        start_time = datetime.now()
        
        try:
            query = request.get("query", "")
            user_id = request.get("user_id", "anonymous")
            
            self.log_activity(f"Researching: {query[:50]}...")
            
            if self.rag_system:
                # Use enhanced RAG system
                result = self.rag_system.generate_rag_response(query, user_id)
                
                response_data = {
                    "research_results": result["response"],
                    "sources": result["sources"],
                    "retrieved_docs": result["retrieved_docs"],
                    "system_used": result["system_used"]
                }
            else:
                # Fallback research
                response_data = {
                    "research_results": self._fallback_research(query),
                    "sources": [],
                    "retrieved_docs": 0,
                    "system_used": "Fallback_Research"
                }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                success=True,
                data=response_data,
                agent_type=self.agent_type.value,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Research agent error: {e}")
            
            return AgentResponse(
                success=False,
                data={"error": str(e)},
                agent_type=self.agent_type.value,
                processing_time=processing_time,
                error=str(e)
            )
    
    def _fallback_research(self, query: str) -> str:
        """Fallback research when RAG is not available"""
        query_lower = query.lower()
        
        if "paris" in query_lower:
            return "Paris Research: Famous for the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and excellent cuisine. Best visited in spring or fall. Budget varies from €50-200+ per day."
        elif "japan" in query_lower:
            return "Japan Research: Known for cherry blossoms, temples, modern cities, and unique culture. Tokyo and Kyoto are must-visit cities. JR Pass recommended for travel."
        elif "budget" in query_lower:
            return "Budget Travel Research: Consider Southeast Asia, Eastern Europe, or Central America for affordable destinations. Hostels, street food, and public transport help reduce costs."
        else:
            return f"Travel research for '{query}': I recommend researching visa requirements, best travel seasons, local customs, and budget considerations for your destination."

class PlanningAgent(BaseAgent):
    """Agent responsible for creating detailed travel plans"""
    
    def __init__(self):
        super().__init__(AgentType.PLANNING)
    
    def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """Create detailed travel plans"""
        start_time = datetime.now()
        
        try:
            query = request.get("query", "")
            research_data = request.get("research_data", {})
            preferences = request.get("preferences", {})
            
            self.log_activity(f"Planning trip: {query[:50]}...")
            
            if self.llm:
                plan = self._generate_llm_plan(query, research_data, preferences)
            else:
                plan = self._generate_fallback_plan(query, preferences)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                success=True,
                data={
                    "travel_plan": plan,
                    "preferences_used": preferences,
                    "system_used": "LLM_Planning" if self.llm else "Template_Planning"
                },
                agent_type=self.agent_type.value,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Planning agent error: {e}")
            
            return AgentResponse(
                success=False,
                data={"error": str(e)},
                agent_type=self.agent_type.value,
                processing_time=processing_time,
                error=str(e)
            )
    
    def _generate_llm_plan(self, query: str, research_data: Dict, preferences: Dict) -> str:
        """Generate travel plan using LLM"""
        planning_prompt = ChatPromptTemplate.from_template(
            """You are an expert travel planner. Create a detailed, practical travel plan based on the user's request and research data.

User Request: {query}

Research Data: {research_data}

User Preferences: {preferences}

Create a comprehensive travel plan including:
1. Itinerary overview
2. Daily activities
3. Accommodation suggestions
4. Transportation options
5. Budget estimates
6. Important tips and considerations

Travel Plan:"""
        )
        
        try:
            chain = planning_prompt | self.llm | StrOutputParser()
            plan = chain.invoke({
                "query": query,
                "research_data": json.dumps(research_data, indent=2),
                "preferences": json.dumps(preferences, indent=2)
            })
            
            return plan.strip()
            
        except Exception as e:
            logger.error(f"LLM planning error: {e}")
            return self._generate_fallback_plan(query, preferences)
    
    def _generate_fallback_plan(self, query: str, preferences: Dict) -> str:
        """Generate basic travel plan template"""
        duration = preferences.get("duration", "7 days")
        budget = preferences.get("budget", "medium")
        
        return f"""Travel Plan for: {query}

Duration: {duration}
Budget Level: {budget}

Day 1-2: Arrival and City Orientation
- Check into accommodation
- Explore main city center
- Visit key landmarks

Day 3-4: Cultural Experiences
- Museums and historical sites
- Local cuisine experiences
- Cultural activities

Day 5-6: Adventure and Exploration
- Day trips or outdoor activities
- Shopping and local markets
- Relaxation time

Final Day: Departure
- Last-minute shopping
- Airport transfer

Budget Considerations:
- Accommodation: Varies by budget level
- Food: Mix of local and international options
- Transportation: Public transport recommended
- Activities: Balance of free and paid attractions

Tips:
- Book accommodation in advance
- Learn basic local phrases
- Keep copies of important documents
- Check visa requirements"""

class ChatAgent(BaseAgent):
    """Agent responsible for natural conversation and user interaction"""

    def __init__(self):
        super().__init__(AgentType.CHAT)

    def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """Handle conversational interactions"""
        start_time = datetime.now()

        try:
            query = request.get("query", "")
            context = request.get("context", {})

            self.log_activity(f"Chat interaction: {query[:50]}...")

            if self.llm:
                response = self._generate_chat_response(query, context)
            else:
                response = self._generate_fallback_chat(query)

            processing_time = (datetime.now() - start_time).total_seconds()

            return AgentResponse(
                success=True,
                data={
                    "chat_response": response,
                    "system_used": "LLM_Chat" if self.llm else "Template_Chat"
                },
                agent_type=self.agent_type.value,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Chat agent error: {e}")

            return AgentResponse(
                success=False,
                data={"error": str(e)},
                agent_type=self.agent_type.value,
                processing_time=processing_time,
                error=str(e)
            )

    def _generate_chat_response(self, query: str, context: Dict) -> str:
        """Generate conversational response using LLM"""
        chat_prompt = ChatPromptTemplate.from_template(
            """You are a friendly, knowledgeable travel assistant. Engage in natural conversation while providing helpful travel advice.

Context: {context}

User Message: {query}

Respond in a conversational, helpful manner. If the user is asking about travel, provide useful information. If they're just chatting, be friendly and try to guide the conversation toward how you can help with their travel needs.

Response:"""
        )

        try:
            chain = chat_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "query": query,
                "context": json.dumps(context, indent=2)
            })

            return response.strip()

        except Exception as e:
            logger.error(f"Chat LLM error: {e}")
            return self._generate_fallback_chat(query)

    def _generate_fallback_chat(self, query: str) -> str:
        """Generate fallback chat response"""
        query_lower = query.lower()

        greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
        if any(greeting in query_lower for greeting in greetings):
            return "Hello! I'm your AI travel assistant. I'm here to help you plan amazing trips, find great destinations, and answer any travel questions you might have. What kind of adventure are you thinking about?"

        thanks = ["thank", "thanks", "appreciate"]
        if any(thank in query_lower for thank in thanks):
            return "You're very welcome! I'm glad I could help. Is there anything else about your travel plans I can assist you with?"

        if "help" in query_lower:
            return "I'd be happy to help! I can assist you with travel planning, destination recommendations, budget advice, itinerary creation, and answering questions about different places around the world. What would you like to know?"

        return "That's interesting! I'm here to help with all your travel needs. Whether you're planning a weekend getaway, a long vacation, or just dreaming about future trips, I can provide recommendations, create itineraries, and share travel tips. What destination are you curious about?"

class CoordinatorAgent(BaseAgent):
    """Master agent that coordinates all other agents"""

    def __init__(self):
        super().__init__(AgentType.COORDINATOR, "llama3-70b-8192")  # Use larger model

        # Initialize specialist agents
        self.research_agent = ResearchAgent()
        self.planning_agent = PlanningAgent()
        self.chat_agent = ChatAgent()

        self.active_workflows = {}

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate complex user requests across multiple agents"""
        try:
            user_id = request.get("user_id", "anonymous")
            user_request = request.get("request", "")
            workflow_type = request.get("workflow_type", "auto")

            self.log_activity(f"Coordinating request", {
                "user_id": user_id,
                "workflow_type": workflow_type,
                "request_length": len(user_request)
            })

            # Analyze request type
            request_type = self._analyze_request_type(user_request)

            # Execute appropriate workflow
            if request_type == "planning":
                return self._execute_planning_workflow(user_request, user_id)
            elif request_type == "research":
                return self._execute_research_workflow(user_request, user_id)
            elif request_type == "chat":
                return self._execute_chat_workflow(user_request, user_id)
            else:
                return self._execute_comprehensive_workflow(user_request, user_id)

        except Exception as e:
            logger.error(f"Coordinator error: {e}")
            return {
                "success": False,
                "error": str(e),
                "system_used": "Error_Handler",
                "agents_involved": ["CoordinatorAgent"]
            }

    def _analyze_request_type(self, request: str) -> str:
        """Analyze request to determine workflow type"""
        request_lower = request.lower()

        # Planning keywords
        planning_keywords = ["plan", "itinerary", "schedule", "trip", "vacation", "visit", "travel to"]
        if any(keyword in request_lower for keyword in planning_keywords):
            return "planning"

        # Research keywords
        research_keywords = ["information", "about", "tell me", "what is", "research", "find out"]
        if any(keyword in request_lower for keyword in research_keywords):
            return "research"

        # Chat keywords
        chat_keywords = ["hello", "hi", "thanks", "help", "how are you"]
        if any(keyword in request_lower for keyword in chat_keywords):
            return "chat"

        return "comprehensive"

    def _execute_planning_workflow(self, request: str, user_id: str) -> Dict[str, Any]:
        """Execute planning-focused workflow"""
        # First research, then plan
        research_result = self.research_agent.process_request({
            "query": request,
            "user_id": user_id
        })

        planning_result = self.planning_agent.process_request({
            "query": request,
            "research_data": research_result.data if research_result.success else {},
            "preferences": {}
        })

        if planning_result.success:
            return {
                "success": True,
                "response": planning_result.data["travel_plan"],
                "system_used": "Multi_Agent_Planning",
                "agents_involved": ["ResearchAgent", "PlanningAgent"],
                "processing_time": research_result.processing_time + planning_result.processing_time,
                "metadata": {
                    "research_sources": research_result.data.get("sources", []) if research_result.success else [],
                    "planning_system": planning_result.data.get("system_used", "")
                }
            }
        else:
            return {
                "success": False,
                "response": "I encountered some issues creating your travel plan. Let me try a different approach.",
                "system_used": "Fallback_Planning",
                "agents_involved": ["CoordinatorAgent"],
                "error": planning_result.error
            }

    def _execute_research_workflow(self, request: str, user_id: str) -> Dict[str, Any]:
        """Execute research-focused workflow"""
        research_result = self.research_agent.process_request({
            "query": request,
            "user_id": user_id
        })

        if research_result.success:
            return {
                "success": True,
                "response": research_result.data["research_results"],
                "system_used": "Multi_Agent_Research",
                "agents_involved": ["ResearchAgent"],
                "processing_time": research_result.processing_time,
                "metadata": {
                    "sources": research_result.data.get("sources", []),
                    "retrieved_docs": research_result.data.get("retrieved_docs", 0)
                }
            }
        else:
            return {
                "success": False,
                "response": "I had trouble researching that information. Could you try rephrasing your question?",
                "system_used": "Fallback_Research",
                "agents_involved": ["CoordinatorAgent"],
                "error": research_result.error
            }

    def _execute_chat_workflow(self, request: str, user_id: str) -> Dict[str, Any]:
        """Execute chat-focused workflow"""
        chat_result = self.chat_agent.process_request({
            "query": request,
            "context": {}
        })

        if chat_result.success:
            return {
                "success": True,
                "response": chat_result.data["chat_response"],
                "system_used": "Multi_Agent_Chat",
                "agents_involved": ["ChatAgent"],
                "processing_time": chat_result.processing_time
            }
        else:
            return {
                "success": False,
                "response": "Hello! I'm here to help with your travel planning needs. What can I assist you with today?",
                "system_used": "Fallback_Chat",
                "agents_involved": ["CoordinatorAgent"],
                "error": chat_result.error
            }

    def _execute_comprehensive_workflow(self, request: str, user_id: str) -> Dict[str, Any]:
        """Execute comprehensive workflow using multiple agents"""
        # Research first
        research_result = self.research_agent.process_request({
            "query": request,
            "user_id": user_id
        })

        # Then plan based on research
        planning_result = self.planning_agent.process_request({
            "query": request,
            "research_data": research_result.data if research_result.success else {},
            "preferences": {}
        })

        # Combine results
        if research_result.success and planning_result.success:
            combined_response = f"{research_result.data['research_results']}\n\n## Travel Plan\n\n{planning_result.data['travel_plan']}"

            return {
                "success": True,
                "response": combined_response,
                "system_used": "Multi_Agent_Comprehensive",
                "agents_involved": ["ResearchAgent", "PlanningAgent"],
                "processing_time": research_result.processing_time + planning_result.processing_time,
                "metadata": {
                    "research_sources": research_result.data.get("sources", []),
                    "retrieved_docs": research_result.data.get("retrieved_docs", 0)
                }
            }
        elif research_result.success:
            return {
                "success": True,
                "response": research_result.data["research_results"],
                "system_used": "Multi_Agent_Research_Only",
                "agents_involved": ["ResearchAgent"],
                "processing_time": research_result.processing_time
            }
        else:
            # Fallback to chat agent
            return self._execute_chat_workflow(request, user_id)
