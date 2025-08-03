"""
Coordinator Agent - Orchestrates multi-agent interactions
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from .base_agent import BaseAgent, AgentResponse, agent_communication
from .scraping_agent import ScrapingAgent
from .recommendation_agent import RecommendationAgent
from .booking_agent import BookingAgent
from .chat_agent import ChatAgent
import logging

logger = logging.getLogger(__name__)

class CoordinatorAgent(BaseAgent):
    """Master agent that coordinates all other agents"""
    
    def __init__(self, model_name: str = "llama3-70b-8192"):  # Use larger model for coordination
        super().__init__("CoordinatorAgent", model_name)
        
        # Initialize all specialist agents
        self.scraping_agent = ScrapingAgent()
        self.recommendation_agent = RecommendationAgent()
        self.booking_agent = BookingAgent()
        self.chat_agent = ChatAgent()
        
        # Register agents for communication
        agent_communication.register_agent(self.scraping_agent)
        agent_communication.register_agent(self.recommendation_agent)
        agent_communication.register_agent(self.booking_agent)
        agent_communication.register_agent(self.chat_agent)
        agent_communication.register_agent(self)
        
        self.active_workflows = {}
        self.agent_capabilities = {
            "ScrapingAgent": ["data_collection", "web_scraping", "information_gathering"],
            "RecommendationAgent": ["personalization", "suggestions", "travel_planning"],
            "BookingAgent": ["reservations", "payments", "booking_management"],
            "ChatAgent": ["conversation", "user_interaction", "intent_detection"]
        }
    
    def create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for coordination decisions"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are the master coordinator for a multi-agent travel system. Your role is to:
            
            1. Analyze complex user requests and break them down into tasks
            2. Determine which agents should handle each task
            3. Coordinate agent interactions and data flow
            4. Ensure optimal user experience through efficient agent orchestration
            5. Handle edge cases and error scenarios
            
            Available Agents and Capabilities:
            - ScrapingAgent: Web scraping, data collection, real-time information gathering
            - RecommendationAgent: Personalized suggestions, travel planning, user profiling
            - BookingAgent: Reservations, booking management, payment processing
            - ChatAgent: User interaction, conversation management, intent detection
            
            Coordination Principles:
            - Minimize user wait time through parallel processing
            - Ensure data consistency across agents
            - Provide fallback options when agents fail
            - Maintain conversation context throughout workflows
            - Prioritize user satisfaction and experience
            
            Always think step-by-step about the optimal agent workflow for each request."""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """User Request: {user_request}
            
            User Context: {user_context}
            
            Available Agent Data: {agent_data}
            
            Please analyze this request and provide a coordination plan with specific agent assignments and workflow steps."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process and coordinate complex user requests"""
        try:
            user_id = request.get("user_id", "anonymous")
            user_request = request.get("request", "")
            workflow_type = request.get("workflow_type", "auto")
            
            self.log_activity(f"Coordinating request", {
                "user_id": user_id,
                "workflow_type": workflow_type,
                "request_length": len(user_request)
            })
            
            # Create workflow ID
            workflow_id = f"wf_{user_id}_{len(self.active_workflows)}"
            
            # Analyze request and create coordination plan
            coordination_plan = self.create_coordination_plan(request)
            
            # Execute workflow
            if workflow_type == "simple":
                result = self.execute_simple_workflow(workflow_id, coordination_plan, request)
            elif workflow_type == "complex":
                result = self.execute_complex_workflow(workflow_id, coordination_plan, request)
            else:
                result = self.execute_auto_workflow(workflow_id, coordination_plan, request)
            
            response = AgentResponse(
                agent_name=self.agent_name,
                success=True,
                data={
                    "workflow_id": workflow_id,
                    "coordination_plan": coordination_plan,
                    "result": result,
                    "agents_involved": result.get("agents_involved", [])
                },
                metadata={
                    "user_id": user_id,
                    "workflow_type": workflow_type,
                    "execution_time": result.get("execution_time", 0)
                }
            )
            
            return response.to_dict()
            
        except Exception as e:
            logger.error(f"Coordination error: {e}")
            response = AgentResponse(
                agent_name=self.agent_name,
                success=False,
                error=str(e)
            )
            return response.to_dict()
    
    def create_coordination_plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create coordination plan using LLM analysis"""
        try:
            prompt_template = self.create_prompt_template()
            chain = self.create_chain(prompt_template)
            
            context = {
                "user_request": request.get("request", ""),
                "user_context": json.dumps(request.get("context", {}), indent=2),
                "agent_data": json.dumps(self.agent_capabilities, indent=2)
            }
            
            plan_text = chain.invoke(context)
            
            # Parse plan (in real implementation, use structured output)
            plan = self.parse_coordination_plan(plan_text, request)
            
            return plan
            
        except Exception as e:
            logger.error(f"Plan creation error: {e}")
            return self.create_fallback_plan(request)
    
    def parse_coordination_plan(self, plan_text: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM coordination plan into structured format"""
        # Simplified parsing - in real implementation, use structured output
        user_request = request.get("request", "").lower()
        
        plan = {
            "workflow_type": "sequential",
            "steps": [],
            "parallel_tasks": [],
            "fallback_options": []
        }
        
        # Determine workflow based on keywords
        if any(word in user_request for word in ["book", "reserve", "buy"]):
            plan["steps"] = [
                {"agent": "ChatAgent", "task": "understand_booking_requirements"},
                {"agent": "ScrapingAgent", "task": "find_available_options"},
                {"agent": "RecommendationAgent", "task": "suggest_best_options"},
                {"agent": "BookingAgent", "task": "process_booking"}
            ]
        elif any(word in user_request for word in ["recommend", "suggest", "plan"]):
            plan["steps"] = [
                {"agent": "ChatAgent", "task": "extract_preferences"},
                {"agent": "ScrapingAgent", "task": "gather_destination_data"},
                {"agent": "RecommendationAgent", "task": "generate_recommendations"}
            ]
        elif any(word in user_request for word in ["find", "search", "information"]):
            plan["steps"] = [
                {"agent": "ChatAgent", "task": "clarify_information_needs"},
                {"agent": "ScrapingAgent", "task": "collect_requested_data"}
            ]
        else:
            plan["steps"] = [
                {"agent": "ChatAgent", "task": "handle_general_conversation"}
            ]
        
        return plan
    
    def create_fallback_plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback plan when LLM analysis fails"""
        return {
            "workflow_type": "simple",
            "steps": [
                {"agent": "ChatAgent", "task": "handle_request"}
            ],
            "parallel_tasks": [],
            "fallback_options": ["direct_chat_response"]
        }
    
    def execute_simple_workflow(self, workflow_id: str, plan: Dict[str, Any], request: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute simple sequential workflow"""
        start_time = self._get_timestamp()
        results = []
        agents_involved = []
        
        for step in plan.get("steps", []):
            agent_name = step["agent"]
            task = step["task"]

            try:
                agent = self.get_agent(agent_name)
                if agent:
                    # Add user request to step context
                    if request:
                        step["user_request"] = request.get("request", "")
                        step["user_id"] = request.get("user_id", "anonymous")
                    else:
                        step["user_request"] = ""
                        step["user_id"] = "anonymous"

                    # Create task-specific request
                    task_request = self.create_task_request(task, step)
                    result = agent.process_request(task_request)
                    
                    results.append({
                        "agent": agent_name,
                        "task": task,
                        "result": result,
                        "success": result.get("success", False)
                    })
                    
                    agents_involved.append(agent_name)
                    
            except Exception as e:
                logger.error(f"Step execution error: {e}")
                results.append({
                    "agent": agent_name,
                    "task": task,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "workflow_id": workflow_id,
            "results": results,
            "agents_involved": list(set(agents_involved)),
            "execution_time": self._calculate_execution_time(start_time),
            "success": all(r.get("success", False) for r in results)
        }
    
    def execute_complex_workflow(self, workflow_id: str, plan: Dict[str, Any], request: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute complex workflow with parallel tasks"""
        start_time = self._get_timestamp()
        
        # Execute parallel tasks first
        parallel_results = []
        if plan.get("parallel_tasks"):
            parallel_results = self.execute_parallel_tasks(plan["parallel_tasks"])
        
        # Execute sequential steps
        sequential_results = []
        for step in plan.get("steps", []):
            # Include parallel results in context
            step["context"] = {"parallel_results": parallel_results}
            result = self.execute_single_step(step)
            sequential_results.append(result)
        
        all_results = parallel_results + sequential_results
        agents_involved = list(set([r["agent"] for r in all_results]))
        
        return {
            "workflow_id": workflow_id,
            "parallel_results": parallel_results,
            "sequential_results": sequential_results,
            "agents_involved": agents_involved,
            "execution_time": self._calculate_execution_time(start_time),
            "success": all(r.get("success", False) for r in all_results)
        }
    
    def execute_auto_workflow(self, workflow_id: str, plan: Dict[str, Any], request: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute workflow with automatic optimization"""
        # For now, always use simple workflow (sequential execution)
        # TODO: Implement proper parallel execution in complex workflow
        return self.execute_simple_workflow(workflow_id, plan, request)
    
    def execute_parallel_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tasks in parallel"""
        results = []
        
        # In a real implementation, use asyncio for true parallelism
        for task in tasks:
            try:
                result = self.execute_single_step(task)
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel task error: {e}")
                results.append({
                    "agent": task.get("agent", "unknown"),
                    "task": task.get("task", "unknown"),
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def execute_single_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        agent_name = step["agent"]
        task = step["task"]
        
        agent = self.get_agent(agent_name)
        if not agent:
            return {
                "agent": agent_name,
                "task": task,
                "error": f"Agent {agent_name} not found",
                "success": False
            }
        
        task_request = self.create_task_request(task, step)
        result = agent.process_request(task_request)
        
        return {
            "agent": agent_name,
            "task": task,
            "result": result,
            "success": result.get("success", False)
        }
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get agent instance by name"""
        agents = {
            "ScrapingAgent": self.scraping_agent,
            "RecommendationAgent": self.recommendation_agent,
            "BookingAgent": self.booking_agent,
            "ChatAgent": self.chat_agent
        }
        return agents.get(agent_name)
    
    def create_task_request(self, task: str, step: Dict[str, Any]) -> Dict[str, Any]:
        """Create task-specific request for agent"""
        base_request = {
            "task": task,
            "context": step.get("context", {}),
            "user_id": step.get("user_id", "anonymous"),
            "user_request": step.get("user_request", ""),  # Pass the actual user message!
            "message": step.get("user_request", ""),       # For ChatAgent
            "request": step.get("user_request", "")        # For RecommendationAgent
        }

        # Add task-specific parameters
        if task == "find_available_options":
            base_request["type"] = "hotels"
            base_request["destination"] = step.get("destination", "")
        elif task == "generate_recommendations":
            base_request["preferences"] = step.get("preferences", {})
        elif task == "process_booking":
            base_request["booking_details"] = step.get("booking_details", {})

        return base_request
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of active workflow"""
        return self.active_workflows.get(workflow_id, {"status": "not_found"})
    
    def _calculate_execution_time(self, start_time: str) -> float:
        """Calculate execution time in seconds"""
        from datetime import datetime
        start = datetime.fromisoformat(start_time)
        end = datetime.now()
        return (end - start).total_seconds()
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
