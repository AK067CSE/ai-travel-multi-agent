"""
Base Agent Class for Multi-Agent Travel System
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all travel agents"""
    
    def __init__(self, agent_name: str, model_name: str = "llama3-8b-8192"):
        self.agent_name = agent_name
        self.model_name = model_name
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0.7,
            max_tokens=2000
        )
        self.memory = {}  # Agent memory for context
        self.tools = []   # Agent-specific tools
        
        logger.info(f"Initialized {agent_name} with model {model_name}")
    
    @abstractmethod
    def create_prompt_template(self) -> ChatPromptTemplate:
        """Create agent-specific prompt template"""
        pass
    
    @abstractmethod
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request and return response"""
        pass
    
    def add_to_memory(self, key: str, value: Any):
        """Add information to agent memory"""
        self.memory[key] = value
        logger.debug(f"{self.agent_name} stored in memory: {key}")
    
    def get_from_memory(self, key: str) -> Any:
        """Retrieve information from agent memory"""
        return self.memory.get(key)
    
    def clear_memory(self):
        """Clear agent memory"""
        self.memory.clear()
        logger.debug(f"{self.agent_name} memory cleared")
    
    def create_chain(self, prompt_template: ChatPromptTemplate):
        """Create LangChain processing chain"""
        return prompt_template | self.llm | StrOutputParser()
    
    def log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log agent activity"""
        log_msg = f"{self.agent_name}: {activity}"
        if details:
            log_msg += f" - {details}"
        logger.info(log_msg)

class AgentResponse:
    """Standardized response format for all agents"""
    
    def __init__(self, agent_name: str, success: bool, data: Any = None, 
                 error: str = None, metadata: Dict[str, Any] = None):
        self.agent_name = agent_name
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = self._get_timestamp()
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

class AgentCommunication:
    """Handles communication between agents"""
    
    def __init__(self):
        self.message_queue = []
        self.agent_registry = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent for communication"""
        self.agent_registry[agent.agent_name] = agent
        logger.info(f"Registered agent: {agent.agent_name}")
    
    def send_message(self, from_agent: str, to_agent: str, message: Dict[str, Any]):
        """Send message between agents"""
        if to_agent not in self.agent_registry:
            logger.error(f"Agent {to_agent} not found in registry")
            return False
        
        message_envelope = {
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "timestamp": self._get_timestamp()
        }
        
        self.message_queue.append(message_envelope)
        logger.info(f"Message sent from {from_agent} to {to_agent}")
        return True
    
    def get_messages_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all messages for a specific agent"""
        messages = [msg for msg in self.message_queue if msg["to"] == agent_name]
        # Remove processed messages
        self.message_queue = [msg for msg in self.message_queue if msg["to"] != agent_name]
        return messages
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

# Global communication system
agent_communication = AgentCommunication()
