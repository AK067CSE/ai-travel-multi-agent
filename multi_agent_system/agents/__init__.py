"""
Multi-Agent Travel AI System
Built with LangChain + Groq for comprehensive travel assistance
"""

__version__ = "1.0.0"
__author__ = "Travel AI Team"

from .scraping_agent import ScrapingAgent
from .recommendation_agent import RecommendationAgent
from .booking_agent import BookingAgent
from .chat_agent import ChatAgent
from .coordinator_agent import CoordinatorAgent

__all__ = [
    "ScrapingAgent",
    "RecommendationAgent", 
    "BookingAgent",
    "ChatAgent",
    "CoordinatorAgent"
]
