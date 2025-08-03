"""
Production-Level AI Travel Agent System
Enhanced with RAG, Multi-Agent Architecture, and Advanced NLP
Combines the best features from the complex system into a production-ready implementation
"""

import openai
import json
import time
import logging
import os
from typing import Dict, List, Any, Optional
from django.conf import settings

# Import enhanced systems
from .enhanced_rag import EnhancedTravelRAG
from .multi_agent_system import CoordinatorAgent
from .crewai_system import CrewAITravelSystem

# LangChain imports
try:
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available")

logger = logging.getLogger(__name__)


class TravelAIAgent:
    """
    Production-Level Travel AI Agent with RAG, Multi-Agent Architecture, and Advanced NLP
    Features:
    - Enhanced RAG with BGE embeddings and hybrid retrieval
    - Multi-agent coordination for complex queries
    - Multiple LLM providers (Groq, OpenAI, Anthropic)
    - Conversational memory and context awareness
    - Production-ready error handling and fallbacks
    """

    def __init__(self):
        # Load configuration
        self.ai_config = getattr(settings, 'AI_CONFIG', {})

        # API Keys
        self.openai_api_key = self.ai_config.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.groq_api_key = self.ai_config.get('GROQ_API_KEY') or os.getenv('GROQ_API_KEY')
        self.anthropic_api_key = self.ai_config.get('ANTHROPIC_API_KEY') or os.getenv('ANTHROPIC_API_KEY')

        # Model configuration
        self.default_provider = os.getenv('DEFAULT_LLM_PROVIDER', 'groq')
        self.default_model = os.getenv('DEFAULT_MODEL', 'llama3-70b-8192')
        self.fallback_model = os.getenv('FALLBACK_MODEL', 'llama3-8b-8192')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '2000'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))

        # Initialize LLM clients
        self.llm_clients = {}
        self._initialize_llm_clients()

        # Initialize enhanced systems
        self.rag_system = None
        self.multi_agent_system = None
        self.crewai_system = None
        self.enable_multi_agent = os.getenv('ENABLE_MULTI_AGENT', 'True').lower() == 'true'
        self.enable_crewai = os.getenv('ENABLE_CREWAI', 'True').lower() == 'true'

        self._initialize_enhanced_systems()

        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'rag_requests': 0,
            'multi_agent_requests': 0,
            'average_response_time': 0,
            'system_health': 'excellent'
        }

        # Backward compatibility - basic travel knowledge
        self.travel_knowledge = {
            'destinations': {
                'paris': {'country': 'France', 'highlights': ['Eiffel Tower', 'Louvre'], 'budget': 'mid-range'},
                'tokyo': {'country': 'Japan', 'highlights': ['Shibuya', 'Temples'], 'budget': 'mid-range'},
                'bali': {'country': 'Indonesia', 'highlights': ['Beaches', 'Culture'], 'budget': 'budget'}
            },
            'travel_types': {
                'romantic': 'Perfect for couples',
                'adventure': 'For thrill seekers',
                'cultural': 'Rich history and traditions',
                'budget': 'Affordable travel options'
            }
        }

        logger.info("Production Travel AI Agent initialized with enhanced capabilities")

    def _initialize_llm_clients(self):
        """Initialize multiple LLM providers for redundancy"""
        try:
            # Initialize Groq client
            if self.groq_api_key and LANGCHAIN_AVAILABLE:
                self.llm_clients['groq'] = ChatGroq(
                    groq_api_key=self.groq_api_key,
                    model_name=self.default_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                logger.info("Groq LLM client initialized")

            # Initialize OpenAI client
            if self.openai_api_key:
                if LANGCHAIN_AVAILABLE:
                    self.llm_clients['openai'] = ChatOpenAI(
                        openai_api_key=self.openai_api_key,
                        model_name="gpt-3.5-turbo",
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                # Also initialize legacy OpenAI client
                openai.api_key = self.openai_api_key
                logger.info("OpenAI LLM client initialized")

            # Initialize Anthropic client
            if self.anthropic_api_key and LANGCHAIN_AVAILABLE:
                self.llm_clients['anthropic'] = ChatAnthropic(
                    anthropic_api_key=self.anthropic_api_key,
                    model="claude-3-sonnet-20240229",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                logger.info("Anthropic LLM client initialized")

        except Exception as e:
            logger.error(f"Error initializing LLM clients: {e}")

    def _initialize_enhanced_systems(self):
        """Initialize RAG and Multi-Agent systems"""
        try:
            # Initialize Enhanced RAG System
            logger.info("Initializing Enhanced RAG System...")
            self.rag_system = EnhancedTravelRAG()

            # Load and index data
            if self.rag_system.load_and_index_data():
                logger.info("RAG system initialized successfully")
            else:
                logger.warning("RAG system initialized with limited functionality")

            # Initialize CrewAI System (preferred)
            if self.enable_crewai:
                logger.info("Initializing CrewAI Multi-Agent System...")
                try:
                    self.crewai_system = CrewAITravelSystem()
                    logger.info("CrewAI system initialized successfully")
                except Exception as e:
                    logger.error(f"CrewAI initialization failed: {e}")
                    self.crewai_system = None

            # Initialize Custom Multi-Agent System (fallback)
            if self.enable_multi_agent and not self.crewai_system:
                logger.info("Initializing Custom Multi-Agent System...")
                self.multi_agent_system = CoordinatorAgent()
                logger.info("Custom Multi-Agent system initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing enhanced systems: {e}")
            # Continue with basic functionality

    def get_primary_llm(self):
        """Get primary LLM client based on configuration"""
        if self.default_provider in self.llm_clients:
            return self.llm_clients[self.default_provider]

        # Fallback to any available client
        for provider, client in self.llm_clients.items():
            if client:
                logger.info(f"Using fallback LLM provider: {provider}")
                return client

        return None
        
        # Travel knowledge base (simplified)
        self.travel_knowledge = {
            "destinations": {
                "paris": {
                    "country": "France",
                    "best_time": "April-June, September-October",
                    "highlights": ["Eiffel Tower", "Louvre Museum", "Notre-Dame", "Champs-Élysées"],
                    "budget": "Medium to High",
                    "culture": "Art, Fashion, Cuisine"
                },
                "tokyo": {
                    "country": "Japan",
                    "best_time": "March-May, September-November",
                    "highlights": ["Shibuya Crossing", "Senso-ji Temple", "Mount Fuji", "Tsukiji Market"],
                    "budget": "Medium to High",
                    "culture": "Traditional meets Modern"
                },
                "bali": {
                    "country": "Indonesia",
                    "best_time": "April-October",
                    "highlights": ["Ubud Rice Terraces", "Tanah Lot Temple", "Seminyak Beach", "Mount Batur"],
                    "budget": "Low to Medium",
                    "culture": "Hindu temples, Beach life"
                }
            },
            "travel_types": {
                "romantic": ["Paris", "Santorini", "Venice", "Maldives"],
                "adventure": ["Nepal", "New Zealand", "Costa Rica", "Iceland"],
                "cultural": ["Japan", "India", "Egypt", "Peru"],
                "budget": ["Thailand", "Vietnam", "Guatemala", "Eastern Europe"],
                "luxury": ["Maldives", "Dubai", "Switzerland", "Monaco"]
            }
        }
    
    def process_travel_request(self, user_message: str, user_preferences: Dict = None, session_id: str = None) -> Dict[str, Any]:
        """
        Process user travel request using enhanced AI capabilities
        Features:
        - Multi-agent coordination for complex queries
        - Enhanced RAG for contextual responses
        - Multiple LLM providers with fallbacks
        - Intent analysis and personalization
        """
        start_time = time.time()
        self.performance_metrics['total_requests'] += 1

        try:
            # Analyze user intent and complexity
            intent = self._analyze_user_intent(user_message)
            complexity = self._assess_query_complexity(user_message, intent)

            logger.info(f"Processing request - Intent: {intent.get('type', 'general')}, Complexity: {complexity}")

            # Choose processing strategy based on complexity and available systems
            if complexity == 'complex' and self.crewai_system and self.enable_crewai:
                # Use CrewAI system for complex queries (preferred)
                result = self._process_with_crewai(user_message, user_preferences, session_id)
                self.performance_metrics['multi_agent_requests'] += 1

            elif complexity == 'complex' and self.multi_agent_system and self.enable_multi_agent:
                # Use custom multi-agent system for complex queries (fallback)
                result = self._process_with_multi_agent(user_message, user_preferences, session_id)
                self.performance_metrics['multi_agent_requests'] += 1

            elif self.rag_system:
                # Use enhanced RAG system
                result = self._process_with_rag(user_message, user_preferences, session_id)
                self.performance_metrics['rag_requests'] += 1

            else:
                # Use enhanced single-agent approach
                result = self._process_with_enhanced_single_agent(user_message, intent, user_preferences, session_id)

            # Add metadata and performance tracking
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['intent'] = intent
            result['session_id'] = session_id or f"session_{int(time.time())}"
            result['complexity'] = complexity

            # Update performance metrics
            if result.get('success', False):
                self.performance_metrics['successful_requests'] += 1

            self._update_performance_metrics(processing_time)

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"AI Agent error: {e}")

            return {
                'success': False,
                'response': "I apologize, but I'm experiencing some technical difficulties. Let me help you with basic travel information instead.",
                'system_used': 'Error_Handler',
                'agents_involved': ['Error_Handler'],
                'processing_time': processing_time,
                'error': str(e),
                'session_id': session_id or f"session_{int(time.time())}"
            }

    def _assess_query_complexity(self, user_message: str, intent: Dict) -> str:
        """Assess query complexity to determine processing strategy"""
        message_lower = user_message.lower()

        # Complex indicators
        complex_indicators = [
            'plan', 'itinerary', 'schedule', 'detailed', 'comprehensive',
            'multiple', 'several', 'compare', 'best options', 'recommendations'
        ]

        # Simple indicators
        simple_indicators = [
            'hello', 'hi', 'thanks', 'what is', 'where is', 'how much'
        ]

        # Check for complex patterns
        if any(indicator in message_lower for indicator in complex_indicators):
            return 'complex'

        # Check for simple patterns
        if any(indicator in message_lower for indicator in simple_indicators):
            return 'simple'

        # Check intent-based complexity
        if intent.get('travel_style') and intent.get('destinations') and len(user_message.split()) > 10:
            return 'complex'

        return 'medium'

    def _process_with_crewai(self, user_message: str, user_preferences: Dict, session_id: str) -> Dict[str, Any]:
        """Process request using CrewAI multi-agent system"""
        try:
            logger.info("Using CrewAI multi-agent system for complex query")

            result = self.crewai_system.process_travel_request(user_message, user_preferences)

            if result.get('success', False):
                return {
                    'success': True,
                    'response': result['response'],
                    'system_used': result.get('system_used', 'CrewAI_Multi_Agent'),
                    'agents_involved': result.get('agents_involved', ['CrewAI_Agents']),
                    'metadata': {
                        'tasks_executed': result.get('tasks_executed', 0),
                        'processing_time': result.get('processing_time', 0)
                    }
                }
            else:
                # Fallback to custom multi-agent system
                logger.warning("CrewAI system failed, falling back to custom multi-agent")
                if self.multi_agent_system:
                    return self._process_with_multi_agent(user_message, user_preferences, session_id)
                else:
                    return self._process_with_rag(user_message, user_preferences, session_id)

        except Exception as e:
            logger.error(f"CrewAI processing error: {e}")
            # Fallback to custom multi-agent system
            if self.multi_agent_system:
                return self._process_with_multi_agent(user_message, user_preferences, session_id)
            else:
                return self._process_with_rag(user_message, user_preferences, session_id)

    def _process_with_multi_agent(self, user_message: str, user_preferences: Dict, session_id: str) -> Dict[str, Any]:
        """Process request using multi-agent system"""
        try:
            logger.info("Using multi-agent system for complex query")

            result = self.multi_agent_system.process_request({
                'request': user_message,
                'user_id': session_id,
                'preferences': user_preferences or {}
            })

            if result.get('success', False):
                return {
                    'success': True,
                    'response': result['response'],
                    'system_used': result.get('system_used', 'Multi_Agent_System'),
                    'agents_involved': result.get('agents_involved', ['CoordinatorAgent']),
                    'metadata': result.get('metadata', {})
                }
            else:
                # Fallback to RAG system
                logger.warning("Multi-agent system failed, falling back to RAG")
                return self._process_with_rag(user_message, user_preferences, session_id)

        except Exception as e:
            logger.error(f"Multi-agent processing error: {e}")
            return self._process_with_rag(user_message, user_preferences, session_id)

    def _process_with_rag(self, user_message: str, user_preferences: Dict, session_id: str) -> Dict[str, Any]:
        """Process request using enhanced RAG system"""
        try:
            logger.info("Using enhanced RAG system")

            result = self.rag_system.generate_rag_response(user_message, session_id)

            return {
                'success': True,
                'response': result['response'],
                'system_used': result.get('system_used', 'Enhanced_RAG'),
                'agents_involved': ['RAG_Agent'],
                'metadata': {
                    'retrieved_docs': result.get('retrieved_docs', 0),
                    'sources': result.get('sources', []),
                    'context_used': result.get('context_used', False)
                }
            }

        except Exception as e:
            logger.error(f"RAG processing error: {e}")
            return self._process_with_enhanced_single_agent(user_message, {}, user_preferences, session_id)

    def _process_with_enhanced_single_agent(self, user_message: str, intent: Dict, user_preferences: Dict, session_id: str) -> Dict[str, Any]:
        """Process request using enhanced single agent with multiple LLM fallbacks"""
        try:
            logger.info("Using enhanced single agent")

            # Try primary LLM first
            primary_llm = self.get_primary_llm()
            if primary_llm:
                response = self._generate_enhanced_response(user_message, intent, user_preferences, primary_llm)
                if response:
                    return {
                        'success': True,
                        'response': response,
                        'system_used': f'Enhanced_Single_Agent_{self.default_provider}',
                        'agents_involved': ['Enhanced_Travel_Agent'],
                        'metadata': {'llm_provider': self.default_provider}
                    }

            # Try legacy OpenAI if available
            if self.openai_api_key:
                response = self._generate_legacy_openai_response(user_message, intent, user_preferences)
                if response:
                    return {
                        'success': True,
                        'response': response,
                        'system_used': 'Enhanced_Single_Agent_OpenAI_Legacy',
                        'agents_involved': ['Enhanced_Travel_Agent'],
                        'metadata': {'llm_provider': 'openai_legacy'}
                    }

            # Final fallback to knowledge base
            return {
                'success': True,
                'response': self._generate_enhanced_fallback_response(user_message, intent),
                'system_used': 'Enhanced_Knowledge_Base',
                'agents_involved': ['Knowledge_Base_Agent'],
                'metadata': {'fallback_reason': 'no_llm_available'}
            }

        except Exception as e:
            logger.error(f"Enhanced single agent error: {e}")
            return {
                'success': False,
                'response': "I apologize for the technical difficulties. Let me provide some basic travel assistance.",
                'system_used': 'Error_Fallback',
                'agents_involved': ['Error_Handler'],
                'error': str(e)
            }

    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        # Update average response time
        total_requests = self.performance_metrics['total_requests']
        current_avg = self.performance_metrics['average_response_time']

        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.performance_metrics['average_response_time'] = new_avg

        # Update system health based on performance
        success_rate = self.performance_metrics['successful_requests'] / total_requests

        if success_rate > 0.95 and new_avg < 3.0:
            self.performance_metrics['system_health'] = 'excellent'
        elif success_rate > 0.85 and new_avg < 5.0:
            self.performance_metrics['system_health'] = 'good'
        else:
            self.performance_metrics['system_health'] = 'degraded'
    
    def _analyze_user_intent(self, message: str) -> Dict[str, Any]:
        """
        Analyze user message to determine travel intent
        """
        message_lower = message.lower()
        
        intent = {
            'type': 'general',
            'destinations': [],
            'travel_style': None,
            'budget': None,
            'duration': None,
            'interests': []
        }
        
        # Detect travel type
        if any(word in message_lower for word in ['romantic', 'honeymoon', 'couple']):
            intent['travel_style'] = 'romantic'
        elif any(word in message_lower for word in ['adventure', 'hiking', 'extreme']):
            intent['travel_style'] = 'adventure'
        elif any(word in message_lower for word in ['cultural', 'culture', 'museum', 'history']):
            intent['travel_style'] = 'cultural'
        elif any(word in message_lower for word in ['budget', 'cheap', 'affordable']):
            intent['travel_style'] = 'budget'
        elif any(word in message_lower for word in ['luxury', 'premium', 'high-end']):
            intent['travel_style'] = 'luxury'
        
        # Detect destinations
        for dest in self.travel_knowledge['destinations'].keys():
            if dest in message_lower:
                intent['destinations'].append(dest)
        
        # Detect duration
        if any(word in message_lower for word in ['day', 'days']):
            import re
            days_match = re.search(r'(\d+)\s*days?', message_lower)
            if days_match:
                intent['duration'] = f"{days_match.group(1)} days"
        
        return intent
    
    def _generate_ai_response(self, message: str, intent: Dict, preferences: Dict = None) -> str:
        """
        Generate AI response using OpenAI
        """
        system_prompt = """You are an expert travel agent AI assistant. You provide personalized, detailed travel recommendations and planning advice. 

Your responses should be:
- Informative and practical
- Personalized based on user preferences
- Include specific recommendations
- Be enthusiastic but professional
- Format responses in a clear, readable way

Always include practical details like best times to visit, budget considerations, and must-see attractions."""

        user_prompt = f"""
User Request: {message}

Travel Intent Analysis:
- Travel Style: {intent.get('travel_style', 'general')}
- Mentioned Destinations: {', '.join(intent.get('destinations', []))}
- Duration: {intent.get('duration', 'not specified')}

User Preferences: {json.dumps(preferences or {}, indent=2)}

Please provide a comprehensive travel response that addresses their specific needs and includes practical recommendations.
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._generate_fallback_response(message, intent)
    
    def _generate_fallback_response(self, message: str, intent: Dict) -> str:
        """
        Generate fallback response when AI is not available
        """
        travel_style = intent.get('travel_style', 'general')
        destinations = intent.get('destinations', [])
        
        if destinations:
            dest_info = []
            for dest in destinations:
                if dest in self.travel_knowledge['destinations']:
                    info = self.travel_knowledge['destinations'][dest]
                    dest_info.append(f"**{dest.title()}**: {info['highlights'][:2]} are must-sees. Best time to visit: {info['best_time']}.")
            
            if dest_info:
                return f"Great choice! Here's what I know about your destinations:\n\n" + "\n\n".join(dest_info)
        
        if travel_style and travel_style in self.travel_knowledge['travel_types']:
            recommended_destinations = self.travel_knowledge['travel_types'][travel_style][:3]
            return f"For {travel_style} travel, I highly recommend: {', '.join(recommended_destinations)}. Each offers unique experiences perfect for your travel style!"
        
        return "I'd be happy to help you plan your trip! Could you tell me more about your preferred destinations, travel style, or what kind of experiences you're looking for?"
    
    def _get_travel_recommendations(self, intent: Dict, preferences: Dict = None) -> List[Dict]:
        """
        Get specific travel recommendations based on intent
        """
        recommendations = []
        travel_style = intent.get('travel_style')
        
        if travel_style and travel_style in self.travel_knowledge['travel_types']:
            destinations = self.travel_knowledge['travel_types'][travel_style][:3]
            for dest in destinations:
                if dest.lower() in self.travel_knowledge['destinations']:
                    dest_info = self.travel_knowledge['destinations'][dest.lower()]
                    recommendations.append({
                        'destination': dest,
                        'country': dest_info.get('country', 'Unknown'),
                        'highlights': dest_info.get('highlights', [])[:2],
                        'best_time': dest_info.get('best_time', 'Year-round'),
                        'budget': dest_info.get('budget', 'Medium')
                    })
        
        return recommendations
    
    def _format_response(self, ai_response: str, recommendations: List[Dict], intent: Dict) -> str:
        """
        Format the final response with recommendations
        """
        formatted_response = ai_response
        
        if recommendations:
            formatted_response += "\n\n## 🌟 Personalized Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                formatted_response += f"**{i}. {rec['destination']}, {rec['country']}**\n"
                formatted_response += f"- Best Time: {rec['best_time']}\n"
                formatted_response += f"- Budget: {rec['budget']}\n"
                formatted_response += f"- Highlights: {', '.join(rec['highlights'])}\n\n"
        
        return formatted_response
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status for production monitoring
        """
        # Get RAG system stats
        rag_stats = {}
        if self.rag_system:
            try:
                rag_stats = self.rag_system.get_stats()
            except Exception as e:
                logger.error(f"Error getting RAG stats: {e}")
                rag_stats = {"error": str(e)}

        # Check LLM availability
        llm_status = {}
        for provider, client in self.llm_clients.items():
            llm_status[provider] = client is not None

        return {
            'status': 'operational',
            'system_health': self.performance_metrics['system_health'],
            'ai_model': self.default_model,
            'primary_provider': self.default_provider,

            # LLM Provider Status
            'llm_providers': llm_status,
            'openai_available': bool(self.openai_api_key),
            'groq_available': bool(self.groq_api_key),
            'anthropic_available': bool(self.anthropic_api_key),

            # Enhanced System Status
            'rag_system_available': self.rag_system is not None,
            'crewai_system_available': self.crewai_system is not None,
            'crewai_enabled': self.enable_crewai,
            'multi_agent_available': self.multi_agent_system is not None,
            'multi_agent_enabled': self.enable_multi_agent,

            # Performance Metrics
            'performance': {
                'total_requests': self.performance_metrics['total_requests'],
                'successful_requests': self.performance_metrics['successful_requests'],
                'success_rate': (self.performance_metrics['successful_requests'] / max(1, self.performance_metrics['total_requests'])) * 100,
                'average_response_time': round(self.performance_metrics['average_response_time'], 2),
                'rag_requests': self.performance_metrics['rag_requests'],
                'multi_agent_requests': self.performance_metrics['multi_agent_requests']
            },

            # RAG System Details
            'rag_system': rag_stats,

            # Legacy compatibility
            'knowledge_base_destinations': rag_stats.get('total_documents', len(self.travel_knowledge.get('destinations', {}))),
            'travel_types_supported': len(self.travel_knowledge.get('travel_types', {})),

            # System capabilities
            'capabilities': {
                'enhanced_rag': bool(self.rag_system),
                'crewai_multi_agent': bool(self.crewai_system),
                'custom_multi_agent_coordination': bool(self.multi_agent_system),
                'multiple_llm_providers': len(self.llm_clients) > 1,
                'conversational_memory': bool(self.rag_system),
                'intent_analysis': True,
                'query_expansion': bool(self.rag_system),
                'hybrid_retrieval': bool(self.rag_system),
                'production_monitoring': True
            },

            'timestamp': time.time()
        }
