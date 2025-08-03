"""
Multi-LLM Manager for Free APIs
Supports Groq, Gemini, and Hugging Face models with fallback logic
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# LangChain imports
from langchain_groq import ChatGroq
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFacePipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    name: str
    provider: str
    model: str
    api_key_env: str
    max_tokens: int = 2000
    temperature: float = 0.3
    rate_limit: int = 60  # requests per minute
    cost_per_1k_tokens: float = 0.0  # Free tier
    priority: int = 1  # Lower number = higher priority

class MultiLLMManager:
    """
    Manages multiple free LLM providers with intelligent routing and fallback
    """
    
    def __init__(self):
        self.llm_configs = self._get_free_llm_configs()
        self.llms = {}
        self.usage_stats = {}
        self.last_request_time = {}
        
        # Initialize available LLMs
        self._initialize_llms()
        
        logger.info(f"MultiLLMManager initialized with {len(self.llms)} LLMs")
    
    def _get_free_llm_configs(self) -> List[LLMConfig]:
        """Get configurations for free LLM providers"""
        return [
            # Groq (Free tier: 14,400 requests/day)
            LLMConfig(
                name="groq_llama3_70b",
                provider="groq",
                model="llama3-70b-8192",
                api_key_env="GROQ_API_KEY",
                max_tokens=2000,
                temperature=0.3,
                rate_limit=30,  # Conservative rate limit
                priority=1
            ),
            LLMConfig(
                name="groq_llama3_8b",
                provider="groq", 
                model="llama3-8b-8192",
                api_key_env="GROQ_API_KEY",
                max_tokens=2000,
                temperature=0.3,
                rate_limit=60,
                priority=2
            ),
            # Google Gemini (Free tier: 60 requests/minute)
            LLMConfig(
                name="gemini_pro",
                provider="gemini",
                model="gemini-pro",
                api_key_env="GEMINI_API_KEY",
                max_tokens=2000,
                temperature=0.3,
                rate_limit=50,  # Conservative rate limit
                priority=3
            ),
            # Mixtral via Groq
            LLMConfig(
                name="groq_mixtral",
                provider="groq",
                model="mixtral-8x7b-32768",
                api_key_env="GROQ_API_KEY",
                max_tokens=2000,
                temperature=0.3,
                rate_limit=30,
                priority=4
            )
        ]
    
    def _initialize_llms(self) -> None:
        """Initialize available LLM providers"""
        for config in self.llm_configs:
            try:
                api_key = os.getenv(config.api_key_env)
                if not api_key:
                    logger.warning(f"API key not found for {config.name}")
                    continue
                
                if config.provider == "groq":
                    llm = ChatGroq(
                        groq_api_key=api_key,
                        model_name=config.model,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens
                    )
                elif config.provider == "gemini" and GEMINI_AVAILABLE:
                    llm = ChatGoogleGenerativeAI(
                        google_api_key=api_key,
                        model=config.model,
                        temperature=config.temperature,
                        max_output_tokens=config.max_tokens
                    )
                else:
                    logger.warning(f"Provider {config.provider} not available")
                    continue
                
                # Test the LLM with a simple query
                test_response = llm.invoke("Hello")
                if test_response:
                    self.llms[config.name] = {
                        'llm': llm,
                        'config': config
                    }
                    self.usage_stats[config.name] = {
                        'requests': 0,
                        'errors': 0,
                        'total_tokens': 0,
                        'avg_response_time': 0.0
                    }
                    self.last_request_time[config.name] = 0
                    logger.info(f"Successfully initialized {config.name}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize {config.name}: {e}")
    
    def get_best_llm(self, task_type: str = "general") -> Optional[Dict[str, Any]]:
        """Get the best available LLM based on current conditions"""
        if not self.llms:
            logger.error("No LLMs available")
            return None
        
        # Sort LLMs by priority and availability
        available_llms = []
        current_time = time.time()
        
        for name, llm_data in self.llms.items():
            config = llm_data['config']
            stats = self.usage_stats[name]
            
            # Check rate limiting
            time_since_last = current_time - self.last_request_time[name]
            if time_since_last < (60 / config.rate_limit):  # Rate limit check
                continue
            
            # Calculate availability score
            error_rate = stats['errors'] / max(stats['requests'], 1)
            availability_score = (1 - error_rate) * (1 / config.priority)
            
            available_llms.append({
                'name': name,
                'llm_data': llm_data,
                'score': availability_score
            })
        
        if not available_llms:
            # If all are rate limited, use the one with highest priority
            best_name = min(self.llms.keys(), key=lambda x: self.llms[x]['config'].priority)
            return self.llms[best_name]
        
        # Return the best scoring LLM
        best_llm = max(available_llms, key=lambda x: x['score'])
        return best_llm['llm_data']
    
    def generate_response(self, prompt: str, task_type: str = "general", max_retries: int = 3) -> Dict[str, Any]:
        """Generate response with automatic fallback"""
        for attempt in range(max_retries):
            llm_data = self.get_best_llm(task_type)
            if not llm_data:
                return {
                    'response': "I'm currently experiencing technical difficulties. Please try again later.",
                    'success': False,
                    'error': 'No LLMs available',
                    'llm_used': None
                }
            
            llm_name = None
            for name, data in self.llms.items():
                if data == llm_data:
                    llm_name = name
                    break
            
            try:
                start_time = time.time()
                
                # Update request tracking
                self.last_request_time[llm_name] = start_time
                self.usage_stats[llm_name]['requests'] += 1
                
                # Generate response
                llm = llm_data['llm']
                response = llm.invoke(prompt)
                
                # Update stats
                response_time = time.time() - start_time
                stats = self.usage_stats[llm_name]
                stats['avg_response_time'] = (
                    (stats['avg_response_time'] * (stats['requests'] - 1) + response_time) 
                    / stats['requests']
                )
                
                return {
                    'response': response.content if hasattr(response, 'content') else str(response),
                    'success': True,
                    'llm_used': llm_name,
                    'response_time': response_time,
                    'attempt': attempt + 1
                }
                
            except Exception as e:
                logger.warning(f"LLM {llm_name} failed (attempt {attempt + 1}): {e}")
                self.usage_stats[llm_name]['errors'] += 1
                
                if attempt == max_retries - 1:
                    return {
                        'response': f"I apologize, but I'm having trouble generating a response right now. Please try again later.",
                        'success': False,
                        'error': str(e),
                        'llm_used': llm_name,
                        'attempt': attempt + 1
                    }
        
        return {
            'response': "Maximum retries exceeded. Please try again later.",
            'success': False,
            'error': 'Max retries exceeded',
            'llm_used': None
        }
    
    def generate_with_template(self, template: str, variables: Dict[str, Any], task_type: str = "general") -> Dict[str, Any]:
        """Generate response using a template"""
        try:
            prompt_template = ChatPromptTemplate.from_template(template)
            formatted_prompt = prompt_template.format(**variables)
            return self.generate_response(formatted_prompt, task_type)
        except Exception as e:
            logger.error(f"Template generation error: {e}")
            return {
                'response': "Error in template processing.",
                'success': False,
                'error': str(e),
                'llm_used': None
            }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all LLMs"""
        return {
            'available_llms': list(self.llms.keys()),
            'total_llms': len(self.llms),
            'usage_stats': self.usage_stats,
            'configs': {name: {
                'provider': data['config'].provider,
                'model': data['config'].model,
                'priority': data['config'].priority,
                'rate_limit': data['config'].rate_limit
            } for name, data in self.llms.items()}
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all LLM providers"""
        health_status = {}
        
        for name, llm_data in self.llms.items():
            try:
                start_time = time.time()
                llm = llm_data['llm']
                response = llm.invoke("Hello")
                response_time = time.time() - start_time
                
                health_status[name] = {
                    'status': 'healthy',
                    'response_time': response_time,
                    'last_check': time.time()
                }
            except Exception as e:
                health_status[name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'last_check': time.time()
                }
        
        return health_status
