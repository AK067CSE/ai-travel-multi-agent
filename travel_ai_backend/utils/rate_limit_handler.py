"""
Rate Limit Handler for LLM APIs
Intelligent handling of rate limits with fallback strategies
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)

class RateLimitHandler:
    """
    Intelligent rate limit handler with multiple fallback strategies
    """
    
    def __init__(self):
        self.rate_limits = {}
        self.fallback_strategies = []
        self.retry_delays = [1, 2, 5, 10, 30]  # Progressive delays in seconds
        
        logger.info("Rate Limit Handler initialized")
    
    def add_fallback_strategy(self, strategy: Callable):
        """Add a fallback strategy function"""
        self.fallback_strategies.append(strategy)
    
    def handle_rate_limit(self, provider: str, error_message: str) -> Dict[str, Any]:
        """
        Handle rate limit errors intelligently
        """
        try:
            # Parse rate limit information
            rate_info = self._parse_rate_limit_error(error_message)
            
            # Store rate limit info
            self.rate_limits[provider] = {
                'hit_at': datetime.now(),
                'retry_after': rate_info.get('retry_after', 60),
                'limit': rate_info.get('limit', 'unknown'),
                'used': rate_info.get('used', 'unknown'),
                'requested': rate_info.get('requested', 'unknown')
            }
            
            logger.warning(f"Rate limit hit for {provider}: {rate_info}")
            
            # Return fallback strategy
            return {
                'status': 'rate_limited',
                'provider': provider,
                'retry_after': rate_info.get('retry_after', 60),
                'fallback_available': len(self.fallback_strategies) > 0,
                'recommendation': self._get_recommendation(provider, rate_info)
            }
            
        except Exception as e:
            logger.error(f"Rate limit handling error: {e}")
            return {
                'status': 'error',
                'message': 'Failed to handle rate limit'
            }
    
    def _parse_rate_limit_error(self, error_message: str) -> Dict[str, Any]:
        """Parse rate limit error message to extract useful information"""
        rate_info = {}
        
        try:
            # Parse Groq rate limit format
            if "Rate limit reached" in error_message:
                # Extract retry time
                if "Please try again in" in error_message:
                    import re
                    retry_match = re.search(r'try again in ([\d.]+)s', error_message)
                    if retry_match:
                        rate_info['retry_after'] = float(retry_match.group(1))
                
                # Extract limits
                if "Limit" in error_message and "Used" in error_message:
                    limit_match = re.search(r'Limit (\d+)', error_message)
                    used_match = re.search(r'Used (\d+)', error_message)
                    requested_match = re.search(r'Requested (\d+)', error_message)
                    
                    if limit_match:
                        rate_info['limit'] = int(limit_match.group(1))
                    if used_match:
                        rate_info['used'] = int(used_match.group(1))
                    if requested_match:
                        rate_info['requested'] = int(requested_match.group(1))
            
            return rate_info
            
        except Exception as e:
            logger.error(f"Error parsing rate limit message: {e}")
            return {'retry_after': 60}  # Default fallback
    
    def _get_recommendation(self, provider: str, rate_info: Dict[str, Any]) -> str:
        """Get recommendation for handling the rate limit"""
        retry_after = rate_info.get('retry_after', 60)
        
        if retry_after < 30:
            return f"Short wait required ({retry_after}s). Consider using fallback or waiting."
        elif retry_after < 300:  # 5 minutes
            return f"Medium wait required ({retry_after}s). Recommend using fallback system."
        else:
            return f"Long wait required ({retry_after}s). Strongly recommend fallback or alternative provider."
    
    def can_retry(self, provider: str) -> bool:
        """Check if we can retry a provider"""
        if provider not in self.rate_limits:
            return True
        
        rate_limit_info = self.rate_limits[provider]
        hit_time = rate_limit_info['hit_at']
        retry_after = rate_limit_info['retry_after']
        
        # Check if enough time has passed
        time_passed = (datetime.now() - hit_time).total_seconds()
        return time_passed >= retry_after
    
    def get_next_available_time(self, provider: str) -> Optional[datetime]:
        """Get the next time this provider will be available"""
        if provider not in self.rate_limits:
            return datetime.now()  # Available now
        
        rate_limit_info = self.rate_limits[provider]
        hit_time = rate_limit_info['hit_at']
        retry_after = rate_limit_info['retry_after']
        
        return hit_time + timedelta(seconds=retry_after)
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for all providers"""
        status = {}
        
        for provider, info in self.rate_limits.items():
            next_available = self.get_next_available_time(provider)
            can_retry = self.can_retry(provider)
            
            status[provider] = {
                'rate_limited': not can_retry,
                'next_available': next_available.isoformat(),
                'can_retry_now': can_retry,
                'limit_info': {
                    'limit': info.get('limit', 'unknown'),
                    'used': info.get('used', 'unknown'),
                    'requested': info.get('requested', 'unknown')
                }
            }
        
        return status
    
    async def execute_with_retry(self, 
                                func: Callable, 
                                provider: str, 
                                max_retries: int = 3,
                                *args, **kwargs) -> Any:
        """
        Execute a function with intelligent retry logic
        """
        for attempt in range(max_retries):
            try:
                # Check if we can retry this provider
                if not self.can_retry(provider):
                    next_available = self.get_next_available_time(provider)
                    wait_time = (next_available - datetime.now()).total_seconds()
                    
                    if wait_time > 0:
                        logger.info(f"Waiting {wait_time:.1f}s for {provider} rate limit to reset")
                        await asyncio.sleep(min(wait_time, 30))  # Cap wait time
                
                # Execute the function
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Success - clear any rate limit info
                if provider in self.rate_limits:
                    del self.rate_limits[provider]
                
                return result
                
            except Exception as e:
                error_message = str(e)
                
                # Check if it's a rate limit error
                if "rate limit" in error_message.lower() or "RateLimitError" in str(type(e)):
                    self.handle_rate_limit(provider, error_message)
                    
                    if attempt < max_retries - 1:
                        delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                        logger.info(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for {provider}")
                        raise
                else:
                    # Non-rate-limit error, re-raise immediately
                    raise
        
        raise Exception(f"Failed to execute after {max_retries} attempts")

def rate_limit_handler(provider: str, max_retries: int = 3):
    """
    Decorator for automatic rate limit handling
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = RateLimitHandler()
            return await handler.execute_with_retry(func, provider, max_retries, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            handler = RateLimitHandler()
            # For sync functions, we'll use a simple retry without async
            for attempt in range(max_retries):
                try:
                    if not handler.can_retry(provider):
                        next_available = handler.get_next_available_time(provider)
                        wait_time = (next_available - datetime.now()).total_seconds()
                        if wait_time > 0 and wait_time < 30:  # Only wait if reasonable
                            time.sleep(wait_time)
                    
                    result = func(*args, **kwargs)
                    
                    # Success - clear rate limit
                    if provider in handler.rate_limits:
                        del handler.rate_limits[provider]
                    
                    return result
                    
                except Exception as e:
                    error_message = str(e)
                    if "rate limit" in error_message.lower():
                        handler.handle_rate_limit(provider, error_message)
                        if attempt < max_retries - 1:
                            delay = handler.retry_delays[min(attempt, len(handler.retry_delays) - 1)]
                            time.sleep(delay)
                            continue
                    raise
            
            raise Exception(f"Failed after {max_retries} attempts")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# Global rate limit handler instance
global_rate_handler = RateLimitHandler()

def get_rate_limit_status() -> Dict[str, Any]:
    """Get global rate limit status"""
    return global_rate_handler.get_rate_limit_status()

def add_global_fallback(strategy: Callable):
    """Add a global fallback strategy"""
    global_rate_handler.add_fallback_strategy(strategy)
