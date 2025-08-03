"""
Scraping Agent - Handles web scraping and data collection
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import random
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tools import Tool
from .base_agent import BaseAgent, AgentResponse
import logging

logger = logging.getLogger(__name__)

class ScrapingAgent(BaseAgent):
    """Agent responsible for web scraping travel data"""
    
    def __init__(self, model_name: str = "llama3-8b-8192"):
        super().__init__("ScrapingAgent", model_name)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.scraped_cache = {}
        self.setup_tools()
    
    def create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for scraping analysis"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert web scraping analyst for travel data. Your role is to:
            1. Analyze scraped travel data for quality and relevance
            2. Extract key information from travel websites
            3. Identify the most valuable data points for travelers
            4. Suggest improvements for scraping strategies
            
            Focus on: hotels, flights, attractions, restaurants, reviews, and pricing data.
            Provide structured analysis and actionable insights."""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            "Analyze this scraped travel data and provide insights: {scraped_data}"
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def setup_tools(self):
        """Setup scraping tools"""
        self.tools = [
            Tool.from_function(
                func=self.scrape_hotels,
                name="hotel_scraper",
                description="Scrape hotel data from travel websites"
            ),
            Tool.from_function(
                func=self.scrape_attractions,
                name="attraction_scraper", 
                description="Scrape tourist attraction data"
            ),
            Tool.from_function(
                func=self.scrape_restaurants,
                name="restaurant_scraper",
                description="Scrape restaurant and dining data"
            )
        ]
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process scraping request"""
        try:
            scraping_type = request.get("type", "general")
            destination = request.get("destination", "")
            
            self.log_activity(f"Processing scraping request", {
                "type": scraping_type,
                "destination": destination
            })
            
            if scraping_type == "hotels":
                data = self.scrape_hotels(destination)
            elif scraping_type == "attractions":
                data = self.scrape_attractions(destination)
            elif scraping_type == "restaurants":
                data = self.scrape_restaurants(destination)
            else:
                data = self.scrape_general_travel_data(destination)
            
            # Analyze scraped data with LLM
            analysis = self.analyze_scraped_data(data)
            
            response = AgentResponse(
                agent_name=self.agent_name,
                success=True,
                data={
                    "scraped_data": data,
                    "analysis": analysis,
                    "destination": destination,
                    "type": scraping_type
                },
                metadata={"data_count": len(data) if isinstance(data, list) else 1}
            )
            
            return response.to_dict()
            
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            response = AgentResponse(
                agent_name=self.agent_name,
                success=False,
                error=str(e)
            )
            return response.to_dict()
    
    def scrape_hotels(self, destination: str) -> List[Dict[str, Any]]:
        """Scrape hotel data for destination"""
        # Cache check
        cache_key = f"hotels_{destination}"
        if cache_key in self.scraped_cache:
            return self.scraped_cache[cache_key]
        
        hotels = []
        
        # Simulate scraping from multiple sources
        sources = [
            f"https://www.booking.com/searchresults.html?ss={destination}",
            f"https://www.hotels.com/search.do?destination={destination}",
            f"https://www.expedia.com/Hotels-Search?destination={destination}"
        ]
        
        for source in sources:
            try:
                # Add delay to be respectful
                time.sleep(random.uniform(1, 3))
                
                # Simulate hotel data (in real implementation, parse actual HTML)
                sample_hotels = [
                    {
                        "name": f"Hotel {i} in {destination}",
                        "price": random.randint(80, 300),
                        "rating": round(random.uniform(3.5, 5.0), 1),
                        "amenities": ["WiFi", "Pool", "Gym"],
                        "source": source
                    }
                    for i in range(1, 4)
                ]
                
                hotels.extend(sample_hotels)
                
            except Exception as e:
                logger.warning(f"Failed to scrape {source}: {e}")
                continue
        
        # Cache results
        self.scraped_cache[cache_key] = hotels
        return hotels
    
    def scrape_attractions(self, destination: str) -> List[Dict[str, Any]]:
        """Scrape tourist attraction data"""
        cache_key = f"attractions_{destination}"
        if cache_key in self.scraped_cache:
            return self.scraped_cache[cache_key]
        
        attractions = [
            {
                "name": f"Famous Landmark in {destination}",
                "type": "Historical Site",
                "rating": round(random.uniform(4.0, 5.0), 1),
                "price": random.choice([0, 10, 15, 25]),
                "description": f"Must-visit attraction in {destination}"
            },
            {
                "name": f"Museum of {destination}",
                "type": "Museum",
                "rating": round(random.uniform(3.8, 4.8), 1),
                "price": random.randint(12, 30),
                "description": f"Learn about the history and culture of {destination}"
            }
        ]
        
        self.scraped_cache[cache_key] = attractions
        return attractions
    
    def scrape_restaurants(self, destination: str) -> List[Dict[str, Any]]:
        """Scrape restaurant data"""
        cache_key = f"restaurants_{destination}"
        if cache_key in self.scraped_cache:
            return self.scraped_cache[cache_key]
        
        restaurants = [
            {
                "name": f"Local Cuisine Restaurant in {destination}",
                "cuisine": "Local",
                "rating": round(random.uniform(4.0, 5.0), 1),
                "price_range": "$$",
                "specialties": ["Local Dish 1", "Local Dish 2"]
            },
            {
                "name": f"International Restaurant in {destination}",
                "cuisine": "International",
                "rating": round(random.uniform(3.5, 4.5), 1),
                "price_range": "$$$",
                "specialties": ["Pasta", "Steaks", "Seafood"]
            }
        ]
        
        self.scraped_cache[cache_key] = restaurants
        return restaurants
    
    def scrape_general_travel_data(self, destination: str) -> Dict[str, Any]:
        """Scrape general travel information"""
        return {
            "destination": destination,
            "best_time_to_visit": "Spring and Fall",
            "average_temperature": "20-25°C",
            "currency": "Local Currency",
            "language": "Local Language",
            "time_zone": "UTC+0",
            "visa_requirements": "Check with embassy"
        }
    
    def analyze_scraped_data(self, data: Any) -> str:
        """Analyze scraped data using LLM"""
        try:
            prompt_template = self.create_prompt_template()
            chain = self.create_chain(prompt_template)
            
            # Convert data to string for analysis
            data_str = json.dumps(data, indent=2) if not isinstance(data, str) else data
            
            analysis = chain.invoke({"scraped_data": data_str})
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return "Analysis unavailable due to processing error"
    
    def get_cached_data(self, destination: str, data_type: str) -> List[Dict[str, Any]]:
        """Get cached scraped data"""
        cache_key = f"{data_type}_{destination}"
        return self.scraped_cache.get(cache_key, [])
    
    def clear_cache(self):
        """Clear scraping cache"""
        self.scraped_cache.clear()
        self.log_activity("Cache cleared")
