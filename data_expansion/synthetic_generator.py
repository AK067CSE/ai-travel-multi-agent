"""
Advanced Synthetic Travel Data Generator
Uses Groq + LangChain to generate diverse travel Q&A scenarios
"""

import json
import time
import random
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import hashlib
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TravelScenario:
    destination: str
    duration_days: int
    budget_usd: int
    traveler_type: str
    interests: List[str]
    season: str = "any"
    group_size: int = 1
    
    def to_prompt(self) -> str:
        """Convert scenario to a travel planning prompt"""
        interests_str = ", ".join(self.interests)
        
        if self.group_size == 1:
            group_text = "solo traveler"
        elif self.group_size == 2:
            group_text = "couple"
        else:
            group_text = f"group of {self.group_size}"
        
        return f"Plan a {self.duration_days}-day trip to {self.destination} for a {self.traveler_type} {group_text} interested in {interests_str}, under ${self.budget_usd}."

class SyntheticDataGenerator:
    def __init__(self, model_name="llama3-8b-8192"):
        self.generated_data = []
        self.seen_prompts = set()
        self.model_name = model_name

        # Initialize LangChain with Groq
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0.7,
            max_tokens=2000
        )

        # Create the prompt template
        self.prompt_template = self._create_prompt_template()

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        
        # Travel data templates
        self.destinations = [
            "Tokyo", "Paris", "New York", "London", "Barcelona", "Rome", "Amsterdam", 
            "Bangkok", "Sydney", "Dubai", "Istanbul", "Prague", "Vienna", "Berlin",
            "Lisbon", "Copenhagen", "Stockholm", "Oslo", "Helsinki", "Reykjavik",
            "Marrakech", "Cairo", "Cape Town", "Nairobi", "Mumbai", "Delhi", "Beijing",
            "Seoul", "Singapore", "Kuala Lumpur", "Jakarta", "Manila", "Ho Chi Minh City",
            "Hanoi", "Yangon", "Kathmandu", "Colombo", "Male", "Bali", "Phuket",
            "Rio de Janeiro", "Buenos Aires", "Lima", "Bogota", "Mexico City", "Havana",
            "San Jose", "Panama City", "Quito", "La Paz", "Santiago", "Montevideo"
        ]
        
        self.traveler_types = [
            "budget backpacker", "luxury traveler", "business traveler", "family",
            "solo female traveler", "digital nomad", "adventure seeker", "cultural enthusiast",
            "foodie", "photographer", "nature lover", "history buff", "art lover",
            "beach lover", "mountain climber", "city explorer", "rural explorer",
            "wellness seeker", "party traveler", "romantic couple", "honeymoon couple",
            "retirement traveler", "student traveler", "gap year traveler", "volunteer traveler"
        ]
        
        self.interests = [
            "museums", "nightlife", "local cuisine", "shopping", "architecture", "history",
            "art galleries", "music", "festivals", "beaches", "hiking", "cycling",
            "photography", "wildlife", "nature", "adventure sports", "water sports",
            "skiing", "snowboarding", "diving", "snorkeling", "surfing", "sailing",
            "cultural experiences", "local traditions", "street food", "fine dining",
            "wine tasting", "brewery tours", "coffee culture", "markets", "temples",
            "churches", "castles", "palaces", "gardens", "parks", "lakes", "rivers",
            "mountains", "deserts", "forests", "islands", "hot springs", "spas",
            "yoga retreats", "meditation", "wellness", "fitness", "sports events",
            "concerts", "theater", "dance", "cooking classes", "language learning"
        ]
        
        self.seasons = ["spring", "summer", "autumn", "winter", "any"]

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create LangChain prompt template for travel planning"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert travel planner with extensive knowledge of destinations worldwide.
            Create detailed, practical, and engaging travel itineraries that include:

            1. Day-by-day breakdown of activities
            2. Accommodation recommendations with specific names
            3. Transportation options and costs
            4. Detailed budget breakdown
            5. Local tips and cultural insights
            6. Food recommendations with restaurant names
            7. Safety considerations
            8. Packing suggestions
            9. Best times to visit attractions
            10. Local customs and etiquette

            Make your responses comprehensive, practical, and tailored to the specific traveler type and interests.
            Include specific names of places, restaurants, hotels, and attractions when possible.
            Provide realistic budget estimates and practical advice.
            Write in an engaging, helpful tone as if you're personally advising the traveler."""
        )

        human_message = HumanMessagePromptTemplate.from_template(
            "{travel_request}"
        )

        return ChatPromptTemplate.from_messages([system_message, human_message])

    def generate_scenario(self) -> TravelScenario:
        """Generate a random travel scenario"""
        return TravelScenario(
            destination=random.choice(self.destinations),
            duration_days=random.choice([3, 5, 7, 10, 14, 21, 30]),
            budget_usd=random.choice([500, 800, 1200, 1500, 2000, 3000, 5000, 8000, 10000]),
            traveler_type=random.choice(self.traveler_types),
            interests=random.sample(self.interests, random.randint(2, 4)),
            season=random.choice(self.seasons),
            group_size=random.choice([1, 1, 1, 2, 2, 3, 4])  # Weighted towards solo/couple
        )
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response using LangChain + Groq"""
        try:
            response = self.chain.invoke({"travel_request": prompt})
            return response
        except Exception as e:
            logger.error(f"Error with LangChain + Groq: {e}")
            return None
    
    def extract_entities(self, scenario: TravelScenario, response: str) -> Dict[str, Any]:
        """Extract entities from the scenario and response"""
        return {
            "destination": scenario.destination,
            "duration_days": scenario.duration_days,
            "budget_usd": scenario.budget_usd,
            "interests": scenario.interests,
            "user_type": scenario.traveler_type.replace(" ", "_"),
            "season": scenario.season,
            "group_size": scenario.group_size
        }
    
    def generate_qa_pair(self) -> Optional[Dict[str, Any]]:
        """Generate a single Q&A pair"""
        scenario = self.generate_scenario()
        prompt = scenario.to_prompt()
        
        # Skip if we've seen this prompt before
        if prompt in self.seen_prompts:
            return None
        
        response = self.generate_response(prompt)
        if not response:
            return None
        
        self.seen_prompts.add(prompt)
        
        # Create unique ID
        content_hash = hashlib.md5(f"{prompt}{response}".encode()).hexdigest()
        
        return {
            "id": content_hash,
            "prompt": prompt,
            "response": response,
            "entities": self.extract_entities(scenario, response)
        }
    
    def generate_batch(self, batch_size: int = 10, delay_seconds: float = 1.0) -> List[Dict[str, Any]]:
        """Generate a batch of Q&A pairs"""
        batch = []
        
        for i in range(batch_size):
            logger.info(f"Generating Q&A pair {i+1}/{batch_size}")
            
            qa_pair = self.generate_qa_pair()
            if qa_pair:
                batch.append(qa_pair)
                self.generated_data.append(qa_pair)
            
            # Add delay to respect API rate limits
            if i < batch_size - 1:
                time.sleep(delay_seconds)
        
        return batch
    
    def generate_diverse_scenarios(self, total_count: int = 1000, batch_size: int = 10) -> None:
        """Generate diverse travel scenarios"""
        logger.info(f"Generating {total_count} diverse travel scenarios...")
        
        generated_count = 0
        while generated_count < total_count:
            remaining = min(batch_size, total_count - generated_count)
            batch = self.generate_batch(remaining)
            generated_count += len(batch)
            
            logger.info(f"Generated {generated_count}/{total_count} scenarios")
            
            # Save progress periodically
            if generated_count % 50 == 0:
                self.save_data(f"synthetic_data_progress_{generated_count}.jsonl")
    
    def save_data(self, filename: str):
        """Save generated data to JSONL file"""
        logger.info(f"Saving {len(self.generated_data)} records to {filename}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            for qa in self.generated_data:
                json.dump(qa, f, ensure_ascii=False)
                f.write('\n')
    
    def run_generation(self, count: int = 1000, output_file: str = "synthetic_travel_data.jsonl"):
        """Run synthetic data generation"""
        logger.info(f"Starting synthetic data generation for {count} samples...")
        
        self.generate_diverse_scenarios(count)
        self.save_data(output_file)
        
        logger.info(f"Generation completed! Created {len(self.generated_data)} travel Q&A pairs")
        return len(self.generated_data)

if __name__ == "__main__":
    # Check for API keys
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        logger.error("Please set GROQ_API_KEY or OPENAI_API_KEY environment variable")
        exit(1)

    # Use Groq by default (faster and cheaper)
    use_groq = bool(os.getenv("GROQ_API_KEY"))

    generator = SyntheticDataGenerator(use_groq=use_groq)
    generator.run_generation(count=100)  # Start with 100 for testing
