"""
Advanced Data Expansion Orchestrator
Coordinates scraping, synthetic generation, and data quality management
"""

import json
import os
import logging
from typing import List, Dict, Any, Set
import hashlib
from datetime import datetime
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from scraper import TravelDataScraper
from synthetic_generator import SyntheticDataGenerator

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityManager:
    """Manages data quality, deduplication, and validation"""
    
    def __init__(self):
        self.seen_ids = set()
        self.quality_stats = {
            'total_processed': 0,
            'duplicates_removed': 0,
            'quality_filtered': 0,
            'final_count': 0
        }
    
    def generate_content_hash(self, prompt: str, response: str) -> str:
        """Generate hash for content deduplication"""
        content = f"{prompt.lower().strip()}{response.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_high_quality(self, qa_pair: Dict[str, Any]) -> bool:
        """Check if Q&A pair meets quality standards"""
        prompt = qa_pair.get('prompt', '')
        response = qa_pair.get('response', '')
        
        # Basic quality checks
        if len(prompt) < 20:
            return False
        
        if len(response) < 100:
            return False
        
        # Check for meaningful content
        if prompt.count(' ') < 5:  # Too few words
            return False
        
        if response.count(' ') < 20:  # Too few words
            return False
        
        # Check for travel-related content
        travel_keywords = [
            'travel', 'trip', 'vacation', 'holiday', 'destination', 'hotel',
            'flight', 'restaurant', 'attraction', 'tour', 'visit', 'stay',
            'accommodation', 'transport', 'budget', 'itinerary', 'guide'
        ]
        
        combined_text = f"{prompt} {response}".lower()
        if not any(keyword in combined_text for keyword in travel_keywords):
            return False
        
        return True
    
    def deduplicate_and_validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and validate quality"""
        logger.info(f"Processing {len(data)} records for quality and deduplication...")
        
        clean_data = []
        
        for item in data:
            self.quality_stats['total_processed'] += 1
            
            # Generate content hash
            content_hash = self.generate_content_hash(
                item.get('prompt', ''), 
                item.get('response', '')
            )
            
            # Check for duplicates
            if content_hash in self.seen_ids:
                self.quality_stats['duplicates_removed'] += 1
                continue
            
            # Check quality
            if not self.is_high_quality(item):
                self.quality_stats['quality_filtered'] += 1
                continue
            
            # Add to clean data
            self.seen_ids.add(content_hash)
            item['content_hash'] = content_hash
            clean_data.append(item)
            self.quality_stats['final_count'] += 1
        
        logger.info(f"Quality stats: {self.quality_stats}")
        return clean_data

class DataExpansionOrchestrator:
    """Main orchestrator for data expansion process"""
    
    def __init__(self, output_dir: str = "expanded_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.quality_manager = DataQualityManager()
        self.all_data = []
        
        # Configuration
        self.config = {
            'scraping': {
                'enabled': True,
                'reddit_limit': 200,
                'forum_limit': 100,
                'blog_limit': 50
            },
            'synthetic': {
                'enabled': True,
                'target_count': 5000,
                'batch_size': 20,
                'model_name': 'llama3-8b-8192'
            },
            'quality': {
                'min_prompt_length': 20,
                'min_response_length': 100,
                'require_travel_keywords': True
            }
        }
    
    def load_existing_data(self, data_file: str = "../data.jsonl") -> List[Dict[str, Any]]:
        """Load existing data to avoid duplication"""
        existing_data = []
        
        if os.path.exists(data_file):
            logger.info(f"Loading existing data from {data_file}")
            
            with open(data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            existing_data.append(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping line {line_num}: {e}")
            
            logger.info(f"Loaded {len(existing_data)} existing records")
        
        return existing_data
    
    def run_scraping_phase(self) -> List[Dict[str, Any]]:
        """Run web scraping phase"""
        if not self.config['scraping']['enabled']:
            logger.info("Scraping phase disabled")
            return []
        
        logger.info("Starting scraping phase...")
        
        scraper = TravelDataScraper()
        
        # Configure scraper based on config
        scraper.run_scraping(
            output_file=str(self.output_dir / "scraped_raw.jsonl")
        )
        
        # Load scraped data
        scraped_data = []
        scraped_file = self.output_dir / "scraped_raw.jsonl"
        
        if scraped_file.exists():
            with open(scraped_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            scraped_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        
        logger.info(f"Scraped {len(scraped_data)} records")
        return scraped_data
    
    def run_synthetic_phase(self) -> List[Dict[str, Any]]:
        """Run synthetic data generation phase"""
        if not self.config['synthetic']['enabled']:
            logger.info("Synthetic generation phase disabled")
            return []
        
        logger.info("Starting synthetic generation phase...")
        
        # Check for API keys
        if not os.getenv("GROQ_API_KEY"):
            logger.error("GROQ_API_KEY not found for synthetic generation")
            return []

        generator = SyntheticDataGenerator(
            model_name=self.config['synthetic']['model_name']
        )
        
        generator.run_generation(
            count=self.config['synthetic']['target_count'],
            output_file=str(self.output_dir / "synthetic_raw.jsonl")
        )
        
        # Load synthetic data
        synthetic_data = []
        synthetic_file = self.output_dir / "synthetic_raw.jsonl"
        
        if synthetic_file.exists():
            with open(synthetic_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            synthetic_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        
        logger.info(f"Generated {len(synthetic_data)} synthetic records")
        return synthetic_data
    
    def merge_and_clean_data(self, existing_data: List[Dict], scraped_data: List[Dict], 
                           synthetic_data: List[Dict]) -> List[Dict[str, Any]]:
        """Merge all data sources and clean"""
        logger.info("Merging and cleaning all data sources...")
        
        # Combine all data
        all_raw_data = existing_data + scraped_data + synthetic_data
        
        # Clean and deduplicate
        clean_data = self.quality_manager.deduplicate_and_validate(all_raw_data)
        
        return clean_data
    
    def save_final_dataset(self, data: List[Dict[str, Any]]):
        """Save the final expanded dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSONL
        final_file = self.output_dir / f"expanded_travel_data_{timestamp}.jsonl"
        with open(final_file, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        # Save as CSV for analysis
        csv_file = self.output_dir / f"expanded_travel_data_{timestamp}.csv"
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        
        # Save statistics
        stats = {
            'timestamp': timestamp,
            'total_records': len(data),
            'quality_stats': self.quality_manager.quality_stats,
            'config': self.config
        }
        
        stats_file = self.output_dir / f"expansion_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Final dataset saved: {len(data)} records")
        logger.info(f"Files saved in: {self.output_dir}")
        
        return final_file
    
    def run_full_expansion(self) -> str:
        """Run the complete data expansion process"""
        logger.info("Starting full data expansion process...")
        
        # Load existing data
        existing_data = self.load_existing_data()
        
        # Run scraping phase
        scraped_data = self.run_scraping_phase()
        
        # Run synthetic generation phase
        synthetic_data = self.run_synthetic_phase()
        
        # Merge and clean all data
        final_data = self.merge_and_clean_data(existing_data, scraped_data, synthetic_data)
        
        # Save final dataset
        final_file = self.save_final_dataset(final_data)
        
        logger.info("Data expansion process completed!")
        return str(final_file)

if __name__ == "__main__":
    orchestrator = DataExpansionOrchestrator()
    final_dataset = orchestrator.run_full_expansion()
    print(f"Final dataset saved to: {final_dataset}")
