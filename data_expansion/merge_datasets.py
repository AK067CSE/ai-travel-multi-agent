"""
Dataset Merger for Travel AI System
Combines synthetic and scraped data into optimized datasets
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TravelDatasetMerger:
    """Merge and optimize travel datasets"""
    
    def __init__(self, output_dir: str = "final_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.seen_hashes = set()
        
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSONL file"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping line {line_num} in {file_path}: {e}")
            logger.info(f"Loaded {len(data)} records from {file_path}")
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
        return data
    
    def standardize_record(self, record: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """Standardize record format"""
        
        # Create content hash for deduplication
        content = f"{record.get('prompt', '')}{record.get('response', '')}"
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in self.seen_hashes:
            return None  # Skip duplicate
        
        self.seen_hashes.add(content_hash)
        
        # Determine record type and quality
        record_type, quality_score = self.classify_record(record, source_type)
        
        standardized = {
            "id": record.get("id", content_hash),
            "prompt": record.get("prompt", ""),
            "response": record.get("response", ""),
            "metadata": {
                "source": source_type,
                "type": record_type,
                "quality_score": quality_score,
                "original_entities": record.get("entities", {})
            }
        }
        
        # Add destination if available
        entities = record.get("entities", {})
        if "destination" in entities:
            standardized["metadata"]["destination"] = entities["destination"]
        elif "location" in entities:
            standardized["metadata"]["destination"] = entities["location"]
        
        return standardized
    
    def classify_record(self, record: Dict[str, Any], source_type: str) -> tuple:
        """Classify record type and assign quality score"""
        
        prompt = record.get("prompt", "").lower()
        response = record.get("response", "")
        
        # Determine type
        if source_type == "curated":
            # Original dataset - highest quality, well-curated
            if len(response) > 500:
                record_type = "curated_planning"
                quality_score = 5
            else:
                record_type = "curated_advice"
                quality_score = 5
        elif source_type == "synthetic":
            if "plan a" in prompt and len(response) > 1000:
                record_type = "detailed_planning"
                quality_score = 5
            elif "recommend" in prompt or "suggest" in prompt:
                record_type = "recommendation"
                quality_score = 4
            else:
                record_type = "general_advice"
                quality_score = 4
        else:  # scraped
            if len(response) < 100:
                record_type = "short_conversation"
                quality_score = 2
            elif "?" in prompt:
                record_type = "question_answer"
                quality_score = 3
            else:
                record_type = "discussion"
                quality_score = 3
        
        return record_type, quality_score
    
    def create_planning_dataset(self, all_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create high-quality planning dataset"""
        planning_data = []
        
        for record in all_data:
            metadata = record.get("metadata", {})
            record_type = metadata.get("type", "")
            if (record_type in ["detailed_planning", "curated_planning"] or
                metadata.get("quality_score", 0) >= 4):
                planning_data.append(record)
        
        logger.info(f"Created planning dataset with {len(planning_data)} records")
        return planning_data
    
    def create_chat_dataset(self, all_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create conversational dataset"""
        chat_data = []
        
        for record in all_data:
            metadata = record.get("metadata", {})
            record_type = metadata.get("type", "")
            
            if (record_type in ["question_answer", "discussion", "short_conversation"] or
                metadata.get("source") == "scraped"):
                chat_data.append(record)
        
        # Add some high-quality synthetic for variety
        synthetic_count = 0
        for record in all_data:
            if (record.get("metadata", {}).get("source") == "synthetic" and 
                synthetic_count < len(chat_data) // 3):  # 1/3 synthetic
                chat_data.append(record)
                synthetic_count += 1
        
        logger.info(f"Created chat dataset with {len(chat_data)} records")
        return chat_data
    
    def save_dataset(self, data: List[Dict[str, Any]], filename: str):
        """Save dataset to JSONL file"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in data:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(data)} records to {output_path}")
        
        # Also save CSV version for analysis
        self.save_csv_summary(data, filename.replace('.jsonl', '_summary.csv'))
    
    def save_csv_summary(self, data: List[Dict[str, Any]], filename: str):
        """Save dataset summary as CSV"""
        import pandas as pd
        
        summary_data = []
        for record in data:
            metadata = record.get("metadata", {})
            summary_data.append({
                "id": record.get("id", ""),
                "prompt_length": len(record.get("prompt", "")),
                "response_length": len(record.get("response", "")),
                "source": metadata.get("source", ""),
                "type": metadata.get("type", ""),
                "quality_score": metadata.get("quality_score", 0),
                "destination": metadata.get("destination", "")
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / filename
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary to {csv_path}")
    
    def generate_statistics(self, all_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate dataset statistics"""
        stats = {
            "total_records": len(all_data),
            "by_source": {},
            "by_type": {},
            "by_quality": {},
            "avg_prompt_length": 0,
            "avg_response_length": 0,
            "destinations": set()
        }
        
        prompt_lengths = []
        response_lengths = []
        
        for record in all_data:
            metadata = record.get("metadata", {})
            
            # Count by source
            source = metadata.get("source", "unknown")
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            
            # Count by type
            record_type = metadata.get("type", "unknown")
            stats["by_type"][record_type] = stats["by_type"].get(record_type, 0) + 1
            
            # Count by quality
            quality = metadata.get("quality_score", 0)
            stats["by_quality"][quality] = stats["by_quality"].get(quality, 0) + 1
            
            # Lengths
            prompt_lengths.append(len(record.get("prompt", "")))
            response_lengths.append(len(record.get("response", "")))
            
            # Destinations
            if metadata.get("destination"):
                stats["destinations"].add(metadata["destination"])
        
        stats["avg_prompt_length"] = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
        stats["avg_response_length"] = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        stats["unique_destinations"] = len(stats["destinations"])
        stats["destinations"] = list(stats["destinations"])[:20]  # Top 20 for display
        
        return stats
    
    def merge_all_datasets(self):
        """Main method to merge all datasets"""
        logger.info("Starting dataset merger...")
        
        # Load all data files
        original_files = [
            "../data.jsonl"  # Your original 1K dataset
        ]

        synthetic_files = [
            "synthetic_raw.jsonl",
            "synthetic_data_progress_50.jsonl",
            "synthetic_data_progress_100.jsonl",
            "synthetic_data_progress_150.jsonl"
        ]

        scraped_files = [
            "scraped_raw.jsonl"
        ]
        
        all_standardized = []

        # Process original curated data (highest priority)
        for file in original_files:
            if Path(file).exists():
                data = self.load_jsonl(file)
                for record in data:
                    standardized = self.standardize_record(record, "curated")
                    if standardized:
                        all_standardized.append(standardized)

        # Process synthetic data
        for file in synthetic_files:
            if Path(file).exists():
                data = self.load_jsonl(file)
                for record in data:
                    standardized = self.standardize_record(record, "synthetic")
                    if standardized:
                        all_standardized.append(standardized)
        
        # Process scraped data
        for file in scraped_files:
            if Path(file).exists():
                data = self.load_jsonl(file)
                for record in data:
                    standardized = self.standardize_record(record, "scraped")
                    if standardized:
                        all_standardized.append(standardized)
        
        logger.info(f"Total standardized records: {len(all_standardized)}")
        
        # Create specialized datasets
        planning_dataset = self.create_planning_dataset(all_standardized)
        chat_dataset = self.create_chat_dataset(all_standardized)
        
        # Save datasets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.save_dataset(planning_dataset, f"travel_planning_dataset_{timestamp}.jsonl")
        self.save_dataset(chat_dataset, f"travel_chat_dataset_{timestamp}.jsonl") 
        self.save_dataset(all_standardized, f"travel_complete_dataset_{timestamp}.jsonl")
        
        # Generate and save statistics
        stats = self.generate_statistics(all_standardized)
        stats_path = self.output_dir / f"dataset_statistics_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info("Dataset merger completed!")
        
        # Print summary
        print("\n📊 Dataset Merger Results:")
        print("=" * 50)
        print(f"✅ Total Records: {stats['total_records']}")
        print(f"📋 Planning Dataset: {len(planning_dataset)} records")
        print(f"💬 Chat Dataset: {len(chat_dataset)} records")
        print(f"📈 Average Response Length: {stats['avg_response_length']:.0f} chars")
        print(f"🌍 Unique Destinations: {stats['unique_destinations']}")
        print(f"\n📁 Output Directory: {self.output_dir}")
        
        return {
            "planning_dataset": len(planning_dataset),
            "chat_dataset": len(chat_dataset), 
            "total_dataset": len(all_standardized),
            "statistics": stats
        }

if __name__ == "__main__":
    merger = TravelDatasetMerger()
    results = merger.merge_all_datasets()
