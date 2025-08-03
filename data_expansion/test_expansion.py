"""
Test script for data expansion components
"""

import os
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_scraper():
    """Test the scraper component"""
    print("🧪 Testing scraper...")
    
    try:
        from scraper import TravelDataScraper, TravelQA
        
        # Test TravelQA class
        qa = TravelQA(
            question="What's the best time to visit Japan?",
            answer="The best time to visit Japan is during spring (March-May) for cherry blossoms or autumn (September-November) for fall colors.",
            source="test",
            category="travel_advice"
        )
        
        qa_dict = qa.to_dict()
        assert 'id' in qa_dict
        assert qa_dict['prompt'] == qa.question
        assert qa_dict['response'] == qa.answer
        
        print("✅ TravelQA class working")
        
        # Test scraper initialization
        scraper = TravelDataScraper()
        assert scraper.session is not None
        assert len(scraper.scraped_data) == 0
        
        print("✅ TravelDataScraper initialization working")
        
    except Exception as e:
        print(f"❌ Scraper test failed: {e}")
        return False
    
    return True

def test_synthetic_generator():
    """Test the synthetic generator component"""
    print("🧪 Testing synthetic generator...")
    
    try:
        from synthetic_generator import SyntheticDataGenerator, TravelScenario
        
        # Test TravelScenario
        scenario = TravelScenario(
            destination="Paris",
            duration_days=7,
            budget_usd=2000,
            traveler_type="romantic couple",
            interests=["art", "cuisine", "history"]
        )
        
        prompt = scenario.to_prompt()
        assert "Paris" in prompt
        assert "7-day" in prompt
        assert "$2000" in prompt
        
        print("✅ TravelScenario class working")
        
        # Test generator initialization (without API calls)
        generator = SyntheticDataGenerator(use_groq=True)
        assert generator.destinations is not None
        assert len(generator.destinations) > 0
        
        print("✅ SyntheticDataGenerator initialization working")
        
    except Exception as e:
        print(f"❌ Synthetic generator test failed: {e}")
        return False
    
    return True

def test_data_quality():
    """Test the data quality manager"""
    print("🧪 Testing data quality manager...")
    
    try:
        from data_expander import DataQualityManager
        
        quality_manager = DataQualityManager()
        
        # Test quality check
        good_qa = {
            'prompt': 'What are the best attractions in Tokyo for first-time visitors?',
            'response': 'Tokyo offers many amazing attractions for first-time visitors. Start with the iconic Senso-ji Temple in Asakusa, one of the oldest temples in the city. Visit the bustling Shibuya Crossing and the nearby Hachiko statue. Don\'t miss the Tokyo Skytree for panoramic city views, and explore the trendy Harajuku district for unique fashion and culture.'
        }
        
        bad_qa = {
            'prompt': 'Hi',
            'response': 'Hello'
        }
        
        assert quality_manager.is_high_quality(good_qa) == True
        assert quality_manager.is_high_quality(bad_qa) == False
        
        print("✅ Data quality checks working")
        
        # Test deduplication
        test_data = [good_qa, good_qa, bad_qa]  # Duplicate good_qa
        clean_data = quality_manager.deduplicate_and_validate(test_data)
        
        assert len(clean_data) == 1  # Only one good, unique record
        
        print("✅ Deduplication working")
        
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False
    
    return True

def test_orchestrator():
    """Test the main orchestrator"""
    print("🧪 Testing orchestrator...")
    
    try:
        from data_expander import DataExpansionOrchestrator
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = DataExpansionOrchestrator(output_dir=temp_dir)
            
            assert orchestrator.output_dir.exists()
            assert orchestrator.config is not None
            
            # Test loading existing data (empty case)
            existing_data = orchestrator.load_existing_data("nonexistent.jsonl")
            assert len(existing_data) == 0
            
            print("✅ DataExpansionOrchestrator working")
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False
    
    return True

def test_cli():
    """Test CLI imports"""
    print("🧪 Testing CLI...")
    
    try:
        import cli
        assert hasattr(cli, 'cli')
        print("✅ CLI imports working")
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all tests"""
    print("🚀 Running data expansion tests...\n")
    
    tests = [
        test_scraper,
        test_synthetic_generator,
        test_data_quality,
        test_orchestrator,
        test_cli
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Data expansion suite is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
