"""
Travel Data Expansion Suite

Advanced data expansion toolkit for travel AI systems.
Combines web scraping, synthetic generation, and quality management.
"""

__version__ = "1.0.0"
__author__ = "Travel AI Team"

from .scraper import TravelDataScraper, TravelQA
from .synthetic_generator import SyntheticDataGenerator, TravelScenario
from .data_expander import DataExpansionOrchestrator, DataQualityManager

__all__ = [
    "TravelDataScraper",
    "TravelQA", 
    "SyntheticDataGenerator",
    "TravelScenario",
    "DataExpansionOrchestrator",
    "DataQualityManager"
]
