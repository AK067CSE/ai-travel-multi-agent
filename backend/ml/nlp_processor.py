import spacy
from datetime import datetime
import re
from typing import Dict, List, Any

class TravelNLPProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract travel-related entities"""
        doc = self.nlp(text)
        
        entities = {
            'locations': [],
            'dates': [],
            'activities': [],
            'budget': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entities, locations
                entities['locations'].append(ent.text)
            elif ent.label_ in ['DATE', 'TIME']:
                entities['dates'].append(ent.text)
            elif ent.label_ == 'MONEY':
                entities['budget'].append(ent.text)
                
        return entities
    
    def detect_intent(self, query: str) -> str:
        """Classify travel query intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hotel', 'stay', 'accommodation']):
            return 'accommodation'
        elif any(word in query_lower for word in ['flight', 'airline', 'fly']):
            return 'transportation'
        elif any(word in query_lower for word in ['restaurant', 'food', 'eat']):
            return 'dining'
        elif any(word in query_lower for word in ['activity', 'tour', 'visit', 'see']):
            return 'activities'
        else:
            return 'general'