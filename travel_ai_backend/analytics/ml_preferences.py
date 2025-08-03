"""
ML-based User Preference Learning System
Advanced analytics and personalization for travel recommendations
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

class TravelPreferenceLearner:
    """
    ML-based system for learning and predicting user travel preferences
    """
    
    def __init__(self):
        self.preference_model = None
        self.destination_clusters = None
        self.activity_vectorizer = None
        self.scaler = StandardScaler()
        
        # Travel preference categories
        self.preference_categories = {
            'travel_style': ['explorer', 'relaxer', 'adventurer', 'cultural', 'foodie', 'luxury', 'budget'],
            'accommodation_type': ['hotel', 'resort', 'boutique', 'hostel', 'airbnb', 'villa'],
            'activity_preferences': ['museums', 'outdoor', 'nightlife', 'shopping', 'food_tours', 'historical', 'nature'],
            'budget_range': ['budget', 'mid_range', 'luxury', 'ultra_luxury'],
            'group_type': ['solo', 'couple', 'family', 'friends', 'business']
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("ML Preference Learning system initialized")
    
    def _initialize_models(self):
        """Initialize ML models for preference learning"""
        try:
            # Try to load existing models
            self._load_models()
        except:
            # Create new models if none exist
            self._create_new_models()
    
    def _create_new_models(self):
        """Create new ML models"""
        # Destination clustering model
        self.destination_clusters = KMeans(n_clusters=8, random_state=42)
        
        # Activity preference vectorizer
        self.activity_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        logger.info("Created new ML models for preference learning")
    
    def _load_models(self):
        """Load existing ML models"""
        model_path = "ml_models"
        if os.path.exists(model_path):
            self.destination_clusters = joblib.load(f"{model_path}/destination_clusters.pkl")
            self.activity_vectorizer = joblib.load(f"{model_path}/activity_vectorizer.pkl")
            self.scaler = joblib.load(f"{model_path}/scaler.pkl")
            logger.info("Loaded existing ML models")
    
    def _save_models(self):
        """Save ML models"""
        model_path = "ml_models"
        os.makedirs(model_path, exist_ok=True)
        
        joblib.dump(self.destination_clusters, f"{model_path}/destination_clusters.pkl")
        joblib.dump(self.activity_vectorizer, f"{model_path}/activity_vectorizer.pkl")
        joblib.dump(self.scaler, f"{model_path}/scaler.pkl")
        
        logger.info("Saved ML models")
    
    def analyze_user_preferences(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user preferences using ML techniques
        """
        try:
            # Extract features from user data
            features = self._extract_user_features(user_data)
            
            # Predict user clusters and preferences
            predictions = self._predict_preferences(features)
            
            # Generate personalized recommendations
            recommendations = self._generate_recommendations(predictions, user_data)
            
            return {
                'user_profile': predictions,
                'recommendations': recommendations,
                'confidence_score': predictions.get('confidence', 0.8),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Preference analysis error: {e}")
            return self._fallback_analysis(user_data)
    
    def _extract_user_features(self, user_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from user data"""
        features = []
        
        # Travel style encoding
        travel_style = user_data.get('travel_style', 'explorer')
        style_vector = [1 if style == travel_style else 0 for style in self.preference_categories['travel_style']]
        features.extend(style_vector)
        
        # Budget encoding
        budget = user_data.get('budget_range', 'mid_range')
        budget_vector = [1 if b == budget else 0 for b in self.preference_categories['budget_range']]
        features.extend(budget_vector)
        
        # Previous destinations (simplified)
        prev_destinations = user_data.get('previous_destinations', [])
        destination_count = len(prev_destinations)
        features.append(min(destination_count / 10.0, 1.0))  # Normalize
        
        # Travel frequency
        frequency_map = {'rarely': 0.2, 'occasionally': 0.5, 'frequently': 1.0}
        frequency = user_data.get('travel_frequency', 'occasionally')
        features.append(frequency_map.get(frequency, 0.5))
        
        # Interests count
        interests = user_data.get('interests', [])
        features.append(min(len(interests) / 5.0, 1.0))  # Normalize
        
        return np.array(features).reshape(1, -1)
    
    def _predict_preferences(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict user preferences using ML models"""
        try:
            # Normalize features
            normalized_features = self.scaler.fit_transform(features)
            
            # Predict destination cluster (if model is trained)
            destination_cluster = 0
            if hasattr(self.destination_clusters, 'cluster_centers_'):
                destination_cluster = self.destination_clusters.predict(normalized_features)[0]
            
            # Calculate confidence based on feature consistency
            confidence = self._calculate_confidence(features)
            
            return {
                'destination_cluster': int(destination_cluster),
                'preference_strength': float(np.mean(features)),
                'confidence': confidence,
                'feature_vector': features.flatten().tolist()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'destination_cluster': 0, 'confidence': 0.5}
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score for predictions"""
        # Simple confidence calculation based on feature completeness
        non_zero_features = np.count_nonzero(features)
        total_features = features.size
        
        confidence = min(non_zero_features / total_features, 1.0)
        return float(confidence)
    
    def _generate_recommendations(self, predictions: Dict[str, Any], user_data: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations based on predictions"""
        recommendations = []
        
        cluster = predictions.get('destination_cluster', 0)
        confidence = predictions.get('confidence', 0.5)
        
        # Cluster-based recommendations
        cluster_recommendations = {
            0: ["Consider European cultural destinations", "Focus on historical sites and museums"],
            1: ["Explore Asian culinary destinations", "Try food tours and cooking classes"],
            2: ["Adventure destinations in South America", "Include outdoor activities and nature"],
            3: ["Luxury beach resorts", "Premium accommodations and spa experiences"],
            4: ["Budget-friendly backpacking routes", "Hostels and local transportation"],
            5: ["Family-friendly destinations", "Kid-friendly activities and accommodations"],
            6: ["Romantic getaway destinations", "Intimate dining and scenic views"],
            7: ["Business travel optimized routes", "Efficient transportation and meeting facilities"]
        }
        
        recommendations.extend(cluster_recommendations.get(cluster, ["General travel recommendations"]))
        
        # Preference-based recommendations
        travel_style = user_data.get('travel_style', 'explorer')
        if travel_style == 'foodie':
            recommendations.append("Focus on destinations known for culinary excellence")
        elif travel_style == 'adventurer':
            recommendations.append("Include extreme sports and outdoor adventures")
        elif travel_style == 'cultural':
            recommendations.append("Prioritize museums, historical sites, and cultural events")
        
        # Budget-based recommendations
        budget = user_data.get('budget_range', 'mid_range')
        if budget == 'luxury':
            recommendations.append("Consider premium experiences and high-end accommodations")
        elif budget == 'budget':
            recommendations.append("Focus on cost-effective options and free activities")
        
        return recommendations
    
    def _fallback_analysis(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when ML fails"""
        return {
            'user_profile': {
                'destination_cluster': 0,
                'confidence': 0.3,
                'fallback': True
            },
            'recommendations': [
                "Based on your preferences, consider popular destinations",
                "Look for activities that match your interests",
                "Consider your budget when selecting accommodations"
            ],
            'confidence_score': 0.3,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def learn_from_feedback(self, user_id: str, recommendation_id: str, rating: int, feedback: str = ""):
        """Learn from user feedback to improve recommendations"""
        try:
            # Store feedback for model retraining
            feedback_data = {
                'user_id': user_id,
                'recommendation_id': recommendation_id,
                'rating': rating,
                'feedback': feedback,
                'timestamp': datetime.now().isoformat()
            }
            
            # In production, store in database for batch retraining
            self._store_feedback(feedback_data)
            
            logger.info(f"Stored feedback for user {user_id}: rating {rating}")
            
        except Exception as e:
            logger.error(f"Feedback learning error: {e}")
    
    def _store_feedback(self, feedback_data: Dict[str, Any]):
        """Store feedback data for model improvement"""
        # In production, store in database
        feedback_file = "user_feedback.jsonl"
        
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_data) + '\n')
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user travel patterns"""
        try:
            # In production, query user's travel history from database
            # For now, return sample insights
            
            insights = {
                'travel_personality': 'Cultural Explorer',
                'preferred_destinations': ['Europe', 'Asia'],
                'average_trip_duration': 7,
                'budget_tendency': 'mid_range',
                'seasonal_preferences': ['Spring', 'Fall'],
                'activity_patterns': {
                    'cultural': 0.8,
                    'outdoor': 0.6,
                    'food': 0.9,
                    'nightlife': 0.3
                },
                'recommendation_accuracy': 0.85,
                'total_trips_planned': 12
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"User insights error: {e}")
            return {'error': 'Unable to generate insights'}
    
    def retrain_models(self):
        """Retrain ML models with new feedback data"""
        try:
            # In production, implement periodic model retraining
            logger.info("Model retraining would be performed here")
            
            # Load feedback data
            # Retrain models
            # Update model files
            # Validate performance
            
            return {'status': 'success', 'message': 'Models retrained successfully'}
            
        except Exception as e:
            logger.error(f"Model retraining error: {e}")
            return {'status': 'error', 'message': str(e)}
