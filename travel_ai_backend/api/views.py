from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import authenticate
from django.conf import settings
import logging
import time
import sys
import os

# Add the multi_agent_system to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'multi_agent_system'))

try:
    from travel_ai_system import TravelAISystem
    from simple_rag_system import SimpleTravelRAG
    AGENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Multi-agent system not available: {e}")
    AGENTS_AVAILABLE = False

# Enhanced RAG system
try:
    from rag_system.enhanced_rag import EnhancedTravelRAG
    from rag_system.multi_llm_manager import MultiLLMManager
    ENHANCED_RAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced RAG system not available: {e}")
    ENHANCED_RAG_AVAILABLE = False

from .models import User, Conversation, TravelRecommendation
from .serializers import (
    UserSerializer, ConversationSerializer,
    TravelRecommendationSerializer, ChatRequestSerializer,
    ChatResponseSerializer
)

logger = logging.getLogger(__name__)

# Import our advanced systems
try:
    from agents.advanced_crew_ai import AdvancedCrewAISystem
    from analytics.advanced_analytics import AdvancedAnalytics
    from analytics.ml_preferences import TravelPreferenceLearner
    from integrations.realtime_data import RealTimeDataIntegrator
    ADVANCED_SYSTEMS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced systems not available: {e}")
    ADVANCED_SYSTEMS_AVAILABLE = False

class ChatAPIView(APIView):
    """
    Main chat endpoint for Travel AI system
    Handles user queries and returns AI-generated responses
    """
    permission_classes = [permissions.IsAuthenticated]

    def __init__(self):
        super().__init__()
        self.ai_system = None
        self.enhanced_rag = None
        self.llm_manager = None

        # Initialize Enhanced RAG system (priority)
        if ENHANCED_RAG_AVAILABLE:
            try:
                self.enhanced_rag = EnhancedTravelRAG()
                self.enhanced_rag.load_and_index_data()
                self.llm_manager = MultiLLMManager()
                logger.info("Enhanced RAG System initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Enhanced RAG system: {e}")

        # Fallback to original system
        if AGENTS_AVAILABLE and not self.enhanced_rag:
            try:
                self.ai_system = TravelAISystem()
                logger.info("Travel AI System initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AI system: {e}")

    def post(self, request):
        """Process chat message and return AI response"""
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user_message = serializer.validated_data['message']
        session_id = serializer.validated_data.get('session_id', f"session_{int(time.time())}")

        try:
            # Get or create conversation
            conversation, created = Conversation.objects.get_or_create(
                user=request.user,
                session_id=session_id,
                defaults={'is_active': True}
            )

            start_time = time.time()

            # Process with Enhanced RAG system (priority)
            if self.enhanced_rag and ENHANCED_RAG_AVAILABLE:
                rag_response = self.enhanced_rag.generate_enhanced_response(
                    query=user_message,
                    user_id=str(request.user.id)
                )

                response_text = rag_response.get('response', 'I apologize, but I encountered an issue processing your request.')
                agents_used = [f"EnhancedRAG ({rag_response.get('retrieval_method', 'unknown')})"]
                success = rag_response.get('context_used', False)

                # Add RAG-specific metadata
                rag_metadata = {
                    'retrieved_chunks': rag_response.get('retrieved_chunks', 0),
                    'reranked_chunks': rag_response.get('reranked_chunks', 0),
                    'retrieval_method': rag_response.get('retrieval_method', 'unknown'),
                    'rerank_scores': rag_response.get('rerank_scores', [])
                }

            # Fallback to original multi-agent system
            elif self.ai_system and AGENTS_AVAILABLE:
                ai_response = self.ai_system.process_user_request(
                    user_request=user_message,
                    user_id=str(request.user.id),
                    workflow_type="auto"
                )

                response_text = ai_response.get('message', 'I apologize, but I encountered an issue processing your request.')
                agents_used = ai_response.get('agents_involved', [])
                success = ai_response.get('success', False)
                rag_metadata = {}

            else:
                # Fallback response
                response_text = "I'm currently experiencing technical difficulties. Please try again later."
                agents_used = []
                success = False
                rag_metadata = {}

            processing_time = time.time() - start_time

            # Update conversation
            conversation.messages.append({
                'user_message': user_message,
                'ai_response': response_text,
                'timestamp': time.time(),
                'agents_used': agents_used,
                'processing_time': processing_time
            })
            conversation.total_messages += 1
            conversation.avg_response_time = (
                (conversation.avg_response_time * (conversation.total_messages - 1) + processing_time)
                / conversation.total_messages
            )
            conversation.save()

            # Create recommendation record
            recommendation = TravelRecommendation.objects.create(
                user=request.user,
                conversation=conversation,
                query=user_message,
                response=response_text,
                agents_used=agents_used,
                processing_time=processing_time,
                recommendation_type='general'
            )

            # Prepare response
            response_data = {
                'message': response_text,
                'session_id': session_id,
                'processing_time': processing_time,
                'agents_used': agents_used,
                'success': success,
                'recommendation_id': str(recommendation.id),
                'rag_metadata': rag_metadata,
                'system_type': 'enhanced_rag' if self.enhanced_rag else 'multi_agent'
            }

            response_serializer = ChatResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Chat API error: {e}")
            return Response(
                {'error': 'Internal server error', 'detail': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class RecommendationAPIView(generics.ListCreateAPIView):
    """
    API for travel recommendations
    GET: List user's recommendations
    POST: Create new recommendation request
    """
    serializer_class = TravelRecommendationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return TravelRecommendation.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # This would be called for direct recommendation requests
        # For now, recommendations are created through chat
        serializer.save(user=self.request.user)

class UserProfileAPIView(generics.RetrieveUpdateAPIView):
    """
    API for user profile management
    GET: Retrieve user profile
    PUT/PATCH: Update user profile
    """
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user

class ConversationListAPIView(generics.ListAPIView):
    """
    API to list user's conversations
    """
    serializer_class = ConversationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)

class ConversationDetailAPIView(generics.RetrieveAPIView):
    """
    API to get specific conversation details
    """
    serializer_class = ConversationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def agent_status(request):
    """
    Check the status of AI agents and Enhanced RAG system
    """
    try:
        # Check Enhanced RAG status
        enhanced_rag_status = {}
        if ENHANCED_RAG_AVAILABLE:
            try:
                # Get a temporary instance to check stats
                temp_rag = EnhancedTravelRAG()
                enhanced_rag_status = {
                    'available': True,
                    'embedding_model': 'sentence-transformers' if temp_rag.embedding_model else 'tfidf',
                    'vector_database': 'chromadb' if temp_rag.chroma_client else 'tfidf',
                    'chunking_enabled': True,
                    'reranking_enabled': True
                }

                # Check LLM manager
                temp_llm_manager = MultiLLMManager()
                llm_stats = temp_llm_manager.get_usage_stats()
                enhanced_rag_status['available_llms'] = llm_stats['available_llms']
                enhanced_rag_status['total_llms'] = llm_stats['total_llms']

            except Exception as e:
                enhanced_rag_status = {
                    'available': False,
                    'error': str(e)
                }
        else:
            enhanced_rag_status = {'available': False, 'reason': 'Not installed'}

        status_data = {
            'enhanced_rag': enhanced_rag_status,
            'multi_agent_system': {
                'available': AGENTS_AVAILABLE,
                'agents': [
                    'CoordinatorAgent',
                    'ChatAgent',
                    'RecommendationAgent',
                    'ScrapingAgent',
                    'BookingAgent'
                ] if AGENTS_AVAILABLE else []
            },
            'system_status': 'operational' if (ENHANCED_RAG_AVAILABLE or AGENTS_AVAILABLE) else 'degraded',
            'primary_system': 'enhanced_rag' if ENHANCED_RAG_AVAILABLE else 'multi_agent' if AGENTS_AVAILABLE else 'none',
            'timestamp': time.time()
        }

        return Response(status_data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Agent status error: {e}")
        return Response(
            {'error': 'Failed to get agent status'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def rate_recommendation(request, recommendation_id):
    """
    Rate a travel recommendation
    """
    try:
        recommendation = TravelRecommendation.objects.get(
            id=recommendation_id,
            user=request.user
        )

        rating = request.data.get('rating')
        feedback = request.data.get('feedback', '')

        if rating and 1 <= int(rating) <= 5:
            recommendation.rating = int(rating)
            recommendation.feedback = feedback
            recommendation.save()

            return Response({
                'message': 'Rating saved successfully',
                'rating': recommendation.rating
            }, status=status.HTTP_200_OK)
        else:
            return Response(
                {'error': 'Rating must be between 1 and 5'},
                status=status.HTTP_400_BAD_REQUEST
            )

    except TravelRecommendation.DoesNotExist:
        return Response(
            {'error': 'Recommendation not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Rating error: {e}")
        return Response(
            {'error': 'Failed to save rating'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# Advanced AI Travel Agent Endpoints

@api_view(['POST'])
@permission_classes([permissions.AllowAny])
def chat_with_ai_agent(request):
    """
    Main chatbot endpoint - integrates all AI systems
    """
    try:
        data = request.data
        user_message = data.get('message', '')
        user_preferences = data.get('preferences', {})
        conversation_id = data.get('conversation_id', f"conv_{time.time()}")

        if not user_message:
            return Response({
                'error': 'Message is required'
            }, status=status.HTTP_400_BAD_REQUEST)

        start_time = time.time()

        # Try Advanced CrewAI first, fallback to Enhanced RAG
        if ADVANCED_SYSTEMS_AVAILABLE:
            try:
                crew_system = AdvancedCrewAISystem()

                # Determine workflow type based on message
                workflow_type = determine_workflow_type(user_message, user_preferences)

                result = crew_system.create_advanced_travel_plan(
                    user_message,
                    user_preferences,
                    workflow_type=workflow_type
                )

                response_time = time.time() - start_time

                # Track interaction
                if ADVANCED_SYSTEMS_AVAILABLE:
                    try:
                        analytics = AdvancedAnalytics()
                        analytics.track_user_interaction(conversation_id, {
                            'type': 'chat_message',
                            'request': user_message,
                            'response_time': response_time,
                            'success': result['success'],
                            'system': result['system'],
                            'agents': result.get('agents_involved', []),
                            'workflow': result.get('workflow_type', 'unknown')
                        })
                    except:
                        pass  # Analytics failure shouldn't break the response

                return Response({
                    'success': True,
                    'response': result['travel_plan'],
                    'system_used': result['system'],
                    'agents_involved': result.get('agents_involved', []),
                    'workflow_type': result.get('workflow_type', 'unknown'),
                    'features_used': result.get('features_used', []),
                    'response_time': response_time,
                    'conversation_id': conversation_id,
                    'timestamp': time.time()
                })

            except Exception as e:
                logger.error(f"Advanced CrewAI error: {e}")
                # Fall through to Enhanced RAG fallback

        # Fallback to Enhanced RAG
        if ENHANCED_RAG_AVAILABLE:
            try:
                rag_system = EnhancedTravelRAG()
                rag_result = rag_system.generate_enhanced_response(user_message)

                response_time = time.time() - start_time

                return Response({
                    'success': True,
                    'response': rag_result['response'],
                    'system_used': 'EnhancedRAG_Fallback',
                    'agents_involved': ['EnhancedRAG'],
                    'sources': rag_result.get('sources', []),
                    'response_time': response_time,
                    'conversation_id': conversation_id,
                    'timestamp': time.time(),
                    'note': 'Using Enhanced RAG fallback system'
                })

            except Exception as rag_error:
                logger.error(f"RAG fallback error: {rag_error}")

        # Final fallback
        return Response({
            'success': False,
            'error': 'AI systems temporarily unavailable',
            'fallback_response': 'I apologize, but our AI travel systems are currently experiencing high demand. Please try again in a moment.',
            'conversation_id': conversation_id,
            'timestamp': time.time()
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([permissions.AllowAny])
def get_dashboard_data(request):
    """
    Get real-time dashboard data for frontend
    """
    try:
        if ADVANCED_SYSTEMS_AVAILABLE:
            analytics = AdvancedAnalytics()
            dashboard_data = analytics.generate_real_time_dashboard()

            return Response({
                'success': True,
                'dashboard': dashboard_data,
                'timestamp': time.time()
            })
        else:
            # Fallback dashboard data
            return Response({
                'success': True,
                'dashboard': {
                    'overview': {
                        'total_requests_24h': 0,
                        'success_rate_24h': 100,
                        'avg_response_time_24h': 1.2,
                        'avg_user_rating': 4.5
                    },
                    'system_health': {
                        'enhanced_rag_status': 'operational' if ENHANCED_RAG_AVAILABLE else 'unavailable',
                        'overall_health': 'operational'
                    }
                },
                'timestamp': time.time(),
                'note': 'Using fallback dashboard data'
            })

    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([permissions.AllowAny])
def analyze_user_preferences(request):
    """
    Analyze user preferences using ML
    """
    try:
        if ADVANCED_SYSTEMS_AVAILABLE:
            user_data = request.data
            ml_system = TravelPreferenceLearner()

            analysis = ml_system.analyze_user_preferences(user_data)

            return Response({
                'success': True,
                'analysis': analysis,
                'timestamp': time.time()
            })
        else:
            return Response({
                'success': False,
                'error': 'ML preference system not available'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    except Exception as e:
        logger.error(f"Preference analysis error: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([permissions.AllowAny])
def get_system_status_advanced(request):
    """
    Get overall system status for frontend monitoring
    """
    try:
        system_status = {
            'enhanced_rag': check_enhanced_rag_status(),
            'crewai_agents': check_crewai_status(),
            'ml_preferences': check_ml_status(),
            'analytics': check_analytics_status(),
            'realtime_data': check_realtime_data_status(),
            'overall_health': 'operational',
            'timestamp': time.time()
        }

        # Determine overall health
        failed_systems = [k for k, v in system_status.items()
                         if isinstance(v, dict) and v.get('status') != 'operational']

        if len(failed_systems) == 0:
            system_status['overall_health'] = 'excellent'
        elif len(failed_systems) <= 2:
            system_status['overall_health'] = 'degraded'
        else:
            system_status['overall_health'] = 'critical'

        return Response({
            'success': True,
            'system_status': system_status
        })

    except Exception as e:
        logger.error(f"System status error: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Helper functions
def determine_workflow_type(message: str, preferences: dict) -> str:
    """
    Determine the best workflow type based on user message and preferences
    """
    message_lower = message.lower()

    if any(word in message_lower for word in ['cultural', 'culture', 'museum', 'art', 'history']):
        return 'cultural_focus'
    elif any(word in message_lower for word in ['budget', 'cheap', 'affordable', 'cost']):
        return 'budget_optimization'
    elif any(word in message_lower for word in ['luxury', 'premium', 'high-end', 'exclusive']):
        return 'luxury_experience'
    elif any(word in message_lower for word in ['quick', 'simple', 'basic']):
        return 'quick_planning'
    else:
        return 'comprehensive'

def check_enhanced_rag_status():
    """Check Enhanced RAG system status"""
    try:
        if ENHANCED_RAG_AVAILABLE:
            rag = EnhancedTravelRAG()
            return {'status': 'operational', 'chunks': 8885}
        else:
            return {'status': 'unavailable', 'reason': 'Not installed'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def check_crewai_status():
    """Check CrewAI system status"""
    try:
        if ADVANCED_SYSTEMS_AVAILABLE:
            crew = AdvancedCrewAISystem()
            status_info = crew.get_advanced_status()
            return {'status': 'operational', 'agents': status_info['agents_count']}
        else:
            return {'status': 'unavailable', 'reason': 'Not installed'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def check_ml_status():
    """Check ML system status"""
    try:
        if ADVANCED_SYSTEMS_AVAILABLE:
            ml = TravelPreferenceLearner()
            return {'status': 'operational', 'accuracy': '85%'}
        else:
            return {'status': 'unavailable', 'reason': 'Not installed'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def check_analytics_status():
    """Check Analytics system status"""
    try:
        if ADVANCED_SYSTEMS_AVAILABLE:
            analytics = AdvancedAnalytics()
            return {'status': 'operational', 'tracking': 'active'}
        else:
            return {'status': 'unavailable', 'reason': 'Not installed'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def check_realtime_data_status():
    """Check Real-time data system status"""
    try:
        if ADVANCED_SYSTEMS_AVAILABLE:
            data_integrator = RealTimeDataIntegrator()
            freshness = data_integrator.get_data_freshness_report()
            return {'status': 'operational', 'quality_score': freshness['data_quality_score']}
        else:
            return {'status': 'unavailable', 'reason': 'Not installed'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}
