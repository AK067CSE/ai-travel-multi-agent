"""
Simplified Travel AI API Views
Clean, efficient endpoints for the travel AI system
"""

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.http import StreamingHttpResponse
import json
import time
import logging

from .models import Conversation, ChatMessage, TravelRecommendation
from .serializers import (
    ChatRequestSerializer, ChatResponseSerializer,
    ConversationSerializer, SystemStatusSerializer
)
from .ai_agent import TravelAIAgent

logger = logging.getLogger(__name__)

# Initialize AI Agent
ai_agent = TravelAIAgent()


@api_view(['POST'])
@permission_classes([AllowAny])
def chat_with_ai(request):
    """
    Main chat endpoint for Travel AI Agent
    Handles user queries and returns AI-generated responses
    """
    try:
        # Validate request
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"Validation errors: {serializer.errors}")
            return Response({
                'success': False,
                'error': 'Invalid request data',
                'details': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user_message = serializer.validated_data['message']
        session_id = serializer.validated_data.get('session_id', f"session_{int(time.time())}")
        preferences = serializer.validated_data.get('preferences', {})

        logger.info(f"Processing chat request: {user_message[:50]}... Session: {session_id}")
        
        # Get or create conversation
        conversation, created = Conversation.objects.get_or_create(
            session_id=session_id,
            defaults={'is_active': True}
        )
        
        # Process with AI Agent
        ai_response = ai_agent.process_travel_request(
            user_message=user_message,
            user_preferences=preferences,
            session_id=session_id
        )
        
        # Save chat message
        chat_message = ChatMessage.objects.create(
            conversation=conversation,
            user_message=user_message,
            ai_response=ai_response['response'],
            response_time=ai_response['processing_time'],
            system_used=ai_response['system_used'],
            agents_involved=ai_response['agents_involved']
        )
        
        # Update conversation
        conversation.total_messages += 1
        conversation.save()
        
        # Create recommendation if successful
        if ai_response['success']:
            TravelRecommendation.objects.create(
                conversation=conversation,
                query=user_message,
                response=ai_response['response']
            )
        
        # Prepare response
        response_data = {
            'success': ai_response['success'],
            'response': ai_response['response'],
            'session_id': session_id,
            'response_time': ai_response['processing_time'],
            'system_used': ai_response['system_used'],
            'agents_involved': ai_response['agents_involved'],
            'conversation_id': str(conversation.id),
            'timestamp': time.time()
        }
        
        # Add additional metadata if available
        if 'intent' in ai_response:
            response_data['intent'] = ai_response['intent']
        if 'recommendations_count' in ai_response:
            response_data['recommendations_count'] = ai_response['recommendations_count']
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return Response({
            'success': False,
            'error': 'Internal server error',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST', 'OPTIONS'])
@permission_classes([AllowAny])
def chat_stream(request):
    """
    Streaming chat endpoint for real-time responses
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response['Access-Control-Allow-Credentials'] = 'true'
        return response

    try:
        # Validate request
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        user_message = serializer.validated_data['message']
        session_id = serializer.validated_data.get('session_id', f"session_{int(time.time())}")
        preferences = serializer.validated_data.get('preferences', {})
        
        def generate_response():
            """Generator function for streaming response"""
            try:
                # Send initial status
                yield f"data: {json.dumps({'type': 'status', 'message': 'Processing your request...'})}\n\n"
                
                # Get or create conversation
                conversation, created = Conversation.objects.get_or_create(
                    session_id=session_id,
                    defaults={'is_active': True}
                )
                
                yield f"data: {json.dumps({'type': 'status', 'message': 'AI Agent is thinking...'})}\n\n"
                
                # Process with AI Agent
                ai_response = ai_agent.process_travel_request(
                    user_message=user_message,
                    user_preferences=preferences,
                    session_id=session_id
                )
                
                # Stream the response in chunks
                response_text = ai_response['response']
                words = response_text.split()
                
                for i in range(0, len(words), 5):  # Send 5 words at a time
                    chunk = ' '.join(words[i:i+5])
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk + ' '})}\n\n"
                    time.sleep(0.1)  # Small delay for streaming effect
                
                # Save chat message
                chat_message = ChatMessage.objects.create(
                    conversation=conversation,
                    user_message=user_message,
                    ai_response=ai_response['response'],
                    response_time=ai_response['processing_time'],
                    system_used=ai_response['system_used'],
                    agents_involved=ai_response['agents_involved']
                )
                
                # Update conversation
                conversation.total_messages += 1
                conversation.save()
                
                # Send completion data
                completion_data = {
                    'type': 'complete',
                    'success': ai_response['success'],
                    'session_id': session_id,
                    'response_time': ai_response['processing_time'],
                    'system_used': ai_response['system_used'],
                    'agents_involved': ai_response['agents_involved'],
                    'conversation_id': str(conversation.id)
                }
                
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        response = StreamingHttpResponse(
            generate_response(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['Connection'] = 'keep-alive'
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response['Access-Control-Allow-Credentials'] = 'true'

        return response
        
    except Exception as e:
        logger.error(f"Stream setup error: {e}")
        return Response({
            'success': False,
            'error': 'Failed to setup streaming',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def system_status(request):
    """
    Get system status and health information
    """
    try:
        agent_status = ai_agent.get_system_status()
        
        status_data = {
            'ai_agent_status': agent_status['status'],
            'overall_health': 'excellent' if agent_status['openai_available'] else 'good',
            'ai_model': agent_status['ai_model'],
            'features': {
                'openai_integration': agent_status['openai_available'],
                'knowledge_base': True,
                'streaming_responses': True,
                'conversation_memory': True
            },
            'statistics': {
                'destinations_in_db': agent_status['knowledge_base_destinations'],
                'travel_types_supported': agent_status['travel_types_supported'],
                'total_conversations': Conversation.objects.count(),
                'total_messages': ChatMessage.objects.count()
            },
            'timestamp': agent_status['timestamp']
        }
        
        return Response(status_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"System status error: {e}")
        return Response({
            'error': 'Failed to get system status',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def get_conversations(request):
    """
    Get user's conversation history
    """
    try:
        session_id = request.GET.get('session_id')
        
        if session_id:
            conversations = Conversation.objects.filter(session_id=session_id)
        else:
            conversations = Conversation.objects.all()[:10]  # Last 10 conversations
        
        serializer = ConversationSerializer(conversations, many=True)
        return Response({
            'success': True,
            'conversations': serializer.data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Conversations error: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([AllowAny])
def rate_recommendation(request, recommendation_id):
    """
    Rate a travel recommendation
    """
    try:
        recommendation = TravelRecommendation.objects.get(id=recommendation_id)
        
        rating = request.data.get('rating')
        feedback = request.data.get('feedback', '')
        
        if rating and 1 <= int(rating) <= 5:
            recommendation.rating = int(rating)
            recommendation.feedback = feedback
            recommendation.save()
            
            return Response({
                'success': True,
                'message': 'Rating saved successfully',
                'rating': recommendation.rating
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                'success': False,
                'error': 'Rating must be between 1 and 5'
            }, status=status.HTTP_400_BAD_REQUEST)
            
    except TravelRecommendation.DoesNotExist:
        return Response({
            'success': False,
            'error': 'Recommendation not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Rating error: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
