from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# API URL patterns for Travel AI Backend
urlpatterns = [
    # Chat API
    path('chat/', views.ChatAPIView.as_view(), name='chat'),
    
    # Recommendations API
    path('recommendations/', views.RecommendationAPIView.as_view(), name='recommendations-list'),
    path('recommendations/<uuid:recommendation_id>/rate/', views.rate_recommendation, name='rate-recommendation'),
    
    # User Profile API
    path('profile/', views.UserProfileAPIView.as_view(), name='user-profile'),
    
    # Conversations API
    path('conversations/', views.ConversationListAPIView.as_view(), name='conversations-list'),
    path('conversations/<uuid:pk>/', views.ConversationDetailAPIView.as_view(), name='conversation-detail'),
    
    # System Status API
    path('agents/status/', views.agent_status, name='agent-status'),

    # Advanced AI Agent Endpoints
    path('chat/ai-agent/', views.chat_with_ai_agent, name='chat-ai-agent'),
    path('dashboard/', views.get_dashboard_data, name='dashboard-data'),
    path('preferences/analyze/', views.analyze_user_preferences, name='analyze-preferences'),
    path('status/advanced/', views.get_system_status_advanced, name='system-status-advanced'),
]
