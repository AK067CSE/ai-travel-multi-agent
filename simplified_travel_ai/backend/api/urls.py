"""
URL configuration for simplified travel AI API
"""
from django.urls import path
from . import views

urlpatterns = [
    # Main chat endpoints
    path('chat/', views.chat_with_ai, name='chat_with_ai'),
    path('chat/stream/', views.chat_stream, name='chat_stream'),
    
    # System and status
    path('status/', views.system_status, name='system_status'),
    
    # Conversations
    path('conversations/', views.get_conversations, name='get_conversations'),
    
    # Recommendations
    path('recommendations/<uuid:recommendation_id>/rate/', views.rate_recommendation, name='rate_recommendation'),
]
