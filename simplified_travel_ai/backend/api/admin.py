from django.contrib import admin
from .models import Conversation, ChatMessage, TravelRecommendation


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'user', 'total_messages', 'created_at', 'is_active']
    list_filter = ['is_active', 'created_at']
    search_fields = ['session_id', 'user__username']
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'timestamp', 'response_time', 'system_used']
    list_filter = ['system_used', 'timestamp']
    search_fields = ['user_message', 'ai_response']
    readonly_fields = ['id', 'timestamp']


@admin.register(TravelRecommendation)
class TravelRecommendationAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'user', 'rating', 'created_at']
    list_filter = ['rating', 'created_at']
    search_fields = ['query', 'response']
    readonly_fields = ['id', 'created_at']
