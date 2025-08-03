from rest_framework import serializers
from .models import Conversation, ChatMessage, TravelRecommendation


class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=2000)
    session_id = serializers.CharField(max_length=100, required=False)
    preferences = serializers.JSONField(required=False, default=dict)


class ChatResponseSerializer(serializers.Serializer):
    success = serializers.BooleanField()
    response = serializers.CharField()
    session_id = serializers.CharField()
    response_time = serializers.FloatField()
    system_used = serializers.CharField()
    agents_involved = serializers.ListField(child=serializers.CharField())
    conversation_id = serializers.CharField()
    timestamp = serializers.FloatField()


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = '__all__'


class ConversationSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = Conversation
        fields = '__all__'


class TravelRecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = TravelRecommendation
        fields = '__all__'


class SystemStatusSerializer(serializers.Serializer):
    ai_agent_status = serializers.CharField()
    overall_health = serializers.CharField()
    timestamp = serializers.FloatField()
