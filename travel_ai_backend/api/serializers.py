from rest_framework import serializers
from .models import User, Conversation, TravelRecommendation

class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model"""
    
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name',
            'travel_preferences', 'budget_range', 'interests', 'travel_style',
            'previous_destinations', 'travel_frequency', 'dietary_restrictions',
            'accessibility_needs', 'is_premium', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def validate_interests(self, value):
        """Validate interests field"""
        if not isinstance(value, list):
            raise serializers.ValidationError("Interests must be a list")
        return value
    
    def validate_travel_preferences(self, value):
        """Validate travel preferences field"""
        if not isinstance(value, dict):
            raise serializers.ValidationError("Travel preferences must be a dictionary")
        return value

class ConversationSerializer(serializers.ModelSerializer):
    """Serializer for Conversation model"""
    user = serializers.StringRelatedField(read_only=True)
    message_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'user', 'session_id', 'messages', 'context',
            'created_at', 'updated_at', 'is_active', 'total_messages',
            'avg_response_time', 'user_satisfaction', 'message_count', 'last_message'
        ]
        read_only_fields = ['id', 'user', 'created_at', 'updated_at']
    
    def get_message_count(self, obj):
        """Get the number of messages in conversation"""
        return len(obj.messages) if obj.messages else 0
    
    def get_last_message(self, obj):
        """Get the last message in conversation"""
        if obj.messages and len(obj.messages) > 0:
            return obj.messages[-1]
        return None

class TravelRecommendationSerializer(serializers.ModelSerializer):
    """Serializer for TravelRecommendation model"""
    user = serializers.StringRelatedField(read_only=True)
    conversation = serializers.StringRelatedField(read_only=True)
    has_rating = serializers.SerializerMethodField()
    
    class Meta:
        model = TravelRecommendation
        fields = [
            'id', 'user', 'conversation', 'query', 'response',
            'agents_used', 'rag_sources', 'processing_time',
            'destination', 'budget_estimate', 'duration_days',
            'recommendation_type', 'rating', 'feedback',
            'is_bookmarked', 'created_at', 'updated_at', 'has_rating'
        ]
        read_only_fields = ['id', 'user', 'conversation', 'created_at', 'updated_at']
    
    def get_has_rating(self, obj):
        """Check if recommendation has been rated"""
        return obj.rating is not None

class ChatRequestSerializer(serializers.Serializer):
    """Serializer for chat request data"""
    message = serializers.CharField(max_length=2000, required=True)
    session_id = serializers.CharField(max_length=100, required=False)
    context = serializers.DictField(required=False, default=dict)
    
    def validate_message(self, value):
        """Validate message content"""
        if not value.strip():
            raise serializers.ValidationError("Message cannot be empty")
        return value.strip()

class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat response data"""
    message = serializers.CharField()
    session_id = serializers.CharField()
    processing_time = serializers.FloatField()
    agents_used = serializers.ListField(child=serializers.CharField())
    success = serializers.BooleanField()
    recommendation_id = serializers.UUIDField()

class RecommendationRequestSerializer(serializers.Serializer):
    """Serializer for direct recommendation requests"""
    destination = serializers.CharField(max_length=200, required=False)
    budget = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    duration = serializers.IntegerField(required=False)
    interests = serializers.ListField(child=serializers.CharField(), required=False)
    travel_style = serializers.CharField(max_length=50, required=False)
    group_size = serializers.IntegerField(required=False, default=1)
    
    def validate_duration(self, value):
        """Validate trip duration"""
        if value and (value < 1 or value > 365):
            raise serializers.ValidationError("Duration must be between 1 and 365 days")
        return value
    
    def validate_group_size(self, value):
        """Validate group size"""
        if value and (value < 1 or value > 50):
            raise serializers.ValidationError("Group size must be between 1 and 50")
        return value

class UserPreferencesSerializer(serializers.Serializer):
    """Serializer for updating user preferences"""
    budget_range = serializers.ChoiceField(
        choices=['budget', 'mid_range', 'luxury'],
        required=False
    )
    interests = serializers.ListField(
        child=serializers.CharField(max_length=50),
        required=False
    )
    travel_style = serializers.ChoiceField(
        choices=['explorer', 'relaxer', 'adventurer', 'cultural', 'foodie'],
        required=False
    )
    dietary_restrictions = serializers.ListField(
        child=serializers.CharField(max_length=50),
        required=False
    )
    accessibility_needs = serializers.ListField(
        child=serializers.CharField(max_length=50),
        required=False
    )

class AgentStatusSerializer(serializers.Serializer):
    """Serializer for agent status response"""
    agents_available = serializers.BooleanField()
    system_status = serializers.CharField()
    available_agents = serializers.ListField(child=serializers.CharField())
    rag_enabled = serializers.BooleanField()
    timestamp = serializers.FloatField()

class RatingSerializer(serializers.Serializer):
    """Serializer for rating recommendations"""
    rating = serializers.IntegerField(min_value=1, max_value=5)
    feedback = serializers.CharField(max_length=1000, required=False, allow_blank=True)

class ConversationSummarySerializer(serializers.ModelSerializer):
    """Lightweight serializer for conversation summaries"""
    user = serializers.StringRelatedField(read_only=True)
    message_count = serializers.SerializerMethodField()
    last_activity = serializers.DateTimeField(source='updated_at', read_only=True)
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'user', 'session_id', 'created_at', 'last_activity',
            'is_active', 'total_messages', 'avg_response_time',
            'user_satisfaction', 'message_count'
        ]
    
    def get_message_count(self, obj):
        """Get the number of messages in conversation"""
        return len(obj.messages) if obj.messages else 0

class RecommendationSummarySerializer(serializers.ModelSerializer):
    """Lightweight serializer for recommendation summaries"""
    user = serializers.StringRelatedField(read_only=True)
    query_preview = serializers.SerializerMethodField()
    response_preview = serializers.SerializerMethodField()
    
    class Meta:
        model = TravelRecommendation
        fields = [
            'id', 'user', 'query_preview', 'response_preview',
            'destination', 'recommendation_type', 'rating',
            'is_bookmarked', 'created_at', 'processing_time'
        ]
    
    def get_query_preview(self, obj):
        """Get preview of query (first 100 characters)"""
        return obj.query[:100] + "..." if len(obj.query) > 100 else obj.query
    
    def get_response_preview(self, obj):
        """Get preview of response (first 200 characters)"""
        return obj.response[:200] + "..." if len(obj.response) > 200 else obj.response
