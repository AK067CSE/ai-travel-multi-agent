"""
URL configuration for Travel AI Backend project.
Backend AI Developer - Omnibound Compatible
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from django.http import JsonResponse

# API Documentation Schema
schema_view = get_schema_view(
    openapi.Info(
        title="Travel AI Backend API",
        default_version='v1',
        description="""
        Advanced Travel AI Backend with Multi-Agent System and RAG

        Features:
        - Multi-agent AI coordination (5 specialized agents)
        - RAG-enhanced recommendations from 1,160+ travel examples
        - Real-time chat with conversation memory
        - Multiple LLM support (GPT-4, Claude, Llama, Gemini)
        - Vector database integration (Pinecone, Weaviate, ChromaDB)
        - User preference learning and personalization

        Built for Backend AI Developer role at Omnibound.ai
        """,
        terms_of_service="https://www.example.com/terms/",
        contact=openapi.Contact(email="contact@travelai.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)

def api_root(request):
    """API root endpoint with system information"""
    return JsonResponse({
        'message': 'Travel AI Backend API',
        'version': 'v1',
        'features': [
            'Multi-Agent AI System',
            'RAG-Enhanced Recommendations',
            'Real-time Chat Interface',
            'User Preference Learning',
            'Vector Database Integration',
            'Multiple LLM Support'
        ],
        'documentation': '/swagger/',
        'admin': '/admin/',
        'endpoints': {
            'chat': '/api/chat/',
            'recommendations': '/api/recommendations/',
            'profile': '/api/profile/',
            'conversations': '/api/conversations/',
            'agent_status': '/api/agents/status/'
        }
    })

urlpatterns = [
    # Admin
    path("admin/", admin.site.urls),

    # API Root
    path('', api_root, name='api-root'),

    # API Endpoints
    path('api/', include('api.urls')),

    # API Documentation
    path('swagger<format>/', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),

    # Authentication (Django REST Framework)
    path('api-auth/', include('rest_framework.urls')),
]
