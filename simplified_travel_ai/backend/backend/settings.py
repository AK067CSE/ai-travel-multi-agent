"""
Django settings for simplified travel AI backend.
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-your-secret-key-here'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '*']

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# CORS settings
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
    'cache-control',
    'connection',
]
CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'travel_ai.log',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'api': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# Enhanced AI System Configuration - Production Level
AI_CONFIG = {
    # API Keys
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
    'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY', ''),
    'GROQ_API_KEY': os.getenv('GROQ_API_KEY', ''),
    'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY', ''),

    # Model Configuration
    'DEFAULT_LLM_PROVIDER': os.getenv('DEFAULT_LLM_PROVIDER', 'groq'),
    'DEFAULT_MODEL': os.getenv('DEFAULT_MODEL', 'llama3-70b-8192'),
    'FALLBACK_MODEL': os.getenv('FALLBACK_MODEL', 'llama3-8b-8192'),
    'MAX_TOKENS': int(os.getenv('MAX_TOKENS', '2000')),
    'TEMPERATURE': float(os.getenv('TEMPERATURE', '0.7')),

    # RAG Configuration
    'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5'),
    'VECTOR_DB_PATH': os.getenv('VECTOR_DB_PATH', './vectordb'),
    'TOP_K_RETRIEVAL': int(os.getenv('TOP_K_RETRIEVAL', '10')),
    'RERANK_TOP_K': int(os.getenv('RERANK_TOP_K', '5')),
    'ENABLE_QUERY_EXPANSION': os.getenv('ENABLE_QUERY_EXPANSION', 'True').lower() == 'true',
    'ENABLE_HYBRID_SEARCH': os.getenv('ENABLE_HYBRID_SEARCH', 'True').lower() == 'true',

    # Multi-Agent Configuration
    'ENABLE_MULTI_AGENT': os.getenv('ENABLE_MULTI_AGENT', 'True').lower() == 'true',
    'MAX_AGENT_ITERATIONS': int(os.getenv('MAX_AGENT_ITERATIONS', '5')),
    'AGENT_TIMEOUT': int(os.getenv('AGENT_TIMEOUT', '30')),

    # Data Configuration
    'TRAVEL_DATA_PATH': os.getenv('TRAVEL_DATA_PATH', './data/travel_dataset.jsonl'),
    'ENABLE_WEB_SCRAPING': os.getenv('ENABLE_WEB_SCRAPING', 'False').lower() == 'true',
    'SCRAPING_RATE_LIMIT': float(os.getenv('SCRAPING_RATE_LIMIT', '1.0')),

    # Performance Configuration
    'BATCH_SIZE': int(os.getenv('BATCH_SIZE', '32')),
    'MAX_CONCURRENT_REQUESTS': int(os.getenv('MAX_CONCURRENT_REQUESTS', '10')),
    'CACHE_TTL': int(os.getenv('CACHE_TTL', '3600')),
}
