#!/usr/bin/env python
"""
Test environment variables for CrewAI
"""

import os
from dotenv import load_dotenv

print("🔍 Testing Environment Variables")
print("=" * 40)

# Load .env file
load_dotenv()

# Check GROQ API key
groq_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY: {groq_key[:10] if groq_key else 'NOT SET'}...")

# Check Gemini API key
gemini_key = os.getenv("GEMINI_API_KEY")
print(f"GEMINI_API_KEY: {gemini_key[:10] if gemini_key else 'NOT SET'}...")

# Test if keys are valid (not placeholder)
if groq_key and groq_key != "your-groq-api-key-here":
    print("✅ GROQ_API_KEY is set and valid")
else:
    print("❌ GROQ_API_KEY is missing or placeholder")

if gemini_key and gemini_key != "your-gemini-api-key-here":
    print("✅ GEMINI_API_KEY is set and valid")
else:
    print("❌ GEMINI_API_KEY is missing or placeholder")

# Set environment variables for CrewAI
if groq_key and groq_key != "your-groq-api-key-here":
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["OPENAI_MODEL_NAME"] = "groq/llama3-70b-8192"
    print("✅ Environment variables set for CrewAI")
else:
    print("❌ Cannot set environment variables - invalid GROQ key")

print("=" * 40)
