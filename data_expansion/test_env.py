"""
Test script to verify .env file configuration
"""

import os
from dotenv import load_dotenv
from pathlib import Path

def test_env_loading():
    """Test if .env file is loaded correctly"""
    print("🧪 Testing .env file configuration...\n")
    
    # Check if .env file exists
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("❌ .env file not found")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Check Groq API key
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        if groq_key == 'your_groq_api_key_here':
            print("⚠️ GROQ_API_KEY is set to placeholder value")
            print("   Please update .env file with your actual API key")
        else:
            # Mask the key for security
            masked_key = groq_key[:8] + "..." + groq_key[-4:] if len(groq_key) > 12 else "***"
            print(f"✅ GROQ_API_KEY loaded: {masked_key}")
    else:
        print("❌ GROQ_API_KEY not found in .env file")
    
    # Note: We only use Groq with LangChain now
    print("ℹ️ Using LangChain + Groq for synthetic generation")
    
    # Check other settings
    groq_model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
    print(f"📊 GROQ_MODEL: {groq_model}")

    max_retries = os.getenv('MAX_RETRIES', '3')
    print(f"📊 MAX_RETRIES: {max_retries}")

    request_delay = os.getenv('REQUEST_DELAY', '1.0')
    print(f"📊 REQUEST_DELAY: {request_delay}")

    output_dir = os.getenv('OUTPUT_DIR', 'expanded_data')
    print(f"📊 OUTPUT_DIR: {output_dir}")

    backup_enabled = os.getenv('BACKUP_ENABLED', 'true').lower() == 'true'
    print(f"📊 BACKUP_ENABLED: {backup_enabled}")

    print("\n🎉 Environment configuration test completed!")

    # Return True if Groq API key is properly configured
    return groq_key and groq_key != 'your_groq_api_key_here'

def test_api_connection():
    """Test LangChain + Groq connection"""
    print("\n🔌 Testing LangChain + Groq connection...\n")

    load_dotenv()

    # Test LangChain + Groq connection
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key and groq_key != 'your_groq_api_key_here':
        try:
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                groq_api_key=groq_key,
                model_name="llama3-8b-8192",
                temperature=0.7,
                max_tokens=50
            )
            print("✅ LangChain + Groq client initialized successfully")

            # Test a simple API call
            try:
                response = llm.invoke("Hello, how are you?")
                print("✅ LangChain + Groq API connection successful")
                print(f"📝 Test response: {response.content[:100]}...")
                return True
            except Exception as e:
                print(f"❌ LangChain + Groq API test failed: {e}")

        except ImportError:
            print("⚠️ LangChain Groq library not installed")
            print("   Run: pip install langchain-groq")
        except Exception as e:
            print(f"❌ LangChain + Groq client initialization failed: {e}")
    else:
        print("❌ GROQ_API_KEY not configured")

    return False

if __name__ == "__main__":
    print("🚀 Environment Configuration Test\n")
    
    # Test environment loading
    env_success = test_env_loading()
    
    if env_success:
        # Test API connections
        api_success = test_api_connection()
        
        if api_success:
            print("\n🎉 All tests passed! Ready for data expansion.")
        else:
            print("\n⚠️ Environment loaded but API connection failed.")
            print("   Check your API keys and internet connection.")
    else:
        print("\n❌ Environment configuration failed.")
        print("   Please check your .env file and API keys.")
    
    print("\n📝 Next steps:")
    print("   1. Ensure your GROQ_API_KEY is correctly set in .env")
    print("   2. Run: python cli.py setup")
    print("   3. Run: python cli.py generate --count 5")
    print("   4. Run: python cli.py expand --count 1000")
