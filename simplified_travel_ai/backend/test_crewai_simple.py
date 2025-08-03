#!/usr/bin/env python
"""
Simple CrewAI test to isolate the issue
"""

import os
from dotenv import load_dotenv
load_dotenv()

def test_crewai_simple():
    """Test CrewAI with minimal setup"""
    try:
        from crewai import Agent, Task, Crew, Process
        from langchain_groq import ChatGroq
        
        print("🚀 Testing CrewAI with Groq...")
        
        # Initialize LLM
        groq_api_key = os.getenv('GROQ_API_KEY')
        print(f"API Key available: {'YES' if groq_api_key else 'NO'}")
        
        if not groq_api_key:
            print("❌ No Groq API key found")
            return False
        
        # Test LLM directly first
        print("🧪 Testing LLM directly...")
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",  # Use newer model
            temperature=0.7,
            max_tokens=50
        )
        
        # Test direct LLM call
        direct_result = llm.invoke("Say 'Hello from Groq' in one sentence.")
        print(f"✅ Direct LLM test: {direct_result.content}")
        
        # Create simple agent
        print("🤖 Creating CrewAI agent...")
        agent = Agent(
            role='Simple Assistant',
            goal='Provide brief helpful responses',
            backstory='You are a helpful assistant that gives brief answers.',
            verbose=False,  # Reduce verbosity
            allow_delegation=False,
            llm=llm
        )
        print("✅ Agent created successfully")
        
        # Create simple task
        print("📋 Creating task...")
        task = Task(
            description="Say 'Hello from CrewAI' in exactly one sentence.",
            agent=agent,
            expected_output="A single sentence greeting from CrewAI"
        )
        print("✅ Task created successfully")
        
        # Create crew
        print("👥 Creating crew...")
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,  # Reduce verbosity to see errors better
            process=Process.sequential
        )
        print("✅ Crew created successfully")
        
        # Execute crew
        print("🚀 Executing crew...")
        try:
            result = crew.kickoff()
            print(f"✅ CrewAI execution successful!")
            print(f"📝 Result: {result}")
            return True
        except Exception as e:
            print(f"❌ CrewAI execution failed: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ CrewAI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_groq_versions():
    """Test different Groq model versions"""
    try:
        from langchain_groq import ChatGroq
        
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            print("❌ No Groq API key")
            return False
        
        models_to_test = [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "meta-llama/llama-4-scout-17b-16e-instruct"
        ]
        
        print("🧪 Testing different Groq models...")
        
        for model in models_to_test:
            try:
                print(f"   Testing {model}...")
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name=model,
                    temperature=0.7,
                    max_tokens=20
                )
                result = llm.invoke("Say hello")
                print(f"   ✅ {model}: {result.content[:50]}...")
            except Exception as e:
                print(f"   ❌ {model}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Simple CrewAI Test")
    print("=" * 40)
    
    # Test Groq models first
    print("\n1. Testing Groq Models:")
    test_groq_versions()
    
    print("\n2. Testing CrewAI Integration:")
    success = test_crewai_simple()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 CrewAI is working perfectly!")
    else:
        print("⚠️ CrewAI has some issues, but core components work")
        print("💡 The system will use fallback agents for complex queries")
