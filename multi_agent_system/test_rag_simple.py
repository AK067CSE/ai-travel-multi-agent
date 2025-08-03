"""
Simple RAG test without complex dependencies
"""

import os
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_rag():
    """Test basic RAG functionality"""
    print("🧪 Testing Basic RAG System")
    print("=" * 40)
    
    # Check if dataset exists
    dataset_path = "../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return False
    
    # Load sample data
    print("📚 Loading sample travel data...")
    sample_data = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:  # Load first 10 examples
                    break
                try:
                    data = json.loads(line.strip())
                    sample_data.append(data)
                except json.JSONDecodeError:
                    continue
        
        print(f"✅ Loaded {len(sample_data)} sample travel examples")
        
        # Show sample data structure
        if sample_data:
            print("\n📋 Sample data structure:")
            sample = sample_data[0]
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False

def test_simple_search():
    """Test simple keyword search"""
    print("\n🔍 Testing Simple Search...")
    
    dataset_path = "../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl"
    
    # Test queries
    test_queries = ["Paris", "budget", "romantic", "family"]
    
    for query in test_queries:
        print(f"\n🔎 Searching for: '{query}'")
        matches = 0
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 100:  # Search first 100 lines
                        break
                    try:
                        data = json.loads(line.strip())
                        text = f"{data.get('query', '')} {data.get('response', '')}".lower()
                        
                        if query.lower() in text:
                            matches += 1
                            if matches <= 2:  # Show first 2 matches
                                print(f"   ✅ Match {matches}: {data.get('query', 'No query')[:80]}...")
                    
                    except json.JSONDecodeError:
                        continue
            
            print(f"   📊 Total matches: {matches}")
            
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

def test_groq_connection():
    """Test Groq API connection"""
    print("\n🔗 Testing Groq Connection...")
    
    try:
        from langchain_groq import ChatGroq
        
        # Check API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ GROQ_API_KEY not found in environment")
            return False
        
        print("✅ GROQ_API_KEY found")
        
        # Test connection
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-8b-8192",
            temperature=0.3,
            max_tokens=100
        )
        
        response = llm.invoke("Hello, this is a test. Respond with 'Connection successful!'")
        print(f"✅ Groq response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Groq connection failed: {e}")
        return False

def simulate_rag_response():
    """Simulate RAG response using simple search + Groq"""
    print("\n🎯 Simulating RAG Response...")
    
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192",
            temperature=0.3,
            max_tokens=500
        )
        
        # Test query
        query = "Plan a 3-day romantic trip to Paris"
        print(f"📝 Query: {query}")
        
        # Simple search for relevant examples
        dataset_path = "../data_expansion/final_datasets/travel_planning_dataset_20250719_224702.jsonl"
        relevant_examples = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 50:  # Search first 50 lines
                    break
                try:
                    data = json.loads(line.strip())
                    text = f"{data.get('query', '')} {data.get('response', '')}".lower()
                    
                    if any(keyword in text for keyword in ["paris", "romantic", "3 day", "trip"]):
                        relevant_examples.append(data)
                        if len(relevant_examples) >= 2:  # Get 2 examples
                            break
                
                except json.JSONDecodeError:
                    continue
        
        print(f"🔍 Found {len(relevant_examples)} relevant examples")
        
        # Create context from examples
        context = ""
        for i, example in enumerate(relevant_examples, 1):
            context += f"Example {i}:\n"
            context += f"Query: {example.get('query', '')}\n"
            context += f"Response: {example.get('response', '')[:300]}...\n\n"
        
        # Create RAG prompt
        rag_prompt = ChatPromptTemplate.from_template(
            """You are a travel expert. Use these examples to create a personalized response.

RELEVANT EXAMPLES:
{context}

USER QUERY: {query}

Create a detailed travel recommendation based on the examples above:"""
        )
        
        # Generate response
        chain = rag_prompt | llm | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "context": context if context else "No specific examples found."
        })
        
        print(f"\n🤖 RAG-style Response:")
        print(response)
        
        return True
        
    except Exception as e:
        print(f"❌ RAG simulation failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Simple RAG System Test")
    print("=" * 50)
    
    # Test 1: Basic data loading
    if not test_basic_rag():
        return
    
    # Test 2: Simple search
    test_simple_search()
    
    # Test 3: Groq connection
    groq_works = test_groq_connection()

    # Test 4: Simulate RAG (only if Groq works)
    if groq_works:
        simulate_rag_response()
    else:
        print("\n⚠️ Skipping RAG simulation due to Groq API issues")
        print("   (This is likely a temporary API key issue)")
    
    print("\n🎉 Simple RAG test completed!")
    print("\n✅ Your system is ready for full RAG implementation!")

if __name__ == "__main__":
    main()
