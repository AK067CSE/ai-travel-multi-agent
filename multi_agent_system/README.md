# Multi-Agent Travel AI System

Advanced multi-agent travel assistant built with **LangChain + Groq** featuring specialized agents for comprehensive travel planning, booking, and assistance.

## 🏗️ Architecture Overview

### 🤖 **Specialized Agents**

1. **🎛️ CoordinatorAgent** - Master orchestrator
   - Manages multi-agent workflows
   - Optimizes task distribution
   - Handles complex request routing

2. **🕷️ ScrapingAgent** - Data collection specialist
   - Real-time web scraping
   - Hotel, flight, and attraction data
   - Travel information gathering

3. **🎯 RecommendationAgent** - Personalization expert
   - AI-powered travel suggestions
   - User preference learning
   - Customized itinerary planning

4. **📋 BookingAgent** - Reservation specialist
   - Hotel and flight bookings
   - Booking management and modifications
   - Payment processing simulation

5. **💬 ChatAgent** - Conversation manager
   - Natural language interaction
   - Intent detection and routing
   - Context-aware conversations

## 🚀 Key Features

### **Multi-Agent Coordination**
- **Parallel Processing**: Multiple agents work simultaneously
- **Intelligent Routing**: Requests automatically routed to appropriate agents
- **Context Sharing**: Agents share information for better results
- **Fallback Handling**: Graceful error recovery and alternatives

### **Advanced Capabilities**
- **Conversational Memory**: Maintains context across interactions
- **User Profiling**: Learns and adapts to user preferences
- **Real-time Data**: Live scraping of travel information
- **Booking Simulation**: Complete booking workflow simulation
- **Performance Metrics**: Detailed system monitoring

### **LangChain Integration**
- **Structured Prompts**: Professional prompt templates
- **Chain Composition**: Modular, reusable components
- **Memory Management**: Conversation and context persistence
- **Error Handling**: Robust retry and fallback mechanisms

## 📦 Installation

1. **Setup Environment**:
```bash
cd multi_agent_system
pip install -r requirements.txt
```

2. **Configure API Keys**:
```bash
# Create .env file
GROQ_API_KEY=your_groq_api_key_here
```

3. **Test Installation**:
```bash
python cli.py status
```

## 🎯 Usage

### **Interactive Chat**
```bash
# Start interactive conversation
python cli.py chat

# With specific user ID
python cli.py chat --user-id john_doe

# Use complex workflow
python cli.py chat --workflow-type complex
```

### **Demo Mode**
```bash
# Run demonstration
python cli.py demo
```

### **System Testing**
```bash
# Run system tests
python cli.py test --count 10

# Check system status
python cli.py status
```

### **Programmatic Usage**
```python
from travel_ai_system import TravelAISystem

# Initialize system
system = TravelAISystem()

# Process user request
response = system.process_user_request(
    user_request="Plan a 7-day trip to Japan",
    user_id="user123",
    workflow_type="auto"
)

print(response["message"])
```

## 🔄 Workflow Types

### **Auto Workflow** (Default)
- System automatically determines optimal agent coordination
- Balances speed and thoroughness
- Best for most use cases

### **Simple Workflow**
- Sequential agent execution
- Faster response times
- Good for straightforward requests

### **Complex Workflow**
- Parallel agent execution
- Maximum thoroughness
- Best for complex travel planning

## 💬 Example Conversations

### **Trip Planning**
```
👤 User: I want to plan a romantic trip to Paris for 5 days
🤖 Assistant: I'd love to help you plan a romantic Parisian getaway! 

🎯 Our recommendation specialist is creating a personalized 5-day itinerary
🕷️ Our data specialist is finding the best romantic hotels and restaurants
💬 I'm here to guide you through the entire planning process

What's your approximate budget for this romantic escape?
```

### **Hotel Booking**
```
👤 User: Book the hotel you recommended
🤖 Assistant: Perfect! I'm connecting you with our booking specialist.

📋 Booking Details:
- Hotel: Le Romantic Paris (4-star)
- Dates: March 15-20, 2024
- Room: Deluxe Suite with Eiffel Tower view
- Total: $1,200 for 5 nights

Would you like to proceed with this booking?
```

### **Information Gathering**
```
👤 User: What's the weather like in Tokyo in April?
🤖 Assistant: 🕷️ Let me get you the latest weather information for Tokyo in April...

🌸 Tokyo in April is perfect for cherry blossom season!
- Temperature: 15-20°C (59-68°F)
- Rainfall: Moderate (perfect for hanami)
- Best time for: Cherry blossom viewing, outdoor activities
- What to pack: Light layers, umbrella, comfortable walking shoes

Would you like me to help plan activities around the cherry blossom season?
```

## 🔧 System Commands

### **In Chat Mode**
- `/status` - Show system status
- `/session` - Show current session info
- `/clear` - Clear current session
- `/agents` - Show agent information
- `/metrics` - Show performance metrics

### **CLI Commands**
- `python cli.py chat` - Interactive chat
- `python cli.py demo` - Run demonstration
- `python cli.py test` - Run system tests
- `python cli.py status` - Show system status

## 📊 Performance Metrics

The system tracks:
- **Response Times**: Average and per-request timing
- **Success Rates**: Request completion statistics
- **Agent Performance**: Individual agent metrics
- **User Sessions**: Active session monitoring
- **Workflow Efficiency**: Multi-agent coordination metrics

## 🏗️ Architecture Details

### **Agent Communication**
```python
# Agents communicate through structured messages
agent_communication.send_message(
    from_agent="ChatAgent",
    to_agent="RecommendationAgent", 
    message={"user_preferences": preferences}
)
```

### **Workflow Coordination**
```python
# Coordinator creates execution plans
coordination_plan = {
    "workflow_type": "parallel",
    "steps": [
        {"agent": "ScrapingAgent", "task": "gather_data"},
        {"agent": "RecommendationAgent", "task": "generate_suggestions"}
    ]
}
```

### **Memory Management**
```python
# Agents maintain context and memory
agent.add_to_memory("user_preferences", preferences)
context = agent.get_from_memory("conversation_history")
```

## 🔮 Advanced Features

### **User Profiling**
- Learns from conversation history
- Adapts recommendations over time
- Remembers preferences across sessions

### **Intelligent Routing**
- Automatic intent detection
- Optimal agent selection
- Dynamic workflow adjustment

### **Error Recovery**
- Graceful failure handling
- Alternative agent routing
- Fallback response generation

### **Scalability**
- Async processing support
- Agent load balancing
- Horizontal scaling ready

## 🛠️ Customization

### **Adding New Agents**
```python
class CustomAgent(BaseAgent):
    def create_prompt_template(self):
        # Define agent-specific prompts
        pass
    
    def process_request(self, request):
        # Implement agent logic
        pass
```

### **Custom Workflows**
```python
# Define custom coordination logic
def custom_workflow(coordinator, request):
    plan = {
        "steps": [
            {"agent": "CustomAgent", "task": "custom_task"}
        ]
    }
    return coordinator.execute_workflow(plan)
```

## 📈 Monitoring & Analytics

### **Real-time Metrics**
- Request volume and patterns
- Agent performance tracking
- Error rates and types
- Response time analysis

### **User Analytics**
- Session duration and engagement
- Popular request types
- User satisfaction indicators
- Conversion tracking

## 🔒 Security & Privacy

- **API Key Protection**: Secure environment variable handling
- **Data Privacy**: No persistent storage of sensitive data
- **Session Security**: Secure session management
- **Input Validation**: Comprehensive request sanitization

## 🚀 Next Steps

1. **Enhanced NLP**: Add spaCy/NLTK for better intent detection
2. **Real APIs**: Integrate with actual booking APIs
3. **Database**: Add persistent storage for user profiles
4. **Web Interface**: Build React/Streamlit frontend
5. **Voice Support**: Add speech-to-text capabilities
6. **Mobile App**: Develop mobile application
7. **Analytics Dashboard**: Build monitoring interface

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add new agents or enhance existing ones
4. Submit pull request with tests

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ using LangChain + Groq for the future of AI travel assistance**
