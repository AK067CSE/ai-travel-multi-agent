# Production-Level AI Travel Agent

A sophisticated, enterprise-ready AI travel planning system with advanced RAG, multi-agent architecture, and production-level features.

## 🌟 Advanced Features

### 🧠 **Enhanced RAG System**
- **BGE Embeddings**: State-of-the-art `BAAI/bge-base-en-v1.5` semantic search
- **Hybrid Retrieval**: Dense (semantic) + Sparse (BM25) search combination
- **Re-ranking**: Cross-encoder style reranking for precision
- **Query Expansion**: LLM-powered query enhancement
- **Conversational Memory**: Context-aware responses with history

### 👥 **Multi-Agent Architecture**
- **Coordinator Agent**: Orchestrates complex workflows
- **Research Agent**: Enhanced RAG-powered information gathering
- **Planning Agent**: Detailed itinerary and travel plan creation
- **Chat Agent**: Natural conversation handling
- **Intelligent Routing**: Automatic agent selection based on query complexity

### 🤖 **Multiple LLM Providers**
- **Primary**: Groq (Llama3-70B) for fast, high-quality responses
- **Fallback**: OpenAI GPT-3.5/4 for reliability
- **Alternative**: Anthropic Claude for specialized tasks
- **Automatic Failover**: Seamless switching between providers

### 📊 **Production Features**
- **Real-time Streaming**: Dynamic response delivery with typing indicators
- **Performance Monitoring**: Comprehensive metrics and health tracking
- **Error Handling**: Graceful degradation and fallback systems
- **Scalable Architecture**: Designed for production workloads
- **Modern Dashboard**: Beautiful card-format interface with system monitoring

## 🏗️ Architecture

### Backend (Django)
- **Simplified Structure**: Clean Django REST API
- **AI Agent System**: Streamlined travel AI with fallback support
- **Streaming Support**: Server-sent events for real-time responses
- **Database Models**: Conversations, messages, and recommendations

### Frontend (React)
- **Modern UI**: Styled-components with glassmorphism design
- **Real-time Chat**: Streaming chat interface with typing indicators
- **Dashboard**: System status and analytics cards
- **Responsive Design**: Mobile-friendly interface

## 🚀 Quick Start

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd simplified_travel_ai/backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (create `.env` file):
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DJANGO_SECRET_KEY=your_secret_key_here
   DEBUG=True
   ```

4. **Run migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create superuser** (optional):
   ```bash
   python manage.py createsuperuser
   ```

6. **Start the server**:
   ```bash
   python manage.py runserver
   ```

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd simplified_travel_ai/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

4. **Open your browser** to `http://localhost:3000`

## 📡 API Endpoints

### Chat Endpoints
- `POST /api/chat/` - Send message to AI agent
- `POST /api/chat/stream/` - Stream chat responses (SSE)

### System Endpoints
- `GET /api/status/` - Get system status and health
- `GET /api/conversations/` - Get conversation history

### Recommendation Endpoints
- `POST /api/recommendations/{id}/rate/` - Rate a recommendation

## 🎨 UI Components

### Chat Interface
- Real-time streaming responses
- Message history with metadata
- Quick action buttons
- Typing indicators

### Dashboard
- System health monitoring
- Performance metrics
- Feature status cards
- Conversation statistics

### System Status
- Component health checks
- Configuration overview
- Performance graphs
- Real-time updates

## 🔧 Configuration

### AI Configuration
The system supports multiple AI providers with fallback:

1. **OpenAI** (Primary): Set `OPENAI_API_KEY` in environment
2. **Fallback System**: Built-in knowledge base responses

### Customization
- Modify `ai_agent.py` to add new travel knowledge
- Update `travel_knowledge` dictionary for destinations
- Customize UI themes in styled-components

## 📊 System Features

### Advanced Capabilities
- ✅ Intelligent travel planning
- ✅ Real-time response streaming
- ✅ Conversation memory
- ✅ System health monitoring
- ✅ Performance analytics
- ✅ Mobile-responsive design

### Simplified Architecture
- ❌ Removed complex multi-agent systems
- ❌ Eliminated redundant RAG implementations
- ❌ Streamlined database models
- ❌ Simplified API structure

## 🛠️ Development

### Adding New Features
1. Backend: Add new views in `api/views.py`
2. Frontend: Create components in `src/components/`
3. API: Update `apiService.js` for new endpoints

### Customizing AI Responses
Edit the `travel_knowledge` dictionary in `ai_agent.py`:

```python
"destinations": {
    "your_destination": {
        "country": "Country Name",
        "best_time": "Best time to visit",
        "highlights": ["Attraction 1", "Attraction 2"],
        "budget": "Budget range",
        "culture": "Cultural highlights"
    }
}
```

## 🔍 Troubleshooting

### Common Issues

1. **OpenAI API Key**: Ensure your API key is valid and has credits
2. **CORS Issues**: Check Django CORS settings in `settings.py`
3. **Streaming Problems**: Verify SSE support in your browser
4. **Database Issues**: Run migrations if models change

### Performance Tips

1. **Response Time**: Use OpenAI API for best results
2. **Memory Usage**: Clear old conversations periodically
3. **UI Performance**: Enable React production build for deployment

## 📝 License

This project is created for educational and demonstration purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Note**: This is a simplified version of the original complex travel AI system, designed for better maintainability and performance while retaining advanced features.
