import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000/api';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`Making ${config.method.toUpperCase()} request to ${config.url}`);
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        return response.data;
      },
      (error) => {
        console.error('API Error:', error);
        if (error.code === 'ECONNREFUSED' || error.message === 'Network Error') {
          return Promise.reject(new Error('Unable to connect to server. Please make sure the backend is running on http://127.0.0.1:8000'));
        }
        return Promise.reject(error.response?.data || error);
      }
    );
  }

  // Chat with AI Agent
  async chatWithAI(message, preferences = {}, sessionId = null) {
    try {
      const requestData = {
        message: message,
        preferences: preferences || {},
      };

      if (sessionId) {
        requestData.session_id = sessionId;
      }

      const response = await this.client.post('/chat/', requestData);
      return response;
    } catch (error) {
      console.error('Chat API Error:', error);
      throw new Error(error.detail || error.error || error.message || 'Failed to get AI response');
    }
  }

  // Stream chat response
  async streamChatWithAI(message, preferences = {}, sessionId = null, onChunk, onComplete, onError) {
    try {
      const requestData = {
        message: message,
        preferences: preferences || {},
      };

      if (sessionId) {
        requestData.session_id = sessionId;
      }

      const response = await fetch(`${API_BASE_URL}/chat/stream/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'chunk') {
                onChunk(data.content);
              } else if (data.type === 'complete') {
                onComplete(data);
              } else if (data.type === 'error') {
                onError(new Error(data.message));
              } else if (data.type === 'status') {
                // Handle status updates if needed
                console.log('Status:', data.message);
              }
            } catch (e) {
              console.warn('Failed to parse SSE data:', line);
            }
          }
        }
      }
    } catch (error) {
      onError(error);
    }
  }

  // Get system status
  async getSystemStatus() {
    try {
      const response = await this.client.get('/status/');
      return response;
    } catch (error) {
      throw new Error(error.detail || error.error || 'Failed to get system status');
    }
  }

  // Get conversations
  async getConversations(sessionId = null) {
    try {
      const params = sessionId ? { session_id: sessionId } : {};
      const response = await this.client.get('/conversations/', { params });
      return response;
    } catch (error) {
      throw new Error(error.detail || error.error || 'Failed to get conversations');
    }
  }

  // Rate recommendation
  async rateRecommendation(recommendationId, rating, feedback = '') {
    try {
      const response = await this.client.post(`/recommendations/${recommendationId}/rate/`, {
        rating,
        feedback,
      });
      return response;
    } catch (error) {
      throw new Error(error.detail || error.error || 'Failed to rate recommendation');
    }
  }
}

export const apiService = new ApiService();
