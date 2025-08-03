import axios from 'axios';

// Configure axios defaults
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('API Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      if (status === 401) {
        // Unauthorized - clear token and redirect to login
        localStorage.removeItem('authToken');
        // window.location.href = '/login';
      }
      
      throw new Error(data.error || data.message || `HTTP ${status} Error`);
    } else if (error.request) {
      // Network error
      throw new Error('Network error - please check your connection');
    } else {
      // Other error
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
);

export const apiService = {
  // Chat with AI Agent
  async chatWithAI(message, preferences = {}, conversationId = null) {
    try {
      const response = await api.post('/chat/ai-agent/', {
        message,
        preferences,
        conversation_id: conversationId,
      });
      return response;
    } catch (error) {
      console.error('Chat API error:', error);
      throw error;
    }
  },

  // Get dashboard data
  async getDashboardData() {
    try {
      const response = await api.get('/dashboard/');
      return response;
    } catch (error) {
      console.error('Dashboard API error:', error);
      throw error;
    }
  },

  // Get system status
  async getSystemStatus() {
    try {
      const response = await api.get('/status/advanced/');
      return response;
    } catch (error) {
      console.error('System status API error:', error);
      throw error;
    }
  },

  // Analyze user preferences
  async analyzePreferences(userData) {
    try {
      const response = await api.post('/preferences/analyze/', userData);
      return response;
    } catch (error) {
      console.error('Preferences API error:', error);
      throw error;
    }
  },

  // Get agent status (legacy)
  async getAgentStatus() {
    try {
      const response = await api.get('/agents/status/');
      return response;
    } catch (error) {
      console.error('Agent status API error:', error);
      throw error;
    }
  },

  // Submit feedback
  async submitFeedback(feedbackData) {
    try {
      const response = await api.post('/feedback/', feedbackData);
      return response;
    } catch (error) {
      console.error('Feedback API error:', error);
      throw error;
    }
  },

  // Health check
  async healthCheck() {
    try {
      const response = await api.get('/health/');
      return response;
    } catch (error) {
      console.error('Health check API error:', error);
      throw error;
    }
  },

  // Get real-time data
  async getRealTimeData(destination, startDate, endDate) {
    try {
      const response = await api.get('/realtime-data/', {
        params: {
          destination,
          start_date: startDate,
          end_date: endDate,
        },
      });
      return response;
    } catch (error) {
      console.error('Real-time data API error:', error);
      throw error;
    }
  },

  // Get user profile
  async getUserProfile() {
    try {
      const response = await api.get('/profile/');
      return response;
    } catch (error) {
      console.error('Profile API error:', error);
      throw error;
    }
  },

  // Update user profile
  async updateUserProfile(profileData) {
    try {
      const response = await api.put('/profile/', profileData);
      return response;
    } catch (error) {
      console.error('Profile update API error:', error);
      throw error;
    }
  },

  // Get conversations
  async getConversations() {
    try {
      const response = await api.get('/conversations/');
      return response;
    } catch (error) {
      console.error('Conversations API error:', error);
      throw error;
    }
  },

  // Get specific conversation
  async getConversation(conversationId) {
    try {
      const response = await api.get(`/conversations/${conversationId}/`);
      return response;
    } catch (error) {
      console.error('Conversation API error:', error);
      throw error;
    }
  },

  // Rate recommendation
  async rateRecommendation(recommendationId, rating, feedback = '') {
    try {
      const response = await api.post(`/recommendations/${recommendationId}/rate/`, {
        rating,
        feedback,
      });
      return response;
    } catch (error) {
      console.error('Rating API error:', error);
      throw error;
    }
  },
};

// Utility functions
export const formatApiError = (error) => {
  if (typeof error === 'string') {
    return error;
  }
  
  if (error.response && error.response.data) {
    return error.response.data.error || error.response.data.message || 'An error occurred';
  }
  
  return error.message || 'An unexpected error occurred';
};

export const isNetworkError = (error) => {
  return !error.response && error.request;
};

export const isServerError = (error) => {
  return error.response && error.response.status >= 500;
};

export const isClientError = (error) => {
  return error.response && error.response.status >= 400 && error.response.status < 500;
};

export default apiService;
