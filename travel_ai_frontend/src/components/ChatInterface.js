import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { FiSend, FiUser, FiBot, FiSettings, FiClock, FiUsers, FiZap } from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';
import { apiService } from '../services/apiService';

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-height: 800px;
  background: white;
`;

const ChatHeader = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const HeaderInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 15px;
`;

const SystemBadge = styled.div`
  background: rgba(255, 255, 255, 0.2);
  padding: 5px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  display: flex;
  align-items: center;
  gap: 5px;
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 15px;
`;

const Message = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 12px;
  ${props => props.isUser && 'flex-direction: row-reverse;'}
`;

const MessageAvatar = styled.div`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 1.2rem;
  ${props => props.isUser 
    ? 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'
    : 'background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'
  }
`;

const MessageContent = styled.div`
  flex: 1;
  max-width: 70%;
`;

const MessageBubble = styled.div`
  padding: 15px 20px;
  border-radius: 20px;
  ${props => props.isUser 
    ? `
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      margin-left: auto;
    `
    : `
      background: #f8f9fa;
      color: #333;
      border: 1px solid #e9ecef;
    `
  }
  
  p {
    margin: 0 0 10px 0;
    &:last-child { margin-bottom: 0; }
  }
  
  ul, ol {
    margin: 10px 0;
    padding-left: 20px;
  }
  
  strong {
    font-weight: 600;
  }
`;

const MessageMeta = styled.div`
  font-size: 0.75rem;
  color: #666;
  margin-top: 5px;
  display: flex;
  align-items: center;
  gap: 10px;
  ${props => props.isUser && 'justify-content: flex-end;'}
`;

const SystemInfo = styled.div`
  background: rgba(102, 126, 234, 0.1);
  border: 1px solid rgba(102, 126, 234, 0.2);
  border-radius: 10px;
  padding: 10px;
  margin-top: 10px;
  font-size: 0.8rem;
`;

const InputContainer = styled.div`
  padding: 20px;
  border-top: 1px solid #e9ecef;
  background: #f8f9fa;
`;

const InputForm = styled.form`
  display: flex;
  gap: 10px;
  align-items: flex-end;
`;

const MessageInput = styled.textarea`
  flex: 1;
  padding: 15px;
  border: 2px solid #e9ecef;
  border-radius: 15px;
  resize: none;
  font-family: inherit;
  font-size: 1rem;
  min-height: 50px;
  max-height: 120px;
  
  &:focus {
    outline: none;
    border-color: #667eea;
  }
  
  &::placeholder {
    color: #999;
  }
`;

const SendButton = styled.button`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 15px;
  padding: 15px 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  transition: all 0.2s ease;
  
  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const TypingIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  color: #666;
  font-style: italic;
  padding: 10px 0;
`;

const TypingDots = styled.div`
  display: flex;
  gap: 3px;
  
  span {
    width: 6px;
    height: 6px;
    background: #667eea;
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
    
    &:nth-child(1) { animation-delay: -0.32s; }
    &:nth-child(2) { animation-delay: -0.16s; }
  }
  
  @keyframes typing {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
  }
`;

const WelcomeMessage = styled.div`
  text-align: center;
  padding: 40px 20px;
  color: #666;
  
  h2 {
    color: #333;
    margin-bottom: 10px;
  }
  
  p {
    margin-bottom: 20px;
  }
`;

const QuickActions = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
  margin-top: 20px;
`;

const QuickActionButton = styled.button`
  background: white;
  border: 2px solid #667eea;
  color: #667eea;
  padding: 10px 15px;
  border-radius: 20px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.2s ease;
  
  &:hover {
    background: #667eea;
    color: white;
  }
`;

function ChatInterface({ systemStatus, onViewChange }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (onViewChange) {
      onViewChange('chat');
    }
  }, [onViewChange]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    setIsLoading(true);

    // Add user message to chat
    const newUserMessage = {
      id: Date.now(),
      text: userMessage,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, newUserMessage]);

    try {
      // Call AI agent API
      const response = await apiService.chatWithAI(
        userMessage,
        {}, // preferences - could be expanded
        conversationId
      );

      if (!conversationId) {
        setConversationId(response.conversation_id);
      }

      // Add AI response to chat
      const aiMessage = {
        id: Date.now() + 1,
        text: response.response,
        isUser: false,
        timestamp: new Date(),
        systemUsed: response.system_used,
        agentsInvolved: response.agents_involved,
        workflowType: response.workflow_type,
        featuresUsed: response.features_used,
        responseTime: response.response_time,
        sources: response.sources,
        note: response.note,
      };

      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      console.error('Chat error:', error);
      
      // Add error message
      const errorMessage = {
        id: Date.now() + 1,
        text: `I apologize, but I'm experiencing some technical difficulties. ${error.message || 'Please try again in a moment.'}`,
        isUser: false,
        timestamp: new Date(),
        isError: true,
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickAction = (message) => {
    setInputMessage(message);
  };

  const quickActions = [
    "Plan a romantic trip to Paris for 5 days",
    "Find budget-friendly destinations in Asia",
    "Suggest luxury resorts in the Maldives",
    "Plan a cultural tour of Japan",
    "Recommend family-friendly destinations in Europe"
  ];

  return (
    <ChatContainer>
      <ChatHeader>
        <HeaderInfo>
          <FiBot size={24} />
          <div>
            <h2>AI Travel Agent</h2>
            <p style={{ margin: 0, opacity: 0.9, fontSize: '0.9rem' }}>
              Powered by Advanced CrewAI & Enhanced RAG
            </p>
          </div>
        </HeaderInfo>
        
        {systemStatus && (
          <SystemBadge>
            <FiZap size={14} />
            {systemStatus.system_status?.overall_health || 'operational'}
          </SystemBadge>
        )}
      </ChatHeader>

      <MessagesContainer>
        {messages.length === 0 ? (
          <WelcomeMessage>
            <h2>🌍 Welcome to AI Travel Agent</h2>
            <p>I'm your intelligent travel planning companion, powered by advanced AI agents and real-time data.</p>
            <p>Ask me anything about travel planning, destinations, or get personalized recommendations!</p>
            
            <QuickActions>
              {quickActions.map((action, index) => (
                <QuickActionButton
                  key={index}
                  onClick={() => handleQuickAction(action)}
                >
                  {action}
                </QuickActionButton>
              ))}
            </QuickActions>
          </WelcomeMessage>
        ) : (
          messages.map((message) => (
            <Message key={message.id} isUser={message.isUser}>
              <MessageAvatar isUser={message.isUser}>
                {message.isUser ? <FiUser /> : <FiBot />}
              </MessageAvatar>
              
              <MessageContent>
                <MessageBubble isUser={message.isUser} isError={message.isError}>
                  <ReactMarkdown>{message.text}</ReactMarkdown>
                </MessageBubble>
                
                <MessageMeta isUser={message.isUser}>
                  <span>
                    <FiClock size={12} />
                    {message.timestamp.toLocaleTimeString()}
                  </span>
                  
                  {message.responseTime && (
                    <span>{message.responseTime.toFixed(2)}s</span>
                  )}
                </MessageMeta>
                
                {!message.isUser && !message.isError && (
                  <SystemInfo>
                    <div style={{ marginBottom: '5px' }}>
                      <strong>System:</strong> {message.systemUsed}
                    </div>
                    
                    {message.agentsInvolved && (
                      <div style={{ marginBottom: '5px' }}>
                        <FiUsers size={12} style={{ marginRight: '5px' }} />
                        <strong>Agents:</strong> {message.agentsInvolved.join(', ')}
                      </div>
                    )}
                    
                    {message.workflowType && (
                      <div style={{ marginBottom: '5px' }}>
                        <strong>Workflow:</strong> {message.workflowType}
                      </div>
                    )}
                    
                    {message.featuresUsed && (
                      <div style={{ marginBottom: '5px' }}>
                        <strong>Features:</strong> {message.featuresUsed.length} advanced features
                      </div>
                    )}
                    
                    {message.note && (
                      <div style={{ fontStyle: 'italic', color: '#666' }}>
                        {message.note}
                      </div>
                    )}
                  </SystemInfo>
                )}
              </MessageContent>
            </Message>
          ))
        )}
        
        {isLoading && (
          <TypingIndicator>
            <FiBot />
            <span>AI Agent is thinking</span>
            <TypingDots>
              <span></span>
              <span></span>
              <span></span>
            </TypingDots>
          </TypingIndicator>
        )}
        
        <div ref={messagesEndRef} />
      </MessagesContainer>

      <InputContainer>
        <InputForm onSubmit={handleSendMessage}>
          <MessageInput
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask me about travel planning, destinations, or get personalized recommendations..."
            disabled={isLoading}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage(e);
              }
            }}
          />
          <SendButton type="submit" disabled={isLoading || !inputMessage.trim()}>
            <FiSend />
          </SendButton>
        </InputForm>
      </InputContainer>
    </ChatContainer>
  );
}

export default ChatInterface;
