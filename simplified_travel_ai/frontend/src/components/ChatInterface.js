import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { FiSend, FiUser, FiMessageCircle, FiClock, FiZap, FiRefreshCw } from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';
import { apiService } from '../services/apiService';

const ChatContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
  height: calc(100vh - 200px);
`;

const ChatCard = styled.div`
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const ChatHeader = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px 25px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const HeaderInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 15px;
  
  h3 {
    margin: 0;
    font-size: 1.3rem;
    font-weight: 600;
  }
  
  p {
    margin: 0;
    opacity: 0.9;
    font-size: 0.9rem;
  }
`;

const SystemBadge = styled.div`
  background: rgba(255, 255, 255, 0.2);
  padding: 8px 15px;
  border-radius: 20px;
  font-size: 0.8rem;
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 25px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  background: linear-gradient(to bottom, #f8f9fa, #ffffff);
`;

const Message = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 15px;
  ${props => props.isUser && 'flex-direction: row-reverse;'}
  animation: fadeIn 0.5s ease-out;
`;

const MessageAvatar = styled.div`
  width: 45px;
  height: 45px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 1.3rem;
  flex-shrink: 0;
  ${props => props.isUser 
    ? 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'
    : 'background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'
  }
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
`;

const MessageContent = styled.div`
  flex: 1;
  max-width: 75%;
`;

const MessageBubble = styled.div`
  padding: 18px 22px;
  border-radius: 20px;
  ${props => props.isUser 
    ? `
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      margin-left: auto;
      box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    `
    : `
      background: white;
      color: #333;
      border: 1px solid #e9ecef;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    `
  }
  
  p {
    margin: 0 0 12px 0;
    line-height: 1.6;
    &:last-child { margin-bottom: 0; }
  }
  
  ul, ol {
    margin: 12px 0;
    padding-left: 20px;
  }
  
  li {
    margin-bottom: 5px;
  }
  
  strong {
    font-weight: 600;
  }
  
  h1, h2, h3, h4, h5, h6 {
    margin: 15px 0 10px 0;
    &:first-child { margin-top: 0; }
  }
`;

const MessageMeta = styled.div`
  font-size: 0.75rem;
  color: #666;
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 12px;
  ${props => props.isUser && 'justify-content: flex-end;'}
`;

const SystemInfo = styled.div`
  background: rgba(102, 126, 234, 0.1);
  border: 1px solid rgba(102, 126, 234, 0.2);
  border-radius: 12px;
  padding: 12px 15px;
  margin-top: 12px;
  font-size: 0.8rem;
  
  .info-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    
    &:last-child {
      margin-bottom: 0;
    }
  }
  
  .label {
    font-weight: 600;
    color: #667eea;
  }
`;

const InputContainer = styled.div`
  padding: 25px;
  border-top: 1px solid #e9ecef;
  background: #f8f9fa;
`;

const InputForm = styled.form`
  display: flex;
  gap: 15px;
  align-items: flex-end;
`;

const MessageInput = styled.textarea`
  flex: 1;
  padding: 18px 20px;
  border: 2px solid #e9ecef;
  border-radius: 20px;
  resize: none;
  font-family: inherit;
  font-size: 1rem;
  min-height: 55px;
  max-height: 120px;
  transition: all 0.3s ease;
  
  &:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }
  
  &::placeholder {
    color: #999;
  }
`;

const SendButton = styled.button`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 20px;
  padding: 18px 25px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
  
  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const TypingIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  color: #666;
  font-style: italic;
  padding: 15px 0;
  animation: fadeIn 0.3s ease-out;
`;

const TypingDots = styled.div`
  display: flex;
  gap: 4px;
  
  span {
    width: 8px;
    height: 8px;
    background: #667eea;
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
    
    &:nth-child(1) { animation-delay: -0.32s; }
    &:nth-child(2) { animation-delay: -0.16s; }
  }
  
  @keyframes typing {
    0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
    40% { transform: scale(1); opacity: 1; }
  }
`;

const WelcomeMessage = styled.div`
  text-align: center;
  padding: 50px 30px;
  color: #666;
  
  h2 {
    color: #333;
    margin-bottom: 15px;
    font-size: 1.8rem;
    font-weight: 600;
  }
  
  p {
    margin-bottom: 25px;
    font-size: 1.1rem;
    line-height: 1.6;
  }
`;

const QuickActions = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 15px;
  margin-top: 30px;
`;

const QuickActionCard = styled.button`
  background: white;
  border: 2px solid #667eea;
  color: #667eea;
  padding: 15px 20px;
  border-radius: 15px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.3s ease;
  text-align: left;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);

  &:hover {
    background: #667eea;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
  }
`;

function ChatInterface({ systemStatus }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage]);

  const handleSendMessage = async (e) => {
    e.preventDefault();

    if (!inputMessage.trim() || isLoading || isStreaming) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    setIsLoading(true);
    setIsStreaming(true);
    setStreamingMessage('');

    // Add user message to chat
    const newUserMessage = {
      id: Date.now(),
      text: userMessage,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, newUserMessage]);

    try {
      // Use regular API for now (streaming disabled due to CORS)
      const response = await apiService.chatWithAI(userMessage, {}, conversationId);

      if (!conversationId) {
        setConversationId(response.session_id);
      }

      const aiMessage = {
        id: Date.now() + 1,
        text: response.response,
        isUser: false,
        timestamp: new Date(),
        systemUsed: response.system_used,
        agentsInvolved: response.agents_involved,
        responseTime: response.response_time,
        conversationId: response.conversation_id,
      };

      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      console.error('Chat error:', error);

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
      setIsStreaming(false);
      setStreamingMessage('');
    }
  };

  const handleQuickAction = (message) => {
    setInputMessage(message);
  };

  const quickActions = [
    "Plan a romantic 5-day trip to Paris with a budget of $3000",
    "Find the best budget-friendly destinations in Southeast Asia",
    "Suggest luxury beach resorts in the Maldives for honeymoon",
    "Create a cultural itinerary for 10 days in Japan",
    "Recommend family-friendly destinations in Europe for summer",
    "Plan an adventure trip to New Zealand for 2 weeks"
  ];

  return (
    <ChatContainer>
      <ChatCard>
        <ChatHeader>
          <HeaderInfo>
            <FiMessageCircle size={24} />
            <div>
              <h3>AI Travel Assistant</h3>
              <p>Intelligent travel planning with real-time streaming</p>
            </div>
          </HeaderInfo>

          {systemStatus && (
            <SystemBadge>
              <FiZap size={14} />
              {systemStatus.overall_health || 'operational'}
            </SystemBadge>
          )}
        </ChatHeader>

        <MessagesContainer>
          {messages.length === 0 ? (
            <WelcomeMessage>
              <h2>🌍 Welcome to AI Travel Agent</h2>
              <p>I'm your intelligent travel planning companion with advanced AI capabilities.</p>
              <p>Ask me anything about destinations, create detailed itineraries, or get personalized recommendations!</p>

              <QuickActions>
                {quickActions.map((action, index) => (
                  <QuickActionCard
                    key={index}
                    onClick={() => handleQuickAction(action)}
                  >
                    {action}
                  </QuickActionCard>
                ))}
              </QuickActions>
            </WelcomeMessage>
          ) : (
            <>
              {messages.map((message) => (
                <Message key={message.id} isUser={message.isUser}>
                  <MessageAvatar isUser={message.isUser}>
                    {message.isUser ? <FiUser /> : <FiMessageCircle />}
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
                        <div className="info-row">
                          <span className="label">System:</span>
                          <span>{message.systemUsed}</span>
                        </div>

                        {message.agentsInvolved && (
                          <div className="info-row">
                            <span className="label">Agents:</span>
                            <span>{message.agentsInvolved.join(', ')}</span>
                          </div>
                        )}

                        {message.conversationId && (
                          <div className="info-row">
                            <span className="label">Session:</span>
                            <span>{message.conversationId.slice(-8)}</span>
                          </div>
                        )}
                      </SystemInfo>
                    )}
                  </MessageContent>
                </Message>
              ))}

              {/* Streaming message */}
              {isStreaming && streamingMessage && (
                <Message isUser={false}>
                  <MessageAvatar isUser={false}>
                    <FiMessageCircle />
                  </MessageAvatar>

                  <MessageContent>
                    <MessageBubble isUser={false}>
                      <ReactMarkdown>{streamingMessage}</ReactMarkdown>
                    </MessageBubble>

                    <MessageMeta isUser={false}>
                      <span>
                        <FiRefreshCw size={12} className="pulse" />
                        Streaming...
                      </span>
                    </MessageMeta>
                  </MessageContent>
                </Message>
              )}
            </>
          )}

          {(isLoading || isStreaming) && !streamingMessage && (
            <TypingIndicator>
              <FiMessageCircle />
              <span>AI Agent is processing your request</span>
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
              disabled={isLoading || isStreaming}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e);
                }
              }}
            />
            <SendButton type="submit" disabled={isLoading || isStreaming || !inputMessage.trim()}>
              <FiSend />
            </SendButton>
          </InputForm>
        </InputContainer>
      </ChatCard>
    </ChatContainer>
  );
}

export default ChatInterface;
