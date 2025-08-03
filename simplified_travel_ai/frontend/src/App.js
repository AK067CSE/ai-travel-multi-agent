import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { FiSettings, FiActivity, FiMessageCircle } from 'react-icons/fi';
import ChatInterface from './components/ChatInterface';
import Dashboard from './components/Dashboard';
import SystemStatus from './components/SystemStatus';
import { apiService } from './services/apiService';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
`;

const Header = styled.header`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 20px 30px;
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: white;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 15px;
  
  h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
  }
  
  p {
    margin: 0;
    opacity: 0.9;
    font-size: 0.9rem;
  }
`;

const Navigation = styled.nav`
  display: flex;
  gap: 15px;
`;

const NavButton = styled.button`
  background: ${props => props.active ? 'rgba(255, 255, 255, 0.2)' : 'transparent'};
  border: 2px solid rgba(255, 255, 255, 0.3);
  color: white;
  padding: 12px 20px;
  border-radius: 15px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
  }
`;

const MainContent = styled.main`
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
  max-width: 1400px;
  margin: 0 auto;
`;

const StatusBar = styled.div`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 15px 20px;
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.9rem;
  margin-bottom: 20px;
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: ${props => {
      switch(props.status) {
        case 'excellent': return '#4ade80';
        case 'good': return '#fbbf24';
        case 'degraded': return '#f87171';
        default: return '#6b7280';
      }
    }};
    animation: pulse 2s infinite;
  }
`;

function App() {
  const [currentView, setCurrentView] = useState('chat');
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSystemStatus();
    // Refresh status every 30 seconds
    const interval = setInterval(loadSystemStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadSystemStatus = async () => {
    try {
      const status = await apiService.getSystemStatus();
      setSystemStatus(status);
    } catch (error) {
      console.error('Failed to load system status:', error);
      setSystemStatus({
        overall_health: 'unknown',
        ai_agent_status: 'unknown'
      });
    } finally {
      setLoading(false);
    }
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case 'dashboard':
        return <Dashboard systemStatus={systemStatus} />;
      case 'status':
        return <SystemStatus systemStatus={systemStatus} onRefresh={loadSystemStatus} />;
      case 'chat':
      default:
        return <ChatInterface systemStatus={systemStatus} />;
    }
  };

  if (loading) {
    return (
      <AppContainer>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh',
          color: 'white',
          fontSize: '1.2rem'
        }}>
          <div className="pulse">🌍 Loading AI Travel Agent...</div>
        </div>
      </AppContainer>
    );
  }

  return (
    <AppContainer>
      <Header>
        <Logo>
          <div>
            <h1>🌍 AI Travel Agent</h1>
            <p>Your intelligent travel planning companion</p>
          </div>
        </Logo>
        
        <Navigation>
          <NavButton 
            active={currentView === 'chat'} 
            onClick={() => setCurrentView('chat')}
          >
            <FiMessageCircle />
            Chat
          </NavButton>
          <NavButton 
            active={currentView === 'dashboard'} 
            onClick={() => setCurrentView('dashboard')}
          >
            <FiActivity />
            Dashboard
          </NavButton>
          <NavButton 
            active={currentView === 'status'} 
            onClick={() => setCurrentView('status')}
          >
            <FiSettings />
            System
          </NavButton>
        </Navigation>
      </Header>

      <StatusBar>
        <StatusIndicator status={systemStatus?.overall_health}>
          <div className="status-dot"></div>
          <span>System Status: {systemStatus?.overall_health || 'Unknown'}</span>
        </StatusIndicator>
        
        <div>
          AI Model: {systemStatus?.ai_model || 'Unknown'} | 
          Features: {systemStatus?.features ? Object.keys(systemStatus.features).length : 0} active
        </div>
      </StatusBar>

      <MainContent>
        {renderCurrentView()}
      </MainContent>
    </AppContainer>
  );
}

export default App;
