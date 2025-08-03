import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import styled from 'styled-components';
import ChatInterface from './components/ChatInterface';
import Dashboard from './components/Dashboard';
import SystemStatus from './components/SystemStatus';
import Navigation from './components/Navigation';
import { apiService } from './services/apiService';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  flex-direction: column;
`;

const MainContent = styled.div`
  flex: 1;
  display: flex;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  padding: 20px;
  gap: 20px;
  
  @media (max-width: 768px) {
    flex-direction: column;
    padding: 10px;
  }
`;

const ContentArea = styled.div`
  flex: 1;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 20px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
  overflow: hidden;
`;

const LoadingScreen = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  color: white;
  font-size: 1.5rem;
  font-weight: 600;
`;

const Spinner = styled.div`
  width: 40px;
  height: 40px;
  border: 4px solid rgba(255, 255, 255, 0.3);
  border-top: 4px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-right: 1rem;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ErrorMessage = styled.div`
  background: rgba(255, 255, 255, 0.95);
  color: #e74c3c;
  padding: 20px;
  border-radius: 10px;
  margin: 20px;
  text-align: center;
  font-weight: 500;
`;

function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentView, setCurrentView] = useState('chat');

  useEffect(() => {
    checkSystemStatus();
    // Check system status every 30 seconds
    const interval = setInterval(checkSystemStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkSystemStatus = async () => {
    try {
      const status = await apiService.getSystemStatus();
      setSystemStatus(status);
      setError(null);
    } catch (err) {
      console.error('System status check failed:', err);
      setError('Failed to connect to AI Travel Agent backend');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <AppContainer>
        <LoadingScreen>
          <Spinner />
          Initializing AI Travel Agent...
        </LoadingScreen>
      </AppContainer>
    );
  }

  if (error) {
    return (
      <AppContainer>
        <ErrorMessage>
          <h3>⚠️ Connection Error</h3>
          <p>{error}</p>
          <p>Please ensure the Django backend is running on port 8000</p>
          <button 
            onClick={() => window.location.reload()}
            style={{
              marginTop: '10px',
              padding: '10px 20px',
              background: '#3498db',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer'
            }}
          >
            Retry Connection
          </button>
        </ErrorMessage>
      </AppContainer>
    );
  }

  return (
    <Router>
      <AppContainer>
        <Navigation 
          currentView={currentView} 
          setCurrentView={setCurrentView}
          systemStatus={systemStatus}
        />
        
        <MainContent>
          <ContentArea>
            <Routes>
              <Route 
                path="/" 
                element={<Navigate to="/chat" replace />} 
              />
              <Route 
                path="/chat" 
                element={
                  <ChatInterface 
                    systemStatus={systemStatus}
                    onViewChange={setCurrentView}
                  />
                } 
              />
              <Route 
                path="/dashboard" 
                element={
                  <Dashboard 
                    systemStatus={systemStatus}
                    onViewChange={setCurrentView}
                  />
                } 
              />
              <Route 
                path="/status" 
                element={
                  <SystemStatus 
                    systemStatus={systemStatus}
                    onRefresh={checkSystemStatus}
                    onViewChange={setCurrentView}
                  />
                } 
              />
            </Routes>
          </ContentArea>
        </MainContent>
      </AppContainer>
    </Router>
  );
}

export default App;
