import React, { useState } from 'react';
import styled from 'styled-components';
import { 
  FiSettings, FiRefreshCw, FiCheck, FiX, FiAlertCircle, 
  FiServer, FiDatabase, FiCpu, FiWifi 
} from 'react-icons/fi';

const StatusContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 20px;
  padding: 20px 0;
`;

const StatusCard = styled.div`
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 25px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
  }
`;

const CardHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 25px;
  
  h3 {
    margin: 0;
    color: #333;
    font-size: 1.3rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 12px;
  }
`;

const RefreshButton = styled.button`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 10px;
  padding: 10px 15px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const StatusItem = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 15px 20px;
  margin-bottom: 12px;
  background: #f8f9fa;
  border-radius: 12px;
  border-left: 4px solid ${props => {
    switch(props.status) {
      case 'operational': return '#10b981';
      case 'warning': return '#f59e0b';
      case 'error': return '#ef4444';
      default: return '#6b7280';
    }
  }};
  
  .item-info {
    display: flex;
    align-items: center;
    gap: 12px;
    
    .item-icon {
      width: 35px;
      height: 35px;
      border-radius: 8px;
      background: ${props => {
        switch(props.status) {
          case 'operational': return 'rgba(16, 185, 129, 0.1)';
          case 'warning': return 'rgba(245, 158, 11, 0.1)';
          case 'error': return 'rgba(239, 68, 68, 0.1)';
          default: return 'rgba(107, 114, 128, 0.1)';
        }
      }};
      color: ${props => {
        switch(props.status) {
          case 'operational': return '#10b981';
          case 'warning': return '#f59e0b';
          case 'error': return '#ef4444';
          default: return '#6b7280';
        }
      }};
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.1rem;
    }
    
    .item-details {
      .item-name {
        font-weight: 600;
        color: #333;
        margin-bottom: 4px;
      }
      
      .item-description {
        font-size: 0.85rem;
        color: #666;
      }
    }
  }
  
  .item-status {
    display: flex;
    align-items: center;
    gap: 8px;
    
    .status-badge {
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 500;
      background: ${props => {
        switch(props.status) {
          case 'operational': return '#dcfce7';
          case 'warning': return '#fef3c7';
          case 'error': return '#fee2e2';
          default: return '#f3f4f6';
        }
      }};
      color: ${props => {
        switch(props.status) {
          case 'operational': return '#166534';
          case 'warning': return '#92400e';
          case 'error': return '#991b1b';
          default: return '#374151';
        }
      }};
    }
  }
`;

const SystemOverview = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 15px;
  padding: 25px;
  margin-bottom: 25px;
  
  .overview-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    
    h4 {
      margin: 0;
      font-size: 1.2rem;
      font-weight: 600;
    }
    
    .overall-status {
      padding: 8px 16px;
      background: rgba(255, 255, 255, 0.2);
      border-radius: 20px;
      font-size: 0.9rem;
      font-weight: 500;
    }
  }
  
  .overview-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 20px;
    
    .stat-item {
      text-align: center;
      
      .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 5px;
      }
      
      .stat-label {
        font-size: 0.85rem;
        opacity: 0.9;
      }
    }
  }
`;

const DetailSection = styled.div`
  margin-top: 25px;
  
  h4 {
    color: #333;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
`;

function SystemStatus({ systemStatus, onRefresh }) {
  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await onRefresh();
    } finally {
      setTimeout(() => setRefreshing(false), 1000);
    }
  };

  const getStatusIcon = (status) => {
    switch(status) {
      case 'operational': return <FiCheck />;
      case 'warning': return <FiAlertCircle />;
      case 'error': return <FiX />;
      default: return <FiAlertCircle />;
    }
  };

  const getOverallHealthStatus = () => {
    const health = systemStatus?.overall_health;
    switch(health) {
      case 'excellent': return 'All Systems Operational';
      case 'good': return 'Systems Running Well';
      case 'degraded': return 'Some Issues Detected';
      default: return 'Status Unknown';
    }
  };

  const systemComponents = [
    {
      name: 'AI Agent System',
      description: 'Core travel AI processing engine',
      status: systemStatus?.ai_agent_status === 'operational' ? 'operational' : 'warning',
      icon: <FiCpu />
    },
    {
      name: 'OpenAI Integration',
      description: 'External AI model connectivity',
      status: systemStatus?.features?.openai_integration ? 'operational' : 'warning',
      icon: <FiWifi />
    },
    {
      name: 'Knowledge Base',
      description: 'Travel information database',
      status: systemStatus?.features?.knowledge_base ? 'operational' : 'error',
      icon: <FiDatabase />
    },
    {
      name: 'Streaming Responses',
      description: 'Real-time response delivery',
      status: systemStatus?.features?.streaming_responses ? 'operational' : 'warning',
      icon: <FiServer />
    }
  ];

  return (
    <StatusContainer>
      <StatusCard>
        <CardHeader>
          <h3>
            <FiSettings />
            System Overview
          </h3>
          <RefreshButton onClick={handleRefresh} disabled={refreshing}>
            <FiRefreshCw className={refreshing ? 'pulse' : ''} />
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </RefreshButton>
        </CardHeader>

        <SystemOverview>
          <div className="overview-header">
            <h4>Overall System Health</h4>
            <div className="overall-status">
              {getOverallHealthStatus()}
            </div>
          </div>
          
          <div className="overview-stats">
            <div className="stat-item">
              <div className="stat-value">{systemStatus?.statistics?.destinations_in_db || 0}</div>
              <div className="stat-label">Destinations</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">{systemStatus?.statistics?.travel_types_supported || 0}</div>
              <div className="stat-label">Travel Types</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">{systemStatus?.statistics?.total_conversations || 0}</div>
              <div className="stat-label">Conversations</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">{systemStatus?.statistics?.total_messages || 0}</div>
              <div className="stat-label">Messages</div>
            </div>
          </div>
        </SystemOverview>

        <DetailSection>
          <h4>
            <FiServer />
            System Components
          </h4>
          
          {systemComponents.map((component, index) => (
            <StatusItem key={index} status={component.status}>
              <div className="item-info">
                <div className="item-icon">
                  {component.icon}
                </div>
                <div className="item-details">
                  <div className="item-name">{component.name}</div>
                  <div className="item-description">{component.description}</div>
                </div>
              </div>
              
              <div className="item-status">
                <div className="status-badge">
                  {component.status === 'operational' ? 'Operational' : 
                   component.status === 'warning' ? 'Warning' : 'Error'}
                </div>
                {getStatusIcon(component.status)}
              </div>
            </StatusItem>
          ))}
        </DetailSection>
      </StatusCard>

      <StatusCard>
        <CardHeader>
          <h3>
            <FiDatabase />
            System Configuration
          </h3>
        </CardHeader>

        <DetailSection>
          <h4>AI Configuration</h4>
          
          <div style={{ background: '#f8f9fa', padding: '20px', borderRadius: '12px' }}>
            <div style={{ display: 'grid', gap: '15px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#666', fontWeight: '500' }}>AI Model:</span>
                <span style={{ color: '#333', fontWeight: '600' }}>
                  {systemStatus?.ai_model || 'Not configured'}
                </span>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#666', fontWeight: '500' }}>OpenAI Integration:</span>
                <span style={{ 
                  color: systemStatus?.features?.openai_integration ? '#10b981' : '#ef4444',
                  fontWeight: '600'
                }}>
                  {systemStatus?.features?.openai_integration ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#666', fontWeight: '500' }}>Knowledge Base:</span>
                <span style={{ 
                  color: systemStatus?.features?.knowledge_base ? '#10b981' : '#ef4444',
                  fontWeight: '600'
                }}>
                  {systemStatus?.features?.knowledge_base ? 'Active' : 'Inactive'}
                </span>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#666', fontWeight: '500' }}>Streaming:</span>
                <span style={{ 
                  color: systemStatus?.features?.streaming_responses ? '#10b981' : '#ef4444',
                  fontWeight: '600'
                }}>
                  {systemStatus?.features?.streaming_responses ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            </div>
          </div>
        </DetailSection>

        <DetailSection>
          <h4>Performance Metrics</h4>
          
          <div style={{ background: '#f8f9fa', padding: '20px', borderRadius: '12px' }}>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ color: '#666' }}>System Uptime</span>
                <span style={{ fontWeight: '600', color: '#10b981' }}>99.9%</span>
              </div>
              <div style={{ 
                height: '6px', 
                background: '#e5e7eb', 
                borderRadius: '3px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  width: '99.9%', 
                  height: '100%', 
                  background: 'linear-gradient(90deg, #10b981, #059669)',
                  borderRadius: '3px'
                }}></div>
              </div>
            </div>
            
            <div style={{ marginBottom: '15px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ color: '#666' }}>Response Quality</span>
                <span style={{ fontWeight: '600', color: '#667eea' }}>95%</span>
              </div>
              <div style={{ 
                height: '6px', 
                background: '#e5e7eb', 
                borderRadius: '3px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  width: '95%', 
                  height: '100%', 
                  background: 'linear-gradient(90deg, #667eea, #764ba2)',
                  borderRadius: '3px'
                }}></div>
              </div>
            </div>
            
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ color: '#666' }}>User Satisfaction</span>
                <span style={{ fontWeight: '600', color: '#f59e0b' }}>4.8/5</span>
              </div>
              <div style={{ 
                height: '6px', 
                background: '#e5e7eb', 
                borderRadius: '3px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  width: '96%', 
                  height: '100%', 
                  background: 'linear-gradient(90deg, #f59e0b, #d97706)',
                  borderRadius: '3px'
                }}></div>
              </div>
            </div>
          </div>
        </DetailSection>
      </StatusCard>
    </StatusContainer>
  );
}

export default SystemStatus;
