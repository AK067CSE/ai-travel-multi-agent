import React, { useEffect } from 'react';
import styled from 'styled-components';
import { 
  FiCheck, FiX, FiAlertTriangle, FiRefreshCw, FiCpu, 
  FiDatabase, FiZap, FiUsers, FiBarChart3, FiGlobe 
} from 'react-icons/fi';

const StatusContainer = styled.div`
  padding: 20px;
  background: white;
  min-height: 100vh;
`;

const StatusHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 2px solid #f0f0f0;
`;

const HeaderTitle = styled.div`
  h1 {
    color: #333;
    margin: 0 0 5px 0;
    font-size: 2rem;
    font-weight: 700;
  }
  
  p {
    color: #666;
    margin: 0;
    font-size: 1rem;
  }
`;

const RefreshButton = styled.button`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 10px;
  padding: 10px 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  transition: all 0.2s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const OverallStatus = styled.div`
  background: ${props => {
    switch(props.status) {
      case 'excellent': return 'linear-gradient(135deg, #27ae60 0%, #2ecc71 100%)';
      case 'operational': return 'linear-gradient(135deg, #27ae60 0%, #2ecc71 100%)';
      case 'degraded': return 'linear-gradient(135deg, #f39c12 0%, #e67e22 100%)';
      case 'critical': return 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)';
      default: return 'linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%)';
    }
  }};
  color: white;
  padding: 30px;
  border-radius: 15px;
  text-align: center;
  margin-bottom: 30px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
`;

const StatusIcon = styled.div`
  font-size: 3rem;
  margin-bottom: 15px;
`;

const StatusTitle = styled.h2`
  margin: 0 0 10px 0;
  font-size: 1.8rem;
  font-weight: 600;
`;

const StatusDescription = styled.p`
  margin: 0;
  font-size: 1.1rem;
  opacity: 0.9;
`;

const SystemsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const SystemCard = styled.div`
  background: white;
  border-radius: 15px;
  padding: 25px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
  border: 1px solid #f0f0f0;
  transition: all 0.2s ease;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
  }
`;

const SystemHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
`;

const SystemInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 15px;
`;

const SystemIcon = styled.div`
  width: 50px;
  height: 50px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  color: white;
  background: ${props => props.gradient || 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'};
`;

const SystemName = styled.div`
  h3 {
    margin: 0 0 5px 0;
    color: #333;
    font-size: 1.2rem;
    font-weight: 600;
  }
  
  p {
    margin: 0;
    color: #666;
    font-size: 0.9rem;
  }
`;

const StatusBadge = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 15px;
  border-radius: 20px;
  font-size: 0.85rem;
  font-weight: 500;
  background: ${props => {
    switch(props.status) {
      case 'operational': return '#d4edda';
      case 'degraded': return '#fff3cd';
      case 'error': return '#f8d7da';
      default: return '#e2e3e5';
    }
  }};
  color: ${props => {
    switch(props.status) {
      case 'operational': return '#155724';
      case 'degraded': return '#856404';
      case 'error': return '#721c24';
      default: return '#383d41';
    }
  }};
`;

const SystemDetails = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 15px;
  margin-top: 15px;
`;

const DetailItem = styled.div`
  text-align: center;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 10px;
  
  .value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #333;
    margin-bottom: 5px;
  }
  
  .label {
    font-size: 0.8rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
`;

const LastUpdated = styled.div`
  text-align: center;
  color: #666;
  font-size: 0.9rem;
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #f0f0f0;
`;

function SystemStatus({ systemStatus, onRefresh, onViewChange }) {
  useEffect(() => {
    if (onViewChange) {
      onViewChange('status');
    }
  }, [onViewChange]);

  if (!systemStatus) {
    return (
      <StatusContainer>
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <FiRefreshCw className="spinning" size={40} color="#667eea" />
          <p style={{ marginTop: '20px', color: '#666' }}>Loading system status...</p>
        </div>
      </StatusContainer>
    );
  }

  const getOverallStatusInfo = () => {
    const health = systemStatus.system_status?.overall_health;
    switch(health) {
      case 'excellent':
        return {
          icon: <FiCheck />,
          title: 'All Systems Excellent',
          description: 'All AI systems are running optimally with peak performance'
        };
      case 'operational':
        return {
          icon: <FiCheck />,
          title: 'Systems Operational',
          description: 'All core systems are functioning normally'
        };
      case 'degraded':
        return {
          icon: <FiAlertTriangle />,
          title: 'Degraded Performance',
          description: 'Some systems are experiencing issues but core functionality remains available'
        };
      case 'critical':
        return {
          icon: <FiX />,
          title: 'Critical Issues',
          description: 'Multiple systems are experiencing problems. Some features may be unavailable'
        };
      default:
        return {
          icon: <FiAlertTriangle />,
          title: 'Status Unknown',
          description: 'Unable to determine system status'
        };
    }
  };

  const statusInfo = getOverallStatusInfo();
  const systems = systemStatus.system_status || {};

  const systemComponents = [
    {
      key: 'enhanced_rag',
      name: 'Enhanced RAG System',
      description: 'AI knowledge retrieval with 8,885 chunks',
      icon: <FiDatabase />,
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      data: systems.enhanced_rag
    },
    {
      key: 'crewai_agents',
      name: 'CrewAI Agents',
      description: '5 specialized AI agents',
      icon: <FiUsers />,
      gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      data: systems.crewai_agents
    },
    {
      key: 'ml_preferences',
      name: 'ML Preferences',
      description: 'Machine learning personalization',
      icon: <FiCpu />,
      gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      data: systems.ml_preferences
    },
    {
      key: 'analytics',
      name: 'Analytics System',
      description: 'Real-time monitoring and insights',
      icon: <FiBarChart3 />,
      gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
      data: systems.analytics
    },
    {
      key: 'realtime_data',
      name: 'Real-time Data',
      description: 'Weather, prices, and live information',
      icon: <FiGlobe />,
      gradient: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
      data: systems.realtime_data
    }
  ];

  return (
    <StatusContainer>
      <StatusHeader>
        <HeaderTitle>
          <h1>🔍 System Status</h1>
          <p>Real-time monitoring of all AI systems</p>
        </HeaderTitle>
        
        <RefreshButton onClick={onRefresh}>
          <FiRefreshCw />
          Refresh Status
        </RefreshButton>
      </StatusHeader>

      <OverallStatus status={systemStatus.system_status?.overall_health}>
        <StatusIcon>{statusInfo.icon}</StatusIcon>
        <StatusTitle>{statusInfo.title}</StatusTitle>
        <StatusDescription>{statusInfo.description}</StatusDescription>
      </OverallStatus>

      <SystemsGrid>
        {systemComponents.map((system) => (
          <SystemCard key={system.key}>
            <SystemHeader>
              <SystemInfo>
                <SystemIcon gradient={system.gradient}>
                  {system.icon}
                </SystemIcon>
                <SystemName>
                  <h3>{system.name}</h3>
                  <p>{system.description}</p>
                </SystemName>
              </SystemInfo>
              
              <StatusBadge status={system.data?.status || 'unknown'}>
                {system.data?.status === 'operational' && <FiCheck size={14} />}
                {system.data?.status === 'error' && <FiX size={14} />}
                {system.data?.status === 'unavailable' && <FiAlertTriangle size={14} />}
                {system.data?.status || 'Unknown'}
              </StatusBadge>
            </SystemHeader>

            <SystemDetails>
              <DetailItem>
                <div className="value">
                  {system.data?.chunks || system.data?.agents || system.data?.accuracy || 'N/A'}
                </div>
                <div className="label">
                  {system.key === 'enhanced_rag' ? 'Chunks' :
                   system.key === 'crewai_agents' ? 'Agents' :
                   system.key === 'ml_preferences' ? 'Accuracy' :
                   'Status'}
                </div>
              </DetailItem>
              
              <DetailItem>
                <div className="value">
                  {system.data?.quality_score ? `${(system.data.quality_score * 100).toFixed(0)}%` :
                   system.data?.tracking || 'Active'}
                </div>
                <div className="label">
                  {system.key === 'realtime_data' ? 'Quality' : 'Health'}
                </div>
              </DetailItem>
            </SystemDetails>

            {system.data?.error && (
              <div style={{ 
                marginTop: '15px', 
                padding: '10px', 
                background: '#f8d7da', 
                color: '#721c24', 
                borderRadius: '5px',
                fontSize: '0.85rem'
              }}>
                Error: {system.data.error}
              </div>
            )}
          </SystemCard>
        ))}
      </SystemsGrid>

      <LastUpdated>
        Last updated: {new Date(systemStatus.timestamp * 1000).toLocaleString()}
      </LastUpdated>
    </StatusContainer>
  );
}

export default SystemStatus;
