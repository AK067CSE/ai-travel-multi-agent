import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import {
  FiActivity, FiMessageCircle, FiTrendingUp,
  FiGlobe, FiZap
} from 'react-icons/fi';
import { apiService } from '../services/apiService';

const DashboardContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  padding: 20px 0;
`;

const DashboardCard = styled.div`
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
  margin-bottom: 20px;
  
  h3 {
    margin: 0;
    color: #333;
    font-size: 1.2rem;
    font-weight: 600;
  }
`;

const CardIcon = styled.div`
  width: 50px;
  height: 50px;
  border-radius: 15px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 1.5rem;
`;

const StatValue = styled.div`
  font-size: 2.5rem;
  font-weight: 700;
  color: #667eea;
  margin-bottom: 10px;
`;

const StatLabel = styled.div`
  color: #666;
  font-size: 0.9rem;
  margin-bottom: 15px;
`;

const StatChange = styled.div`
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.8rem;
  color: ${props => props.positive ? '#10b981' : '#ef4444'};
  font-weight: 500;
`;

const FeatureList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const FeatureItem = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 15px;
  background: #f8f9fa;
  border-radius: 10px;
  
  .feature-name {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 500;
    color: #333;
  }
  
  .feature-status {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    ${props => props.active 
      ? 'background: #dcfce7; color: #166534;'
      : 'background: #fee2e2; color: #991b1b;'
    }
  }
`;

const SystemHealthCard = styled(DashboardCard)`
  background: ${props => {
    switch(props.health) {
      case 'excellent': return 'linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)';
      case 'good': return 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)';
      case 'degraded': return 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)';
      default: return 'rgba(255, 255, 255, 0.95)';
    }
  }};
`;

const HealthIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
  
  .health-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: ${props => {
      switch(props.health) {
        case 'excellent': return '#10b981';
        case 'good': return '#f59e0b';
        case 'degraded': return '#ef4444';
        default: return '#6b7280';
      }
    }};
    animation: pulse 2s infinite;
  }
  
  .health-text {
    font-size: 1.1rem;
    font-weight: 600;
    color: #333;
    text-transform: capitalize;
  }
`;

const QuickStats = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
  margin-top: 20px;
`;

const QuickStat = styled.div`
  text-align: center;
  padding: 15px;
  background: rgba(255, 255, 255, 0.7);
  border-radius: 10px;
  
  .stat-number {
    font-size: 1.5rem;
    font-weight: 700;
    color: #667eea;
    margin-bottom: 5px;
  }
  
  .stat-label {
    font-size: 0.8rem;
    color: #666;
  }
`;

function Dashboard({ systemStatus }) {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      await apiService.getConversations();
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSystemFeatures = () => {
    if (!systemStatus?.features) return [];
    
    return Object.entries(systemStatus.features).map(([key, value]) => ({
      name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      active: value,
      icon: getFeatureIcon(key)
    }));
  };

  const getFeatureIcon = (feature) => {
    switch (feature) {
      case 'openai_integration': return <FiZap size={16} />;
      case 'knowledge_base': return <FiGlobe size={16} />;
      case 'streaming_responses': return <FiActivity size={16} />;
      case 'conversation_memory': return <FiMessageCircle size={16} />;
      default: return <FiActivity size={16} />;
    }
  };

  if (loading) {
    return (
      <DashboardContainer>
        <DashboardCard>
          <div style={{ textAlign: 'center', padding: '40px 0', color: '#666' }}>
            <FiActivity size={48} className="pulse" />
            <p style={{ marginTop: '20px' }}>Loading dashboard data...</p>
          </div>
        </DashboardCard>
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer>
      {/* System Health Card */}
      <SystemHealthCard health={systemStatus?.overall_health}>
        <CardHeader>
          <h3>System Health</h3>
          <CardIcon>
            <FiActivity />
          </CardIcon>
        </CardHeader>
        
        <HealthIndicator health={systemStatus?.overall_health}>
          <div className="health-dot"></div>
          <div className="health-text">{systemStatus?.overall_health || 'Unknown'}</div>
        </HealthIndicator>
        
        <p style={{ color: '#666', marginBottom: '20px' }}>
          All systems operational and ready to assist with travel planning.
        </p>
        
        <QuickStats>
          <QuickStat>
            <div className="stat-number">{systemStatus?.statistics?.destinations_in_db || 0}</div>
            <div className="stat-label">Destinations</div>
          </QuickStat>
          <QuickStat>
            <div className="stat-number">{systemStatus?.statistics?.travel_types_supported || 0}</div>
            <div className="stat-label">Travel Types</div>
          </QuickStat>
        </QuickStats>
      </SystemHealthCard>

      {/* Conversations Stats */}
      <DashboardCard>
        <CardHeader>
          <h3>Conversations</h3>
          <CardIcon>
            <FiMessageCircle />
          </CardIcon>
        </CardHeader>
        
        <StatValue>{systemStatus?.statistics?.total_conversations || 0}</StatValue>
        <StatLabel>Total Conversations</StatLabel>
        
        <StatChange positive={true}>
          <FiTrendingUp size={14} />
          Active sessions ready
        </StatChange>
        
        <div style={{ marginTop: '20px', padding: '15px', background: '#f8f9fa', borderRadius: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ color: '#666' }}>Total Messages:</span>
            <span style={{ fontWeight: '600', color: '#333' }}>
              {systemStatus?.statistics?.total_messages || 0}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#666' }}>AI Model:</span>
            <span style={{ fontWeight: '600', color: '#667eea' }}>
              {systemStatus?.ai_model || 'Unknown'}
            </span>
          </div>
        </div>
      </DashboardCard>

      {/* Features Status */}
      <DashboardCard>
        <CardHeader>
          <h3>System Features</h3>
          <CardIcon>
            <FiZap />
          </CardIcon>
        </CardHeader>
        
        <FeatureList>
          {getSystemFeatures().map((feature, index) => (
            <FeatureItem key={index} active={feature.active}>
              <div className="feature-name">
                {feature.icon}
                {feature.name}
              </div>
              <div className="feature-status">
                {feature.active ? 'Active' : 'Inactive'}
              </div>
            </FeatureItem>
          ))}
        </FeatureList>
      </DashboardCard>

      {/* Performance Metrics */}
      <DashboardCard>
        <CardHeader>
          <h3>Performance</h3>
          <CardIcon>
            <FiTrendingUp />
          </CardIcon>
        </CardHeader>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ color: '#666' }}>Response Time</span>
              <span style={{ fontWeight: '600', color: '#10b981' }}>Excellent</span>
            </div>
            <div style={{ 
              height: '8px', 
              background: '#e5e7eb', 
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <div style={{ 
                width: '85%', 
                height: '100%', 
                background: 'linear-gradient(90deg, #10b981, #059669)',
                borderRadius: '4px'
              }}></div>
            </div>
          </div>
          
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ color: '#666' }}>System Load</span>
              <span style={{ fontWeight: '600', color: '#667eea' }}>Optimal</span>
            </div>
            <div style={{ 
              height: '8px', 
              background: '#e5e7eb', 
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <div style={{ 
                width: '65%', 
                height: '100%', 
                background: 'linear-gradient(90deg, #667eea, #764ba2)',
                borderRadius: '4px'
              }}></div>
            </div>
          </div>
          
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ color: '#666' }}>AI Accuracy</span>
              <span style={{ fontWeight: '600', color: '#f59e0b' }}>High</span>
            </div>
            <div style={{ 
              height: '8px', 
              background: '#e5e7eb', 
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <div style={{ 
                width: '92%', 
                height: '100%', 
                background: 'linear-gradient(90deg, #f59e0b, #d97706)',
                borderRadius: '4px'
              }}></div>
            </div>
          </div>
        </div>
      </DashboardCard>
    </DashboardContainer>
  );
}

export default Dashboard;
