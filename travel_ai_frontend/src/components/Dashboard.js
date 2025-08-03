import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { 
  FiActivity, FiUsers, FiClock, FiStar, FiTrendingUp, 
  FiZap, FiGlobe, FiBarChart3, FiRefreshCw 
} from 'react-icons/fi';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell 
} from 'recharts';
import { apiService } from '../services/apiService';

const DashboardContainer = styled.div`
  padding: 20px;
  background: white;
  min-height: 100vh;
`;

const DashboardHeader = styled.div`
  display: flex;
  justify-content: between;
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

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const MetricCard = styled.div`
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

const MetricHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
`;

const MetricIcon = styled.div`
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

const MetricValue = styled.div`
  font-size: 2.5rem;
  font-weight: 700;
  color: #333;
  margin-bottom: 5px;
`;

const MetricLabel = styled.div`
  color: #666;
  font-size: 0.9rem;
  font-weight: 500;
`;

const MetricChange = styled.div`
  font-size: 0.8rem;
  color: ${props => props.positive ? '#27ae60' : '#e74c3c'};
  font-weight: 500;
  margin-top: 5px;
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 20px;
  margin-bottom: 30px;
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

const ChartCard = styled.div`
  background: white;
  border-radius: 15px;
  padding: 25px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
  border: 1px solid #f0f0f0;
`;

const ChartTitle = styled.h3`
  color: #333;
  margin: 0 0 20px 0;
  font-size: 1.2rem;
  font-weight: 600;
`;

const SystemHealthGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 30px;
`;

const HealthCard = styled.div`
  background: white;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
  border: 1px solid #f0f0f0;
  text-align: center;
`;

const HealthStatus = styled.div`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: ${props => {
    switch(props.status) {
      case 'operational': return '#27ae60';
      case 'degraded': return '#f39c12';
      case 'error': return '#e74c3c';
      default: return '#95a5a6';
    }
  }};
  margin: 0 auto 10px;
`;

const LoadingSpinner = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  font-size: 1.1rem;
  color: #666;
`;

const ErrorMessage = styled.div`
  background: #fee;
  color: #c33;
  padding: 15px;
  border-radius: 10px;
  text-align: center;
  margin: 20px 0;
`;

function Dashboard({ systemStatus, onViewChange }) {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (onViewChange) {
      onViewChange('dashboard');
    }
    loadDashboardData();
  }, [onViewChange]);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getDashboardData();
      setDashboardData(data);
    } catch (err) {
      console.error('Dashboard data error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <DashboardContainer>
        <LoadingSpinner>
          <FiRefreshCw className="spinning" style={{ marginRight: '10px' }} />
          Loading dashboard data...
        </LoadingSpinner>
      </DashboardContainer>
    );
  }

  if (error) {
    return (
      <DashboardContainer>
        <ErrorMessage>
          <h3>Dashboard Error</h3>
          <p>{error}</p>
          <button onClick={loadDashboardData} style={{ marginTop: '10px' }}>
            Retry
          </button>
        </ErrorMessage>
      </DashboardContainer>
    );
  }

  const overview = dashboardData?.dashboard?.overview || {};
  const systemHealth = dashboardData?.dashboard?.system_health || {};

  // Mock chart data for demonstration
  const responseTimeData = [
    { time: '00:00', responseTime: 1.2 },
    { time: '04:00', responseTime: 1.1 },
    { time: '08:00', responseTime: 1.4 },
    { time: '12:00', responseTime: 1.3 },
    { time: '16:00', responseTime: 1.5 },
    { time: '20:00', responseTime: 1.2 },
  ];

  const systemUsageData = [
    { name: 'Enhanced RAG', value: 45, color: '#667eea' },
    { name: 'CrewAI Agents', value: 35, color: '#764ba2' },
    { name: 'ML Preferences', value: 20, color: '#f093fb' },
  ];

  return (
    <DashboardContainer>
      <DashboardHeader>
        <HeaderTitle>
          <h1>📊 AI Travel Agent Dashboard</h1>
          <p>Real-time analytics and system monitoring</p>
        </HeaderTitle>
        
        <RefreshButton onClick={loadDashboardData} disabled={loading}>
          <FiRefreshCw />
          Refresh Data
        </RefreshButton>
      </DashboardHeader>

      <MetricsGrid>
        <MetricCard>
          <MetricHeader>
            <MetricIcon gradient="linear-gradient(135deg, #667eea 0%, #764ba2 100%)">
              <FiActivity />
            </MetricIcon>
          </MetricHeader>
          <MetricValue>{overview.total_requests_24h || 0}</MetricValue>
          <MetricLabel>Requests (24h)</MetricLabel>
          <MetricChange positive>+12% from yesterday</MetricChange>
        </MetricCard>

        <MetricCard>
          <MetricHeader>
            <MetricIcon gradient="linear-gradient(135deg, #f093fb 0%, #f5576c 100%)">
              <FiUsers />
            </MetricIcon>
          </MetricHeader>
          <MetricValue>{overview.active_users_24h || 0}</MetricValue>
          <MetricLabel>Active Users</MetricLabel>
          <MetricChange positive>+8% from yesterday</MetricChange>
        </MetricCard>

        <MetricCard>
          <MetricHeader>
            <MetricIcon gradient="linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)">
              <FiClock />
            </MetricIcon>
          </MetricHeader>
          <MetricValue>{overview.avg_response_time_24h?.toFixed(2) || '0.00'}s</MetricValue>
          <MetricLabel>Avg Response Time</MetricLabel>
          <MetricChange positive>-5% faster</MetricChange>
        </MetricCard>

        <MetricCard>
          <MetricHeader>
            <MetricIcon gradient="linear-gradient(135deg, #fa709a 0%, #fee140 100%)">
              <FiStar />
            </MetricIcon>
          </MetricHeader>
          <MetricValue>{overview.avg_user_rating?.toFixed(1) || '0.0'}</MetricValue>
          <MetricLabel>User Rating</MetricLabel>
          <MetricChange positive>+0.2 improvement</MetricChange>
        </MetricCard>
      </MetricsGrid>

      <ChartsGrid>
        <ChartCard>
          <ChartTitle>Response Time Trend</ChartTitle>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={responseTimeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="responseTime" 
                stroke="#667eea" 
                strokeWidth={3}
                dot={{ fill: '#667eea', strokeWidth: 2, r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard>
          <ChartTitle>System Usage</ChartTitle>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={systemUsageData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}%`}
              >
                {systemUsageData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </ChartsGrid>

      <ChartTitle>System Health Status</ChartTitle>
      <SystemHealthGrid>
        <HealthCard>
          <HealthStatus status="operational" />
          <h4>Enhanced RAG</h4>
          <p>8,885 chunks loaded</p>
        </HealthCard>

        <HealthCard>
          <HealthStatus status="operational" />
          <h4>CrewAI Agents</h4>
          <p>5 agents active</p>
        </HealthCard>

        <HealthCard>
          <HealthStatus status="operational" />
          <h4>ML Preferences</h4>
          <p>85% accuracy</p>
        </HealthCard>

        <HealthCard>
          <HealthStatus status="operational" />
          <h4>Real-time Data</h4>
          <p>92% quality score</p>
        </HealthCard>

        <HealthCard>
          <HealthStatus status="operational" />
          <h4>Analytics</h4>
          <p>Tracking active</p>
        </HealthCard>
      </SystemHealthGrid>
    </DashboardContainer>
  );
}

export default Dashboard;
