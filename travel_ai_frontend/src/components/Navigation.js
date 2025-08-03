import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import styled from 'styled-components';
import { FiMessageCircle, FiBarChart3, FiActivity, FiGlobe, FiZap } from 'react-icons/fi';

const NavContainer = styled.nav`
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  padding: 15px 0;
  position: sticky;
  top: 0;
  z-index: 100;
`;

const NavContent = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 1.5rem;
  font-weight: 700;
  color: #333;
  text-decoration: none;
  
  .logo-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
  }
`;

const NavLinks = styled.div`
  display: flex;
  gap: 5px;
  
  @media (max-width: 768px) {
    gap: 2px;
  }
`;

const NavLink = styled(Link)`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  border-radius: 12px;
  text-decoration: none;
  color: ${props => props.active ? 'white' : '#666'};
  background: ${props => props.active 
    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    : 'transparent'
  };
  font-weight: 500;
  transition: all 0.2s ease;
  
  &:hover {
    background: ${props => props.active 
      ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      : 'rgba(102, 126, 234, 0.1)'
    };
    color: ${props => props.active ? 'white' : '#667eea'};
    transform: translateY(-2px);
  }
  
  @media (max-width: 768px) {
    padding: 10px 15px;
    font-size: 0.9rem;
    
    span {
      display: none;
    }
  }
`;

const SystemStatus = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 15px;
  background: rgba(102, 126, 234, 0.1);
  border-radius: 20px;
  font-size: 0.85rem;
  color: #667eea;
  font-weight: 500;
  
  @media (max-width: 768px) {
    display: none;
  }
`;

const StatusIndicator = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${props => {
    switch(props.status) {
      case 'excellent': return '#27ae60';
      case 'operational': return '#27ae60';
      case 'degraded': return '#f39c12';
      case 'critical': return '#e74c3c';
      default: return '#95a5a6';
    }
  }};
  animation: ${props => props.status === 'excellent' || props.status === 'operational' 
    ? 'pulse 2s infinite' : 'none'
  };
  
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
`;

const MobileMenuButton = styled.button`
  display: none;
  background: none;
  border: none;
  font-size: 1.5rem;
  color: #667eea;
  cursor: pointer;
  
  @media (max-width: 768px) {
    display: block;
  }
`;

function Navigation({ currentView, setCurrentView, systemStatus }) {
  const location = useLocation();

  const navItems = [
    {
      path: '/chat',
      icon: FiMessageCircle,
      label: 'AI Chat',
      key: 'chat'
    },
    {
      path: '/dashboard',
      icon: FiBarChart3,
      label: 'Dashboard',
      key: 'dashboard'
    },
    {
      path: '/status',
      icon: FiActivity,
      label: 'System Status',
      key: 'status'
    }
  ];

  const getSystemStatusText = () => {
    if (!systemStatus) return 'Checking...';
    
    const health = systemStatus.system_status?.overall_health;
    switch(health) {
      case 'excellent': return 'All Systems Optimal';
      case 'operational': return 'Systems Operational';
      case 'degraded': return 'Some Issues Detected';
      case 'critical': return 'Critical Issues';
      default: return 'Status Unknown';
    }
  };

  const getSystemStatusLevel = () => {
    if (!systemStatus) return 'unknown';
    return systemStatus.system_status?.overall_health || 'unknown';
  };

  return (
    <NavContainer>
      <NavContent>
        <Logo>
          <div className="logo-icon">
            <FiGlobe />
          </div>
          <span>AI Travel Agent</span>
        </Logo>

        <NavLinks>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <NavLink
                key={item.key}
                to={item.path}
                active={isActive}
                onClick={() => setCurrentView && setCurrentView(item.key)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </NavLinks>

        <SystemStatus>
          <StatusIndicator status={getSystemStatusLevel()} />
          <FiZap size={14} />
          <span>{getSystemStatusText()}</span>
        </SystemStatus>
      </NavContent>
    </NavContainer>
  );
}

export default Navigation;
