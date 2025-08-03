"""
Command Line Interface for Multi-Agent Travel AI System
"""

import click
import json
import os
from pathlib import Path
from travel_ai_system import TravelAISystem
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Multi-Agent Travel AI System CLI"""
    pass

@cli.command()
@click.option('--user-id', default='cli_user', help='User ID for the session')
@click.option('--workflow-type', default='auto', help='Workflow type (auto, simple, complex)')
def chat(user_id, workflow_type):
    """Start interactive chat with the travel AI system"""
    
    click.echo("🤖 Multi-Agent Travel AI System")
    click.echo("=" * 50)
    click.echo("Welcome! I'm your AI travel assistant powered by multiple specialized agents.")
    click.echo("Type 'quit', 'exit', or 'bye' to end the conversation.")
    click.echo("Type 'help' for available commands.")
    click.echo("")
    
    # Initialize system
    system = TravelAISystem()
    session_id = None
    
    while True:
        try:
            # Get user input
            user_input = click.prompt("You", type=str)
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                click.echo("🤖 Thank you for using the Travel AI System! Safe travels! ✈️")
                break
            
            # Check for help
            if user_input.lower() in ['help', '?']:
                show_help()
                continue
            
            # Check for system commands
            if user_input.lower().startswith('/'):
                handle_system_command(user_input, system, session_id)
                continue
            
            # Process travel request
            response = system.process_user_request(
                user_request=user_input,
                user_id=user_id,
                session_id=session_id,
                workflow_type=workflow_type
            )
            
            # Update session ID
            if not session_id:
                session_id = response.get("workflow_id", f"session_{user_id}")
            
            # Display response
            display_response(response)
            
        except KeyboardInterrupt:
            click.echo("\n🤖 Goodbye! Safe travels! ✈️")
            break
        except Exception as e:
            click.echo(f"❌ Error: {e}")
            logger.error(f"CLI error: {e}")

def show_help():
    """Show help information"""
    click.echo("""
🔧 Available Commands:
    
Travel Requests:
    - "Plan a trip to Paris for 5 days"
    - "Recommend hotels in Tokyo under $200/night"
    - "Find flights from NYC to London"
    - "Book a hotel for next weekend"
    
System Commands:
    /status     - Show system status
    /session    - Show current session info
    /clear      - Clear current session
    /agents     - Show agent information
    /metrics    - Show system metrics
    
General:
    help        - Show this help
    quit/exit   - End conversation
    """)

def handle_system_command(command: str, system: TravelAISystem, session_id: str):
    """Handle system commands"""
    command = command.lower().strip()
    
    if command == '/status':
        status = system.get_system_status()
        click.echo(f"📊 System Status: {status['status']}")
        click.echo(f"🔢 Total Requests: {status['metrics']['total_requests']}")
        click.echo(f"✅ Success Rate: {status['metrics']['successful_requests']}/{status['metrics']['total_requests']}")
        click.echo(f"⚡ Avg Response Time: {status['metrics']['average_response_time']:.2f}s")
        click.echo(f"👥 Active Sessions: {status['active_sessions']}")
    
    elif command == '/session':
        if session_id:
            session_info = system.get_session_info(session_id)
            if 'error' not in session_info:
                click.echo(f"🔗 Session ID: {session_info['session_id']}")
                click.echo(f"👤 User ID: {session_info['user_id']}")
                click.echo(f"📅 Created: {session_info['created_at']}")
                click.echo(f"🔄 Requests: {session_info['request_count']}")
            else:
                click.echo("❌ Session not found")
        else:
            click.echo("ℹ️ No active session")
    
    elif command == '/clear':
        if session_id and system.clear_session(session_id):
            click.echo("✅ Session cleared")
        else:
            click.echo("❌ No session to clear")
    
    elif command == '/agents':
        click.echo("""
🤖 Available Agents:
    
🕷️ ScrapingAgent
    - Web scraping and data collection
    - Real-time travel information
    - Hotel, flight, and attraction data
    
🎯 RecommendationAgent  
    - Personalized travel suggestions
    - User preference learning
    - Destination planning
    
📋 BookingAgent
    - Hotel and flight reservations
    - Booking management
    - Payment processing
    
💬 ChatAgent
    - Natural conversation
    - Intent detection
    - User interaction
    
🎛️ CoordinatorAgent
    - Multi-agent orchestration
    - Workflow management
    - System coordination
        """)
    
    elif command == '/metrics':
        status = system.get_system_status()
        metrics = status['metrics']
        click.echo("📈 System Metrics:")
        click.echo(f"  Total Requests: {metrics['total_requests']}")
        click.echo(f"  Successful: {metrics['successful_requests']}")
        click.echo(f"  Failed: {metrics['failed_requests']}")
        click.echo(f"  Average Response Time: {metrics['average_response_time']:.2f}s")
    
    else:
        click.echo(f"❓ Unknown command: {command}")
        click.echo("Type 'help' for available commands")

def display_response(response: dict):
    """Display system response in a formatted way"""
    
    if not response.get("success"):
        click.echo(f"❌ Error: {response.get('message', 'Unknown error')}")
        return
    
    # Main message
    click.echo(f"🤖 {response['message']}")
    
    # Show agents involved
    agents = response.get("agents_involved", [])
    if agents:
        agent_icons = {
            "ScrapingAgent": "🕷️",
            "RecommendationAgent": "🎯", 
            "BookingAgent": "📋",
            "ChatAgent": "💬",
            "CoordinatorAgent": "🎛️"
        }
        agent_display = " ".join([f"{agent_icons.get(agent, '🤖')}{agent}" for agent in agents])
        click.echo(f"🔧 Agents: {agent_display}")
    
    # Show response time
    response_time = response.get("response_time", 0)
    click.echo(f"⚡ Response time: {response_time:.2f}s")
    
    # Show suggested actions
    suggestions = response.get("suggested_actions", [])
    if suggestions:
        click.echo("💡 Suggestions:")
        for suggestion in suggestions[:3]:  # Show max 3 suggestions
            click.echo(f"   • {suggestion}")
    
    # Show additional data if available
    additional_data = response.get("additional_data", {})
    if additional_data.get("booking_details"):
        booking = additional_data["booking_details"]
        click.echo(f"📋 Booking ID: {booking.get('booking_id')}")
        click.echo(f"💰 Total Cost: ${booking.get('total_cost', 0):.2f}")
    
    click.echo("")  # Empty line for readability

@cli.command()
def demo():
    """Run a demo conversation"""
    click.echo("🎬 Running Multi-Agent Travel AI Demo...")
    
    from travel_ai_system import demo_conversation
    demo_conversation()

@cli.command()
@click.option('--user-id', default='test_user', help='User ID for testing')
@click.option('--count', default=5, help='Number of test requests')
def test(user_id, count):
    """Run system tests"""
    click.echo(f"🧪 Running {count} test requests...")
    
    system = TravelAISystem()
    
    test_requests = [
        "Hello, I need help planning a trip",
        "I want to visit Japan for 10 days in spring",
        "What are the best hotels in Tokyo?",
        "Can you recommend some restaurants?",
        "I'd like to book a hotel for my trip"
    ]
    
    session_id = None
    successful = 0
    
    for i in range(min(count, len(test_requests))):
        request = test_requests[i]
        click.echo(f"\n🧪 Test {i+1}: {request}")
        
        try:
            response = system.process_user_request(
                user_request=request,
                user_id=user_id,
                session_id=session_id
            )
            
            if not session_id:
                session_id = response.get("workflow_id")
            
            if response.get("success"):
                successful += 1
                click.echo(f"✅ Success ({response.get('response_time', 0):.2f}s)")
            else:
                click.echo(f"❌ Failed: {response.get('message')}")
                
        except Exception as e:
            click.echo(f"❌ Error: {e}")
    
    click.echo(f"\n📊 Test Results: {successful}/{count} successful")
    
    # Show system status
    status = system.get_system_status()
    click.echo(f"📈 System Metrics:")
    click.echo(f"  Total Requests: {status['metrics']['total_requests']}")
    click.echo(f"  Success Rate: {status['metrics']['successful_requests']}/{status['metrics']['total_requests']}")

@cli.command()
def status():
    """Show system status"""
    system = TravelAISystem()
    status = system.get_system_status()
    
    click.echo("📊 Multi-Agent Travel AI System Status")
    click.echo("=" * 40)
    click.echo(f"Status: {status['status']}")
    click.echo(f"Active Sessions: {status['active_sessions']}")
    click.echo(f"Total Requests: {status['metrics']['total_requests']}")
    click.echo(f"Success Rate: {status['metrics']['successful_requests']}/{status['metrics']['total_requests']}")
    click.echo(f"Average Response Time: {status['metrics']['average_response_time']:.2f}s")
    
    click.echo("\n🤖 Agent Status:")
    for agent, status_val in status['agents_status'].items():
        icon = "✅" if status_val == "active" else "❌"
        click.echo(f"  {icon} {agent.title()}: {status_val}")

if __name__ == '__main__':
    cli()
