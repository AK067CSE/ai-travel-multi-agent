"""
Booking Agent - Handles travel bookings and reservations
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from .base_agent import BaseAgent, AgentResponse
import logging

logger = logging.getLogger(__name__)

class BookingAgent(BaseAgent):
    """Agent responsible for handling travel bookings and reservations"""
    
    def __init__(self, model_name: str = "llama3-8b-8192"):
        super().__init__("BookingAgent", model_name)
        self.bookings = {}
        self.booking_history = {}
        self.payment_methods = ["credit_card", "debit_card", "paypal", "bank_transfer"]
        self.booking_status_options = ["pending", "confirmed", "cancelled", "completed"]
    
    def create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for booking assistance"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a professional travel booking assistant. Your role is to:
            
            1. Help users understand booking processes and requirements
            2. Explain booking terms, conditions, and policies
            3. Provide guidance on payment options and security
            4. Assist with booking modifications and cancellations
            5. Offer alternative options when bookings are unavailable
            
            Always prioritize:
            - Clear communication about costs and fees
            - Transparency about cancellation policies
            - Security and privacy of user information
            - Helpful alternatives when primary options aren't available
            
            Be professional, trustworthy, and detail-oriented in all booking matters."""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Booking Request: {booking_request}
            
            Available Options: {available_options}
            
            User Preferences: {user_preferences}
            
            Please assist with this booking request and provide detailed guidance."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process booking request"""
        try:
            booking_type = request.get("type", "general")
            user_id = request.get("user_id", "anonymous")
            
            self.log_activity(f"Processing booking request", {
                "type": booking_type,
                "user_id": user_id
            })
            
            if booking_type == "hotel":
                result = self.handle_hotel_booking(request)
            elif booking_type == "flight":
                result = self.handle_flight_booking(request)
            elif booking_type == "package":
                result = self.handle_package_booking(request)
            elif booking_type == "modify":
                result = self.modify_booking(request)
            elif booking_type == "cancel":
                result = self.cancel_booking(request)
            else:
                result = self.provide_booking_guidance(request)
            
            response = AgentResponse(
                agent_name=self.agent_name,
                success=True,
                data=result,
                metadata={"booking_type": booking_type, "user_id": user_id}
            )
            
            return response.to_dict()
            
        except Exception as e:
            logger.error(f"Booking error: {e}")
            response = AgentResponse(
                agent_name=self.agent_name,
                success=False,
                error=str(e)
            )
            return response.to_dict()
    
    def handle_hotel_booking(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hotel booking request"""
        hotel_details = request.get("hotel_details", {})
        user_details = request.get("user_details", {})
        dates = request.get("dates", {})
        
        # Generate booking ID
        booking_id = str(uuid.uuid4())[:8]
        
        # Calculate total cost
        nights = self.calculate_nights(dates.get("check_in"), dates.get("check_out"))
        room_rate = hotel_details.get("price_per_night", 100)
        total_cost = nights * room_rate
        taxes = total_cost * 0.15  # 15% taxes
        final_cost = total_cost + taxes
        
        # Create booking record
        booking = {
            "booking_id": booking_id,
            "type": "hotel",
            "hotel_name": hotel_details.get("name", "Hotel"),
            "check_in": dates.get("check_in"),
            "check_out": dates.get("check_out"),
            "nights": nights,
            "room_type": hotel_details.get("room_type", "Standard"),
            "guests": user_details.get("guests", 2),
            "total_cost": final_cost,
            "currency": "USD",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "user_id": request.get("user_id", "anonymous")
        }
        
        # Store booking
        self.bookings[booking_id] = booking
        
        # Generate booking confirmation using LLM
        confirmation = self.generate_booking_confirmation(booking)
        
        return {
            "booking_id": booking_id,
            "booking_details": booking,
            "confirmation_message": confirmation,
            "next_steps": [
                "Complete payment within 24 hours",
                "Check email for confirmation details",
                "Review cancellation policy",
                "Save booking reference number"
            ]
        }
    
    def handle_flight_booking(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle flight booking request"""
        flight_details = request.get("flight_details", {})
        user_details = request.get("user_details", {})
        
        booking_id = str(uuid.uuid4())[:8]
        
        # Calculate flight cost
        base_price = flight_details.get("price", 300)
        passengers = user_details.get("passengers", 1)
        taxes_fees = base_price * 0.25  # 25% taxes and fees
        total_cost = (base_price + taxes_fees) * passengers
        
        booking = {
            "booking_id": booking_id,
            "type": "flight",
            "airline": flight_details.get("airline", "Airline"),
            "flight_number": flight_details.get("flight_number", "FL123"),
            "departure": flight_details.get("departure"),
            "arrival": flight_details.get("arrival"),
            "departure_time": flight_details.get("departure_time"),
            "arrival_time": flight_details.get("arrival_time"),
            "passengers": passengers,
            "class": flight_details.get("class", "Economy"),
            "total_cost": total_cost,
            "currency": "USD",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "user_id": request.get("user_id", "anonymous")
        }
        
        self.bookings[booking_id] = booking
        confirmation = self.generate_booking_confirmation(booking)
        
        return {
            "booking_id": booking_id,
            "booking_details": booking,
            "confirmation_message": confirmation,
            "next_steps": [
                "Complete payment to secure seats",
                "Check-in online 24 hours before departure",
                "Review baggage allowance",
                "Arrive at airport 2 hours early for international flights"
            ]
        }
    
    def handle_package_booking(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle travel package booking"""
        package_details = request.get("package_details", {})
        
        booking_id = str(uuid.uuid4())[:8]
        
        booking = {
            "booking_id": booking_id,
            "type": "package",
            "package_name": package_details.get("name", "Travel Package"),
            "destination": package_details.get("destination"),
            "duration": package_details.get("duration", "7 days"),
            "includes": package_details.get("includes", ["Hotel", "Flights", "Tours"]),
            "total_cost": package_details.get("price", 1500),
            "currency": "USD",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "user_id": request.get("user_id", "anonymous")
        }
        
        self.bookings[booking_id] = booking
        confirmation = self.generate_booking_confirmation(booking)
        
        return {
            "booking_id": booking_id,
            "booking_details": booking,
            "confirmation_message": confirmation,
            "next_steps": [
                "Review package inclusions and exclusions",
                "Complete payment to confirm booking",
                "Receive detailed itinerary via email",
                "Contact support for any special requests"
            ]
        }
    
    def modify_booking(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Modify existing booking"""
        booking_id = request.get("booking_id")
        modifications = request.get("modifications", {})
        
        if booking_id not in self.bookings:
            return {"error": "Booking not found", "booking_id": booking_id}
        
        booking = self.bookings[booking_id]
        
        # Check if modification is allowed
        if booking["status"] in ["cancelled", "completed"]:
            return {"error": "Cannot modify cancelled or completed booking"}
        
        # Apply modifications
        for key, value in modifications.items():
            if key in booking:
                booking[key] = value
        
        booking["modified_at"] = datetime.now().isoformat()
        
        return {
            "booking_id": booking_id,
            "message": "Booking modified successfully",
            "updated_booking": booking
        }
    
    def cancel_booking(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel existing booking"""
        booking_id = request.get("booking_id")
        reason = request.get("reason", "User requested")
        
        if booking_id not in self.bookings:
            return {"error": "Booking not found", "booking_id": booking_id}
        
        booking = self.bookings[booking_id]
        booking["status"] = "cancelled"
        booking["cancelled_at"] = datetime.now().isoformat()
        booking["cancellation_reason"] = reason
        
        # Calculate refund amount (simplified)
        refund_amount = booking.get("total_cost", 0) * 0.8  # 80% refund
        
        return {
            "booking_id": booking_id,
            "message": "Booking cancelled successfully",
            "refund_amount": refund_amount,
            "refund_timeline": "5-7 business days"
        }
    
    def provide_booking_guidance(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Provide general booking guidance using LLM"""
        try:
            prompt_template = self.create_prompt_template()
            chain = self.create_chain(prompt_template)
            
            context = {
                "booking_request": request.get("query", "General booking assistance"),
                "available_options": json.dumps(request.get("options", {}), indent=2),
                "user_preferences": json.dumps(request.get("preferences", {}), indent=2)
            }
            
            guidance = chain.invoke(context)
            
            return {
                "guidance": guidance,
                "booking_tips": [
                    "Book in advance for better prices",
                    "Read cancellation policies carefully",
                    "Keep booking confirmations safe",
                    "Check visa requirements for international travel"
                ]
            }
            
        except Exception as e:
            logger.error(f"Guidance generation error: {e}")
            return {"guidance": "General booking assistance available. Please contact support for specific help."}
    
    def generate_booking_confirmation(self, booking: Dict[str, Any]) -> str:
        """Generate booking confirmation message"""
        booking_type = booking.get("type", "travel")
        booking_id = booking.get("booking_id")
        total_cost = booking.get("total_cost", 0)
        
        confirmation = f"""
        🎉 Booking Confirmation
        
        Booking ID: {booking_id}
        Type: {booking_type.title()}
        Total Cost: ${total_cost:.2f} USD
        Status: {booking.get("status", "pending").title()}
        
        Thank you for your booking! You will receive a detailed confirmation email shortly.
        """
        
        return confirmation.strip()
    
    def calculate_nights(self, check_in: str, check_out: str) -> int:
        """Calculate number of nights between dates"""
        try:
            check_in_date = datetime.fromisoformat(check_in.replace('Z', '+00:00'))
            check_out_date = datetime.fromisoformat(check_out.replace('Z', '+00:00'))
            return (check_out_date - check_in_date).days
        except:
            return 1  # Default to 1 night
    
    def get_booking(self, booking_id: str) -> Dict[str, Any]:
        """Get booking details"""
        return self.bookings.get(booking_id, {})
    
    def get_user_bookings(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all bookings for a user"""
        return [booking for booking in self.bookings.values() 
                if booking.get("user_id") == user_id]
