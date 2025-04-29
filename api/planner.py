from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Add parent directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import from existing modules if needed
try:
    # Try to import agent components as in the original file
    from agents import Agent, Runner
    from tools import (
        get_existing_leads,
        search_hubspot_contacts,
        get_website_visits,
        get_crm_activities
    )
    agent_imports_successful = True
except ImportError as e:
    print(f"Warning: Could not import agent components: {str(e)}")
    agent_imports_successful = False

# Create FastAPI app
app = FastAPI()

# Define request and response models
class PlannerRequest(BaseModel):
    company: str

class ContactPerson(BaseModel):
    name: str
    role: str
    reason: str
    linkedin_url: Optional[str] = None
    email: Optional[str] = None

class PlanStep(BaseModel):
    day: int
    action: str
    content: str
    condition: Optional[str] = None
    next_step_if_true: Optional[int] = None
    next_step_if_false: Optional[int] = None

class PlannerResponse(BaseModel):
    company: str
    company_status: str  # "new", "existing_lead", "showed_interest"
    contact_person: ContactPerson
    plan: List[PlanStep]
    recommendations: List[str]

# Company context API
COMPANY_CONTEXT_API_URL = "https://app.rebounds.ai/api/companycontextsven"

async def get_company_context():
    """Get company context information from the API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(COMPANY_CONTEXT_API_URL)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error fetching company context: {str(e)}")
        return []

async def check_company_status(company: str) -> Dict[str, Any]:
    """Check if company exists in our systems or has shown interest"""
    status = "new"  # Default status
    details = {}
    
    # If agent imports were successful, try to use the tools
    if agent_imports_successful:
        try:
            # Check if company exists in leads
            leads_result = await get_existing_leads(company)
            if leads_result and len(leads_result) > 0:
                status = "existing_lead"
                details["leads"] = leads_result
            
            # Check contacts in Hubspot
            contacts_result = await search_hubspot_contacts(company)
            if contacts_result and len(contacts_result) > 0:
                details["contacts"] = contacts_result
            
            # Check website visits
            visits_result = await get_website_visits(company)
            if visits_result and len(visits_result) > 0:
                status = "showed_interest"
                details["visits"] = visits_result
            
            # Check CRM activities
            activities_result = await get_crm_activities(company)
            if activities_result and len(activities_result) > 0:
                details["activities"] = activities_result
                
        except Exception as e:
            print(f"Error checking company status: {str(e)}")
    
    return {
        "status": status,
        "details": details
    }

async def identify_best_contact(company: str, company_context: List[Dict[str, Any]]) -> ContactPerson:
    """Find the best person to contact based on company and context"""
    # Use the company context to determine target personas
    if not company_context:
        # Default persona if no context available
        return ContactPerson(
            name="Unknown",
            role="Sales Manager",
            reason="Default target persona based on general company profile"
        )
    
    # Get the first context (assuming it's the most relevant)
    context = company_context[0]
    
    # Extract target job titles from context
    target_job_titles = context.get("targetJobTitles", ["Sales Manager", "Marketing Manager"])
    
    # For this example, we'll just use the first job title
    # In a real implementation, you would use LinkedIn/sales tools to find actual people
    best_role = target_job_titles[0] if target_job_titles else "Sales Manager"
    
    return ContactPerson(
        name=f"Decision Maker at {company}",
        role=best_role,
        reason=f"This role is a primary target based on your company profile: {best_role} is in your target job titles"
    )

async def create_outreach_plan(company: str, status: str, contact_person: ContactPerson, 
                            company_context: List[Dict[str, Any]]) -> List[PlanStep]:
    """Create a step-by-step outreach plan"""
    plan = []
    
    # Different approach based on company status
    if status == "existing_lead":
        # For existing leads, start with an email
        plan = [
            PlanStep(
                day=0,
                action="Send Email",
                content=f"Follow up on previous interaction with {company}. Mention specific pain points and solutions.",
                condition="email responded to or not",
                next_step_if_true=4,
                next_step_if_false=1
            ),
            PlanStep(
                day=2,
                action="Send LinkedIn Connection Request",
                content=f"Connection request to {contact_person.name} mentioning previous email and interaction with company.",
                condition="connection request accepted or not",
                next_step_if_true=2,
                next_step_if_false=3
            ),
            PlanStep(
                day=4,
                action="Send LinkedIn Message",
                content=f"Thank for connecting. Follow up on email and offer a quick 15-minute meeting.",
                condition=None,
                next_step_if_true=None,
                next_step_if_false=None
            ),
            PlanStep(
                day=5,
                action="Send Email",
                content=f"Second follow-up email with more specific value propositions for {company}.",
                condition=None,
                next_step_if_true=None,
                next_step_if_false=None
            ),
            PlanStep(
                day=7,
                action="Start Campaign",
                content=f"Add to nurture campaign with regular updates about products and services.",
                condition=None,
                next_step_if_true=None,
                next_step_if_false=None
            )
        ]
    
    elif status == "showed_interest":
        # For companies that showed interest, be more direct
        plan = [
            PlanStep(
                day=0,
                action="Send LinkedIn Connection Request",
                content=f"Connection request to {contact_person.name} mentioning their interest in your website/content.",
                condition="connection request accepted or not",
                next_step_if_true=1,
                next_step_if_false=2
            ),
            PlanStep(
                day=1,
                action="Send LinkedIn Message",
                content=f"Thank for connecting. Mention specific content they viewed and offer a 15-minute call.",
                condition=None,
                next_step_if_true=None,
                next_step_if_false=None
            ),
            PlanStep(
                day=2,
                action="Send Email",
                content=f"Introduction email with specific pain points and solutions for {company}.",
                condition="email responded to or not",
                next_step_if_true=4,
                next_step_if_false=3
            ),
            PlanStep(
                day=5,
                action="Send Email",
                content=f"Follow-up email with case study relevant to {company}'s industry.",
                condition=None,
                next_step_if_true=None,
                next_step_if_false=None
            ),
            PlanStep(
                day=7,
                action="Start Campaign",
                content=f"Add to high-interest nurture campaign.",
                condition=None,
                next_step_if_true=None,
                next_step_if_false=None
            )
        ]
    
    else:  # New companies
        # For new companies, start with LinkedIn research
        plan = [
            PlanStep(
                day=0,
                action="Visit LinkedIn Profile",
                content=f"Research {company} and {contact_person.name} on LinkedIn to gather more information.",
                condition=None,
                next_step_if_true=None,
                next_step_if_false=None
            ),
            PlanStep(
                day=1,
                action="Send LinkedIn Connection Request",
                content=f"Connection request to {contact_person.name} with personalized message about their industry challenges.",
                condition="connection request accepted or not",
                next_step_if_true=2,
                next_step_if_false=3
            ),
            PlanStep(
                day=3,
                action="Send LinkedIn Message",
                content=f"Thank for connecting. Share relevant content and start conversation about their challenges.",
                condition=None,
                next_step_if_true=None,
                next_step_if_false=None
            ),
            PlanStep(
                day=4,
                action="Send Email",
                content=f"Introduction email with value proposition tailored to {company}'s industry.",
                condition="email responded to or not",
                next_step_if_true=5,
                next_step_if_false=4
            ),
            PlanStep(
                day=7,
                action="Send Email",
                content=f"Follow-up email with more specific solutions for {company}.",
                condition=None,
                next_step_if_true=None,
                next_step_if_false=None
            ),
            PlanStep(
                day=10,
                action="Start Campaign",
                content=f"Add to introductory nurture campaign.",
                condition=None,
                next_step_if_true=None,
                next_step_if_false=None
            )
        ]
    
    # Customize plan based on company context if available
    if company_context:
        context = company_context[0]
        
        # Get pain points and solutions for personalization
        pain_points = context.get("painPoints", [])
        if pain_points:
            # Use pain points to customize outreach messages
            for step in plan:
                if "Send Email" in step.action or "Send LinkedIn Message" in step.action:
                    # Add personalized pain point and solution to the message
                    pain_point = pain_points[0] if pain_points else None
                    if pain_point:
                        step.content += f" Address pain point: {pain_point.get('pain', '')} with solution: {pain_point.get('solution', '')}"
    
    return plan

async def generate_recommendations(company: str, status: str, company_context: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations for approach based on company information"""
    recommendations = []
    
    # Default recommendations
    recommendations.append(f"Research {company} on LinkedIn before initial contact")
    recommendations.append("Prepare for objections about pricing and implementation time")
    
    # Add specific recommendations based on company context
    if company_context:
        context = company_context[0]
        
        # Use tone of voice from context
        tone = context.get("toneOfVoice", "")
        if tone:
            recommendations.append(f"Use this tone in communications: {tone}")
        
        # Add sales pitch as a recommendation
        sales_pitch = context.get("salesPitch", "")
        if sales_pitch:
            recommendations.append(f"Use this key message: {sales_pitch}")
        
        # Add industry-specific recommendations
        industries = context.get("targetIndustries", [])
        if industries:
            recommendations.append(f"Tailor messaging to these industries: {', '.join(industries[:3])}")
        
        # Add language recommendation
        language = context.get("language", "")
        if language:
            recommendations.append(f"Communicate in {language}")
    
    return recommendations

@app.post("/api/planner", response_model=PlannerResponse)
async def planner(request: PlannerRequest):
    try:
        company = request.company
        
        # Get company context information
        company_context = await get_company_context()
        
        # Check company status
        status_info = await check_company_status(company)
        status = status_info["status"]
        
        # Identify best contact person
        contact_person = await identify_best_contact(company, company_context)
        
        # Create outreach plan
        plan = await create_outreach_plan(company, status, contact_person, company_context)
        
        # Generate recommendations
        recommendations = await generate_recommendations(company, status, company_context)
        
        # Create response
        response = PlannerResponse(
            company=company,
            company_status=status,
            contact_person=contact_person,
            plan=plan,
            recommendations=recommendations
        )
        
        return response
    
    except Exception as e:
        error_msg = f"Error processing planner request: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

# Health check endpoint
@app.get("/api/planner/health")
async def health_check():
    return {
        "status": "ok", 
        "message": "Planner API is running",
        "agent_imports_successful": agent_imports_successful
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Use a different port than the main API
