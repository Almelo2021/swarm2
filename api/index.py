from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import sys
import os
from pathlib import Path
import httpx
from typing import List, Dict, Any, Optional

# Add parent directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import error handling wrapper
def import_agent_components():
    try:
        # Try to import from specific module paths
        from agents import Agent, WebSearchTool, Runner
        
        from tools import (
            get_existing_leads,
            search_hubspot_contacts,
            get_website_visits,
            get_crm_activities
        )
        
        # Create the agent
        agent = Agent(
            name="Assistant",
            model="gpt-4.1",
            tools=[
                WebSearchTool(),
                get_existing_leads,
                search_hubspot_contacts,
                get_website_visits,
                get_crm_activities
            ],
        )
        
        return agent, Runner
    except ImportError as e:
        print(f"Error importing agent components: {str(e)}")
        raise e

# Try to import agent components
try:
    agent, Runner = import_agent_components()
    agent_initialized = True
except Exception as e:
    print(f"Failed to initialize agent: {str(e)}")
    agent_initialized = False

app = FastAPI()

class QueryRequest(BaseModel):
    company: str
    query: str

# Main route with HTML response
@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Sales Intelligence Agent API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 40px;
                    line-height: 1.6;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                }
                h1 {
                    color: #333;
                }
                .endpoint {
                    background-color: #f4f4f4;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                code {
                    background-color: #e7e7e7;
                    padding: 3px 5px;
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Hello World! 👋</h1>
                <p>Welcome to the Sales Intelligence Agent API</p>
                
                <div class="endpoint">
                    <h2>API Endpoint</h2>
                    <p>Send POST requests to <code>/api</code> with the following JSON structure:</p>
                    <pre><code>{
  "company": "example.com",
  "query": "Do they have a marketing team?"
}</code></pre>
                </div>
                
                <div class="endpoint">
                    <h2>Planner API</h2>
                    <p>Send POST requests to <code>/api/planner</code> with the following JSON structure:</p>
                    <pre><code>{
  "company": "example.com"
}</code></pre>
                </div>
                
                <div class="endpoint">
                    <h2>Health Check</h2>
                    <p>Check API status at <a href="/api/health">/api/health</a></p>
                </div>
            </div>
        </body>
    </html>
    """
    return html_content

@app.post("/api")
async def process_query(request: QueryRequest):
    if not agent_initialized:
        return {"error": "OpenAI Agents SDK not properly initialized. Check server logs for details."}
    
    try:
        formatted_query = f"Company: {request.company} Query: {request.query}"
        print(f"Processing query: {formatted_query}")
        
        # If query is about website visits, log extra info
        if "visit" in request.query.lower() or "website" in request.query.lower():
            print(f"Query about website visits detected for domain: {request.company}")
        
        result = await Runner.run(agent, formatted_query)
        print(f"Query completed successfully")
        return {"result": result.final_output}
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg}

# Add a health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok", 
        "message": "API is running",
        "agent_initialized": agent_initialized
    }

# ------------------------------------------------------
# PLANNER API INTEGRATION
# ------------------------------------------------------

# Define Planner models
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
    if agent_initialized:
        try:
            # Safely access the agent's tools
            if 'agent' in globals() and hasattr(agent, 'tools'):
                # Find the tools we need
                leads_tool = None
                visits_tool = None
                activities_tool = None
                
                for tool in agent.tools:
                    if hasattr(tool, 'name'):
                        if tool.name == 'get_existing_leads':
                            leads_tool = tool
                        elif tool.name == 'get_website_visits':
                            visits_tool = tool
                        elif tool.name == 'get_crm_activities':
                            activities_tool = tool
                
                # Use the get_existing_leads tool if available
                if leads_tool:
                    try:
                        # Access the underlying function through the tool
                        if hasattr(leads_tool, 'function'):
                            leads_result = await leads_tool.function({"query": {"company": company, "checkExists": True}})
                        elif hasattr(leads_tool, '__call__'):
                            leads_result = await leads_tool({"query": {"company": company, "checkExists": True}})
                        else:
                            print(f"Warning: leads_tool has no callable method")
                            leads_result = None
                            
                        # Check if company exists in leads
                        if leads_result and isinstance(leads_result, dict) and leads_result.get("exists", False):
                            status = "existing_lead"
                            details["leads"] = leads_result
                        elif leads_result and isinstance(leads_result, list) and len(leads_result) > 0:
                            status = "existing_lead"
                            details["leads"] = leads_result
                    except Exception as lead_error:
                        print(f"Warning when checking for existing leads: {str(lead_error)}")
                
                # Check website visits if tool is available
                if visits_tool:
                    try:
                        # Access the underlying function through the tool
                        if hasattr(visits_tool, 'function'):
                            visits_result = await visits_tool.function({"domain": company})
                        elif hasattr(visits_tool, '__call__'):
                            visits_result = await visits_tool({"domain": company})
                        else:
                            print(f"Warning: visits_tool has no callable method")
                            visits_result = None
                            
                        if visits_result and len(visits_result) > 0:
                            status = "showed_interest"
                            details["visits"] = visits_result
                    except Exception as visit_error:
                        print(f"Warning when checking website visits: {str(visit_error)}")
                
                # Check CRM activities if tool is available
                if activities_tool:
                    try:
                        # Access the underlying function through the tool
                        if hasattr(activities_tool, 'function'):
                            activities = await activities_tool.function({"company": company})
                        elif hasattr(activities_tool, '__call__'):
                            activities = await activities_tool({"company": company})
                        else:
                            print(f"Warning: activities_tool has no callable method")
                            activities = None
                            
                        if activities and len(activities) > 0:
                            details["activities"] = activities
                            # If they have activities but weren't marked as existing lead, mark them now
                            if status == "new":
                                status = "existing_lead"
                    except Exception as activity_error:
                        print(f"Warning when checking CRM activities: {str(activity_error)}")
            else:
                print("Warning: agent or agent.tools not available")
                
        except Exception as e:
            print(f"Error checking company status: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # If we couldn't determine status through tools, fallback to simple logic
    # This ensures the API continues to work even if tools are unavailable
    
    return {
        "status": status,
        "details": details
    }

async def identify_best_contact(company: str, company_context: List[Dict[str, Any]]) -> ContactPerson:
    """Find the best person to contact based on company and context"""
    # First try to find existing contacts in Hubspot
    contact_name = f"Decision Maker at {company}"
    contact_role = "Sales Manager"  # Default role
    contact_reason = "Default target persona based on company profile"
    contact_linkedin = None
    contact_email = None
    
    # If agent imports were successful, try to use the tools
    if agent_initialized:
        try:
            # Safely access the agent's tools
            if 'agent' in globals() and hasattr(agent, 'tools'):
                # Find the contacts tool
                contacts_tool = None
                
                for tool in agent.tools:
                    if hasattr(tool, 'name') and tool.name == 'search_hubspot_contacts':
                        contacts_tool = tool
                        break
                
                # Use the contacts tool if available
                if contacts_tool:
                    try:
                        # Access the underlying function through the tool
                        if hasattr(contacts_tool, 'function'):
                            contacts = await contacts_tool.function({"company": company})
                        elif hasattr(contacts_tool, '__call__'):
                            contacts = await contacts_tool({"company": company})
                        else:
                            print(f"Warning: contacts_tool has no callable method")
                            contacts = None
                        
                        if contacts and len(contacts) > 0:
                            # Find the best contact based on job title
                            best_contact = None
                            
                            # Get target job titles from context
                            target_titles = []
                            if company_context and len(company_context) > 0:
                                target_titles = company_context[0].get("targetJobTitles", [])
                            
                            if not target_titles:
                                target_titles = ["Sales Manager", "Marketing Manager", "CEO", "Founder", "Owner"]
                            
                            # Look for contacts with matching job titles
                            for contact in contacts:
                                job_title = contact.get("jobTitle", "").lower()
                                
                                # Check if this contact's job title matches any target title
                                for target in target_titles:
                                    if target.lower() in job_title:
                                        best_contact = contact
                                        break
                                
                                if best_contact:
                                    break
                            
                            # If no match found, just use the first contact
                            if not best_contact and contacts:
                                best_contact = contacts[0]
                            
                            # If we found a contact, use their info
                            if best_contact:
                                contact_name = f"{best_contact.get('firstName', '')} {best_contact.get('lastName', '')}".strip()
                                if not contact_name:
                                    contact_name = best_contact.get("email", f"Contact at {company}")
                                
                                contact_role = best_contact.get("jobTitle", "Unknown Role")
                                contact_reason = "Existing contact in Hubspot"
                                contact_email = best_contact.get("email")
                                contact_linkedin = best_contact.get("linkedinUrl")
                    except Exception as contact_error:
                        print(f"Error when searching contacts: {str(contact_error)}")
                        import traceback
                        traceback.print_exc()
            else:
                print("Warning: agent or agent.tools not available for contact search")
        except Exception as e:
            print(f"Error finding contacts: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # If no contacts found, use company context to determine target personas
    if contact_name == f"Decision Maker at {company}" and company_context:
        # Get the first context (assuming it's the most relevant)
        context = company_context[0]
        
        # Extract target job titles from context
        target_job_titles = context.get("targetJobTitles", ["Sales Manager", "Marketing Manager"])
        
        # Use the first job title
        best_role = target_job_titles[0] if target_job_titles else "Sales Manager"
        contact_role = best_role
        contact_reason = f"This role is a primary target based on your company profile: {best_role} is in your target job titles"
    
    return ContactPerson(
        name=contact_name,
        role=contact_role,
        reason=contact_reason,
        linkedin_url=contact_linkedin,
        email=contact_email
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

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
