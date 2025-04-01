from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import sys
import os
import json
import requests
import asyncio
from typing import List, Optional, Dict, Any

app = FastAPI()

# Since we don't have the agents module, let's implement a simple agent directly
class SimpleAgent:
    def __init__(self, name: str, tools: List[Any] = None):
        self.name = name
        self.tools = tools or []
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    async def run(self, query: str) -> str:
        """Simple implementation to call tools based on the query"""
        result = f"Processing query: {query}"
        
        # Parse company from query format
        company = None
        if query.startswith("Company:"):
            parts = query.split("Query:")
            if len(parts) == 2:
                company = parts[0].replace("Company:", "").strip()
                question = parts[1].strip()
                result += f"\n\nCompany: {company}\nQuestion: {question}"
                
                # Call the relevant tools based on the question
                if "email" in question.lower():
                    result += "\n\nChecking Hubspot contacts..."
                    hubspot_data = await search_hubspot_contacts(company=company)
                    result += f"\n\nHubspot data: {hubspot_data}"
                
                if "lead" in question.lower() or "people" in question.lower():
                    result += "\n\nChecking existing leads..."
                    leads_data = await get_existing_leads(company=company)
                    result += f"\n\nLeads data: {leads_data}"
                
                if "visit" in question.lower() or "website" in question.lower():
                    domain = company if "." in company else f"{company}.com"
                    result += "\n\nChecking website visits..."
                    visits_data = await get_website_visits(domain=domain)
                    result += f"\n\nWebsite visits: {visits_data}"
                
                if "crm" in question.lower() or "activit" in question.lower():
                    domain = company if "." in company else f"{company}.com"
                    result += "\n\nChecking CRM activities..."
                    crm_data = await get_crm_activities(domain=domain)
                    result += f"\n\nCRM activities: {crm_data}"
        
        return result

# Import tool functions from tools.py
from tools import get_existing_leads, search_hubspot_contacts, get_website_visits, get_crm_activities

# Create a simple agent
agent = SimpleAgent(name="Assistant")

class QueryRequest(BaseModel):
    company: str
    query: str

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
    try:
        formatted_query = f"Company: {request.company} Query: {request.query}"
        result = await agent.run(formatted_query)
        return {"result": result}
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)  # Log the error
        return {"error": error_msg}

# Add a health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok", 
        "message": "API is running",
        "tools_available": ["get_existing_leads", "search_hubspot_contacts", 
                           "get_website_visits", "get_crm_activities"]
    }
