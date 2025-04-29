from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import sys
import os
from pathlib import Path
import json

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
            get_crm_activities,
            company_context,
            researcher
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
                get_crm_activities,
                company_context,
                researcher
            ],
        )
        
        return agent, Runner, researcher
    except ImportError as e:
        print(f"Error importing agent components: {str(e)}")
        raise e

# Try to import agent components
try:
    agent, Runner, researcher_function = import_agent_components()
    agent_initialized = True
except Exception as e:
    print(f"Failed to initialize agent: {str(e)}")
    agent_initialized = False
    researcher_function = None

app = FastAPI()

class QueryRequest(BaseModel):
    company: str
    query: str

class ResearcherRequest(BaseModel):
    domain: str

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
                    <h2>Researcher Endpoint</h2>
                    <p>Send POST requests to <code>/api/researcher</code> with the following JSON structure:</p>
                    <pre><code>{
  "domain": "example.com"
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

@app.post("/api/researcher")
async def run_researcher(request: ResearcherRequest):
    if not agent_initialized or researcher_function is None:
        return {"error": "Researcher function not properly initialized. Check server logs for details."}
    
    try:
        print(f"Running researcher for domain: {request.domain}")
        
        # Call the researcher function directly
        result = await researcher_function(domain=request.domain)
        
        # Parse the result to return as JSON
        try:
            parsed_result = json.loads(result)
            print(f"Researcher completed successfully for domain: {request.domain}")
            return parsed_result
        except json.JSONDecodeError:
            # If the result is not valid JSON, return it as a string
            print(f"Researcher completed but returned non-JSON result")
            return {"result": result}
            
    except Exception as e:
        error_msg = f"Error running researcher: {str(e)}"
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
        "agent_initialized": agent_initialized,
        "researcher_available": researcher_function is not None
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
