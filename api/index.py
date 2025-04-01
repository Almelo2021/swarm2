from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from tools import (
        get_existing_leads,
        search_hubspot_contacts,
        get_website_visits,
        get_crm_activities
    )
    from agents import Agent, WebSearchTool, Runner
    
    # Create the agent
    agent = Agent(
        name="Assistant",
        tools=[
            WebSearchTool(),
            get_existing_leads,
            search_hubspot_contacts,
            get_website_visits,
            get_crm_activities
        ],
    )
except ImportError as e:
    print(f"Warning: Some imports failed - {str(e)}")
    # We'll still allow the app to start for basic functionality

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
        result = await Runner.run(agent, formatted_query)
        return {"result": result.final_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Add a health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "API is running"}

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"An error occurred: {str(e)}"}
    )

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
