from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from tools import (
    get_existing_leads,
    search_hubspot_contacts,
    get_website_visits,
    get_crm_activities
)
from agents import Agent, WebSearchTool, Runner

app = FastAPI()

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

class QueryRequest(BaseModel):
    company: str
    query: str

@app.post("/api")
async def process_query(request: QueryRequest):
    formatted_query = f"Company: {request.company} Query: {request.query}"
    result = await Runner.run(agent, formatted_query)
    return {"result": result.final_output}

# Add a health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"An error occurred: {str(exc)}"}
    )

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
