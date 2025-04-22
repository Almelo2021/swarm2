from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import sys
import os
import asyncio
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import error handling wrapper
def import_agent_components():
    try:
        # Import necessary components
        from agents import Agent, WebSearchTool, Runner, trace
        
        from tools import (
            get_existing_leads,
            search_hubspot_contacts,
            get_website_visits,
            get_crm_activities
        )
        
        # Create specialized agents for different tasks
        
        # 1. Web search agent with full model support
        web_search_agent = Agent(
            name="web_search_agent",
            instructions="You perform web searches to find information relevant to the query about the specified company.",
            tools=[WebSearchTool()],
            model="o4"  # Use full model for web search
        )
        
        # 2. CRM data agent (using o4-mini)
        crm_agent = Agent(
            name="crm_agent",
            instructions="You analyze CRM data for the specified company to extract relevant insights.",
            tools=[
                search_hubspot_contacts,
                get_crm_activities
            ],
            model="o4-mini"
        )
        
        # 3. Lead finder agent (using o4-mini)
        leads_agent = Agent(
            name="leads_agent",
            instructions="You find leads at the specified company and analyze their relevance.",
            tools=[get_existing_leads],
            model="o4-mini"
        )
        
        # 4. Website analytics agent (using o4-mini)
        website_agent = Agent(
            name="website_agent",
            instructions="You analyze website visit data for the specified company.",
            tools=[get_website_visits],
            model="o4-mini"
        )
        
        # 5. Main orchestrator agent that uses all specialized agents as tools
        orchestrator_agent = Agent(
            name="orchestrator_agent",
            instructions=(
                "You are a sales intelligence assistant that helps analyze companies. "
                "Based on the query, determine which specialized tools to use. "
                "For questions requiring internet information, use the web search tool. "
                "For questions about CRM data, use the CRM tools. "
                "For questions about leads at a company, use the leads finder tool. "
                "For questions about website visits, use the website analytics tool. "
                "Always be thorough and provide comprehensive answers."
            ),
            tools=[
                web_search_agent.as_tool(
                    tool_name="web_search",
                    tool_description="Search the web for information about the company"
                ),
                crm_agent.as_tool(
                    tool_name="analyze_crm_data",
                    tool_description="Analyze CRM data for the company"
                ),
                leads_agent.as_tool(
                    tool_name="find_company_leads",
                    tool_description="Find leads at the specified company"
                ),
                website_agent.as_tool(
                    tool_name="analyze_website_visits",
                    tool_description="Analyze website visit data for the company"
                )
            ],
            model="o4-mini"  # Orchestrator uses o4-mini for efficiency
        )
        
        # 6. Synthesizer agent to ensure high-quality final responses
        synthesizer_agent = Agent(
            name="synthesizer_agent",
            instructions=(
                "You review and synthesize information from various sources to create a "
                "comprehensive, well-structured response. Ensure the final answer directly "
                "addresses the original query and presents information in a clear, concise manner."
            ),
            model="o4-mini"
        )
        
        return orchestrator_agent, synthesizer_agent, Runner
    except ImportError as e:
        print(f"Error importing agent components: {str(e)}")
        raise e

# Try to import agent components
try:
    orchestrator_agent, synthesizer_agent, Runner = import_agent_components()
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
        formatted_query = f"Company: {request.company} Query: {request.query} -- Be as thorough as possible."
        print(f"Processing query: {formatted_query}")
        
        # Run the entire process in a trace for better debugging
        from agents import trace
        with trace("Sales Intelligence Query"):
            # First, use the orchestrator to determine which specialized agents to use
            orchestrator_result = await Runner.run(orchestrator_agent, formatted_query)
            print(f"Orchestrator step completed")
            
            # Then use the synthesizer to create the final response
            synthesizer_result = await Runner.run(
                synthesizer_agent, orchestrator_result.to_input_list()
            )
            print(f"Synthesizer step completed")
            
        print(f"Query completed successfully")
        return {"result": synthesizer_result.final_output}
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
        "agent_initialized": agent_initialized,
        "orchestration": "agents-as-tools",
        "primary_model": "o4-mini",
        "web_search_model": "o4"
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
