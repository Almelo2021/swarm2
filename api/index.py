from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import sys
import os
from pathlib import Path

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

# Import tracing components
from agents.tracing import (
    add_trace_processor, set_tracing_export_api_key, 
    trace, force_flush, custom_span
)
try:
    from agents.tracing.exporters import APIExporter
    tracing_available = True
except ImportError:
    print("Warning: Tracing exporters not available")
    tracing_available = False

# Try to import agent components
try:
    agent, Runner = import_agent_components()
    agent_initialized = True
except Exception as e:
    print(f"Failed to initialize agent: {str(e)}")
    agent_initialized = False

app = FastAPI()

# Set up tracing if agent is initialized
if agent_initialized and tracing_available:
    try:
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            set_tracing_export_api_key(api_key)
            add_trace_processor(APIExporter())
            print("Tracing initialized successfully")
        else:
            print("No OpenAI API key found for tracing")
    except Exception as e:
        print(f"Failed to initialize tracing: {str(e)}")

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
    
    # Create a trace for this API call
    with trace(workflow_name="sales_intelligence", 
               metadata={"company": request.company, "query": request.query}) as current_trace:
        try:
            formatted_query = f"Company: {request.company} Query: {request.query}"
            print(f"Processing query: {formatted_query}")
            
            # If query is about website visits, add a custom span for tracking
            if "visit" in request.query.lower() or "website" in request.query.lower():
                with custom_span(name="website_visits_query", 
                                data={"domain": request.company}) as visit_span:
                    print(f"Query about website visits detected for domain: {request.company}")
                    result = await Runner.run(agent, formatted_query)
            else:
                result = await Runner.run(agent, formatted_query)
            
            print(f"Query completed successfully")
            
            # Explicitly flush traces to ensure they're sent to the backend
            try:
                force_flush()
            except Exception as e:
                print(f"Warning: Failed to flush traces: {str(e)}")
                
            return {"result": result.final_output}
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Try to flush traces even on error
            try:
                force_flush()
            except:
                pass
                
            return {"error": error_msg}

# Add a health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok", 
        "message": "API is running",
        "agent_initialized": agent_initialized,
        "tracing_available": tracing_available
    }

# Add shutdown event handler to flush any pending traces
@app.on_event("shutdown")
def shutdown_event():
    if agent_initialized and tracing_available:
        try:
            print("Flushing any pending traces on shutdown...")
            force_flush()
            print("Trace flush complete")
        except Exception as e:
            print(f"Warning: Failed to flush traces on shutdown: {str(e)}")

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
