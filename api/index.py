from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import sys
import os
import importlib.util
from pathlib import Path

app = FastAPI()

# Basic response for the main page
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>Agent API</title></head>
        <body>
            <h1>Hello World! 👋</h1>
            <p>API is running, but the agent is still being configured.</p>
            <p>Check <a href="/api/debug">/api/debug</a> for more information.</p>
        </body>
    </html>
    """

# Add a debug endpoint to check the environment
@app.get("/api/debug")
async def debug_info():
    # Check what modules are available
    installed_packages = []
    try:
        import pkg_resources
        installed_packages = [d.project_name for d in pkg_resources.working_set]
    except:
        pass
    
    # Check system paths
    paths = sys.path
    
    # Check environment variables
    env_vars = {k: v for k, v in os.environ.items() if not k.startswith('VERCEL') and not k.startswith('AWS')}
    
    # List files in the current directory
    try:
        files = os.listdir('.')
    except:
        files = ["Unable to list files"]
        
    # Try to find the agents module
    agents_found = "agents" in installed_packages or any("agents" in p for p in paths)
    
    return {
        "status": "debugging",
        "installed_packages": installed_packages,
        "paths": paths,
        "files": files,
        "env_vars": env_vars,
        "agents_found": agents_found
    }

# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "API is running but agent not initialized"}
