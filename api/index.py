from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Type
import asyncio
import sys
import os
from pathlib import Path

# Third‑party
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
#  Local imports & dynamic path handling
# ──────────────────────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))

try:
    from agents import Agent, WebSearchTool, Runner, ModelSettings
    #from tools import (
        #get_existing_leads,
        #search_hubspot_contacts,
        #get_website_visits,
        #get_crm_activities,
    #)
except ImportError as e:
    # Falling back makes local development easier if the Agents SDK or tools
    # are not present in the environment.
    print(f"Error importing agent components: {e}")
    Agent = WebSearchTool = Runner = None  # type: ignore

# Import the LangGraph chatbot
try:
    from graph import graph, format_final_output
    from langchain_core.messages import HumanMessage
    graph_available = True
except ImportError as e:
    print(f"Error importing graph: {e}")
    graph = None
    format_final_output = None
    graph_available = False

# ──────────────────────────────────────────────────────────────────────────────
#  OpenAI client (reads key from ENV)
# ──────────────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ──────────────────────────────────────────────────────────────────────────────
#  Structured‑output Pydantic models
# ──────────────────────────────────────────────────────────────────────────────
class _BaseConfig:
    extra = "forbid"


class _SourcesMixin(BaseModel):
    """Optional list of URLs or free‑text citations supporting the answer."""

    sources: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of source URLs or citations that back up the answer. "
            "Omit the field when no sources are available."
        ),
    )

    class Config(_BaseConfig):
        pass


class IntegerOutput(_SourcesMixin):
    answer: int

    class Config(_BaseConfig):
        pass


class FloatOutput(_SourcesMixin):
    answer: float

    class Config(_BaseConfig):
        pass


class BooleanOutput(_SourcesMixin):
    answer: bool

    class Config(_BaseConfig):
        pass


class StringOutput(_SourcesMixin):
    answer: str

    class Config(_BaseConfig):
        pass


class StringListOutput(_SourcesMixin):
    answer: List[str]

    class Config(_BaseConfig):
        pass


class KVPair(BaseModel):
    key: str
    value: str  # keep values as strings to satisfy strict schema rules

    class Config(_BaseConfig):
        pass


class KVListOutput(_SourcesMixin):
    """Dictionary represented as a list of key/value string pairs."""

    answer: List[KVPair]

    class Config(_BaseConfig):
        pass


# New models for the sheet endpoint
class SheetRequest(BaseModel):
    query: str
    model: Optional[str] = "openai:gpt-4.1"
    max_search_results: Optional[int] = 2

    class Config(_BaseConfig):
        pass


OUTPUT_MODELS: Dict[str, Type[BaseModel]] = {
    "integer": IntegerOutput,
    "float": FloatOutput,
    "boolean": BooleanOutput,
    "string": StringOutput,
    "string_list": StringListOutput,
    "dict": KVListOutput,
}

# ──────────────────────────────────────────────────────────────────────────────
#  FastAPI setup
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI()


class QueryRequest(BaseModel):
    company: str
    query: str
    outputType: Optional[str] = None  # e.g. "integer", "dict", "boolean", ...

    def output_type_normalised(self) -> Optional[str]:
        return self.outputType.lower() if self.outputType else None


class BulkQuery(BaseModel):
    companies: List[str]
    query: str
    outputType: Optional[str] = None

    def output_type_normalised(self) -> Optional[str]:
        return self.outputType.lower() if self.outputType else None


# ──────────────────────────────────────────────────────────────────────────────
#  Initialise base Agent (if SDK present)
# ──────────────────────────────────────────────────────────────────────────────
if Agent is not None:
    base_agent = Agent(
        name="Assistant",
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0.5),
        instructions=(
            "Always log the search phrases you used.\n\n"
            "Tip: sometimes when you search for a company's vacancy for a webdesigner you find nothing, "
            "but when you search for their vacancies/careers page you find the webdesigner listing there."
        ),
        tools=[
            WebSearchTool(),
            #get_existing_leads,
            #search_hubspot_contacts,
            #get_website_visits,
            #get_crm_activities,
        ],
    )
    agent_initialized = True
else:
    base_agent = None  # type: ignore
    agent_initialized = False

# ──────────────────────────────────────────────────────────────────────────────
#  Helper: agent‑driven structured query
# ──────────────────────────────────────────────────────────────────────────────
async def run_agent_structured(prompt: str, model_cls: Type[BaseModel]):
    """Run the agent and ensure the final output matches `model_cls`."""

    if not agent_initialized:
        raise HTTPException(status_code=500, detail="Agents SDK not initialised.")

    # Add extra formatting guardrails only for plain‑string responses.
    extra_instr = (
        "\n\nWhen answering as a *string* you must:\n"
        "• Use plain text (no markdown headings, bold, lists, links).\n"
        "• Escape every newline as \\n (two characters).\n"
        "• Do not start the string with a newline.\n\n"
        "When asked for example for the link of a LinkedIn page, do not add superfluous text. Return the link and nothing more."
        if model_cls is StringOutput
        else ""
    )

    agent = base_agent.clone(output_type=model_cls, instructions=base_agent.instructions + extra_instr)

    try:
        result = await Runner.run(agent, prompt)
        final = result.final_output
        if isinstance(final, BaseModel):
            return final.model_dump(mode="json")
        return final
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent processing error: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: run LangGraph chatbot
# ──────────────────────────────────────────────────────────────────────────────
async def run_graph_chatbot(query: str, model: str = "openai:gpt-4.1", max_search_results: int = 2):
    """Run the LangGraph chatbot with the given query."""
    
    if not graph_available:
        raise HTTPException(status_code=500, detail="LangGraph chatbot not available.")
    
    try:
        # Create the initial state with the user's message
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Configuration for the graph
        config = {
            "configurable": {
                "model": model,
                "max_search_results": max_search_results
            }
        }
        
        # Run the graph
        final_state = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: graph.invoke(initial_state, config)
        )
        
        # Format the output
        formatted_output = format_final_output(final_state)
        
        return final_state
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph processing error: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
#  Landing page
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    types_list = ", ".join(OUTPUT_MODELS.keys())

    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Sales Intelligence Agent API</title>
            <style>
                body{{font-family:Arial,Helvetica,sans-serif;max-width:800px;margin:40px auto;line-height:1.6}}
                pre{{background:#f4f4f4;padding:16px;border-radius:8px}}
                code{{background:#e7e7e7;padding:2px 4px;border-radius:4px}}
            </style>
        </head>
        <body>
            <h1>Hello World 👋</h1>
            <p>Welcome to the Sales Intelligence Agent API.</p>
            <h2>Single query → POST /api</h2>
            <pre><code>{{"company": "example.com", "query": "…", "outputType": "string"}}</code></pre>

            <h2>Bulk query → POST /api/bulk</h2>
            <pre><code>{{
  "companies": ["a.com", "b.com"],
  "query": "…",
  "outputType": "dict"  // optional – {types_list}
}}</code></pre>

            <h2>LangGraph Chatbot → POST /api/sheet</h2>
            <pre><code>{{
  "query": "What is the weather like today?",
  "model": "openai:gpt-4.1",  // optional
  "max_search_results": 2    // optional
}}</code></pre>

            <h2>Health Check</h2>
            <p><a href="/api/health">/api/health</a></p>
        </body>
    </html>
    """


# ──────────────────────────────────────────────────────────────────────────────
#  Single‑company endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api")
async def process_query(req: QueryRequest):
    prompt = f"Company: {req.company} | Query: {req.query}"

    # Structured requests – validate via the cloned agent.
    if req.outputType:
        output_type_key = req.output_type_normalised()
        model_cls = OUTPUT_MODELS.get(output_type_key)
        if model_cls is None:
            raise HTTPException(status_code=400, detail=f"Unsupported outputType '{req.outputType}'.")
        return {"result": await run_agent_structured(prompt, model_cls)}

    # Unstructured path – still routed through the base agent.
    if not agent_initialized:
        raise HTTPException(status_code=500, detail="Agents SDK not initialised.")

    try:
        result = await Runner.run(base_agent, prompt)
        return {"result": result.final_output}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent processing error: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
#  Bulk endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/bulk")
async def bulk_process(req: BulkQuery):
    async def handle_company(company: str):
        prompt = f"Company: {company} | Query: {req.query}"
        try:
            if req.outputType:
                output_type_key = req.output_type_normalised()
                model_cls = OUTPUT_MODELS.get(output_type_key)
                if model_cls is None:
                    raise ValueError(f"Unsupported outputType '{req.outputType}'.")
                res = await run_agent_structured(prompt, model_cls)
            else:
                if not agent_initialized:
                    raise RuntimeError("Agents SDK not initialised.")
                res = (await Runner.run(base_agent, prompt)).final_output
            return {"company": company, "result": res}
        except Exception as exc:
            return {"company": company, "error": str(exc)}

    tasks = [handle_company(c) for c in req.companies]
    results = await asyncio.gather(*tasks)
    return {"results": results}


# ──────────────────────────────────────────────────────────────────────────────
#  LangGraph Chatbot endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/sheet")
async def process_sheet_query(req: SheetRequest):
    """Process a query using the LangGraph chatbot with web search capabilities."""
    
    if not graph_available:
        raise HTTPException(
            status_code=500, 
            detail="LangGraph chatbot not available. Make sure graph.py is properly configured."
        )
    
    try:
        result = await run_graph_chatbot(
            query=req.query,
            model=req.model or "openai:gpt-4.1",
            max_search_results=req.max_search_results or 2
        )
        return result
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sheet processing error: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
#  Researcher passthrough endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/researcher")
async def process_research_query(req: QueryRequest):
    try:
        from tools2 import researcher
        res = await researcher(req.company)
        return {"result": res}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Researcher error: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
#  Health check
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok", 
        "agent_initialized": agent_initialized,
        "graph_available": graph_available
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Local dev entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
