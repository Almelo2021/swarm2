from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Type
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
    from agents import Agent, WebSearchTool, Runner
    from tools import (
        get_existing_leads,
        search_hubspot_contacts,
        get_website_visits,
        get_crm_activities,
    )
except ImportError as e:
    print(f"Error importing agent components: {e}")
    Agent = WebSearchTool = Runner = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
#  OpenAI client (reads key from ENV)
# ──────────────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ──────────────────────────────────────────────────────────────────────────────
#  Structured‑output Pydantic models
# ──────────────────────────────────────────────────────────────────────────────
class _BaseConfig:
    extra = "forbid"


class IntegerOutput(BaseModel):
    answer: int

    class Config(_BaseConfig):
        pass


class FloatOutput(BaseModel):
    answer: float

    class Config(_BaseConfig):
        pass


class StringOutput(BaseModel):
    answer: str

    class Config(_BaseConfig):
        pass


class StringListOutput(BaseModel):
    answer: List[str]

    class Config(_BaseConfig):
        pass


class KVPair(BaseModel):
    key: str
    value: str  # keep values as strings to satisfy strict schema rules

    class Config(_BaseConfig):
        pass


class KVListOutput(BaseModel):
    """Dictionary represented as a list of key/value string pairs."""

    answer: List[KVPair]

    class Config(_BaseConfig):
        pass


OUTPUT_MODELS: Dict[str, Type[BaseModel]] = {
    "integer": IntegerOutput,
    "float": FloatOutput,
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
    outputType: Optional[str] = None  # e.g. "integer", "dict", ...

    def output_type_normalised(self) -> Optional[str]:
        return self.outputType.lower() if self.outputType else None


class BulkQuery(BaseModel):
    companies: List[str]
    query: str
    outputType: Optional[str] = None

    def output_type_normalised(self) -> Optional[str]:
        return self.outputType.lower() if self.outputType else None


# ──────────────────────────────────────────────────────────────────────────────
#  Initialise Agent (if SDK present)
# ──────────────────────────────────────────────────────────────────────────────
if Agent is not None:
    agent = Agent(
        name="Assistant",
        model="gpt-4.1",
        tools=[
            WebSearchTool(),
            get_existing_leads,
            search_hubspot_contacts,
            get_website_visits,
            get_crm_activities,
        ],
    )
    agent_initialized = True
else:
    agent_initialized = False


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

            <h2>Health Check</h2>
            <p><a href="/api/health">/api/health</a></p>
        </body>
    </html>
    """


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: run Structured‑Output query
# ──────────────────────────────────────────────────────────────────────────────
async def run_structured_query(prompt: str, output_type: str):
    model_cls = OUTPUT_MODELS.get(output_type)
    if model_cls is None:
        raise HTTPException(status_code=400, detail=f"Unsupported outputType '{output_type}'.")

    try:
        response = client.responses.parse(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": "You are a helpful assistant. Follow the schema strictly."},
                {"role": "user", "content": prompt},
            ],
            text_format=model_cls,
            temperature=0,
        )
        return response.output_parsed.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Structured query failed: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
#  Single‑company endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api")
async def process_query(req: QueryRequest):
    prompt = f"Company: {req.company} | Query: {req.query}"

    if req.outputType:
        return {"result": await run_structured_query(prompt, req.output_type_normalised())}

    if not agent_initialized:
        raise HTTPException(status_code=500, detail="Agents SDK not initialised.")

    try:
        result = await Runner.run(agent, prompt)
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
                res = await run_structured_query(prompt, req.output_type_normalised())
            else:
                if not agent_initialized:
                    raise RuntimeError("Agents SDK not initialised.")
                res = (await Runner.run(agent, prompt)).final_output
            return {"company": company, "result": res}
        except Exception as exc:
            return {"company": company, "error": str(exc)}

    tasks = [handle_company(c) for c in req.companies]
    results = await asyncio.gather(*tasks)
    return {"results": results}


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
    return {"status": "ok", "agent_initialized": agent_initialized}


# ──────────────────────────────────────────────────────────────────────────────
#  Local dev entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
