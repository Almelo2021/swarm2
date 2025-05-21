from fastapi import FastAPI, HTTPException, Request
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
        #get_existing_leads,
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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # keep the client for future use

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
#  Helper: agent‑driven structured query
# ──────────────────────────────────────────────────────────────────────────────
async def run_agent_structured(prompt: str, output_type: str):
    """Execute the agent *with* tool access and validate the final output against
    a strict Pydantic schema. If the returned JSON does not match, an HTTP 500
    is raised so the caller immediately sees the mismatch.
    """
    if not agent_initialized:
        raise HTTPException(status_code=500, detail="Agents SDK not initialised.")

    model_cls = OUTPUT_MODELS.get(output_type)
    if model_cls is None:
        raise HTTPException(status_code=400, detail=f"Unsupported outputType '{output_type}'.")

    # Tell the assistant to answer exclusively with JSON matching the schema.
    schema_json = model_cls.schema_json(indent=2)

    agent_prompt = (
        f"{prompt}\n\n"
        "When answering, respond ONLY with a JSON object that follows *exactly* the schema below.\n"
        "Do not add markdown code fences, do not include any explanatory text.\n"
        f"Schema:\n{schema_json}"
    )

    try:
        result = await Runner.run(agent, agent_prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent processing error: {exc}") from exc

    output_text = result.final_output.strip()

    try:
        parsed = model_cls.model_validate_json(output_text)
        return parsed.model_dump(mode="json")
    except Exception as exc:
        # Provide visibility on what the model actually produced for easier debugging.
        detail = (
            "Structured response did not match the expected schema. "
            f"Raw output was: {output_text}. Error: {exc}"
        )
        raise HTTPException(status_code=500, detail=detail) from exc


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
#  Single‑company endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api")
async def process_query(req: QueryRequest):
    print("fuck de opps")
    prompt = f"Company: {req.company} | Query: {req.query}"

    # Route *all* requests through the Agent so tools are always available.
    if req.outputType:
        return {"result": await run_agent_structured(prompt, req.output_type_normalised())}

    # Unstructured – we still use the agent but we do not validate the output.
    if not agent_initialized:
        raise HTTPException(status_code=500, detail="Agents SDK not initialised.")

    try:
        result = await Runner.run(agent, prompt)
        print(result.new_items)
        print(result.last_agent)
        print(result.raw_responses)
        print(result._last_agent)
        print("swen")
        print(result.final_output)
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
                res = await run_agent_structured(prompt, req.output_type_normalised())
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
