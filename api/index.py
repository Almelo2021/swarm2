from __future__ import annotations

"""Sales‑Intelligence Agent API
───────────────────────────────
Rewritten so that *all* requests (structured or free‑form) go through the
Agents SDK, letting the model call tools such as `search_hubspot_contacts`.

If the client specifies an `outputType`, the agent is instructed to return a
JSON object conforming to the matching Pydantic schema.  The server validates
that output before returning it to the caller.

Optionally, the client can set `includeSources=true` to receive any tool
citations the agent emitted (e.g. links returned by `WebSearchTool`).  A tool
can expose its citations by attaching them to the `sources` field of its
return value, or by using the standard `.sources` attribute in the Agents SDK.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

# ──────────────────────────────────────────────────────────────────────────────
#  Dynamic import of local packages
# ──────────────────────────────────────────────────────────────────────────────

sys.path.append(str(Path(__file__).parent.parent))
try:
    from agents import Agent, Runner, WebSearchTool  # noqa: WPS433 (import outside top‑level)
    from tools import (
        get_existing_leads,
        get_crm_activities,
        get_website_visits,
        search_hubspot_contacts,
    )
except ImportError as exc:  # pragma: no cover
    print(f"Warning: Agent SDK not available: {exc}")
    Agent = WebSearchTool = Runner = None  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────────────
#  Structured‑output Pydantic models
# ──────────────────────────────────────────────────────────────────────────────

class _BaseConfig:  # noqa: WPS110 (name < 3 chars)
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
    value: str

    class Config(_BaseConfig):
        pass


class KVListOutput(BaseModel):
    """Dictionary represented as a list of key/value pairs."""

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

app = FastAPI(title="Sales‑Intelligence Agent API", version="2.0")


class QueryRequest(BaseModel):
    company: str
    query: str
    outputType: Optional[str] = None  # noqa: N815  (keep camelCase for client)
    includeSources: Optional[bool] = False  # noqa: N815

    def output_type_normalised(self) -> Optional[str]:
        return self.outputType.lower() if self.outputType else None


class BulkQuery(BaseModel):
    companies: List[str]
    query: str
    outputType: Optional[str] = None
    includeSources: Optional[bool] = False

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
    agent_initialised = True
else:  # pragma: no cover
    agent_initialised = False


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: build schema instruction
# ──────────────────────────────────────────────────────────────────────────────

def _schema_instruction(model_cls: Type[BaseModel]) -> str:
    """Return a prompt fragment instructing the assistant to output JSON."""
    schema = json.dumps(model_cls.model_json_schema()["properties"], indent=2)
    return (
        "Return **only** a JSON object fulfilling this schema (no markdown, no code‑block):\n"
        f"```json\n{schema}\n```"
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: run an Agent query and (optionally) validate structured output
# ──────────────────────────────────────────────────────────────────────────────

async def _run_agent(
    prompt: str,
    *,
    output_type: Optional[str] = None,
    include_sources: bool = False,
) -> Dict[str, Any]:
    """Execute the Agents SDK and post‑process the result."""

    if not agent_initialised:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Agents SDK not initialised.")

    # Add schema instruction when structured output requested
    if output_type is not None:
        model_cls = OUTPUT_MODELS.get(output_type)
        if model_cls is None:
            raise HTTPException(status_code=400, detail=f"Unsupported outputType '{output_type}'.")
        prompt += "\n\n" + _schema_instruction(model_cls)

    result = await Runner.run(agent, prompt)
    final_output: str = result.final_output  # type: ignore[attr-defined]

    # Validate / parse if needed
    parsed_output: Any
    if output_type is not None:
        try:
            parsed_output = OUTPUT_MODELS[output_type].model_validate_json(final_output).model_dump(mode="json")
        except (ValidationError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"Structured output did not match schema: {exc}") from exc
    else:
        parsed_output = final_output

    response: Dict[str, Any] = {"result": parsed_output}

    if include_sources:
        # Runner.run exposes tool call metadata on the .sources attr (SDK ≥ 2025‑04‑10)
        # Fallback to empty list if not present.
        response["sources"] = getattr(result, "sources", [])

    return response


# ──────────────────────────────────────────────────────────────────────────────
#  Landing page
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> str:  # noqa: D401 (imperative mood)
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
        <h1>Hello World 👋</h1>
        <p>Welcome to the Sales‑Intelligence Agent API.</p>

        <h2>Single query → POST /api</h2>
        <pre><code>{{"company": "example.com", "query": "…", "outputType": "string", "includeSources": true}}</code></pre>

        <h2>Bulk query → POST /api/bulk</h2>
        <pre><code>{{
  "companies": ["a.com", "b.com"],
  "query": "…",
  "outputType": "dict",  // optional – {types_list}
  "includeSources": true
}}</code></pre>

        <h2>Health Check</h2>
        <p><a href="/api/health">/api/health</a></p>
    </body>
    </html>
    """


# ──────────────────────────────────────────────────────────────────────────────
#  Single‑company endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api")
async def process_query(req: QueryRequest):  # noqa: D401
    prompt = f"Company: {req.company} | Query: {req.query}"
    return await _run_agent(
        prompt,
        output_type=req.output_type_normalised(),
        include_sources=bool(req.includeSources),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Bulk endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/bulk")
async def bulk_process(req: BulkQuery):  # noqa: D401
    async def handle_company(company: str):
        prompt = f"Company: {company} | Query: {req.query}"
        try:
            return await _run_agent(
                prompt,
                output_type=req.output_type_normalised(),
                include_sources=bool(req.includeSources),
            ) | {"company": company}
        except HTTPException as exc:
            return {"company": company, "error": exc.detail}

    results: Sequence[Dict[str, Any]] = await asyncio.gather(*[handle_company(c) for c in req.companies])
    return {"results": list(results)}


# ──────────────────────────────────────────────────────────────────────────────
#  Researcher passthrough endpoint (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/researcher")
async def process_research_query(req: QueryRequest):  # noqa: D401
    try:
        from tools2 import researcher  # local import to keep optional
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Researcher tool missing: {exc}") from exc

    try:
        res = await researcher(req.company)
        return {"result": res}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Researcher error: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
#  Health check
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():  # noqa: D401
    return {"status": "ok", "agent_initialised": agent_initialised}


# ──────────────────────────────────────────────────────────────────────────────
#  Local dev entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
