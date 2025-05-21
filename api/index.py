from __future__ import annotations

"""Sales-Intelligence Agent API (v2.1)
────────────────────────────────────────
•  All requests run through Agents SDK so every tool is available.
•  Robust post‑processing that tolerates the model accidentally wrapping its
   answer in ```json fences or adding chatter, preventing schema‑mismatch
   errors like the one you saw.
•  Optional `includeSources` flag unchanged.
"""

import asyncio
import json
import os
import re
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
    from agents import Agent, Runner, WebSearchTool  # noqa: WPS433
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

class _BaseConfig:  # noqa: WPS110
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
#  FastAPI init
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Sales‑Intelligence Agent API", version="2.1")


class QueryRequest(BaseModel):
    company: str
    query: str
    outputType: Optional[str] = None
    includeSources: Optional[bool] = False

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
#  Agent bootstrap
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
#  Prompt helpers
# ──────────────────────────────────────────────────────────────────────────────

def _schema_instruction(model_cls: Type[BaseModel]) -> str:
    """Tell the assistant to output raw JSON only (no back‑ticks)."""

    schema = json.dumps(model_cls.model_json_schema()["properties"], indent=2)
    return (
        "Return **only** a JSON object that satisfies this schema. No markdown, "
        "no code‑fences, no commentary.\n" + schema
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Post‑processing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Best‑effort extraction of a raw JSON string from model output."""

    text = text.strip()

    # 1) Strip ```json fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()

    # 2) Trim leading chatter until first opening brace
    brace = text.find("{")
    if brace != -1:
        candidate = text[brace:]
        return candidate.strip()

    return text  # fallback: give caller original string


# ──────────────────────────────────────────────────────────────────────────────
#  Core execution
# ──────────────────────────────────────────────────────────────────────────────

async def _run_agent(*, prompt: str, output_type: Optional[str], include_sources: bool) -> Dict[str, Any]:
    if not agent_initialised:
        raise HTTPException(status_code=500, detail="Agents SDK not initialised.")

    if output_type is not None:
        model_cls = OUTPUT_MODELS.get(output_type)
        if model_cls is None:
            raise HTTPException(status_code=400, detail=f"Unsupported outputType '{output_type}'.")
        prompt += "\n\n" + _schema_instruction(model_cls)

    run = await Runner.run(agent, prompt)
    raw_output: str = run.final_output  # type: ignore[attr-defined]

    if output_type is None:
        parsed: Any = raw_output
    else:
        cleaned = _extract_json(raw_output)
        try:
            parsed = OUTPUT_MODELS[output_type].model_validate_json(cleaned).model_dump(mode="json")
        except (ValidationError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Structured output did not match schema after cleaning: {exc}",
            ) from exc

    response: Dict[str, Any] = {"result": parsed}
    if include_sources:
        response["sources"] = getattr(run, "sources", [])
    return response


# ──────────────────────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    types_list = ", ".join(OUTPUT_MODELS)
    return f"""
    <!DOCTYPE html><html><head><title>Sales Intelligence Agent API</title><style>
        body{{font-family:Arial,Helvetica,sans-serif;max-width:800px;margin:40px auto;line-height:1.6}}
        pre,code{{background:#f4f4f4;padding:16px;border-radius:8px}}
    </style></head><body>
        <h1>Hello World 👋</h1><p>Welcome to the Sales‑Intelligence Agent API.</p>
        <h2>Single query → POST /api</h2><pre><code>{{"company":"example.com","query":"…","outputType":"string","includeSources":true}}</code></pre>
        <h2>Bulk query → POST /api/bulk</h2><pre><code>{{"companies":["a.com","b.com"],"query":"…","outputType":"dict","includeSources":true}}</code></pre>
        <h2>Health Check</h2><p><a href="/api/health">/api/health</a></p>
    </body></html>"""


@app.post("/api")
async def process_query(req: QueryRequest):
    prompt = f"Company: {req.company} | Query: {req.query}"
    return await _run_agent(
        prompt=prompt,
        output_type=req.output_type_normalised(),
        include_sources=bool(req.includeSources),
    )


@app.post("/api/bulk")
async def bulk_process(req: BulkQuery):
    async def handle(company: str):
        prompt = f"Company: {company} | Query: {req.query}"
        try:
            return await _run_agent(
                prompt=prompt,
                output_type=req.output_type_normalised(),
                include_sources=bool(req.includeSources),
            ) | {"company": company}
        except HTTPException as exc:
            return {"company": company, "error": exc.detail}

    results: Sequence[Dict[str, Any]] = await asyncio.gather(*[handle(c) for c in req.companies])
    return {"results": list(results)}


@app.post("/api/researcher")
async def process_research_query(req: QueryRequest):
    try:
        from tools2 import researcher  # late import
        res = await researcher(req.company)
        return {"result": res}
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Researcher tool missing: {exc}") from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Researcher error: {exc}") from exc


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "agent_initialised": agent_initialised}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
