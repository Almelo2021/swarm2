"""
Sales Intelligence Agent API
----------------------------
Single-company endpoint  :  POST /api
Bulk (multi‑company)     :  POST /api/bulk
Health check             :  GET  /api/health

Key features
============
* **Tool calling** – the model can invoke `search_hubspot_contacts` (and any other
  functions you register) via OpenAI’s function‑calling interface.
* **Structured Outputs** – when `outputType` is supplied we enforce the response
  against a JSON Schema, so the final assistant message is guaranteed to match.
* **Fallback Agent SDK** – if you omit `outputType` we keep using the original
  OpenAI Agent (so existing traces and tools still work as before).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

# ────────────────────────────── Local imports ────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))

try:
    # Original Agent SDK pieces (fallback path)
    from agents import Agent, WebSearchTool, Runner  # type: ignore
    from tools import (
        get_existing_leads,  # noqa: F401  (still available to the Agent)
        search_hubspot_contacts,
        get_website_visits,  # noqa: F401
        get_crm_activities,  # noqa: F401
    )

    AGENT_AVAILABLE = True
except ImportError as exc:  # pragma: no cover – e.g. during CI without SDK
    print(f"Agent SDK unavailable: {exc}")
    search_hubspot_contacts = None  # type: ignore
    Runner = Agent = WebSearchTool = None  # type: ignore
    AGENT_AVAILABLE = False

# ────────────────────────────── OpenAI client ────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───────────────────── Structured‑output Pydantic models ─────────────────────
class _Cfg:
    extra = "forbid"


class IntegerOutput(BaseModel):
    answer: int

    class Config(_Cfg):
        pass


class FloatOutput(BaseModel):
    answer: float

    class Config(_Cfg):
        pass


class StringOutput(BaseModel):
    answer: str

    class Config(_Cfg):
        pass


class StringListOutput(BaseModel):
    answer: List[str]

    class Config(_Cfg):
        pass


class KVPair(BaseModel):
    key: str
    value: str

    class Config(_Cfg):
        pass


class KVListOutput(BaseModel):
    """Dictionary represented as list of key/value strings (schema‑safe)."""

    answer: List[KVPair]

    class Config(_Cfg):
        pass


OUTPUT_MODELS: Dict[str, Type[BaseModel]] = {
    "integer": IntegerOutput,
    "float": FloatOutput,
    "string": StringOutput,
    "string_list": StringListOutput,
    "dict": KVListOutput,
}

# ───────────────────────────── Tool registry ────────────────────────────────
FUNCTION_SCHEMAS: List[Dict[str, Any]] = []
TOOL_MAP: Dict[str, Any] = {}

if search_hubspot_contacts is not None:
    FUNCTION_SCHEMAS.append(
        {
            "name": "search_hubspot_contacts",
            "description": "Search HubSpot for contacts at the given company domain and return basic info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Company domain to search, e.g. example.com",
                    }
                },
                "required": ["domain"],
                "additionalProperties": False,
            },
        }
    )
    TOOL_MAP["search_hubspot_contacts"] = search_hubspot_contacts

# Add more tool schemas & mappings here if needed.

# ─────────────────────── Helper – maybe‑await sync/async ─────────────────────
async def _call_tool(func, **kwargs):  # type: ignore[no-untyped-def]
    if inspect.iscoroutinefunction(func):
        return await func(**kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(**kwargs))


# ─────────────────── core: ChatCompletion with tools & schema ────────────────
async def chat_with_tools(prompt: str, output_type: str) -> dict[str, Any]:
    model_cls = OUTPUT_MODELS.get(output_type)
    if not model_cls:
        raise HTTPException(status_code=400, detail=f"Unsupported outputType '{output_type}'.")

    schema = model_cls.model_json_schema()

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to external tools. "
                "If needed, call the appropriate tool. "
                "When you answer the user, you MUST follow the provided JSON Schema strictly."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # max 3 rounds: tool call → tool response → final answer
    for _ in range(3):
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=FUNCTION_SCHEMAS or None,
            response_format={"type": "json_schema", "schema": schema, "strict": True},
            temperature=0,
        )

        msg = response.choices[0].message

        # Tool call?
        if msg.tool_calls:
            for call in msg.tool_calls:
                name = call.function.name
                arguments = json.loads(call.function.arguments or "{}")
                tool_fn = TOOL_MAP.get(name)
                if not tool_fn:
                    raise HTTPException(status_code=500, detail=f"Tool '{name}' not implemented.")
                result = await _call_tool(tool_fn, **arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
            continue  # go to next loop – allow assistant to use tool result

        # No tool call → should be final structured answer
        try:
            parsed = model_cls.model_validate_json(msg.content)
            return parsed.model_dump(mode="json")
        except Exception as exc:  # pragma: no cover – schema mismatch
            raise HTTPException(status_code=500, detail=f"Schema validation failed: {exc}") from exc

    raise HTTPException(status_code=500, detail="Too many tool‑call iterations")


# ───────────────────────────── FastAPI plumbing ─────────────────────────────
app = FastAPI()


class QueryRequest(BaseModel):
    company: str
    query: str
    outputType: Optional[str] = None

    def output_type_norm(self) -> Optional[str]:
        return self.outputType.lower() if self.outputType else None


class BulkQuery(BaseModel):
    companies: List[str]
    query: str
    outputType: Optional[str] = None

    def output_type_norm(self) -> Optional[str]:
        return self.outputType.lower() if self.outputType else None


# Initialise fallback Agent once (for non‑structured queries)
if AGENT_AVAILABLE:
    agent = Agent(
        name="Assistant",
        model="gpt-4.1",
        tools=[WebSearchTool(), get_existing_leads, search_hubspot_contacts, get_website_visits, get_crm_activities],
    )
else:
    agent = None


# ────────────────────────────── HTML landing page ───────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    types_list = ", ".join(OUTPUT_MODELS)
    return f"""
    <!DOCTYPE html>
    <html><head><title>Sales Intelligence Agent API</title>
    <style>body{{font-family:Arial;max-width:800px;margin:40px auto;line-height:1.6}} pre{{background:#f4f4f4;padding:12px;border-radius:6px}}</style>
    </head><body>
      <h1>Sales Intelligence Agent API</h1>
      <h2>Single query</h2>
      <pre>{{"company":"example.com","query":"…","outputType":"string"}}</pre>
      <h2>Bulk query</h2>
      <pre>{{"companies":["a.com","b.com"],"query":"…","outputType":"dict"}}</pre>
      <p>Allowed <code>outputType</code> values: {types_list}</p>
      <p><a href="/api/health">Health check</a></p>
    </body></html>"""


# ───────────────────────────── Single‑company API ───────────────────────────
@app.post("/api")
async def process_query(req: QueryRequest):
    prompt = f"Company: {req.company} | Query: {req.query}"

    if req.outputType:
        return {"result": await chat_with_tools(prompt, req.output_type_norm())}

    # fallback to legacy Agent (no strict schema)
    if not agent:
        raise HTTPException(status_code=500, detail="Agent SDK not available and outputType not provided.")
    try:
        res = await Runner.run(agent, prompt)
        return {"result": res.final_output}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc


# ───────────────────────────── Bulk API ─────────────────────────────────────
@app.post("/api/bulk")
async def bulk_process(req: BulkQuery):
    async def one(company: str):
        prompt = f"Company: {company} | Query: {req.query}"
        try:
            if req.outputType:
                res = await chat_with_tools(prompt, req.output_type_norm())
            else:
                if not agent:
                    raise RuntimeError("Agent SDK not available.")
                res = (await Runner.run(agent, prompt)).final_output
            return {"company": company, "result": res}
        except Exception as exc:
            return {"company": company, "error": str(exc)}

    results = await asyncio.gather(*(one(c) for c in req.companies))
    return {"results": results}


# ───────────────────────────── Health check ────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "agent": bool(agent), "tools": list(TOOL_MAP)}


# ────────────────────────── Local dev entry point ───────────────────────────
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
