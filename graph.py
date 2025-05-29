"""LangGraph chatbot with Tavily search integration and structured output.

A conversational agent that can search the web using Tavily and return structured responses.

This version also exposes *batch helpers* so the same graph can be run over many
queries concurrently (via the Runnable `.abatch` helper that LangGraph inherits
from LangChain).
"""

from __future__ import annotations

import time
import asyncio
from typing import Annotated, TypedDict, Literal, List, Sequence

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────────────────────────
#   State & schema definitions
# ────────────────────────────────────────────────

class Configuration(TypedDict):
    """Configurable parameters for the agent."""

    model: str
    max_search_results: int


class State(TypedDict):
    """State for the chatbot agent."""

    messages: Annotated[list, add_messages]
    steps_taken: list[str]  # Track all steps
    start_time: float  # Track execution time
    response: str  # Final response
    reasoning: str  # Reasoning behind the response
    confidence: Literal["very low", "low", "medium", "high", "very high"]  # Confidence level


class StructuredResponse(BaseModel):
    """Structured response format."""

    response: str = Field(description="The main answer to the user's question")
    reasoning: str = Field(description="Explanation of how the answer was derived")
    confidence: Literal[
        "very low", "low", "medium", "high", "very high"
    ] = Field(description="Confidence level in the answer")


# ────────────────────────────────────────────────
#   Graph construction helpers
# ────────────────────────────────────────────────

def create_graph() -> StateGraph:
    """Create and compile the chatbot graph."""

    # ─── Node implementations ──────────────────

    def initialize_state(state: State) -> State:  # type: ignore[override]
        """Add bookkeeping fields to the initial state."""
        return {
            "steps_taken": [],
            "start_time": time.time(),
            "response": "",
            "reasoning": "",
            "confidence": "medium",
        }

    def chatbot(state: State, config: RunnableConfig):  # type: ignore[override]
        """Chatbot node that handles conversation and decides on tool usage."""

        configuration = config.get("configurable", {})
        model_name = configuration.get("model", "openai:gpt-4.1")
        max_results = configuration.get("max_search_results", 2)

        # Initialize LLM
        llm = init_chat_model(model_name)

        system_message = (
            "You are a helpful assistant with web search capabilities. When searching for "
            "information:\n"
            "1. If a site‑specific search fails, try broader searches\n"
            "2. Try multiple search variations if needed\n"
            "3. Be persistent in finding the information\n"
            "4. Always provide clear, factual responses based on your findings"
        )

        tool = TavilySearch(max_results=max_results)
        llm_with_tools = llm.bind_tools([tool])

        # Ensure the first message is a SystemMessage with search strategy
        if state["messages"] and state["messages"][0].type != "system":
            messages_with_system = [SystemMessage(content=system_message)] + state[
                "messages"
            ]
        else:
            messages_with_system = state["messages"]

        response = llm_with_tools.invoke(messages_with_system)

        steps_taken = state.get("steps_taken", []) + [
            f"Generated AI response: {response.content[:100]}..."
        ]

        return {"messages": [response], "steps_taken": steps_taken}

    def tools_node(state: State, config: RunnableConfig):  # type: ignore[override]
        """Run the Tavily search tool and log the step."""

        configuration = config.get("configurable", {})
        max_results = configuration.get("max_search_results", 2)

        tool = TavilySearch(max_results=max_results)
        tool_node_instance = ToolNode(tools=[tool])

        # Peek at the query for logging
        last_message = state["messages"][-1]
        search_query = ""
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            search_query = last_message.tool_calls[0].get("args", {}).get("query", "")

        result = tool_node_instance.invoke(state, config)

        search_step = (
            f'Searched for: "{search_query}"\nFound results: '
            f"{len(result.get('messages', []))} items"
        )
        if result.get("messages"):
            tool_result = result["messages"][-1].content
            search_step += f"\nSample result: {tool_result[:200]}..."

        steps_taken = state.get("steps_taken", []) + [search_step]
        return {**result, "steps_taken": steps_taken}

    def should_continue(state: State) -> Literal["tools", "finalize"]:  # type: ignore[override]
        """Route either to the Tools node or to finalisation."""

        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "finalize"

    def finalize_response(state: State, config: RunnableConfig) -> State:  # type: ignore[override]
        """Convert chat+search history into a structured JSON response."""

        configuration = config.get("configurable", {})
        model_name = configuration.get("model", "openai:gpt-4.1")
        llm = init_chat_model(model_name)

        conversation_summary = "\n".join(
            [
                (
                    f"{msg.type}: {msg.content[:200]}..."
                    if len(msg.content) > 200
                    else f"{msg.type}: {msg.content}"
                )
                for msg in state["messages"][-5:]
                if hasattr(msg, "content")
            ]
        )

        structured_prompt = (
            "Based on the following conversation and search results, provide a structured response:\n\n"
            f"Conversation Summary:\n{conversation_summary}\n\n"
            "Please analyse this conversation and provide:\n"
            "1. A clear, direct response to the user's question\n"
            "2. Your reasoning for this response\n"
            "3. Your confidence level (very low, low, medium, high, very high)\n\n"
            "Format your response as JSON with these exact keys:\n"
            "- response\n- reasoning\n- confidence"
        )

        try:
            structured_llm = llm.with_structured_output(StructuredResponse)
            structured_result = structured_llm.invoke([HumanMessage(content=structured_prompt)])
            response, reasoning, confidence = (
                structured_result.response,
                structured_result.reasoning,
                structured_result.confidence,
            )
        except Exception:
            # Fallback path – try to parse JSON manually
            fallback = llm.invoke([HumanMessage(content=structured_prompt)]).content
            try:
                import json

                parsed = json.loads(fallback)
                response = parsed.get("response", fallback)
                reasoning = parsed.get("reasoning", "Based on search results and conversation")
                confidence = parsed.get("confidence", "medium")
            except Exception:
                response, reasoning, confidence = (
                    fallback,
                    "Based on search results and conversation",
                    "medium",
                )

        return {
            "response": response,
            "reasoning": reasoning,
            "confidence": confidence,
        }

    # ─── Build the DAG ──────────────────────────────────────────

    builder = StateGraph(State, config_schema=Configuration)

    builder.add_node("initialize", initialize_state)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", tools_node)
    builder.add_node("finalize", finalize_response)

    builder.set_entry_point("initialize")
    builder.add_edge("initialize", "chatbot")

    builder.add_conditional_edges(
        "chatbot",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    builder.add_edge("tools", "chatbot")
    builder.add_edge("finalize", END)

    return builder.compile()


# Instantiate the **Runnable** graph once at import time
graph = create_graph()


# ────────────────────────────────────────────────
#   Utility helpers
# ────────────────────────────────────────────────

def format_final_output(final_state: State) -> dict:
    """Convert the graph's final state dictionary into the API‑friendly shape."""

    end_time = time.time()
    start_time = final_state.get("start_time", end_time)
    time_taken = round(end_time - start_time, 2)

    return {
        "response": final_state.get("response", ""),
        "reasoning": final_state.get("reasoning", ""),
        "confidence": final_state.get("confidence", "medium"),
        "stepsTaken": final_state.get("steps_taken", []),
        "timeTakenInSeconds": str(time_taken),
    }


# ────────────────────────────────────────────────
#   ✨  NEW: batch helpers  ✨
# ────────────────────────────────────────────────

async def run_graph_batch(
    queries: Sequence[str],
    *,
    model: str = "openai:gpt-4.1",
    max_search_results: int = 2,
    max_concurrency: int = 4,
):
    """Run the LangGraph over many queries concurrently.

    Parameters
    ----------
    queries            A list/sequence of user utterances.
    model              The model name to pass through the config.
    max_search_results Maximum number of Tavily results per search.
    max_concurrency    Soft cap on parallel executions (passed to RunnableConfig).

    Returns
    -------
    List[dict]
        One formatted‑output dict per query, in the same order.
    """

    # Build initial‑state objects for each query
    states = [{"messages": [HumanMessage(content=q)]} for q in queries]

    base_config = {
        "configurable": {"model": model, "max_search_results": max_search_results},
        "max_concurrency": max_concurrency,
    }

    # abatch preserves order and runs inside the current event loop
    final_states: List[State] = await graph.abatch(states, config=base_config)
    return final_states


def run_graph_batch_sync(
    queries: Sequence[str],
    **kwargs,
):
    """Synchronous wrapper around `run_graph_batch` for scripts/notebooks."""

    return asyncio.run(run_graph_batch(list(queries), **kwargs))


# Convenience single‑query helper (still used by existing /api/sheet)
async def run_graph_single(
    query: str,
    *,
    model: str = "openai:gpt-4.1",
    max_search_results: int = 2,
):
    """Run the graph for one query and return the formatted output."""

    state = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"model": model, "max_search_results": max_search_results}}
    final_state: State = await graph.invoke(state, config)  # type: ignore[arg-type]
    return format_final_output(final_state)


__all__ = [
    "graph",
    "format_final_output",
    "run_graph_batch",
    "run_graph_batch_sync",
    "run_graph_single",
]
