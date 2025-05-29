"""LangGraph chatbot with Tavily search integration and structured output.

A conversational agent that can search the web using Tavily and return structured responses.
"""

from __future__ import annotations

import time
from typing import Annotated, TypedDict, Literal
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
    confidence: Literal["very low", "low", "medium", "high", "very high"] = Field(
        description="Confidence level in the answer"
    )


def create_graph() -> StateGraph:
    """Create and compile the chatbot graph."""
    
    def initialize_state(state: State) -> State:
        """Initialize tracking variables."""
        return {
            "steps_taken": [],
            "start_time": time.time(),
            "response": "",
            "reasoning": "",
            "confidence": "medium"
        }
    
    def chatbot(state: State, config: RunnableConfig):
        """Chatbot node that handles conversation and decides on tool usage."""
        configuration = config.get("configurable", {})
        model_name = configuration.get("model", "openai:gpt-4.1")
        print(model_name)
        print(configuration)
        max_results = configuration.get("max_search_results", 2)
        
        # Initialize LLM
        llm = init_chat_model(model_name)
        
        # System message for better search strategy
        system_message = """You are a helpful assistant with web search capabilities. When searching for information:
1. If a site-specific search fails, try broader searches
2. Try multiple search variations if needed
3. Be persistent in finding the information
4. Always provide clear, factual responses based on your findings"""
        
        tool = TavilySearch(max_results=max_results)
        llm_with_tools = llm.bind_tools([tool])
        
        # Prepare messages with system context
        if state["messages"] and state["messages"][0].type != "system":
            messages_with_system = [SystemMessage(content=system_message)] + state["messages"]
        else:
            messages_with_system = state["messages"]
        
        # Get LLM response
        response = llm_with_tools.invoke(messages_with_system)
        
        # Track this step
        new_step = f"Generated AI response: {response.content[:100]}..."
        steps_taken = state.get("steps_taken", []) + [new_step]
        
        return {
            "messages": [response],
            "steps_taken": steps_taken
        }
    
    def tools_node(state: State, config: RunnableConfig):
        """Enhanced tools node that tracks search operations."""
        configuration = config.get("configurable", {})
        max_results = configuration.get("max_search_results", 2)
        
        tool = TavilySearch(max_results=max_results)
        tool_node_instance = ToolNode(tools=[tool])
        
        # Get the last message to understand what search is being performed
        last_message = state["messages"][-1]
        search_query = ""
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            search_query = last_message.tool_calls[0].get('args', {}).get('query', '')
        
        # Execute the tool
        result = tool_node_instance.invoke(state, config)
        
        # Track this search step
        search_step = f"Searched for: \"{search_query}\"\n\nFound results: {len(result.get('messages', []))} items"
        if result.get('messages'):
            # Add snippet of results
            tool_result = result['messages'][-1].content
            search_step += f"\n\nSample result: {tool_result[:200]}..."
        
        steps_taken = state.get("steps_taken", []) + [search_step]
        
        return {
            **result,
            "steps_taken": steps_taken
        }
    
    def should_continue(state: State) -> Literal["tools", "finalize"]:
        """Determine if we should continue with tools or finalize."""
        last_message = state["messages"][-1]
        
        # If the last message has tool calls, go to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Otherwise, finalize the response
        return "finalize"
    
    def finalize_response(state: State, config: RunnableConfig) -> State:
        """Generate final structured response."""
        configuration = config.get("configurable", {})
        model_name = configuration.get("model", "openai:gpt-4.1")
        
        llm = init_chat_model(model_name)
        
        # Get the conversation history
        conversation_summary = "\n".join([
            f"{msg.type}: {msg.content[:200]}..." if len(msg.content) > 200 else f"{msg.type}: {msg.content}"
            for msg in state["messages"][-5:]  # Last 5 messages
            if hasattr(msg, 'content')
        ])
        
        # Create structured response prompt
        structured_prompt = f"""Based on the following conversation and search results, provide a structured response:

Conversation Summary:
{conversation_summary}

Please analyze this conversation and provide:
1. A clear, direct response to the user's question
2. Your reasoning for this response
3. Your confidence level (very low, low, medium, high, very high)

Format your response as JSON with these exact keys:
- response: (string) Direct answer to the question
- reasoning: (string) Explanation of how you arrived at this answer
- confidence: (string) One of: very low, low, medium, high, very high
"""
        
        try:
            # Use structured output if available
            structured_llm = llm.with_structured_output(StructuredResponse)
            structured_result = structured_llm.invoke([HumanMessage(content=structured_prompt)])
            
            response = structured_result.response
            reasoning = structured_result.reasoning
            confidence = structured_result.confidence
            
        except Exception as e:
            # Fallback to regular LLM call
            fallback_result = llm.invoke([HumanMessage(content=structured_prompt)])
            
            # Try to parse JSON from response, or use defaults
            try:
                import json
                parsed = json.loads(fallback_result.content)
                response = parsed.get("response", fallback_result.content)
                reasoning = parsed.get("reasoning", "Based on search results and conversation")
                confidence = parsed.get("confidence", "medium")
            except:
                response = fallback_result.content
                reasoning = "Based on search results and conversation"
                confidence = "medium"
        
        return {
            "response": response,
            "reasoning": reasoning,
            "confidence": confidence
        }
    
    # Build the graph
    graph_builder = StateGraph(State, config_schema=Configuration)
    
    # Add nodes
    graph_builder.add_node("initialize", initialize_state)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_node("finalize", finalize_response)
    
    # Add edges
    graph_builder.set_entry_point("initialize")
    graph_builder.add_edge("initialize", "chatbot")
    
    graph_builder.add_conditional_edges(
        "chatbot",
        should_continue,
        {
            "tools": "tools",
            "finalize": "finalize"
        }
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("finalize", END)
    
    return graph_builder.compile()


def format_final_output(final_state: State) -> dict:
    """Format the final state into the desired JSON structure."""
    print("formatting...")
    end_time = time.time()
    start_time = final_state.get("start_time", end_time)
    time_taken = round(end_time - start_time, 2)
    
    return {
        "response": final_state.get("response", ""),
        "reasoning": final_state.get("reasoning", ""),
        "confidence": final_state.get("confidence", "medium"),
        "stepsTaken": final_state.get("steps_taken", []),
        "timeTakenInSeconds": str(time_taken)
    }


# Create the default graph instance
graph = create_graph()
