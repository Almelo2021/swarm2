import json
from typing import Dict, List, Any
from dotenv import load_dotenv
import os
from agents import Agent, Runner, WebSearchTool
from openai import OpenAI

load_dotenv()

def research_company_for_sales(target_url: str, company_context: List[Dict]) -> Dict[str, Any]:
    """
    Research a target company using OpenAI's Agents SDK with real web search
    and generate sales angles based on your company context
    
    Args:
        target_url: URL of the company to research
        company_context: Your company's context (the JSON array you provided)
    
    Returns:
        Dictionary with research findings and sales angles
    """
    
    # Create research agent with web search capabilities
    research_agent = Agent(
        name="Sales Research Agent",
        instructions=f"""
        You are an expert B2B sales researcher and strategist. Your task is to:
        1. Research the provided company URL thoroughly using web search
        2. Analyze their business model, challenges, and opportunities
        3. Generate strategic sales angles based on the seller's company context
        4. Provide actionable insights with proper source attribution

        CRITICAL: You MUST use the web search tool to research the target company.
        Do not make assumptions or hallucinate information.
        
        Company context for generating sales angles:
        {json.dumps(company_context, indent=2)}
        
        Return your findings in this JSON structure:
        {{
            "target_company": {{
                "name": "Company Name from research",
                "industry": "Primary Industry from research",
                "size": "Employee count or size category from research",
                "business_model": "Brief description from research",
                "key_products_services": ["List from research"],
                "recent_news": ["Recent developments from research"]
            }},
            "research_findings": {{
                "pain_points_identified": [
                    {{
                        "pain": "Specific challenge from research",
                        "evidence": "What from your research suggests this",
                        "severity": "High/Medium/Low"
                    }}
                ],
                "growth_initiatives": ["From research"],
                "technology_stack": ["From research"],
                "competitive_landscape": ["From research"]
            }},
            "sales_angles": [
                {{
                    "angle_title": "Compelling angle name",
                    "approach": "How to position solution",
                    "value_proposition": "Specific value for this prospect",
                    "pain_point_addressed": "Which pain this solves",
                    "proof_point_to_use": "Relevant proof point from context",
                    "conversation_starter": "Specific opener for outreach",
                    "priority": "High/Medium/Low"
                }}
            ],
            "outreach_strategy": {{
                "best_contact_titles": ["Job titles to target"],
                "recommended_channels": ["Email, LinkedIn, etc."],
                "timing_considerations": ["Based on research"],
                "personalization_hooks": ["Specific details from research"]
            }},
            "sources": [
                {{
                    "url": "Actual source URL from web search",
                    "type": "Website/News/Social/etc.",
                    "key_info": "Information gathered from this source"
                }}
            ]
        }}
        """,
        tools=[WebSearchTool()]
    )
    
    # Research prompt
    research_prompt = f"""
    Research the company at this URL: {target_url}
    
    Please conduct thorough research including:
    1. Visit their main website and analyze their business
    2. Search for recent news about the company
    3. Look for information about their challenges and growth initiatives
    4. Find details about their industry and competitive landscape
    5. Identify their technology stack if possible
    
    Then generate strategic sales angles based on how my company's capabilities 
    can address their specific challenges and opportunities.
    
    Focus on creating 3-5 high-priority sales angles that:
    - Address specific pain points you discovered
    - Leverage relevant proof points from my company context
    - Include specific conversation starters
    - Are backed by evidence from your research
    
    Make sure all information comes from your actual web search results.
    """
    
    try:
        # Run the research agent
        result = Runner.run_sync(research_agent, research_prompt)
        
        # Extract the JSON from the agent's response
        response_text = result.final_output
        
        # Try to parse JSON from the response
        try:
            # Look for JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx+1]
                research_data = json.loads(json_str)
            else:
                # If no JSON found, structure the response
                research_data = {
                    "error": "No structured JSON found in response",
                    "raw_response": response_text
                }
        
        except json.JSONDecodeError:
            # If JSON parsing fails, return structured error
            research_data = {
                "error": "Failed to parse JSON from agent response",
                "raw_response": response_text
            }
        
        # Add metadata
        research_data["metadata"] = {
            "research_date": "2025-08-01",
            "target_url": target_url,
            "seller_company": company_context[0]["companyName"],
            "agent_used": "OpenAI Agents SDK with WebSearchTool",
            "analysis_confidence": "High"
        }
        
        return research_data
        
    except Exception as e:
        return {
            "error": "Research agent execution failed",
            "details": str(e),
            "metadata": {
                "target_url": target_url,
                "seller_company": company_context[0]["companyName"] if company_context else "Unknown"
            }
        }

