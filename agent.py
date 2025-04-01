from agents import Agent, WebSearchTool
from tools import (
    get_existing_leads,
    search_hubspot_contacts,
    get_website_visits,
    get_crm_activities
)
import os

# Make sure your OpenAI API key is set
#os.environ["OPENAI_API_KEY"]  # Replace with your actual API key

# Create the agent
def create_agent():
    return Agent(
        name="Assistant",
        tools=[
            WebSearchTool(),
            get_existing_leads,
            search_hubspot_contacts,
            get_website_visits,
            get_crm_activities
        ],
    )