from agents import function_tool
import asyncio
import requests
import json
from typing import List, Optional

@function_tool
async def get_existing_leads(company: str, titles: Optional[List[str]] = None) -> str:
    """Find people working at a specific company with optional title filtering.
    
    Args:
        company: Company name or domain to search for
        titles: Optional list of job titles to filter by
    """
    base_url = "https://app.rebounds.ai/api/agentpeople"
    params = {"query": company}
    
    # Add titles if provided
    if titles and len(titles) > 0:
        for title in titles:
            params["title"] = title
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Format the response for better readability
        result = {
            "total_results": data.get("pagination", {}).get("total_entries", 0),
            "people": []
        }
        
        for person in data.get("people", []):
            result["people"].append({
                "name": person.get("name"),
                "title": person.get("title"),
                "company": person.get("organization", {}).get("name"),
                "linkedin_url": person.get("linkedin_url")
            })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching people data: {str(e)}"

@function_tool
async def search_hubspot_contacts(search_term: Optional[str] = None, company: Optional[str] = None) -> str:
    """Search Hubspot contacts with optional filtering by name or company.
    
    Args:
        search_term: Optional text to search in contact names or emails
        company: Optional company name to filter contacts
    """
    url = "https://app.rebounds.ai/api/hubspot/excludesven"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        contacts = response.json()
        
        # Apply filters if specified
        filtered_contacts = contacts
        
        if company:
            company_lower = company.lower()
            filtered_contacts = [
                contact for contact in filtered_contacts
                if (
                    # Check company property
                    ("company" in contact.get("properties", {}) and 
                     company_lower in contact.get("properties", {}).get("company", {}).get("value", "").lower()) or
                    # Check email domain
                    any(
                        identity.get("type") == "EMAIL" and 
                        company_lower in identity.get("value", "").lower()
                        for profile in contact.get("identity-profiles", [])
                        for identity in profile.get("identities", [])
                    )
                )
            ]
        
        if search_term:
            search_lower = search_term.lower()
            filtered_contacts = [
                contact for contact in filtered_contacts
                if (
                    ("firstname" in contact.get("properties", {}) and 
                     search_lower in contact.get("properties", {}).get("firstname", {}).get("value", "").lower()) or
                    ("lastname" in contact.get("properties", {}) and 
                     search_lower in contact.get("properties", {}).get("lastname", {}).get("value", "").lower()) or
                    any(
                        identity.get("type") == "EMAIL" and 
                        search_lower in identity.get("value", "").lower()
                        for profile in contact.get("identity-profiles", [])
                        for identity in profile.get("identities", [])
                    )
                )
            ]
        
        # Limit the number of results to avoid token overflow
        limited_contacts = filtered_contacts[:5]
        
        # Format the response
        result = []
        for contact in limited_contacts:
            properties = contact.get("properties", {})
            
            # Get primary email
            email = None
            for profile in contact.get("identity-profiles", []):
                for identity in profile.get("identities", []):
                    if identity.get("type") == "EMAIL" and identity.get("is-primary", False):
                        email = identity.get("value")
                        break
                if email:
                    break
            
            result.append({
                "first_name": properties.get("firstname", {}).get("value", ""),
                "last_name": properties.get("lastname", {}).get("value", ""),
                "company": properties.get("company", {}).get("value", ""),
                "email": email
            })
        
        return json.dumps({
            "total_found": len(filtered_contacts),
            "showing": len(limited_contacts),
            "contacts": result
        }, indent=2)
    
    except Exception as e:
        return f"Error searching Hubspot contacts: {str(e)}"

@function_tool
async def get_website_visits(domain: str) -> str:
    """Get website visits from a specific company domain.
    
    Args:
        domain: Company domain to check for website visits
    """
    url = f"https://app.rebounds.ai/api/agent/visits?domain={domain}"
    
    try:
        # Make sure domain is formatted correctly
        if not domain.startswith("http") and "." not in domain:
            domain = f"{domain}.com"  # Add default TLD if missing
        
        # Make sure the domain is properly formatted
        if not domain.startswith("http"):
            domain = domain.split("//")[-1]  # Remove protocol if present
        
        url = f"https://app.rebounds.ai/api/agent/visits?domain={domain}"
        
        # Add timeout to prevent hanging
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        visits = response.json()
        
        # Check if response is empty
        if not visits:
            return json.dumps({"message": "No website visits found for this domain"})
        
        # Format the response to highlight important information
        result = {
            "total_visits": len(visits),
            "company_info": {},
            "visits": []
        }
        
        # Extract company info from the first visit if available
        if visits:
            first_visit = visits[0]
            result["company_info"] = {
                "name": first_visit.get("name"),
                "domain": first_visit.get("domain"),
                "employees": first_visit.get("employees"),
                "industry": first_visit.get("industry"),
                "location": {
                    "city": first_visit.get("City"),
                    "state": first_visit.get("state"),
                    "country": first_visit.get("country")
                },
                "website": first_visit.get("web")
            }
        
        # Include visit details
        for visit in visits:
            result["visits"].append({
                "date": visit.get("created_at"),
                "visited_page": visit.get("ref"),
                "session_time_seconds": visit.get("sessionTime"),
                "session_id": visit.get("sessionId")
            })
        
        return json.dumps(result, indent=2)
    
    except requests.exceptions.Timeout:
        return json.dumps({"error": "Request timed out when accessing website visits API"})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Error accessing website visits API: {str(e)}"})
    except ValueError as e:
        return json.dumps({"error": f"Error parsing response from website visits API: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error fetching website visits: {str(e)}"})
    
    
@function_tool
async def get_crm_activities(domain: str) -> str:
    """Get logged CRM activities for contacts from a specific company domain.
    
    Args:
        domain: Company domain to check for CRM activities
    """
    url = f"https://app.rebounds.ai/api/agent/loggedactions?domain={domain}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Format the response for better readability
        result = {
            "total_contacts": len(data),
            "contacts": []
        }
        
        for contact in data:
            contact_info = {
                "name": f"{contact.get('properties', {}).get('firstname', {}).get('value', '')} {contact.get('properties', {}).get('lastname', {}).get('value', '')}",
                "email": contact.get("email"),
                "created_date": contact.get("createDate"),
                "activities": []
            }
            
            # Process activities
            for activity in contact.get("activities", []):
                engagement = activity.get("engagement", {})
                metadata = activity.get("metadata", {})
                
                contact_info["activities"].append({
                    "type": engagement.get("type"),
                    "date": timestamp_to_date(engagement.get("timestamp")),
                    "description": metadata.get("body") or engagement.get("bodyPreview")
                })
            
            result["contacts"].append(contact_info)
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        return f"Error fetching CRM activities: {str(e)}"

def timestamp_to_date(timestamp):
    """Convert timestamp to readable date if available"""
    if not timestamp:
        return None
    
    from datetime import datetime
    try:
        # Convert milliseconds to seconds and format
        return datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(timestamp)
