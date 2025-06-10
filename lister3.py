#!/usr/bin/env python3
"""
Simple company scraper - no classes, just scrape and ask OpenAI
"""

import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv

load_dotenv()

def scrape_page(url):
    """Scrape HTML and save to file"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    # Disable SSL verification
    response = requests.get(url, headers=headers, verify=False)
    response.raise_for_status()
    
    # Save HTML
    #filename = url.split('/')[-1] + '.html'
    #with open(filename, 'w', encoding='utf-8') as f:
        #f.write(response.text)
    #print(f"HTML saved to {filename}")
    
    return response.text

def ask_openai_strategy(html_content):
    """Ask OpenAI what strategy to use"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("No OpenAI API key found")
        return None
    
    # Get page sample
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.title.get_text() if soup.title else ""
    text_sample = soup.get_text()[:1000]
    
    prompt = f"""
Analyze this webpage and tell me the best strategy to extract companies:

Title: {title}
Content sample: {text_sample}

Choose ONE strategy:
1. "scrape_domains" - extract domains from links
2. "extract_table" - companies are in HTML tables  
3. "analyze_structure" - need detailed structure analysis

Return only JSON: {{"strategy": "your_choice", "reason": "why"}}
"""
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'gpt-4.1',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.1,
        'max_tokens': 200
    }
    
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        print("OpenAI Response:")
        print(content)
        return content
    else:
        print(f"OpenAI Error: {response.status_code} - {response.text}")
        return None

def extract_with_regex(html_content, regex_pattern):
    """Extract companies using regex pattern"""
    import re
    
    soup = BeautifulSoup(html_content, 'html.parser')
    text = get_visible_html(soup)
    
    companies = []
    matches = re.finditer(regex_pattern, text, re.MULTILINE)
    
    for match in matches:
        company = match.group(1).strip()
        if company and len(company) > 1:
            companies.append(company)
    
    print(f"Found {len(companies)} companies:")
    for i, company in enumerate(companies, 1):
        print(f"{i:3d}. {company}")
    
    return companies


def scrape_domains(html_content, current_url):
    """Extract domains from all links on the page"""
    from urllib.parse import urlparse
    
    soup = BeautifulSoup(html_content, 'html.parser')
    current_domain = urlparse(current_url).netloc.lower()
    
    domains = set()
    links = soup.find_all('a', href=True)
    
    for link in links:
        href = link.get('href', '')
        
        if href.startswith('http'):
            try:
                parsed = urlparse(href)
                domain = parsed.netloc.lower()
                
                # Skip current domain and subdomains
                if domain and not domain.endswith(current_domain) and domain != current_domain:
                    domains.add(domain)
                    
            except Exception:
                continue
    
    companies = list(domains)
    print(f"Found {len(companies)} external domains:")
    for i, company in enumerate(companies, 1):
        print(f"{i:3d}. {company}")
    
    return companies


def get_visible_html(soup):
    """Extract only HTML elements that contribute to visible text"""
    from bs4 import BeautifulSoup, NavigableString, Tag
    
    # Tags that don't contribute visible content
    invisible_tags = {'script', 'style', 'meta', 'link', 'head', 'title', 'noscript'}
    
    def has_visible_content(element):
        """Check if element or its children have visible text content"""
        if isinstance(element, NavigableString):
            return bool(element.strip())
        
        if isinstance(element, Tag):
            # Skip invisible tags
            if element.name in invisible_tags:
                return False
            
            # Check if element has direct text content
            direct_text = ''.join([str(s) for s in element.strings if isinstance(s, NavigableString)])
            if direct_text.strip():
                return True
            
            # Check if any children have visible content
            for child in element.children:
                if has_visible_content(child):
                    return True
        
        return False
    
    def extract_visible_elements(element):
        """Recursively extract elements with visible content"""
        if isinstance(element, NavigableString):
            text = element.strip()
            return text if text else None
        
        if isinstance(element, Tag):
            if element.name in invisible_tags:
                return None
            
            # Create new tag
            new_tag = soup.new_tag(element.name)
            
            # Copy important attributes (you can expand this list)
            for attr in ['class', 'id']:
                if element.has_attr(attr):
                    new_tag[attr] = element[attr]
            
            has_content = False
            
            # Process children
            for child in element.children:
                visible_child = extract_visible_elements(child)
                if visible_child is not None:
                    if isinstance(visible_child, str):
                        new_tag.append(visible_child)
                    else:
                        new_tag.append(visible_child)
                    has_content = True
            
            return new_tag if has_content else None
        
        return None
    
    # Create new soup with only visible elements
    visible_soup = BeautifulSoup('', 'html.parser')
    
    # Find the main content area (usually body, but could be a specific div)
    body = soup.find('body') or soup
    
    for element in body.children:
        visible_element = extract_visible_elements(element)
        if visible_element is not None:
            visible_soup.append(visible_element)
    
    return str(visible_soup)

def analyze_structure(html_content):
    """Ask OpenAI to analyze page structure for company extraction"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("No OpenAI API key found")
        return "FAILED"
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    visible_html = get_visible_html(soup)
    
    # Save for debugging
    #with open('visible_structure.html', 'w', encoding='utf-8') as f:
        #f.write(visible_html)
    #print("Visible HTML structure saved to visible_structure.html")

    prompt = f"""
Analyze this webpage content and return ONLY a regex pattern to extract company names:

{visible_html}

IMPORTANT: This text is all on one continuous line, NOT separate lines per company.
Look for patterns like "45. CompanyName description 46. NextCompany description"

The regex should:
- NOT use ^ (start of line) since companies are mid-text
- Use word boundaries \\b or just \\d+ to find numbers
- Capture the company name in group 1
- Stop before the next numbered item


Return ONLY the regex pattern or "FAILED":
"""
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'gpt-4.1',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.1,
        'max_tokens': 100
    }
    
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        pattern = result['choices'][0]['message']['content'].strip()
        print("OpenAI returned pattern:", pattern)
        return pattern
    else:
        print(f"OpenAI Error: {response.status_code}")
        return "FAILED"


def save_to_csv(companies, filename="companies.csv"):
    """Save companies to CSV file"""
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Company'])  # Header
        for company in companies:
            writer.writerow([company])
    
    print(f"Saved {len(companies)} companies to {filename}")


def main(url):
    #url = "https://www.volta.ventures/our-portfolio/"
    
    print("Scraping page...")
    html = scrape_page(url)
    
    print("\nAsking OpenAI for strategy...")
    strategy_response = ask_openai_strategy(html)
    
    if strategy_response and "scrape_domains" in strategy_response:
        print("\nScraping domains...")
        companies = scrape_domains(html, url)
        #save_to_csv(companies, "zdomains.csv")
        return companies
    elif strategy_response and "analyze_structure" in strategy_response:
        print("\nAnalyzing structure...")
        pattern = analyze_structure(html)
        
        if pattern != "FAILED":
            print(f"\nExtracting companies with pattern: {pattern}")
            companies = extract_with_regex(html, pattern)
            #save_to_csv(companies, "zcompanies.csv")
            return companies
        else:
            print("Structure analysis failed")

#main()
