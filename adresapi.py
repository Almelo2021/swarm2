import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import time
from googlesearch import search
import re

def track_company_address_change(company_name, domain, address):
    """
    Track when a company's address changed on their website using Wayback Machine.
    
    Args:
        company_name (str): Name of the company
        domain (str): Website domain (e.g., 'example.com')
        address (str): Current address to search for
        
    Returns:
        tuple: (last_no_address_date, first_with_address_date) or (None, None) if not found
               Dates are in YYYY-MM-DD format
    """
    
    def google_search(domain, query, num_results=10):
        """Simple Google search within a specific domain."""
        domain = domain.replace('www.', '')
        search_query = f"site:{domain} {query}"
        print(f"Zoeken via Google: {search_query}")
        
        try:
            results = list(search(search_query, num_results=num_results, advanced=True))
            formatted_results = []
            for result in results:
                if hasattr(result, 'url') and result.url:
                    formatted_results.append({
                        'url': result.url,
                        'title': result.title or "No title available",
                        'description': result.description or "No description available"
                    })
            
            print(f"Gevonden: {len(formatted_results)} resultaten")
            return formatted_results
            
        except Exception as e:
            print(f"Fout bij Google zoeken: {e}")
            return []

    def normalize_address(address):
        """Normalize an address by standardizing format and extracting key components."""
        normalized = address.lower().strip()
        
        # Extract postal/zip code with regex
        postal_code = None
        postal_patterns = [
            r'\b\d{4}\s*[a-z]{2}\b',  # Dutch format: 1234 AB
            r'\b[a-z]\d[a-z]\s*\d[a-z]\d\b',  # Canadian format: A1A 1A1
            r'\b\d{5}(?:-\d{4})?\b'  # US format: 12345 or 12345-6789
        ]
        
        for pattern in postal_patterns:
            match = re.search(pattern, normalized)
            if match:
                postal_code = match.group(0)
                break
        
        # Extract street number and name
        clean_address = re.sub(r'[,:]', ' ', normalized)
        parts = clean_address.split()
        
        number_candidates = [part for part in parts if part.isdigit() or (part and part[0].isdigit() and part[-1].isalpha())]
        word_candidates = [part for part in parts if not part.isdigit() and len(part) > 1 and part not in ['st', 'rd', 'nd', 'th']]
        
        street_name = None
        street_number = None
        
        if number_candidates and word_candidates:
            street_number = number_candidates[0]
            street_words = []
            for word in word_candidates:
                if word not in ['apt', 'unit', 'suite', 'floor', 'department']:
                    street_words.append(word)
            
            if street_words:
                street_name = ' '.join(street_words)
        
        return {
            'full': normalized,
            'postal_code': postal_code,
            'street_name': street_name,
            'street_number': street_number,
            'parts': parts
        }

    def find_contact_page(domain, adres):
        """Find the contact page of a domain with improved address detection."""
        print(f"Zoeken naar contactpagina voor domein: {domain}")
        
        domain = domain.replace('www.', '')
        address_info = normalize_address(adres)
        print(f"Genormaliseerd adres: {address_info}")
        
        adres_clean = adres.lower().strip()
        
        # Standard contact pages to try
        standard_pages = [
            f"https://{domain}/",
            f"https://{domain}/contact",
            f"https://{domain}/contactus",
            f"https://{domain}/contact-us",
            f"https://{domain}/over-ons",
            f"https://{domain}/about-us",
            f"https://{domain}/about",
            f"https://{domain}/locatie",
            f"https://{domain}/location",
            f"https://{domain}/contactgegevens",
            f"https://{domain}/nl/contact",
            f"https://{domain}/en/contact",
            f"https://{domain}/wie-zijn-wij",
            f"https://{domain}/over-ons/contact",
            f"https://{domain}/nl/over-ons",
            f"https://{domain}/en/about-us",
            f"https://{domain}/contact-opnemen",
            f"https://{domain}/adres"
        ]
        
        # Try www. versions
        www_domain = f"www.{domain}"
        for page in [f"https://{www_domain}/", f"https://{www_domain}/contact"]:
            if page not in standard_pages:
                standard_pages.append(page)
        
        session = requests.Session()
        retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        session.headers.update(headers)
        
        def check_for_address(html_content):
            """Check if the address is present on the page."""
            soup = BeautifulSoup(html_content, "html.parser")
            page_text = soup.get_text().lower()
            
            # Method 1: Check for exact address
            if adres_clean in page_text:
                print("Adres gevonden: exacte match")
                return True
                
            # Method 2: Check for street name and number in different orders
            if address_info['street_name'] and address_info['street_number']:
                pattern1 = f"{address_info['street_name']}\\s*{address_info['street_number']}"
                pattern2 = f"{address_info['street_number']}\\s*{address_info['street_name']}"
                
                if (re.search(pattern1, page_text) or re.search(pattern2, page_text)):
                    print(f"Adres gevonden: straatnaam en nummer match")
                    return True
            
            # Method 3: Check if postal code is present along with street name
            if address_info['postal_code'] and address_info['street_name']:
                postal_pattern = re.escape(address_info['postal_code'])
                street_pattern = re.escape(address_info['street_name'])
                
                if re.search(postal_pattern, page_text) and re.search(street_pattern, page_text):
                    postal_pos = page_text.find(address_info['postal_code'])
                    street_pos = page_text.find(address_info['street_name'])
                    
                    if abs(postal_pos - street_pos) < 100:
                        print(f"Adres gevonden: postcode en straatnaam binnen nabije tekst")
                        return True
            
            # Method 4: Check for key parts in address blocks
            address_blocks = []
            for tag in soup.find_all(['p', 'div', 'span', 'address']):
                tag_text = tag.get_text().lower().strip()
                if len(tag_text) > 0 and any(part in tag_text for part in address_info['parts'] if len(part) > 1):
                    address_blocks.append(tag_text)
            
            for block in address_blocks:
                if address_info['street_name'] and address_info['street_number']:
                    if address_info['street_name'] in block and address_info['street_number'] in block:
                        print(f"Adres gevonden: straatnaam en nummer in hetzelfde tekstblok")
                        return True
                
                significant_parts = [part for part in address_info['parts'] if len(part) > 2]
                if len(significant_parts) >= 2:
                    matches = sum(1 for part in significant_parts if part in block)
                    if matches >= 2:
                        print(f"Adres gevonden: meerdere delen in hetzelfde tekstblok")
                        return True
            
            return False
        
        # Try standard contact pages first
        for page in standard_pages:
            try:
                print(f"Proberen: {page}")
                response = session.get(page, timeout=15)
                if response.status_code == 200:
                    if check_for_address(response.content):
                        print(f"Contactpagina gevonden: {page}")
                        return page
            except requests.exceptions.RequestException as e:
                print(f"Fout bij ophalen van {page}: {e}")
                continue
        
        # If standard pages don't work, use Google search
        try:
            print("Standaard pagina's niet gevonden, zoeken via Google...")
            
            search_queries = ["contact"]
            if address_info['street_name']:
                search_queries.append(address_info['street_name'])
            if address_info['postal_code']:
                search_queries.append(address_info['postal_code'])
            if address_info['street_name'] and address_info['street_number']:
                search_queries.append(f"{address_info['street_name']} {address_info['street_number']}")
                search_queries.append(f"{address_info['street_number']} {address_info['street_name']}")
            search_queries.append(adres_clean)
            
            search_queries = list(set(search_queries))
            
            for query in search_queries:
                try:
                    print(f"Google zoeken: {query}")
                    results = google_search(domain, query, num_results=5)
                    
                    for result in results:
                        url = result['url']
                        if domain in url.replace('www.', ''):
                            try:
                                response = session.get(url, timeout=15)
                                if response.status_code == 200:
                                    if check_for_address(response.content):
                                        print(f"Contactpagina gevonden via Google: {url}")
                                        return url
                            except requests.exceptions.RequestException:
                                continue
                except Exception as e:
                    print(f"Fout bij Google zoekopdracht '{query}': {e}")
                    continue
                    
        except Exception as e:
            print(f"Fout bij zoeken via Google: {e}")
        
        # Last resort: search through all links on main page
        try:
            main_url = f"https://{domain}"
            try:
                response = session.get(main_url, timeout=15)
            except:
                main_url = f"https://www.{domain}"
                response = session.get(main_url, timeout=15)
                
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                
                if check_for_address(response.content):
                    print(f"Adres gevonden op hoofdpagina: {main_url}")
                    return main_url
                    
                contact_links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    link_text = link.get_text().lower()
                    
                    if any(keyword in link_text for keyword in ['contact', 'over ons', 'about', 'locatie', 'adres', 'location']):
                        if href.startswith('/'):
                            href = f"{main_url}{href}"
                        elif not href.startswith(('http://', 'https://')):
                            href = f"{main_url}/{href}"
                        
                        if domain in href.replace('www.', '') and href not in contact_links:
                            contact_links.append(href)
                
                for link in contact_links:
                    try:
                        response = session.get(link, timeout=15)
                        if response.status_code == 200:
                            if check_for_address(response.content):
                                print(f"Adres gevonden op pagina: {link}")
                                return link
                    except:
                        continue
                        
        except requests.exceptions.RequestException as e:
            print(f"Fout bij doorzoeken van de website: {e}")
        
        print("Geen pagina gevonden met het opgegeven adres.")
        return None

    print(f"\n---------- Verwerken van {company_name} ({domain}) ----------")
    print(f"Zoeken naar adres: {address}")
    
    # Remove www. from domain if present
    domain = domain.replace('www.', '')
    
    # Step 1: Find the relevant page
    url = find_contact_page(domain, address)
    if not url:
        print("Kan geen pagina vinden met het opgegeven adres. Script wordt gestopt.")
        return None, None
    
    # Set up session with retries for Wayback Machine requests
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    session.headers.update(headers)

    # Step 2: Get snapshots via CDX API
    cdx_url = f"https://web.archive.org/cdx/search/cdx?url={url}&output=json&fl=timestamp,original&limit=1000"
    try:
        response = session.get(cdx_url, timeout=30)
        response.raise_for_status()
        all_data = json.loads(response.text)
        if len(all_data) <= 1:
            print(f"Geen snapshots gevonden voor URL: {url}")
            return None, None
        
        snapshots = sorted(all_data[1:], key=lambda x: x[0])
        print(f"Snapshots succesvol opgehaald: {len(snapshots)} snapshots")
    except requests.exceptions.RequestException as e:
        print(f"Fout bij het ophalen van CDX-data: {e}")
        return None, None

    # Normalize address for better detection
    address_info = normalize_address(address)

    def has_address(timestamp):
        """Check if the address is present in a specific Wayback Machine snapshot."""
        wayback_url = f"https://web.archive.org/web/{timestamp}/{url}"
        try:
            page = session.get(wayback_url, timeout=30, headers=headers)
            page.raise_for_status()
            soup = BeautifulSoup(page.content, "html.parser")
            page_text = soup.get_text().lower()
            
            # Method 1: Check for exact address
            if address_info['full'] in page_text:
                print(f"Geanalyseerd: {timestamp} - Adres aanwezig: True (exacte match)")
                return True
                
            # Method 2: Check for street name and number in different orders
            if address_info['street_name'] and address_info['street_number']:
                pattern1 = f"{address_info['street_name']}\\s*{address_info['street_number']}"
                pattern2 = f"{address_info['street_number']}\\s*{address_info['street_name']}"
                
                if (re.search(pattern1, page_text) or re.search(pattern2, page_text)):
                    print(f"Geanalyseerd: {timestamp} - Adres aanwezig: True (straatnaam/nummer match)")
                    return True
            
            # Method 3: Check for key parts in address blocks
            address_blocks = []
            for tag in soup.find_all(['p', 'div', 'span', 'address']):
                tag_text = tag.get_text().lower().strip()
                if len(tag_text) > 0 and any(part in tag_text for part in address_info['parts'] if len(part) > 1):
                    address_blocks.append(tag_text)
            
            for block in address_blocks:
                if address_info['street_name'] and address_info['street_number']:
                    if address_info['street_name'] in block and address_info['street_number'] in block:
                        print(f"Geanalyseerd: {timestamp} - Adres aanwezig: True (in tekstblok)")
                        return True
                
                significant_parts = [part for part in address_info['parts'] if len(part) > 2]
                if len(significant_parts) >= 2:
                    matches = sum(1 for part in significant_parts if part in block)
                    if matches >= 2:
                        print(f"Geanalyseerd: {timestamp} - Adres aanwezig: True (meerdere delen in blok)")
                        return True
            
            print(f"Geanalyseerd: {timestamp} - Adres aanwezig: False")
            return False
            
        except requests.exceptions.RequestException as e:
            print(f"Fout bij ophalen van snapshot {wayback_url}: {e}")
            return None

    # Binary search to find the transition point
    all_results = []
    left, right = 0, len(snapshots) - 1
    last_no_address = None
    first_with_address = None

    while left <= right:
        mid = (left + right) // 2
        timestamp = snapshots[mid][0]
        
        address_present = has_address(timestamp)
        
        if address_present is not None:
            all_results.append((timestamp, address_present))
            
        if address_present is None:
            # On errors, try a few other snapshots linearly
            continue_search = False
            for offset in [1, -1, 2, -2]:
                new_idx = mid + offset
                if 0 <= new_idx < len(snapshots):
                    new_timestamp = snapshots[new_idx][0]
                    new_result = has_address(new_timestamp)
                    if new_result is not None:
                        all_results.append((new_timestamp, new_result))
                        mid = new_idx
                        address_present = new_result
                        continue_search = True
                        break
            
            if not continue_search:
                print("Te veel fouten bij het ophalen van snapshots. Overschakelen naar lineair zoeken...")
                break
        
        if address_present:
            if first_with_address is None or timestamp < first_with_address:
                first_with_address = timestamp
            right = mid - 1
        else:
            if last_no_address is None or timestamp > last_no_address:
                last_no_address = timestamp
            left = mid + 1

    # Verify and correct results with all collected data
    all_results.sort(key=lambda x: x[0])
    
    # Find the latest 'False' before the first 'True'
    if first_with_address is not None:
        latest_false_before_true = None
        for timestamp, result in all_results:
            if not result and (latest_false_before_true is None or timestamp > latest_false_before_true) and timestamp < first_with_address:
                latest_false_before_true = timestamp
        
        if latest_false_before_true is not None:
            last_no_address = latest_false_before_true
    
    # Find the first 'True' after the last 'False'
    if last_no_address is not None:
        earliest_true_after_false = None
        for timestamp, result in all_results:
            if result and (earliest_true_after_false is None or timestamp < earliest_true_after_false) and timestamp > last_no_address:
                earliest_true_after_false = timestamp
        
        if earliest_true_after_false is not None:
            first_with_address = earliest_true_after_false

    # If binary search is incomplete, check a few additional snapshots
    if (last_no_address is None or first_with_address is None) and len(snapshots) > 0:
        print("Binair zoeken onvolledig. Enkele aanvullende snapshots controleren...")
        
        sample_indices = [0, len(snapshots)-1]
        if len(snapshots) > 2:
            sample_indices.append(len(snapshots) // 4)
            sample_indices.append(len(snapshots) // 2)
            sample_indices.append(3 * len(snapshots) // 4)
        
        for idx in sample_indices:
            if idx >= 0 and idx < len(snapshots):
                timestamp = snapshots[idx][0]
                if not any(timestamp == t for t, _ in all_results):
                    address_present = has_address(timestamp)
                    if address_present is not None:
                        all_results.append((timestamp, address_present))
                        
                        if address_present and (first_with_address is None or timestamp < first_with_address):
                            first_with_address = timestamp
                        elif not address_present and (last_no_address is None or timestamp > last_no_address):
                            last_no_address = timestamp
        
        all_results.sort(key=lambda x: x[0])
        
        if last_no_address is None:
            for timestamp, result in all_results:
                if not result:
                    last_no_address = timestamp
        
        if first_with_address is None:
            for timestamp, result in all_results:
                if result:
                    first_with_address = timestamp
                    break

    # Step 4: Results
    if last_no_address and first_with_address:
        # Ensure correct chronological order
        if last_no_address > first_with_address:
            correct_last_no = None
            for timestamp, result in all_results:
                if not result and timestamp < first_with_address:
                    if correct_last_no is None or timestamp > correct_last_no:
                        correct_last_no = timestamp
            
            if correct_last_no:
                last_no_address = correct_last_no
        
        last_no_address_date = f"{last_no_address[:4]}-{last_no_address[4:6]}-{last_no_address[6:8]}"
        first_with_address_date = f"{first_with_address[:4]}-{first_with_address[4:6]}-{first_with_address[6:8]}"
        
        print(f"\nLaatste snapshot zonder huidig adres: {last_no_address} ({last_no_address_date})")
        print(f"Eerste snapshot met huidig adres: {first_with_address} ({first_with_address_date})")
        print(f"Verhuisrange: tussen {last_no_address_date} en {first_with_address_date}")
        
        return last_no_address_date, first_with_address_date
    else:
        # Special case: check for True snapshots before 2022 and in 2024/2025
        true_before_2022 = None
        true_in_2024_2025 = None
        
        for timestamp, result in all_results:
            if result:
                year = int(timestamp[:4])
                if year < 2022 and (true_before_2022 is None or timestamp < true_before_2022):
                    true_before_2022 = timestamp
                if (year >= 2024) and (true_in_2024_2025 is None or timestamp < true_in_2024_2025):
                    true_in_2024_2025 = timestamp
        
        if true_before_2022 and true_in_2024_2025:
            oldest_true_date = f"{true_before_2022[:4]}-{true_before_2022[4:6]}-{true_before_2022[6:8]}"
            print(f"\nSpeciaal geval: adres gevonden in snapshot van voor 2022 ({oldest_true_date}) en in 2024/2025")
            print(f"Verhuisrange: tussen 1-1-2000 en {oldest_true_date}")
            return "2000-01-01", oldest_true_date
        else:
            print("\nNiet genoeg data om een verhuisrange te bepalen. Mogelijk is het adres niet gevonden of zijn er te weinig snapshots.")
            return None, None


# Example usage:
if __name__ == "__main__":
    # Example function call
    company_name = "Flynth"
    domain = "flynth.nl"
    address = "Brouwerijstraat 1, 7523 XC Enschede, Netherlands"
    
    last_no_address_date, first_with_address_date = track_company_address_change(company_name, domain, address)
    
    if last_no_address_date and first_with_address_date:
        print(f"\nResult: Address change occurred between {last_no_address_date} and {first_with_address_date}")
    else:
        print("\nNo address change period could be determined.")
