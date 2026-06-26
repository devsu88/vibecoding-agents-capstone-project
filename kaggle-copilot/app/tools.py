"""
External Tools Module

This module encapsulates all interactions with external systems and APIs, such as 
fetching Kaggle metadata, downloading datasets via the Kaggle API, and performing 
web searches. These functions are designed to be used as tools by the LLM agents 
or directly by the workflow orchestrator.
"""

import os
import re
import socket
import ipaddress
import urllib.request
import urllib.parse
import urllib.error
import json
from ddgs import DDGS

def fetch_kaggle_competition_metadata(url: str) -> str:
    """
    Fetches the title and description from a Kaggle competition webpage.
    
    This function performs a lightweight HTTP GET request to scrape the HTML metadata 
    of the Kaggle competition. It avoids heavy scraping libraries by using regex.
    
    Args:
        url (str): The full Kaggle competition URL to fetch metadata for 
            (e.g., https://www.kaggle.com/competitions/titanic).
            
    Returns:
        str: A formatted string containing the competition title and description, 
            or an error message if the fetch fails.
    """
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
            
            # Extract title tag
            title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Unknown Title"
            
            # Extract meta description tag (fallback to og:description)
            desc_match = re.search(r'<meta\s+name="description"\s+content="([^"]+)"', html, re.IGNORECASE)
            if not desc_match:
                desc_match = re.search(r'<meta\s+property="og:description"\s+content="([^"]+)"', html, re.IGNORECASE)
            desc = desc_match.group(1).strip() if desc_match else "No description available"
            
            return f"Title: {title}\nDescription: {desc}"
    except Exception as e:
        return f"Error fetching webpage: {e}"

def custom_web_search(query: str) -> str:
    """
    Searches the web for the given query and returns a JSON string.
    
    This uses the DuckDuckGo Search (DDGS) library to avoid Google's strict API 
    limits. The returned JSON string contains titles, URLs, and text snippets 
    which the LLM can use to cite its sources correctly.
    
    Args:
        query (str): The search query (e.g., "LightGBM Kaggle best practices").
        
    Returns:
        str: A JSON-encoded string representing a list of search result dictionaries.
    """
    try:
        results = DDGS().text(query, max_results=3)
        return json.dumps(results)
    except Exception as e:
        return f"Error performing search: {str(e)}"

def extract_and_validate_kaggle_url(text: str) -> tuple[str | None, bool]:
    """
    Extracts a Kaggle URL from the user's chat input and validates its format and reachability.
    
    This function parses the text, finds potential URLs, ensures they point to Kaggle 
    competitions or datasets, and verifies that the URL returns a HTTP 200 OK status.
    
    Args:
        text (str): The raw text input from the user.
        
    Returns:
        tuple[str | None, bool]: A tuple containing the validated URL (or None) and a 
            boolean indicating whether the validation was successful.
    """
    urls = re.findall(r'(https?://[^\s]+)', text)
    for url in urls:
        url = url.rstrip('.,;:)("')
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                continue
            if parsed.username or parsed.password:
                continue
            
            host = parsed.hostname
            if not host:
                continue
            host = host.lower()
            # Enforce kaggle.com domain
            if host != 'kaggle.com' and not host.endswith('.kaggle.com'):
                continue
                
            # Ensure it's a competition or dataset path
            if not re.search(r'/(competitions|c|datasets)/[\w-]+', parsed.path.lower()):
                continue
                
            # SSRF Protection: ensure IP is public
            ip_str = socket.gethostbyname(host)
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                continue
                
            # Final reachability check
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.getcode() == 200:
                    return url, True
        except Exception:
            continue
            
    # Fallback to extract URL without explicit http/https scheme
    path_match = re.search(r'(?:https?://)?(?:www\.)?(kaggle\.com/(?:competitions|c|datasets)/[\w-]+)', text, re.IGNORECASE)
    if path_match:
        fallback_url = "https://www." + path_match.group(1)
        try:
            parsed = urllib.parse.urlparse(fallback_url)
            host = parsed.hostname
            ip_str = socket.gethostbyname(host)
            ip = ipaddress.ip_address(ip_str)
            if not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast):
                req = urllib.request.Request(
                    fallback_url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.getcode() == 200:
                        return fallback_url, True
        except Exception:
            pass
            
    return None, False

