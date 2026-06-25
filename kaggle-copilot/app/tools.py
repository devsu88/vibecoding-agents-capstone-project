import os
import re
import socket
import ipaddress
import urllib.request
import urllib.parse
import urllib.error
import zipfile
import json
from kaggle.api.kaggle_api_extended import KaggleApi
from ddgs import DDGS

def fetch_kaggle_competition_metadata(url: str) -> str:
    """Fetches the title and description from a Kaggle competition webpage."""
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
            title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Unknown Title"
            
            desc_match = re.search(r'<meta\s+name="description"\s+content="([^"]+)"', html, re.IGNORECASE)
            if not desc_match:
                desc_match = re.search(r'<meta\s+property="og:description"\s+content="([^"]+)"', html, re.IGNORECASE)
            desc = desc_match.group(1).strip() if desc_match else "No description available"
            
            return f"Title: {title}\nDescription: {desc}"
    except Exception as e:
        return f"Error fetching webpage: {e}"

def custom_web_search(query: str) -> str:
    """Searches the web for the given query and returns a JSON string containing titles, URLs, and snippets."""
    try:
        results = DDGS().text(query, max_results=3)
        return json.dumps(results)
    except Exception as e:
        return f"Error performing search: {str(e)}"

def extract_and_validate_kaggle_url(text: str) -> tuple[str | None, bool]:
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
            if host != 'kaggle.com' and not host.endswith('.kaggle.com'):
                continue
                
            if not re.search(r'/(competitions|c|datasets)/[\w-]+', parsed.path.lower()):
                continue
                
            ip_str = socket.gethostbyname(host)
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                continue
                
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.getcode() == 200:
                    return url, True
        except Exception:
            continue
            
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

def download_kaggle_competition_data(url: str) -> str:
    """Downloads and extracts the dataset for a Kaggle competition using the Kaggle API."""
    slug_match = re.search(r'/(?:competitions|c|datasets)/([\w-]+)', url)
    if not slug_match:
        return "Error: Could not extract competition slug from URL."
    
    slug = slug_match.group(1)
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        api.competition_download_files(slug, path='.', force=False, quiet=True)
        
        zip_path = f"{slug}.zip"
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            return f"Successfully downloaded and extracted dataset for {slug}."
        else:
            return f"Successfully downloaded dataset for {slug}, but no zip file was found."
    except Exception as e:
        return f"Error downloading dataset: {str(e)}"
