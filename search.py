import requests

from config import SERPER_API_KEY

SERPER_API_URL = "https://google.serper.dev/search"


def web_search(query: str) -> str:
    """
    Search the web for information using Serper (Google Search) API.
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing search results with titles, URLs, and content snippets
    """
    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        
        payload = {
            "q": query,
            "gl": "us",
            "num": 10
        }
        response = requests.post(SERPER_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Extract organic results
        organic_results = data.get("organic", [])
        
        if not organic_results:
            return "No search results found for this query."
        
        # Format results
        formatted = []
        for result in organic_results:
            title = result.get("title", "")
            url = result.get("link", "")
            snippet = result.get("snippet", "")
            formatted.append(f"Title: {title}\nURL: {url}\nContent: {snippet}")
        
        return "\n\n---\n\n".join(formatted) if formatted else "No search results found."
        
    except requests.exceptions.RequestException as e:
        return f"Search error: {str(e)}"
    except Exception as e:
        return f"Search error: {str(e)}"
