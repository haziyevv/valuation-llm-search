import requests

from config import WEBSEARCH_PLUS_API_KEY


def web_search(query: str) -> str:
    """
    Search the web for information using Web Search Plus API.
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing search results with titles, URLs, and content snippets
    """
    try:
        url = "https://api.websearch.plus/v1/web_search_plus"
        
        headers = {
            "Authorization": f"Bearer {WEBSEARCH_PLUS_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "search_context_size": "high",
            "language": "en",
            "mode": "smart",
            "type": "search",
            "qdr": "m"
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Check for errors in response
        if "error" in data:
            return f"Search error: {data['error']}"
        
        # Web Search Plus returns results as a pre-formatted string
        results = data.get("results", "")
        
        if not results:
            return "No search results found for this query."
        
        # Results is already a formatted string from the API
        if isinstance(results, str):
            return results
        
        # Fallback: if results is a list, format it
        if isinstance(results, list):
            formatted = []
            for result in results:
                title = result.get("title", "")
                result_url = result.get("url", "")
                content = result.get("content", "")
                formatted.append(f"Title: {title}\nURL: {result_url}\nContent: {content}")
            return "\n\n---\n\n".join(formatted) if formatted else "No search results found."
        
        return str(results)
        
    except requests.exceptions.RequestException as e:
        return f"Search error: {str(e)}"
    except Exception as e:
        return f"Search error: {str(e)}"
