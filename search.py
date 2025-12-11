import os
from tavily import TavilyClient

from config import TAVILY_API_KEY


tavily = TavilyClient(api_key=TAVILY_API_KEY)

def web_search(query: str) -> str:
    """
    Search the web for information using Tavily API.
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing search results with titles, URLs, and content snippets
    """
    try:
        response = tavily.search(
            query=query,
            max_results=10,
            include_raw_content=True,
            search_depth="advanced"
        )
        
        results = []
        for result in response.get("results", []):
            title = result.get("title", "")
            url = result.get("url", "")
            snippet = result.get("content", "")
            raw_content = result.get("raw_content", "")
            
            # Truncate raw content if too long
            if raw_content and len(raw_content) > 2000:
                raw_content = raw_content[:2000] + "..."
            
            result_text = f"Title: {title}\nURL: {url}\nSnippet: {snippet}"
            if raw_content:
                result_text += f"\nPage Content: {raw_content}"
            results.append(result_text)
        
        if not results:
            return "No search results found for this query."
        
        return "\n\n---\n\n".join(results)
        
    except Exception as e:
        return f"Search error: {str(e)}"
