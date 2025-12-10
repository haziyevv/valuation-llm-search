"""
Price Research Agent - Enhanced Version (Azure OpenAI + Tavily Search)
=======================================================================
An intelligent agent for international trade price research with:
- ISO3166 country codes
- ISO4217 currency codes
- Country of origin research validation
- Wholesale vs retail discrimination
- Currency conversion with fallback
- Confidence scoring
- Tavily Search API integration (optimized for AI agents)
"""

from openai import AzureOpenAI
from tavily import TavilyClient
import json
import os
from typing import Optional, Literal
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from utils import ISO3166_COUNTRIES, ISO4217_CURRENCIES, COUNTRY_TO_CURRENCY, UNIT_MAP, THRESHOLDS
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# Tavily Search Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
@dataclass
class PricePrediction:
    """Price prediction response."""
    unit_price: Optional[float]
    currency: str
    unit_of_measure: str
    quantity_searched: float
    quantity_unit: str
    coo_research: bool
    source_type: Literal["retail", "wholesale", "mixed", "unknown"]
    currency_converted: bool
    currency_fallback: Optional[str]
    fx_rate: Optional[dict]
    sources: list
    confidence: float
    notes: str
    error: Optional[str]


def normalize_unit(unit: str) -> str:
    return UNIT_MAP.get(unit.lower().strip(), unit.lower().strip())


def get_country_name(code: str) -> str:
    return ISO3166_COUNTRIES.get(code.upper(), code)


def get_default_currency(country_code: str) -> str:
    return COUNTRY_TO_CURRENCY.get(country_code.upper(), "USD")


def determine_source_type(quantity: float, unit: str) -> Literal["retail", "wholesale", "mixed"]:
    """Determine retail/wholesale based on quantity."""
    norm_unit = normalize_unit(unit)
    
    weight_units = ["kg", "g", "tonne", "lb"]
    volume_units = ["litre", "ml", "m3"]
    count_units = ["piece", "unit", "set"]
    
    if norm_unit in weight_units:
        qty = quantity * 1000 if norm_unit == "tonne" else quantity / 1000 if norm_unit == "g" else quantity
        retail_max, wholesale_min = THRESHOLDS["weight"]
    elif norm_unit in volume_units:
        qty = quantity * 1000 if norm_unit == "m3" else quantity / 1000 if norm_unit == "ml" else quantity
        retail_max, wholesale_min = THRESHOLDS["volume"]
    elif norm_unit in count_units:
        qty = quantity
        retail_max, wholesale_min = THRESHOLDS["count"]
    else:
        return "mixed"
    
    if qty < retail_max:
        return "retail"
    elif qty >= wholesale_min:
        return "wholesale"
    return "mixed"


class PriceResearchAgent:
    """Price research agent with web search capability using Azure OpenAI."""
    
    SYSTEM_PROMPT = """You are an expert international trade price analyst specializing in commodity valuation, customs declarations, and cross-border pricing research.

## YOUR MISSION
Find accurate, verifiable unit prices for goods in international trade contexts.
Your research directly impacts customs valuations and trade decisions, so accuracy and source quality are paramount.
Analyze the web search results provided to find the most accurate pricing information.

RULES:
## SEARCH PROTOCOL (Strict "Waterfall" Logic)
Analyze the provided search results following this strict order:

**STEP 1: Bilateral Trade (Country of Origin -> Country of Destination)**
* Look for export prices specifically from the Country of Origin to the Country of Destination for the [Data Source Type].
* *IF FOUND:* Use this data, mark `search_tier` as "1".

**STEP 2: Global Export (COO -> World)**
* If Step 1 yields no verifiable data, look for the general export price of the product from the Country of Origin to *any* country for the [Data Source Type].
* *IF FOUND:* Use this data, mark `search_tier` as "2".

**STEP 3: Global Market Price (Fallback)**
* If Steps 1 and 2 yield no data, use the global market price (commodity benchmarks, major international marketplaces) for the [Data Source Type].
* *IF FOUND:* Use this data, mark `search_tier` as "3".

2. Set coo_research=true in the response if you found any sources from the origin country to the destination country, false otherwise
3. Data Source Type is the type of data source you are using to find the price. It can be retail, wholesale, mixed, or unknown. For unknown source just use mixed.
4. Convert prices to target currency. If conversion fails, fallback to USD
5. All currencies should be written in ISO4217, countries in ISO3166
6. All prices should be in the Target currency

OUTPUT JSON ONLY:
{
  "unit_price": <number|null>,
  "currency": "<ISO4217>",
  "unit_of_measure": "<normalized unit>",
  "quantity_searched": <number>,
  "quantity_unit": "<original unit>",
  "coo_research": <boolean>,
  "source_type": "<retail|wholesale|mixed|unknown>",
  "currency_converted": <boolean>,
  "currency_fallback": "<ISO4217|null>",
  "fx_rate": {"rate": <number|null>, "from": "<ISO4217>", "to": "<ISO4217>", "timestamp_utc": "<ISO8601|null>", "source": "<string|null>"},
  "sources": [{"title": "<string>", "url": "<url>", "country": "<ISO3166>", "type": "<retail|wholesale|official|market|customs|other>", "price_raw": "<string>", "extracted_price": <number|null>, "extracted_currency": "<ISO4217|null>", "extracted_unit": "<string|null>"}],
  "confidence": <0.0-1.0>,
  "notes": "<string>",
  "error": <null|"INVALID_INPUT"|"NO_DATA"|"FX_FAIL">
}"""

    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION
        )
        if TAVILY_API_KEY:
            self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        else:
            self.tavily = None
    
    def _web_search(self, query: str, max_results: int = 10) -> list[dict]:
        """Perform web search using Tavily API."""
        if not self.tavily:
            print("Warning: TAVILY_API_KEY not set")
            return []
        
        try:
            import pdb; pdb.set_trace()
            response = self.tavily.search(
                query=query,
                max_results=max_results,
                include_raw_content=True,
                search_depth="advanced"
            )
            
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", ""),
                    "raw_content": result.get("raw_content", "")[:2000] if result.get("raw_content") else "",
                    "score": result.get("score", 0)
                })
            return results
            
        except Exception as e:
            print(f"Tavily Search error: {e}")
            return []
    
    def research_price(
        self,
        country_of_origin: str,
        country_of_destination: str,
        description: str,
        quantity: float,
        unit_of_measure: str,
        preferred_currency: Optional[str] = None
    ) -> PricePrediction:
        """Research price for goods."""
        # Validate country codes
        origin = country_of_origin.upper().strip()
        dest = country_of_destination.upper().strip()
        
        if origin not in ISO3166_COUNTRIES:
            return self._error("INVALID_INPUT", f"Invalid origin: {origin}", quantity, unit_of_measure)
        if dest not in ISO3166_COUNTRIES:
            return self._error("INVALID_INPUT", f"Invalid destination: {dest}", quantity, unit_of_measure)
        
        # Determine target currency
        target_currency = preferred_currency.upper() if preferred_currency else get_default_currency(dest)
        if target_currency not in ISO4217_CURRENCIES:
            return self._error("INVALID_INPUT", f"Invalid currency: {target_currency}", quantity, unit_of_measure)
        
        # Determine source type
        source_type = determine_source_type(quantity, unit_of_measure)
        origin_name = get_country_name(origin)
        dest_name = get_country_name(dest)
        
        # Perform tiered web searches
        search_results = []
        
        # Tier 1: Bilateral trade search
        tier1_query = f"{description} price {origin_name} to {dest_name} {source_type} export"
        tier1_results = self._web_search(tier1_query)
        search_results.extend([{**r, "tier": 1} for r in tier1_results])
        
        # Tier 2: COO export search
        tier2_query = f"{description} price {origin_name} export {source_type}"
        tier2_results = self._web_search(tier2_query)
        search_results.extend([{**r, "tier": 2} for r in tier2_results])
        
        # Tier 3: Global market search
        tier3_query = f"{description} {source_type} price per {normalize_unit(unit_of_measure)} international market"
        tier3_results = self._web_search(tier3_query)
        search_results.extend([{**r, "tier": 3} for r in tier3_results])
        
        # Format search results for the prompt (include raw content when available)
        search_context = "\n\n".join([
            f"[Tier {r['tier']}] {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}" + 
            (f"\nPage Content: {r.get('raw_content', '')}" if r.get('raw_content') else "")
            for r in search_results
        ])
        
        # Build prompt with search results
        prompt = f"""Analyze these web search results and find the price for:
- Product: {description}
- Country of Origin: {origin} ({origin_name})
- Country of Destination: {dest} ({dest_name})
- Data Source Type: {source_type.upper()}
- Target currency: {target_currency}
- Quantity: {quantity} {unit_of_measure}

Return price per {normalize_unit(unit_of_measure)} in {target_currency}.

WEB SEARCH RESULTS:
{search_context}

Based on these search results, extract pricing information and return the JSON response."""

        try:
            response = self.client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return PricePrediction(
                unit_price=result.get("unit_price"),
                currency=result.get("currency", target_currency),
                unit_of_measure=result.get("unit_of_measure", normalize_unit(unit_of_measure)),
                quantity_searched=result.get("quantity_searched", quantity),
                quantity_unit=result.get("quantity_unit", unit_of_measure),
                coo_research=result.get("coo_research", False),
                source_type=result.get("source_type", "unknown"),
                currency_converted=result.get("currency_converted", False),
                currency_fallback=result.get("currency_fallback"),
                fx_rate=result.get("fx_rate"),
                sources=result.get("sources", []),
                confidence=result.get("confidence", 0.0),
                notes=result.get("notes", ""),
                error=result.get("error")
            )
        except json.JSONDecodeError as e:
            return self._error("INVALID_INPUT", f"Parse error: {e}", quantity, unit_of_measure)
        except Exception as e:
            return self._error("NO_DATA", str(e), quantity, unit_of_measure)
    
    def _error(self, code: str, msg: str, qty: float, unit: str) -> PricePrediction:
        return PricePrediction(
            unit_price=None, currency="USD", unit_of_measure=normalize_unit(unit),
            quantity_searched=qty, quantity_unit=unit, coo_research=False,
            source_type="unknown", currency_converted=False, currency_fallback=None,
            fx_rate=None, sources=[], confidence=0.0, notes=msg, error=code
        )
    
    def to_dict(self, prediction: PricePrediction) -> dict:
        return asdict(prediction)


if __name__ == "__main__":
    agent = PriceResearchAgent()
    result = agent.research_price(
        country_of_origin="QA",
        country_of_destination="PK",
        description="LOW DENSITY POLYETHYLENE (LDPE) LOTRENE MG70",
        quantity=5000,
        unit_of_measure="kg"
    )
    print(json.dumps(agent.to_dict(result), indent=2))
