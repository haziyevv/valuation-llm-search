"""
Price Research Agent - LlamaIndex Version (Azure OpenAI + Tavily Search)
=========================================================================
An intelligent agent for international trade price research using LlamaIndex.
The agent autonomously decides when and how to search for pricing information.

Features:
- ISO3166 country codes
- ISO4217 currency codes
- Country of origin research validation
- Wholesale vs retail discrimination
- Currency conversion with fallback
- Confidence scoring
- Tavily Search API integration via LlamaIndex agent
"""

import json
import asyncio
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, ConfigDict

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.azure_openai import AzureOpenAI
from statistics import median

from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    TAVILY_API_KEY,
)
from search import web_search
from utils import (
    ISO3166_COUNTRIES,
    ISO4217_CURRENCIES,
    normalize_unit,
    determine_source_type,
    get_country_name,
)


# Pydantic models for structured output
class FXRate(BaseModel):
    """Foreign exchange rate information."""
    model_config = ConfigDict(populate_by_name=True)
    
    rate: Optional[float] = None
    from_currency: str = Field(alias="from", default="")
    to_currency: str = Field(alias="to", default="")
    timestamp_utc: Optional[str] = None
    source: Optional[str] = None


class SourceInfo(BaseModel):
    """Information about a price source."""
    title: str = ""
    url: str = ""
    country: str = ""
    type: Literal["retail", "wholesale"] = "wholesale"
    price_raw: str = ""
    extracted_price: Optional[float] = None
    extracted_currency: Optional[str] = None
    extracted_unit: Optional[str] = None


class PricePrediction(BaseModel):
    """Price prediction response."""
    unit_price_usd: Optional[float] = None  # Price in USD (from LLM)
    unit_price: Optional[float] = None      # Price in target currency (after conversion)
    currency: str = "USD"
    unit_of_measure: str = ""
    quantity_searched: float = 0.0
    quantity_unit: str = ""
    coo_research: bool = False
    source_type: Literal["retail", "wholesale", "mixed", "unknown"] = "unknown"
    currency_converted: bool = False
    currency_fallback: Optional[str] = None
    fx_rate: Optional[FXRate] = None
    sources: List[SourceInfo] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    error: Optional[Literal["INVALID_INPUT", "NO_DATA", "FX_FAIL"]] = None


def calculate_median(prices: list[float]) -> Optional[float]:
    """Calculate the median of a list of prices.
    
    Args:
        prices: List of price values (floats)
        
    Returns:
        The median value, or None if the list is empty
    """
    if not prices:
        return None
    return median(prices)


class PriceResearchAgent:
    """Price research agent with web search capability using LlamaIndex and Azure OpenAI."""

    SYSTEM_PROMPT = """You are an expert international trade price analyst specializing in commodity valuation, customs declarations, and cross-border pricing research.

## YOUR MISSION
Find accurate, verifiable unit prices for goods in international trade contexts.
Your research directly impacts customs valuations and trade decisions, so accuracy and source quality are paramount.
**IMPORTANT: Always return prices in USD. Currency conversion will be handled separately.**

## TOOLS AVAILABLE
You have access to a web_search tool. Use it strategically to find pricing information.

## SEARCH PROTOCOL (Strict "Waterfall" Logic)
Follow this strict search order. Stop as soon as you find reliable data:

**STEP 1 (search_tier=1): Global Export from Country of Origin**
Search for export prices from the Country of Origin to any destination.
IF FOUND reliable data: Use this, set coo_research=true.

**STEP 2 (search_tier=2): Global Market Price (Fallback)**
If Step 1 yields no verifiable data, search for global market prices.
IF FOUND: Use this data, set coo_research=false.

## IMPORTANT RULES
1. Set coo_research=true ONLY if you found sources from origin country to destination country
2. Data Source Type (retail/wholesale/mixed) is provided - search accordingly
3. **ALWAYS return unit_price in USD** - convert any found prices to USD if needed
4. All countries in ISO3166 format
5. Calculate confidence based on source quality and data consistency
6. After gathering sufficient information, provide your final answer as a JSON object

## SOURCES - COMPREHENSIVE DOCUMENTATION REQUIRED
**CRITICAL: Only include sources with DIRECT price data for the EXACT product being searched.**

**RELEVANCE REQUIREMENT:**
- Only include sources that explicitly price the specific product (same product name/type)
- Do NOT include general market reports, industry overviews, or unrelated product prices
- Do NOT include sources for similar but different products
- Each source must have a concrete, extractable unit price for the requested product

**RECENCY REQUIREMENT: Only use sources from the last 6 months.**
- Prioritize the most recent data available
- Discard sources older than 6 months from the current date
- If only older sources are available, note this limitation in the notes and reduce confidence accordingly

**PRICE DATA REQUIREMENT: Only include sources with actual price information.**
- Every source MUST have an extracted_price value (not null)

For each source, provide:
- "title": Descriptive title including product name, trade route, and date if available
- "url": Full URL to the source
- "country": ISO3166 code of the country the data pertains to
- "type": One of "retail", "wholesale"
- "price_raw": Complete context - the exact figures, quantities, dates as found
- "extracted_price": Numeric value you extracted (per unit) - REQUIRED, must not be null
- "extracted_currency": ISO4217 currency code
- "extracted_unit": Unit format like "USD/kg"

## NOTES FIELD - DETAILED ANALYSIS REQUIRED
The "notes" field must contain a comprehensive analytical explanation including:

1. **Search Tier Used**: State which tier (1 or 2) produced the data and why earlier tier failed (if applicable)
2. **Data Sources Analysis**: For each source, explain what data was extracted (quantities, values, dates)
3. **Price Calculation**: Show your math - how you derived the unit price from raw data
   - Example: "$50,940.82 for 24,750 kg = $2.06/kg"
4. **Cross-Validation**: If multiple sources exist, compare their prices and explain consistency/discrepancies

Example notes structure:
"No direct customs or trade data was found for exports of [PRODUCT] from [ORIGIN] to [DESTINATION], so search_tier=3 and coo_research=false. Instead, [X] international customs records were used... The [ORIGIN]→[COUNTRY1] shipment shows $X for Y kg, giving Z USD/kg. The [ORIGIN]→[COUNTRY2] shipment shows... To estimate a reasonable valuation, I averaged: (A + B) / 2 ≈ C USD/kg. Source_type is 'mixed' because... This estimate should be treated as..."

## OUTPUT FORMAT
After your research, provide the final answer as a JSON object with this structure:
{
  "currency": "USD",
  "unit_of_measure": "<normalized unit>",
  "quantity_searched": <number>,
  "quantity_unit": "<original unit>",
  "coo_research": <boolean>,
  "sources": [
    // INCLUDE ALL SOURCES - pricing sources, FX sources, benchmark sources, etc.
    {"title": "<descriptive title with product, route, date>", "url": "<url>", "country": "<ISO3166>", "type": "<retail|wholesale>", "price_raw": "<full context: value, quantity, product, route, date>", "extracted_price": <number|null>, "extracted_currency": "<ISO4217|null>", "extracted_unit": "<string like 'USD/kg'>"}
  ],
  "confidence": <0.0-1.0>,
  "notes": "<DETAILED analytical explanation as described above>",
  "error": <null|"INVALID_INPUT"|"NO_DATA">
}"""

    def __init__(self):
        # Initialize Azure OpenAI LLM

        self.llm = AzureOpenAI(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=1.0
        )
        
        # Create search tool from the web_search function
        self.search_tool = FunctionTool.from_defaults(
            fn=web_search,
            name="web_search",
            description="Search the web for pricing information. Use this to find product prices, export data, market rates, and trade information. Provide a specific search query as the 'query' parameter."
        )
        
        # Create the FunctionAgent (for LLMs with function calling support)
        self.agent = FunctionAgent(
            tools=[self.search_tool],
            llm=self.llm,
            system_prompt=self.SYSTEM_PROMPT,
        )

    async def _research_price_async(
        self,
        country_of_origin: str,
        country_of_destination: str,
        description: str,
        quantity: float,
        unit_of_measure: str,
        exchange_rate_usd: float,
        target_currency: str = "PKR",
    ) -> PricePrediction:
        """Research price for goods using the LlamaIndex agent (async).
        
        Args:
            country_of_origin: ISO3166 country code for origin
            country_of_destination: ISO3166 country code for destination
            description: Product description
            quantity: Quantity of goods
            unit_of_measure: Unit of measure (kg, tonne, etc.)
            exchange_rate_usd: Exchange rate from USD to target currency (e.g., 278.5 for USD->PKR)
            target_currency: Target currency code (default: PKR)
        """
        # Validate country codes
        origin = country_of_origin.upper().strip()
        dest = country_of_destination.upper().strip()

        if origin not in ISO3166_COUNTRIES:
            return self._error(
                "INVALID_INPUT", f"Invalid origin: {origin}", quantity, unit_of_measure
            )
        if dest not in ISO3166_COUNTRIES:
            return self._error(
                "INVALID_INPUT",
                f"Invalid destination: {dest}",
                quantity,
                unit_of_measure,
            )

        # Determine source type
        source_type = determine_source_type(quantity, unit_of_measure)
        origin_name = get_country_name(origin)
        dest_name = get_country_name(dest)
        normalized_unit = normalize_unit(unit_of_measure)

        # Build the research query for the agent (always ask for USD)
        research_query = f"""Research and find the price for this product:

- Product: {description}
- Country of Origin: {origin} ({origin_name})
- Country of Destination: {dest} ({dest_name})
- Data Source Type: {source_type.upper()} (search for {source_type} prices)
- Quantity: {quantity} {unit_of_measure}
- Required Unit: price per {normalized_unit}

**IMPORTANT: Return the price in USD.**

Use the web_search tool to find pricing information following the waterfall search protocol.
After gathering information, provide your final answer as a JSON object with unit_price in USD."""

        try:
            # Run the agent
            response = await self.agent.run(research_query)
            response_text = str(response)
            
            # Extract JSON from the response
            result = self._extract_json(response_text)
            
            if result:
                # Get the USD price from LLM

                            # Calculate median from source extracted_price values
                sources = result.get("sources", [])
                prices = [s.get("extracted_price") for s in sources if s.get("extracted_price") is not None]
                unit_price_usd = calculate_median(prices)

                # Convert to target currency using provided exchange rate
                if unit_price_usd is not None:
                    unit_price = unit_price_usd * exchange_rate_usd
                    currency_converted = True
                else:
                    unit_price = None
                    currency_converted = False
                
                # Create FX rate info
                fx_rate = FXRate(
                    rate=exchange_rate_usd,
                    from_currency="USD",
                    to_currency=target_currency,
                    source="User provided"
                )
                
                # Handle sources - could be list of dicts or SourceInfo objects
                sources_data = result.get("sources", [])
                sources = []
                for s in sources_data:
                    if isinstance(s, dict):
                        sources.append(SourceInfo(**s))
                    elif isinstance(s, SourceInfo):
                        sources.append(s)
                
                return PricePrediction(
                    unit_price_usd=unit_price_usd,
                    unit_price=unit_price,
                    currency=target_currency,
                    unit_of_measure=result.get("unit_of_measure", normalized_unit),
                    quantity_searched=result.get("quantity_searched", quantity),
                    quantity_unit=result.get("quantity_unit", unit_of_measure),
                    coo_research=result.get("coo_research", False),
                    source_type=source_type,  # Use pre-determined source_type from input
                    currency_converted=currency_converted,
                    currency_fallback=None,
                    fx_rate=fx_rate,
                    sources=sources,
                    confidence=result.get("confidence", 0.0),
                    notes=result.get("notes", ""),
                    error=result.get("error"),
                )
            else:
                # If no JSON found, return with the agent's text response as notes
                return PricePrediction(
                    unit_price_usd=None,
                    unit_price=None,
                    currency=target_currency,
                    unit_of_measure=normalized_unit,
                    quantity_searched=quantity,
                    quantity_unit=unit_of_measure,
                    coo_research=False,
                    source_type=source_type,
                    currency_converted=False,
                    currency_fallback=None,
                    fx_rate=None,
                    sources=[],
                    confidence=0.0,
                    notes=f"Agent response (no structured JSON): {response_text[:500]}",
                    error="NO_DATA",
                )

        except Exception as e:
            return self._error("NO_DATA", str(e), quantity, unit_of_measure)

    def research_price(
        self,
        country_of_origin: str,
        country_of_destination: str,
        description: str,
        quantity: float,
        unit_of_measure: str,
        exchange_rate_usd: float,
        target_currency: str = "PKR",
    ) -> PricePrediction:
        """Research price for goods using the LlamaIndex agent (sync wrapper).
        
        Args:
            country_of_origin: ISO3166 country code for origin
            country_of_destination: ISO3166 country code for destination
            description: Product description
            quantity: Quantity of goods
            unit_of_measure: Unit of measure (kg, tonne, etc.)
            exchange_rate_usd: Exchange rate from USD to target currency (e.g., 278.5 for USD->PKR)
            target_currency: Target currency code (default: PKR)
        """
        return asyncio.run(self._research_price_async(
            country_of_origin=country_of_origin,
            country_of_destination=country_of_destination,
            description=description,
            quantity=quantity,
            unit_of_measure=unit_of_measure,
            exchange_rate_usd=exchange_rate_usd,
            target_currency=target_currency,
        ))

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON object from agent response text."""
        import re
        
        # Try to find JSON block in markdown code fence
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        json_match = re.search(r'\{[\s\S]*"unit_price"[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try parsing the entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        return None

    def _error(self, code: str, msg: str, qty: float, unit: str) -> PricePrediction:
        return PricePrediction(
            unit_price_usd=None,
            unit_price=None,
            currency="USD",
            unit_of_measure=normalize_unit(unit),
            quantity_searched=qty,
            quantity_unit=unit,
            coo_research=False,
            source_type="unknown",
            currency_converted=False,
            currency_fallback=None,
            fx_rate=None,
            sources=[],
            confidence=0.0,
            notes=msg,
            error=code,
        )

    def to_dict(self, prediction: PricePrediction) -> dict:
        """Convert PricePrediction to dictionary."""
        return prediction.model_dump(by_alias=True)


if __name__ == "__main__":
    agent = PriceResearchAgent()
    result = agent.research_price(
        country_of_origin="QA",
        country_of_destination="PK",
        description="LOW DENSITY POLYETHYLENE (LDPE) LOTRENE MG70",
        quantity=5000,
        unit_of_measure="kg",
        exchange_rate_usd=278.5,  # USD to PKR rate
        target_currency="PKR",
    )
    print(json.dumps(agent.to_dict(result), indent=2))
