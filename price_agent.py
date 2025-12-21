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
import csv
from datetime import datetime, timedelta
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

SOURCE_MAX_AGE_DAYS = 90

# Pydantic models for structured output
class FXRate(BaseModel):
    """Foreign exchange rate information."""
    model_config = ConfigDict(populate_by_name=True)
    
    rate: Optional[float] = None
    from_currency: str = Field(alias="from", default="")
    to_currency: str = Field(alias="to", default="")
    timestamp_utc: Optional[str] = None
    source: Optional[str] = None


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
    sources: list
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


def filter_recent_sources(sources: list[dict], max_age_days: int = SOURCE_MAX_AGE_DAYS) -> list[dict]:
    """Filter sources to only include those within the maximum age limit.
    
    Args:
        sources: List of source dictionaries, each should have a "date" field (YYYY-MM-DD)
        max_age_days: Maximum age in days for a source to be considered valid
        
    Returns:
        List of sources that are within the age limit
    """
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    valid_sources = []
    
    for source in sources:
        source_date_str = source.get("date")
        if source_date_str:
            try:
                # Parse the date (supports YYYY-MM-DD format)
                source_date = datetime.strptime(source_date_str, "%Y-%m-%d")
                if source_date >= cutoff_date:
                    valid_sources.append(source)
            except ValueError:
                # If date parsing fails, skip this source
                pass
        # Sources without a date are excluded (strict mode)
    
    return valid_sources


class PriceResearchAgent:
    """Price research agent with web search capability using LlamaIndex and Azure OpenAI."""

    SYSTEM_PROMPT = """You are an expert international trade price analyst specializing in commodity valuation, customs declarations, and cross-border pricing research.

## YOUR MISSION
Find accurate, verifiable unit prices for goods in international trade contexts.
Your research directly impacts customs valuations and trade decisions, so accuracy and source quality are paramount.
**IMPORTANT: Always return prices in USD. Currency conversion will be handled separately.**

## TOOLS AVAILABLE
You have access to a web_search tool. Use it strategically to find pricing information.

Only include sources published on or after the Source Cutoff Date.

## RECENCY GATE (HARD REQUIREMENT)
You will be given a **Source Cutoff Date** in the user prompt (YYYY-MM-DD).
- Every source MUST include a `date` field in YYYY-MM-DD.
- If a source page does not clearly show a publish/record date, treat its date as UNKNOWN and **do not include it**.
- If a source only provides month/year (no day), treat it as UNKNOWN and **do not include it**.

## IMPORTANT RULES
1. Data Source Type (retail/wholesale) is provided - search accordingly
2. Convert any found prices to USD
3. All countries in ISO3166 format
4. Calculate confidence based on source quality and data consistency
5. After gathering sufficient information, provide your final answer as a JSON object

## UNIT CONVERSION (CRITICAL)
**You MUST convert all prices to the EXACT unit requested by the user.**
- If user requests price per "kg", convert ALL prices to USD/kg
- If source shows price per tonne, DIVIDE by 1000 to get per kg
- If source shows price per gram, MULTIPLY by 1000 to get per kg
- The "extracted_price" field MUST be in the requested unit, NOT the source's original unit
- Example: Source shows "$1200/tonne" → extracted_price = 1.20 (USD/kg)

## SOURCES - STRICT RELEVANCE REQUIRED
**CRITICAL: Only include sources with DIRECT price data for the EXACT product being searched.**

**RELEVANCE REQUIREMENT:**
- Only include sources that explicitly price the specific product (same product name/type)
- Do NOT include general market reports, industry overviews, or unrelated product prices
- Do NOT include sources for similar but different products
- Each source must have a concrete, extractable unit price for the requested product

**EXCLUDE EXCHANGE RATE SOURCES:**
- Do NOT include exchange rate or FX sources in the sources list
- You may use exchange rates internally for conversion, but do NOT list them as sources

**PRICE DATA REQUIREMENT: Only include sources with actual price information.**
- Every source MUST have an extracted_price value (not null)

For each source, provide:
- "title": Descriptive title including product name, trade route, and date if available
- "url": Full URL to the source
- "country": ISO3166 code of the country the data pertains to
- "type": One of "retail", "wholesale"
- "date": ISO8601 date (YYYY-MM-DD) when the price data was published/recorded - REQUIRED for validation
- "price_raw": Complete context - the exact figures, quantities, dates as found
- "extracted_price": Numeric value you extracted (per unit) - REQUIRED, must not be null
- "extracted_currency": ISO4217 currency code
- "extracted_unit": Unit format like "USD/kg"

## NOTES FIELD - DETAILED ANALYSIS REQUIRED
The "notes" field must contain a comprehensive analytical explanation including:

1. **Data Sources Analysis**: For each source, explain what data was extracted (quantities, values, dates)
2. **Price Calculation**: Show your math - how you derived the unit price from raw data
   - Example: "$50,940.82 for 24,750 kg = $2.06/kg"

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
    {"title": "<descriptive title with product, route, date>", "url": "<url>", "country": "<ISO3166>", "type": "<retail|wholesale>", "date": "<YYYY-MM-DD>", "price_raw": "<full context: value, quantity, product, route, date>", "extracted_price": <number>, "extracted_currency": "<ISO4217>", "extracted_unit": "<string like 'USD/kg'>"}
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
        description: str,
        quantity: float,
        unit_of_measure: str,
        exchange_rate_usd: float,
        target_currency: str = "PKR",
    ) -> PricePrediction:
        """Research price for goods using the LlamaIndex agent (async).
        
        Args:
            country_of_origin: ISO3166 country code for origin
            description: Product description
            quantity: Quantity of goods
            unit_of_measure: Unit of measure (kg, tonne, etc.)
            exchange_rate_usd: Exchange rate from USD to target currency (e.g., 278.5 for USD->PKR)
            target_currency: Target currency code (default: PKR)
        """
        # Validate country codes
        origin = country_of_origin.upper().strip()

        if origin not in ISO3166_COUNTRIES:
            return self._error(
                "INVALID_INPUT", f"Invalid origin: {origin}", quantity, unit_of_measure
            )
        

        # Determine source type
        source_type = determine_source_type(quantity, unit_of_measure)
        origin_name = get_country_name(origin)
        normalized_unit = normalize_unit(unit_of_measure)
        target_unit = normalize_unit(unit_of_measure)
        current_date = datetime.now().strftime("%Y-%m-%d")
        cutoff_date = datetime.now() - timedelta(days=SOURCE_MAX_AGE_DAYS)

        # Build the research query for the agent (always ask for USD)
        research_query = f"""Research and find the price for this product:

- Product: {description}
- Data Source Type: {source_type.upper()} (search for {source_type} prices)
- Target unit: {target_unit} (CONVERT all prices to USD/{target_unit})
- Current Date: {current_date}
- Source Cutoff Date: {cutoff_date}

**IMPORTANT: Return the price in USD.**

Use the web_search tool to find pricing information .
After gathering information, provide your final answer as a JSON object with unit_price in USD/{target_unit}."""
        coo_research = True
        try:
            # Run the agent
            tier1_prompt = (
                research_query
                + f"\n\nCountry of Origin: {origin} ({origin_name})"
                + f"\n\nInvestigate unit price per {target_unit} of {description} in USD exported from {get_country_name(origin)}.\n"
                + f"Hard rule: ONLY include sources with date >= {cutoff_date}.\n"
            )
            response = await self.agent.run(tier1_prompt)
            response_text = str(response)
            
            # Extract JSON from the response
            result1 = self._extract_json(response_text)
            
            raw_source1 = result1.get("sources", [])
            sources1 = filter_recent_sources(raw_source1, SOURCE_MAX_AGE_DAYS)
            notes1 = result1.get("notes", "")

            # If Tier 1 has not more than 3 recent sources, also run Tier 2 and append its recent sources.
            sources2: list[dict] = []
            notes2: str = ""
            result2: dict = {}

            def _dedupe_sources(items: list[dict]) -> list[dict]:
                """De-dupe sources, preferring unique URLs, else title+date."""
                seen: set[str] = set()
                out: list[dict] = []
                for s in items:
                    url = (s.get("url") or "").strip()
                    title = (s.get("title") or "").strip()
                    date = (s.get("date") or "").strip()
                    key = url or f"{title}::{date}"
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(s)
                return out

            if len(sources1) < 3:
                coo_research = False
                tier2_prompt = (
                    research_query
                    + f"\n\nInvestigate global market unit price per {target_unit} of {description} in USD\n"
                    + f"Hard rule: ONLY include sources with date >= {cutoff_date}.\n"
                )

                response = await self.agent.run(tier2_prompt)
                response_text = str(response)
                result2 = self._extract_json(response_text)

                raw_source2 = result2.get("sources", [])
                sources2 = filter_recent_sources(raw_source2, SOURCE_MAX_AGE_DAYS)
                notes2 = result2.get("notes", "")
            
            combined_sources = _dedupe_sources(sources1 + sources2)
            combined_prices = [
                s.get("extracted_price")
                for s in combined_sources
                if s.get("extracted_price") is not None
            ]
            unit_price_usd = calculate_median(combined_prices)
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
            
            merged_notes = notes1
            if notes2:
                merged_notes = (notes1 or "") + "\n\n--- Tier 2 (fallback) ---\n" + notes2

            if not combined_prices or unit_price is None:
                return self._error("NO_DATA", merged_notes, quantity, unit_of_measure)

            # Prefer tier-1 metadata if available; fall back to tier-2.
            primary_result = result1 if sources1 else (result2 or result1)

            prediction = PricePrediction(
                unit_price_usd=unit_price_usd,
                unit_price=unit_price,
                currency=target_currency,
                unit_of_measure=primary_result.get("unit_of_measure", normalized_unit),
                quantity_searched=primary_result.get("quantity_searched", quantity),
                quantity_unit=primary_result.get("quantity_unit", unit_of_measure),
                coo_research=coo_research,
                source_type=source_type,  # Use pre-determined source_type from input
                currency_converted=currency_converted,
                currency_fallback=None,
                fx_rate=fx_rate,
                sources=combined_sources,
                confidence=primary_result.get("confidence", 0.0),
                notes=primary_result.get("notes", ""),
                error=primary_result.get("error"),
            )
            self.save_csv(prediction)

            return prediction

        except Exception as e:
            return self._error("NO_DATA", str(e), quantity, unit_of_measure)

    def research_price(
        self,
        country_of_origin: str,
        description: str,
        quantity: float,
        unit_of_measure: str,
        exchange_rate_usd: float,
        target_currency: str = "PKR",
    ) -> PricePrediction:
        """Research price for goods using the LlamaIndex agent (sync wrapper).
        
        Args:
            country_of_origin: ISO3166 country code for origin
            description: Product description
            quantity: Quantity of goods
            unit_of_measure: Unit of measure (kg, tonne, etc.)
            exchange_rate_usd: Exchange rate from USD to target currency (e.g., 278.5 for USD->PKR)
        """
        return asyncio.run(self._research_price_async(
            country_of_origin=country_of_origin,
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

    def save_csv(self, prediction: PricePrediction, filepath: str = "results.csv") -> None:
        """Save prediction to CSV file.
        
        Args:
            prediction: PricePrediction object to export
            filepath: File path to save the CSV (default: results.csv)
        """
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Predicted Price row
            if prediction.unit_price is not None:
                price_str = f"{prediction.unit_price:.2f} {prediction.currency}/{prediction.unit_of_measure}"
            else:
                price_str = "N/A"
            writer.writerow(["Predicted Price", price_str])
            writer.writerow([])  # Empty row for separation
            
            # Sources table header
            writer.writerow(["Title", "URL", "Raw Text", "Date", "Extracted Price"])
            
            # Sources data rows
            for source in prediction.sources:
                title = source.get("title", "")
                url = source.get("url", "")
                raw_text = source.get("price_raw", "")
                date = source.get("date", "")
                extracted_price = source.get("extracted_price", "")
                if extracted_price:
                     extracted_price = f"{extracted_price} {source.get('extracted_unit', '')}"
                writer.writerow([title, url, raw_text, date, extracted_price])

if __name__ == "__main__":
    agent = PriceResearchAgent()
    result = agent.research_price(
        country_of_origin="QA",
        description="LOW DENSITY POLYETHYLENE (LDPE) LOTRENE MG70",
        quantity=5000,
        unit_of_measure="kg",
        exchange_rate_usd=278.5,  # USD to PKR rate
        target_currency="PKR",
    )
    print(json.dumps(agent.to_dict(result), indent=2))
