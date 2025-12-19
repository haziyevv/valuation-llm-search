"""
Price Research Agent - Enhanced Version
=========================================
An intelligent agent for international trade price research with:
- ISO3166 country codes
- ISO4217 currency codes
- Country of origin research validation
- Wholesale vs retail discrimination
- Currency conversion with fallback
- Confidence scoring
"""

from openai import OpenAI
import json
import os
import re
from statistics import median
from typing import Optional, Literal
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from utils import ISO3166_COUNTRIES, ISO4217_CURRENCIES, COUNTRY_TO_CURRENCY, UNIT_MAP, THRESHOLDS
load_dotenv()

MODEL_NAME = "gpt-5.1"
@dataclass
class PricePrediction:
    """Price prediction response."""
    unit_price: Optional[float]
    currency: str
    unit_of_measure: str
    quantity_searched: float
    quantity_unit: str
    coo_research: bool
    source_type: Literal["retail", "wholesale"]
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


def determine_source_type(quantity: float, unit: str) -> Literal["retail", "wholesale"]:
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
        return "retail"
    
    if qty < retail_max:
        return "retail"
    return "wholesale"


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
    """Price research agent with web search capability."""
    
    SYSTEM_PROMPT = """You are an expert international trade price analyst specializing in commodity valuation, customs declarations, and cross-border pricing research.

## YOUR MISSION
Find accurate, verifiable unit prices for goods in international trade contexts.
Your research directly impacts customs valuations and trade decisions, so accuracy and source quality are paramount.
**IMPORTANT: Always return prices in USD.**

## SEARCH PROTOCOL (Strict "Waterfall" Logic)
Follow this strict search order. Stop as soon as you find reliable data:

**STEP 1 (search_tier=1): Global Export from Country of Origin**
Search for export prices from the Country of Origin to any destination.
IF FOUND reliable data: Use this, set coo_research=true.

**STEP 2 (search_tier=2): Global Market Price (Fallback)**
If Step 1 yields no verifiable data, search for global market prices.
IF FOUND: Use this data, set coo_research=false.

## IMPORTANT RULES
1. Set coo_research=true ONLY if you found sources from the origin country
2. Data Source Type (retail/wholesale) is provided - search accordingly
3. Convert any found prices to USD
4. All countries in ISO3166 format
5. Calculate confidence based on source quality and data consistency
6. After gathering sufficient information, provide your final answer as a JSON object

## SOURCES - STRICT RELEVANCE REQUIRED
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
- "type": One of "retail", "wholesale", "official", "market", "customs", "other"
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
"No direct customs or trade data was found for exports of [PRODUCT] from [ORIGIN], so search_tier=2 and coo_research=false. Instead, [X] international customs records were used... The [ORIGIN]→[COUNTRY1] shipment shows $X for Y kg, giving Z USD/kg. The [ORIGIN]→[COUNTRY2] shipment shows... To estimate a reasonable valuation, I selected the median of prices [A, B, C] = B USD/kg."

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
    {"title": "<descriptive title with product, route, date>", "url": "<url>", "country": "<ISO3166>", "type": "<retail|wholesale|official|market|customs|other>", "price_raw": "<full context: value, quantity, product, route, date>", "extracted_price": <number|null>, "extracted_currency": "<ISO4217|null>", "extracted_unit": "<string like 'USD/kg'>"}
  ],
  "confidence": <0.0-1.0>,
  "notes": "<DETAILED analytical explanation as described above>",
  "error": <null|"INVALID_INPUT"|"NO_DATA">
}"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
        
        # Build prompt
        source_type = determine_source_type(quantity, unit_of_measure)
        prompt = f"""Find FOB (Free On Board) price for:
- Product: {description}
- Country of Origin: {origin} ({get_country_name(origin)})
- Country of Destination: {dest} ({get_country_name(dest)})
- Data Source Type: {source_type.upper()}
- Target currency: {target_currency}

Search {get_country_name(origin)} sources first. Return FOB price per {normalize_unit(unit_of_measure)}.
**Important: Exclude freight, insurance, and shipping costs. Return FOB price only.**"""

        try:
            response = self.client.responses.create(
                model=MODEL_NAME,
                tools=[{"type": "web_search"}],
                instructions=self.SYSTEM_PROMPT,
                input=prompt,
                temperature=0,  # Set to 0 for consistent, deterministic results
            )
            
            result = json.loads(response.output_text)
            
            # Calculate median from source extracted_price values
            sources = result.get("sources", [])
            prices = [s.get("extracted_price") for s in sources if s.get("extracted_price") is not None]
            unit_price = calculate_median(prices)
            
            return PricePrediction(
                unit_price=unit_price,
                currency=result.get("currency", target_currency),
                unit_of_measure=result.get("unit_of_measure", normalize_unit(unit_of_measure)),
                quantity_searched=result.get("quantity_searched", quantity),
                quantity_unit=result.get("quantity_unit", unit_of_measure),
                coo_research=result.get("coo_research", False),
                source_type=source_type,  # Use pre-determined source_type from input
                currency_converted=result.get("currency_converted", False),
                currency_fallback=result.get("currency_fallback"),
                fx_rate=result.get("fx_rate"),
                sources=sources,
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
            source_type="wholesale", currency_converted=False, currency_fallback=None,
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
        quantity=500,
        unit_of_measure="kg"
    )
    print(json.dumps(agent.to_dict(result), indent=2))
