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
        return "wholesale"
    
    if qty < retail_max:
        return "retail"
    return "wholesale"


class PriceResearchAgent:
    """Price research agent with web search capability."""
    
    SYSTEM_PROMPT = """You are an expert international trade price analyst specializing in commodity valuation, customs declarations, and cross-border pricing research.

## YOUR MISSION
Find accurate, verifiable FOB (Free On Board) unit prices for goods in international trade contexts.
Your research directly impacts customs valuations and trade decisions, so accuracy and source quality are paramount.
Use web search to have the most accurate and up to date information. Use your own knowledge and experience to make the best decision.

**CRITICAL: Return FOB prices only. Do NOT include freight, insurance, or any other shipping costs (CIF, CFR, etc.). If you only find CIF prices, estimate the FOB price by deducting typical freight/insurance costs (usually 5-15% depending on route).**

## SEARCH PROTOCOL (Strict "Waterfall" Logic)
You must execute your research in the following strict order. Do not skip to Step 2 unless Step 1 fails.

**STEP 1 (search_tier=1): Global Export from Country of Origin**
Search for FOB export prices from the Country of Origin to any destination.
Example query: "[product] FOB price [origin country] export [wholesale/retail] 2025"
IF FOUND reliable data: Use this, set coo_research=true.

**STEP 2 (search_tier=2): Global Market Price (Fallback)**
If Step 1 yields no verifiable data, search for global FOB market prices.
Example query: "[product] FOB [wholesale/retail] price per [unit] international market 2025"
IF FOUND: Use this data, set coo_research=false.

## IMPORTANT RULES
1. Set coo_research=true in the response if you found any sources from the origin country, false otherwise
2. Data Source Type is the type of data source you are using to find the price. It can be retail or wholesale.
3. Convert prices to target currency. If conversion fails, fallback to USD
4. All currencies should be written in ISO4217, countries in ISO3166
5. All prices should be in the Target currency
6. **Always return FOB prices** - exclude freight, insurance, and delivery costs

**RECENCY REQUIREMENT: Only use sources from the last 6 months.**
- Prioritize the most recent data available
- Discard sources older than 6 months from the current date
- If only older sources are available, note this limitation in the notes and reduce confidence accordingly
- Do not include pricing sources used to convert the price to the target currency



OUTPUT JSON ONLY:
{
  "unit_price": <number|null>,
  "currency": "<ISO4217>",
  "unit_of_measure": "<normalized unit>",
  "quantity_searched": <number>,
  "quantity_unit": "<original unit>",
  "coo_research": <boolean>,
  "currency_converted": <boolean>,
  "currency_fallback": "<ISO4217|null>",
  "fx_rate": {"rate": <number|null>, "from": "<ISO4217>", "to": "<ISO4217>", "timestamp_utc": "<ISO8601|null>", "source": "<string|null>"},
  "sources": [{"title": "<string>", "url": "<url>", "country": "<ISO3166>", "type": "<retail|wholesale|official|market|customs|other>", "price_raw": "<string>", "extracted_price": <number|null>, "extracted_currency": "<ISO4217|null>", "extracted_unit": "<string|null>"}],
  "confidence": <0.0-1.0>,
  "notes": "<string>",
  "error": <null|"INVALID_INPUT"|"NO_DATA"|"FX_FAIL">
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

Search {get_country_name(origin)} sources first. Return FOB price per {normalize_unit(unit_of_measure)} in {target_currency}.
**Important: Exclude freight, insurance, and shipping costs. Return FOB price only.**"""

        try:
            response = self.client.responses.create(
                model=MODEL_NAME,
                tools=[{"type": "web_search"}],
                instructions=self.SYSTEM_PROMPT,
                input=prompt,
            )
            
            result = json.loads(response.output_text)
            return PricePrediction(
                unit_price=result.get("unit_price"),
                currency=result.get("currency", target_currency),
                unit_of_measure=result.get("unit_of_measure", normalize_unit(unit_of_measure)),
                quantity_searched=result.get("quantity_searched", quantity),
                quantity_unit=result.get("quantity_unit", unit_of_measure),
                coo_research=result.get("coo_research", False),
                source_type=source_type,  # Use pre-determined source_type from input
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
        quantity=5000,
        unit_of_measure="kg"
    )
    print(json.dumps(agent.to_dict(result), indent=2))
