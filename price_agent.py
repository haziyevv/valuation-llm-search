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
import csv
import json
import os
import re
from datetime import datetime, timedelta
from statistics import median
from typing import Optional, Literal
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from utils import ISO3166_COUNTRIES, ISO4217_CURRENCIES, COUNTRY_TO_CURRENCY, UNIT_MAP, THRESHOLDS

# Maximum age of sources in days
SOURCE_MAX_AGE_DAYS = 90
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
    """Price research agent with web search capability."""
    
    SYSTEM_PROMPT = """You are an expert international trade price analyst specializing in commodity valuation, customs declarations, and cross-border pricing research.

## YOUR MISSION
Find accurate, verifiable unit prices for goods in international trade contexts.
Your research directly impacts customs valuations and trade decisions, so accuracy and source quality are paramount.
**IMPORTANT: Always return prices in USD.**

Only include sources published on or after the Source Cutoff Date.

## RECENCY GATE (HARD REQUIREMENT)
You will be given a **Source Cutoff Date** in the user prompt (YYYY-MM-DD).

**You MUST NOT include any source older than the Source Cutoff Date.**
- Every source MUST include a `date` field in YYYY-MM-DD.
- If a source page does not clearly show a publish/record date, treat its date as UNKNOWN and **do not include it**.
- If a source only provides month/year (no day), treat it as UNKNOWN and **do not include it**.

**Final self-check before producing JSON (MANDATORY):**
1. Compute `cutoff_date_used` EXACTLY as provided in the user prompt.
2. For each source in `sources`, verify `date >= cutoff_date_used`.
3. If any source fails this check, you MUST remove it from `sources` (do not “keep it anyway”, do not justify it, do not include it).
4. After removal:
   - If `sources` is empty, return `error="NO_DATA"` and explain in `notes` that no sources met the cutoff date.
   - Otherwise continue with remaining sources only.

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
- "type": One of "retail" or "wholesale"
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

**If error="NO_DATA"**, the notes MUST:
- Explicitly state that **no sources met the Source Cutoff Date**
- Briefly describe what you searched (origin export first, then global market)

## OUTPUT FORMAT
After your research, provide the final answer as a JSON object with this structure:
{
  "currency": "USD",
  "unit_of_measure": "<normalized unit>",
  "quantity_searched": <number>,
  "quantity_unit": "<original unit>",
  "sources": [
    // ONLY include PRODUCT PRICING sources - do NOT include exchange rate/FX sources
    {"title": "<descriptive title with product, route, date>", "url": "<url>", "country": "<ISO3166>", "type": "<retail|wholesale>", "date": "<YYYY-MM-DD>", "price_raw": "<full context: value, quantity, product, route, date>", "extracted_price": <number>, "extracted_currency": "<ISO4217>", "extracted_unit": "<string like 'USD/kg'>"}
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
        description: str,
        quantity: float,
        unit_of_measure: str,
        preferred_currency: Optional[str] = None
    ) -> PricePrediction:
        """Research price for goods."""
        # Validate country codes
        origin = country_of_origin.upper().strip()
        
        if origin not in ISO3166_COUNTRIES:
            return self._error("INVALID_INPUT", f"Invalid origin: {origin}", quantity, unit_of_measure)
        
        # Determine target currency
        target_currency = 'PKR'
        if target_currency not in ISO4217_CURRENCIES:
            return self._error("INVALID_INPUT", f"Invalid currency: {target_currency}", quantity, unit_of_measure)
        
        # Build prompt
        source_type = determine_source_type(quantity, unit_of_measure)
        current_date = datetime.now().strftime("%Y-%m-%d")
        cutoff_date = (datetime.now() - timedelta(days=SOURCE_MAX_AGE_DAYS)).strftime("%Y-%m-%d")
        
        target_unit = normalize_unit(unit_of_measure)
        prompt = f"""Find unit price for:
- Product: {description}
- Country of Origin: {get_country_name(origin)} 
- Data Source Type: {source_type.upper()}
- Target currency: {target_currency}
- Target unit: {target_unit} (CONVERT all prices to USD/{target_unit})
- Current Date: {current_date}
- Source Cutoff Date: {cutoff_date}
"""
        coo_research = True
        try:
            # --- Tier 1: Origin export prices only (STEP 1) ---
            tier1_prompt = (
                prompt
                + f"\n\nInvestigate unit price per {target_unit} of {description} in {target_currency} exported from {get_country_name(origin)}.\n"
                + f"Hard rule: ONLY include sources with date >= {cutoff_date}.\n"
            )

            response1 = self.client.responses.create(
                model=MODEL_NAME,
                tools=[{"type": "web_search"}],
                instructions=self.SYSTEM_PROMPT,
                input=tier1_prompt,
            )
            result1 = json.loads(response1.output_text)

            raw_sources1 = result1.get("sources", [])
            sources1 = filter_recent_sources(raw_sources1, SOURCE_MAX_AGE_DAYS)
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

            if len(sources1) <= 3:
                coo_research = False
                # --- Tier 2: Global market price fallback only (STEP 2) ---
                tier2_prompt = (
                    prompt
                    + f"\n\nInvestigate unit price per {target_unit} of {description} in {target_currency}.\n"
                    + f"Hard rule: ONLY include sources with date >= {cutoff_date}.\n"
                )

                response2 = self.client.responses.create(
                    model=MODEL_NAME,
                    tools=[{"type": "web_search"}],
                    instructions=self.SYSTEM_PROMPT,
                    input=tier2_prompt,
                )
                result2 = json.loads(response2.output_text)

                raw_sources2 = result2.get("sources", [])
                sources2 = filter_recent_sources(raw_sources2, SOURCE_MAX_AGE_DAYS)
                notes2 = result2.get("notes", "")

            combined_sources = _dedupe_sources(sources1 + sources2)
            combined_prices = [
                s.get("extracted_price")
                for s in combined_sources
                if s.get("extracted_price") is not None
            ]
            unit_price = calculate_median(combined_prices)

            merged_notes = notes1
            if notes2:
                merged_notes = (notes1 or "") + "\n\n--- Tier 2 (fallback) ---\n" + notes2

            if not combined_prices or unit_price is None:
                return self._error("NO_DATA", merged_notes, quantity, unit_of_measure)

            # Prefer tier-1 metadata if available; fall back to tier-2.
            primary_result = result1 if sources1 else (result2 or result1)

            prediction = PricePrediction(
                unit_price=unit_price,
                currency=primary_result.get("currency", target_currency),
                unit_of_measure=primary_result.get("unit_of_measure", normalize_unit(unit_of_measure)),
                quantity_searched=primary_result.get("quantity_searched", quantity),
                quantity_unit=primary_result.get("quantity_unit", unit_of_measure),
                coo_research=coo_research,
                source_type=source_type,  # Use pre-determined source_type from input
                currency_converted=primary_result.get("currency_converted", False),
                currency_fallback=primary_result.get("currency_fallback"),
                fx_rate=primary_result.get("fx_rate"),
                sources=combined_sources,
                confidence=primary_result.get("confidence", 0.0),
                notes=merged_notes,
                error=primary_result.get("error"),
            )

            self.save_csv(prediction)
            return prediction
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
        quantity=500,
        unit_of_measure="kg"
    )
    print(json.dumps(agent.to_dict(result), indent=2))
    print("\nResults saved to results.csv")
