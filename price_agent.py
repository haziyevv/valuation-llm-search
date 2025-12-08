"""
Price Research Agent - ToolCallingAgent Version
================================================
Uses smolagents ToolCallingAgent with OpenAI's web search as a custom tool.
- ToolCallingAgent: LLM returns structured tool calls (name + args), no code execution
- More reliable than CodeAgent which generates and runs Python code
- Allows using any LLM while leveraging OpenAI's high-quality search
"""

import json
import os
import re
from typing import Optional, Literal
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

from openai import OpenAI
from smolagents import ToolCallingAgent, Tool
from smolagents import OpenAIServerModel  # For OpenAI API
# from smolagents import HfApiModel       # For Hugging Face Inference API
# from smolagents import LiteLLMModel     # For various providers via LiteLLM

from utils import ISO3166_COUNTRIES, ISO4217_CURRENCIES, COUNTRY_TO_CURRENCY, UNIT_MAP, THRESHOLDS

load_dotenv()

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_ID = os.getenv("OPENAI_MODEL", "gpt-5.1")

# Open-source alternatives (set LLM_PROVIDER=huggingface in .env):
# HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Initialize OpenAI client for web search tool
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class OpenAIWebSearchTool(Tool):
    """Web search tool using OpenAI's search capability."""
    name = "openai_web_search"
    description = """Search the web for current prices, market data, and trade information.

IMPORTANT: This tool accepts ONLY the 'query' parameter. Do not pass any other parameters like cursor, page, limit, or filters - they will cause an error.

Example usage:
  openai_web_search(query="LDPE polymer export price Qatar 2024 USD per kg")
"""
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query. Be specific: include product name, country, year, currency, and unit."
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        """Execute the web search."""
        try:
            response = _openai_client.responses.create(
                model="gpt-4o-mini",
                tools=[{"type": "web_search"}],
                input=query,
            )
            return response.output_text
        except Exception as e:
            return f"Search failed: {str(e)}"


# Create tool instance
openai_web_search = OpenAIWebSearchTool()


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


def get_model():
    """Initialize the LLM model based on configuration."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        return OpenAIServerModel(
            model_id=MODEL_ID,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "huggingface":
        from smolagents import HfApiModel
        return HfApiModel(
            model_id=os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            token=os.getenv("HF_TOKEN")
        )
    elif provider == "local":
        from smolagents import TransformersModel
        return TransformersModel(
            model_id=MODEL_ID,
            device_map="auto"
        )
    else:
        # Fallback: use LiteLLM for flexibility
        from smolagents import LiteLLMModel
        return LiteLLMModel(model_id=MODEL_ID)


class PriceResearchAgent:
    """Price research agent using smolagents ToolCallingAgent with OpenAI's web search.
    
    Uses structured tool calling (not code execution) for more reliable operation.
    The LLM returns tool name + arguments, smolagents handles execution.
    """
    
    SYSTEM_PROMPT = """You are an expert international trade price analyst specializing in commodity valuation, customs declarations, and cross-border pricing research.

## YOUR MISSION
Find accurate, verifiable unit prices for goods in international trade contexts.
Your research directly impacts customs valuations and trade decisions, so accuracy and source quality are paramount.

## TOOL USAGE
Use the openai_web_search tool to find pricing information. 
CRITICAL: Pass ONLY the 'query' parameter. Do NOT add cursor, page, limit, or any other parameters.

Example: openai_web_search(query="LDPE polymer wholesale price Qatar export 2024")

## SEARCH PROTOCOL (Strict "Waterfall" Logic)
Execute research in this order. Do not skip to Step 3 unless Steps 1 and 2 fail.

**STEP 1: Bilateral Trade (Country of Origin -> Country of Destination)**
* Search for export prices specifically from the Country of Origin to the Country of Destination.
* IF FOUND: Use this data, mark search_tier as "1", and stop searching.

**STEP 2: Global Export (COO -> World)**
* If Step 1 yields no verifiable data, search for the general export price from the Country of Origin to any country.
* IF FOUND: Use this data, mark search_tier as "2", and stop searching.

**STEP 3: Global Market Price (Fallback)**
* If Steps 1 and 2 yield no data, search for global market price (commodity benchmarks, major marketplaces).
* IF FOUND: Use this data, mark search_tier as "3".

## RULES
1. Set coo_research=true if sources are from origin country, false otherwise
2. Convert prices to target currency. If conversion fails, fallback to USD
3. All currencies in ISO4217, countries in ISO3166

## OUTPUT FORMAT
Return ONLY valid JSON:
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
  "fx_rate": {"rate": <number|null>, "from": "<ISO4217>", "to": "<ISO4217>"},
  "sources": [{"title": "<string>", "url": "<url>", "country": "<ISO3166>", "type": "<retail|wholesale|official|market|customs|other>"}],
  "confidence": <0.0-1.0>,
  "notes": "<string>",
  "error": <null|"INVALID_INPUT"|"NO_DATA"|"FX_FAIL">
}"""

    def __init__(self):
        self.model = get_model()
        self.agent = ToolCallingAgent(
            tools=[openai_web_search],  # OpenAI's web search as a tool
            model=self.model,
            instructions=self.SYSTEM_PROMPT,
            max_steps=10
        )
    
    def research_price(
        self,
        country_of_origin: str,
        country_of_destination: str,
        description: str,
        quantity: float,
        unit_of_measure: str,
        preferred_currency: Optional[str] = None
    ) -> PricePrediction:
        """Research price for goods using OpenAI's web search."""
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
        prompt = f"""Find the current market price for:
- Product: {description}
- Country of Origin: {origin} ({get_country_name(origin)})
- Country of Destination: {dest} ({get_country_name(dest)})
- Data Source Type: {source_type.upper()}
- Target currency: {target_currency}

Search {get_country_name(origin)} sources first. Return price per {normalize_unit(unit_of_measure)} in {target_currency}.

Return your findings as a JSON object matching the required schema."""

        try:
            result_text = self.agent.run(prompt)
            
            # Parse the JSON from agent output
            if isinstance(result_text, str):
                json_match = re.search(r'\{[\s\S]*\}', result_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = json.loads(result_text)
            else:
                result = result_text
            
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
