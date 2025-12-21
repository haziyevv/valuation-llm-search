# Price Research Agent

International trade price prediction with ISO compliance (ISO3166 countries, ISO4217 currencies).

## Features

- Azure OpenAI integration for intelligent price analysis
- Tavily Search API for real-time market prices (optimized for AI agents)
- Country of origin research tracking (`coo_research` flag)
- Wholesale/retail discrimination based on quantity
- Currency conversion with USD fallback
- Confidence scoring (0.0-1.0)

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:
```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Tavily Search API (Get free key at https://tavily.com)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx
```

Run:
```bash
streamlit run main.py
```

## API Usage

```python
from price_agent import PriceResearchAgent

agent = PriceResearchAgent()
result = agent.research_price(
    country_of_origin="QA",        # ISO3166
    description="LDPE LOTRENE MG70",
    quantity=5000,
    unit_of_measure="kg",
    preferred_currency="USD"       # ISO4217 (optional)
)

print(result.unit_price)      # Price per unit
print(result.currency)        # ISO4217 currency
print(result.coo_research)    # True if sourced from origin country
print(result.source_type)     # retail/wholesale/mixed
print(result.confidence)      # 0.0-1.0
```

## Output Schema

```json
{
  "unit_price": 1.60,
  "currency": "USD",
  "unit_of_measure": "kg",
  "quantity_searched": 5000,
  "quantity_unit": "kg",
  "coo_research": true,
  "source_type": "wholesale",
  "currency_converted": true,
  "currency_fallback": null,
  "fx_rate": {"rate": 1.0, "from": "USD", "to": "USD", "timestamp_utc": "...", "source": "..."},
  "sources": [...],
  "confidence": 0.84,
  "notes": "...",
  "error": null
}
```

## Quantity Thresholds

| Type | Retail | Wholesale |
|------|--------|-----------|
| Weight (kg) | < 100 | ≥ 1000 |
| Volume (litre) | < 100 | ≥ 1000 |
| Count (pieces) | < 50 | ≥ 500 |
