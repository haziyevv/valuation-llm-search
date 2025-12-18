# ISO3166-1 Alpha-2 to Country Name mapping (common trading countries)
ISO3166_COUNTRIES = {
    "AF": "Afghanistan", "AE": "United Arab Emirates", "AR": "Argentina",
    "AU": "Australia", "AT": "Austria", "BD": "Bangladesh", "BE": "Belgium",
    "BR": "Brazil", "CA": "Canada", "CH": "Switzerland", "CL": "Chile",
    "CN": "China", "CO": "Colombia", "CZ": "Czech Republic", "DE": "Germany",
    "DK": "Denmark", "EG": "Egypt", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "United Kingdom", "GR": "Greece", "HK": "Hong Kong",
    "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland", "IL": "Israel",
    "IN": "India", "IR": "Iran", "IT": "Italy", "JP": "Japan",
    "KE": "Kenya", "KM": "Comoros", "KR": "South Korea", "KW": "Kuwait",
    "LK": "Sri Lanka", "MA": "Morocco", "MX": "Mexico", "MY": "Malaysia",
    "NG": "Nigeria", "NL": "Netherlands", "NO": "Norway", "NZ": "New Zealand",
    "OM": "Oman", "PH": "Philippines", "PK": "Pakistan", "PL": "Poland",
    "PT": "Portugal", "QA": "Qatar", "RO": "Romania", "RU": "Russia",
    "SA": "Saudi Arabia", "SE": "Sweden", "SG": "Singapore", "TH": "Thailand",
    "TR": "Turkey", "TW": "Taiwan", "UA": "Ukraine", "US": "United States",
    "VE": "Venezuela", "VN": "Vietnam", "ZA": "South Africa",
}

# ISO3166 -> ISO4217 mapping
COUNTRY_TO_CURRENCY = {
    "AF": "AFN", "AE": "AED", "AR": "ARS", "AU": "AUD", "AT": "EUR",
    "BD": "BDT", "BE": "EUR", "BR": "BRL", "CA": "CAD", "CH": "CHF",
    "CL": "CLP", "CN": "CNY", "CO": "COP", "CZ": "CZK", "DE": "EUR",
    "DK": "DKK", "EG": "EGP", "ES": "EUR", "FI": "EUR", "FR": "EUR",
    "GB": "GBP", "GR": "EUR", "HK": "HKD", "HU": "HUF", "ID": "IDR",
    "IE": "EUR", "IL": "ILS", "IN": "INR", "IR": "IRR", "IT": "EUR",
    "JP": "JPY", "KE": "KES", "KM": "KMF", "KR": "KRW", "KW": "KWD",
    "LK": "LKR", "MA": "MAD", "MX": "MXN", "MY": "MYR", "NG": "NGN",
    "NL": "EUR", "NO": "NOK", "NZ": "NZD", "OM": "OMR", "PH": "PHP",
    "PK": "PKR", "PL": "PLN", "PT": "EUR", "QA": "QAR", "RO": "RON",
    "RU": "RUB", "SA": "SAR", "SE": "SEK", "SG": "SGD", "TH": "THB",
    "TR": "TRY", "TW": "TWD", "UA": "UAH", "US": "USD", "VE": "VES",
    "VN": "VND", "ZA": "ZAR",
}

# ISO4217 Currencies
ISO4217_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CNY", "INR", "PKR", "SAR", 
                      "AED", "QAR", "KWD", "OMR", "BDT", "MYR", "SGD", "THB", 
                      "VND", "IDR", "PHP", "KRW", "AUD", "CAD", "CHF", "BRL"]

# Unit normalization
UNIT_MAP = {
    "kg": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gram": "g", "grams": "g",
    "t": "tonne", "ton": "tonne", "tonne": "tonne", "tonnes": "tonne", "mt": "tonne",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "l": "litre", "liter": "litre", "litre": "litre", "litres": "litre",
    "ml": "ml", "milliliter": "ml", "millilitre": "ml",
    "pc": "piece", "pcs": "piece", "piece": "piece", "pieces": "piece",
    "unit": "unit", "units": "unit", "set": "set", "sets": "set",
    "sqm": "sqm", "m2": "sqm", "m3": "m3", "cbm": "m3",
}

# Quantity thresholds
THRESHOLDS = {"weight": (100, 1000), "volume": (100, 1000), "count": (50, 500)}



from typing import Literal

def normalize_unit(unit: str) -> str:
    return UNIT_MAP.get(unit.lower().strip(), unit.lower().strip())


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


def get_country_name(code: str) -> str:
    return ISO3166_COUNTRIES.get(code.upper(), code)


def get_default_currency(country_code: str) -> str:
    return COUNTRY_TO_CURRENCY.get(country_code.upper(), "USD")
