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
