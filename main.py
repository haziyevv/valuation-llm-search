from openai import OpenAI
import streamlit as st
import json

model_name = "gpt-4o"

assistant_prompt = """
You are a helpful assistant that provides accurate and concise information
based on web search results. Use the provided tools to gather information 
as needed to answer the user's query. Please return only the final answer  and the source URLs
without any additional commentary. 

Return in the following format: 

Output format (strict JSON):
{{
  "expected_price": NUMERIC VALUE ONLY,
  "min_price": NUMERIC VALUE ONLY,
  "max_price": NUMERIC VALUE ONLY,
  "source_urls": [LIST OF URL STRINGS ONLY],
  "unit_of_measure": "quantity unit of measure string"
}}
"""



client = OpenAI()

def get_price_info(coo: str, goods_description: str, uom: str):
    response = client.responses.create(
        model=model_name,
        tools=[{ "type": "web_search" }],
        max_tool_calls=3,
        instructions=(
            assistant_prompt
        ),
        input=f'What is the price for the given product: {goods_description} coming from {coo}.',
    )
    result = json.loads(response.output_text)

    expcected_price = result.get("expected_price")
    min_price = result.get("min_price")
    max_price = result.get("max_price")
    source_urls = result.get("source_urls", [])
    uom = result.get("unit_of_measure")
    
    return {
        "expected_price": expcected_price,
        "min_price": min_price,
        "max_price": max_price,
        "source_urls": source_urls,
        "unit_of_measure": uom
    }

# -------------------------
# STREAMLIT UI
# -------------------------

st.set_page_config(page_title="Price Lookup API", page_icon="💰", layout="centered")

st.title("💰 Product Price Lookup (Web-Search Powered)")
st.write("Enter product details to estimate international prices using GPT-5 with web search.")

# Inputs
coo = st.text_input("🌍 Country of Origin", placeholder="Qatar, Saudi Arabia, China...")
goods_description = st.text_input("📦 Goods Description", placeholder="PLASTIC MOULDING COMPOUND LLDPE FILM GRADE...")
uom = st.text_input("📏 Unit of Measure", placeholder="KG, TON, LITER...", value="KG")

if st.button("🔍 Get Price Estimate"):
    if not coo or not goods_description:
        st.warning("Please fill in all required fields.")
    else:
        with st.spinner(f"Fetching price from {model_name} Web Search..."):
            try:
                data = get_price_info(coo, goods_description, uom)

                st.success("Price information retrieved successfully!")

                st.json(data)

                if "source_urls" in data:
                    st.markdown("### 🔗 Source URLs")
                    for url in str(data["source_urls"]).split(","):
                        st.markdown(f"- {url.strip()}")

            except Exception as e:
                st.error(f"Error: {e}")

# get_price_info("China", "PLASTIC MOULDING COMPOUND LLDPE FILM GRADE", "KG")