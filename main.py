"""Price Research Agent - Streamlit UI"""

import streamlit as st
from price_agent import (
    PriceResearchAgent, 
    ISO3166_COUNTRIES, 
    ISO4217_CURRENCIES,
    determine_source_type
)

st.set_page_config(page_title="Price Research Agent", page_icon="💰", layout="centered")

st.title("💰 Price Research Agent")
st.caption("International trade price prediction with ISO compliance")

# Build options
countries = {f"{code} - {name}": code for code, name in sorted(ISO3166_COUNTRIES.items(), key=lambda x: x[1])}
currencies = ["Auto (Destination)"] + ISO4217_CURRENCIES
units = ["kg", "tonne", "g", "lb", "litre", "ml", "piece", "unit", "set", "sqm", "m3"]

# Input form
with st.form("price_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        origin = st.selectbox("Country of Origin", list(countries.keys()), 
                              index=list(countries.keys()).index("QA - Qatar") if "QA - Qatar" in countries else 0)
    with col2:
        destination = st.selectbox("Country of Destination", list(countries.keys()),
                                   index=list(countries.keys()).index("PK - Pakistan") if "PK - Pakistan" in countries else 0)
    
    description = st.text_area("Goods Description", 
                               placeholder="e.g., LOW DENSITY POLYETHYLENE (LDPE) LOTRENE MG70")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        quantity = st.number_input("Quantity", min_value=0.01, value=500.0, step=1.0)
    with col4:
        unit = st.selectbox("Unit", units)
    with col5:
        currency_sel = st.selectbox("Preferred Currency", currencies)
    
    # Show expected source type
    source_type = determine_source_type(quantity, unit)
    st.info(f"Quantity suggests **{source_type.upper()}** sources")
    
    submitted = st.form_submit_button("🔍 Research Price", use_container_width=True)

# Process
if submitted:
    if not description:
        st.warning("Please provide a goods description.")
    else:
        with st.spinner("Researching prices..."):
            try:
                agent = PriceResearchAgent()
                result = agent.research_price(
                    country_of_origin=countries[origin],
                    country_of_destination=countries[destination],
                    description=description,
                    quantity=quantity,
                    unit_of_measure=unit,
                    preferred_currency=currency_sel if currency_sel != "Auto (Destination)" else None
                )
                
                if result.error:
                    st.error(f"**Error:** {result.error}\n\n{result.notes}")
                else:
                    # Display results
                    st.success("Price research completed!")
                    
                    # Main metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Unit Price", f"{result.currency} {result.unit_price:,.4f}" if result.unit_price else "N/A")
                    col2.metric("Per", result.unit_of_measure)
                    col3.metric("Confidence", f"{result.confidence:.0%}")
                    
                    # Status indicators
                    col4, col5, col6 = st.columns(3)
                    col4.metric("COO Research", "✅ Yes" if result.coo_research else "⚠️ No")
                    col5.metric("Source Type", result.source_type.upper())
                    col6.metric("Currency Fallback", result.currency_fallback or "None")
                    
                    # FX Rate
                    if result.fx_rate and result.fx_rate.get("rate"):
                        fx = result.fx_rate
                        st.caption(f"💱 FX: 1 {fx.get('from', 'N/A')} = {fx.get('rate', 'N/A')} {fx.get('to', 'N/A')}")
                    
                    # Sources
                    if result.sources:
                        with st.expander(f"📚 Sources ({len(result.sources)})"):
                            for src in result.sources:
                                st.markdown(f"**{src.get('title', 'Unknown')}** ({src.get('country', 'N/A')} - {src.get('type', 'N/A')})")
                                st.caption(f"Raw: {src.get('price_raw', 'N/A')} → {src.get('extracted_price', 'N/A')} {src.get('extracted_currency', '')} / {src.get('extracted_unit', '')}")
                                st.markdown(f"[{src.get('url', '')[:50]}...]({src.get('url', '#')})")
                                st.divider()
                    
                    # Full JSON
                    with st.expander("📋 Full JSON Response"):
                        st.json(agent.to_dict(result))
                        
            except Exception as e:
                st.error(f"Research failed: {str(e)}")
