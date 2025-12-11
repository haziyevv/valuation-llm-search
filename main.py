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
currencies = ISO4217_CURRENCIES
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
    
    col3, col4 = st.columns(2)
    with col3:
        quantity = st.number_input("Quantity", min_value=0.01, value=500.0, step=1.0)
    with col4:
        unit = st.selectbox("Unit", units)
    
    # Exchange rate and currency settings
    col5, col6 = st.columns(2)
    with col5:
        exchange_rate = st.number_input(
            "Exchange Rate (USD → Target)", 
            min_value=0.01, 
            value=278.5,  # Default USD to PKR
            step=0.1,
            help="Exchange rate from USD to target currency. E.g., 278.5 means 1 USD = 278.5 PKR"
        )
    with col6:
        target_currency = st.selectbox(
            "Target Currency",
            currencies,
            index=currencies.index("PKR") if "PKR" in currencies else 0
        )
    
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
                    exchange_rate_usd=exchange_rate,
                    target_currency=target_currency,
                )
                if result.error:
                    st.error(f"**Error:** {result.error}\n\n{result.notes}")
                else:
                    # Display results
                    st.success("Price research completed!")
                    
                    # Main metrics - show both USD and converted price
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(
                        "Price (USD)", 
                        f"${result.unit_price_usd:,.4f}" if result.unit_price_usd else "N/A"
                    )
                    col2.metric(
                        f"Price ({target_currency})", 
                        f"{result.unit_price:,.2f}" if result.unit_price else "N/A"
                    )
                    col3.metric("Per", result.unit_of_measure)
                    col4.metric("Confidence", f"{result.confidence:.0%}")
                    
                    # Status indicators
                    col5, col6, col7 = st.columns(3)
                    col5.metric("COO Research", "✅ Yes" if result.coo_research else "⚠️ No")
                    col6.metric("Source Type", result.source_type.upper())
                    col7.metric("Converted", "✅ Yes" if result.currency_converted else "❌ No")
                    
                    # FX Rate
                    if result.fx_rate and result.fx_rate.rate:
                        fx = result.fx_rate
                        st.caption(f"💱 FX: 1 {fx.from_currency or 'USD'} = {fx.rate or 'N/A'} {fx.to_currency or target_currency}")
                    
                    # Notes
                    if result.notes:
                        with st.expander("📝 Research Notes"):
                            st.write(result.notes)
                    
                    # Sources
                    if result.sources:
                        with st.expander(f"📚 Sources ({len(result.sources)})"):
                            for src in result.sources:
                                st.markdown(f"**{src.title or 'Unknown'}** ({src.country or 'N/A'} - {src.type or 'N/A'})")
                                st.caption(f"Raw: {src.price_raw or 'N/A'} → {src.extracted_price or 'N/A'} {src.extracted_currency or ''} / {src.extracted_unit or ''}")
                                url = src.url or '#'
                                st.markdown(f"[{url[:50]}...]({url})")
                                st.divider()
                    
                    # Full JSON
                    with st.expander("📋 Full JSON Response"):
                        st.json(agent.to_dict(result))
                        
            except Exception as e:
                st.error(f"Research failed: {str(e)}")
