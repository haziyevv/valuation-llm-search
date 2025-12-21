"""Price Research Agent - Streamlit UI with Redis Caching"""

import streamlit as st
from price_agent import (
    PriceResearchAgent, 
    ISO3166_COUNTRIES, 
    ISO4217_CURRENCIES,
    determine_source_type,
    PricePrediction,
)
from price_cache import get_cached_price_service, SIMILARITY_THRESHOLD

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
        with st.spinner("Checking cache and researching prices..."):
            try:
                # Use the cached price service
                service = get_cached_price_service()
                response = service.get_price(
                    description=description,
                    country_of_origin=countries[origin],
                    unit_of_measure=unit,
                    quantity=quantity,
                    exchange_rate_usd=exchange_rate,
                    target_currency=target_currency,
                )
                
                # Extract prediction from response
                prediction = response["prediction"]
                cache_hit = response["cache_hit"]
                similarity = response.get("similarity")
                cached_at = response.get("cached_at")
                
                # Check for errors
                if prediction.get("error"):
                    st.error(f"**Error:** {prediction['error']}\n\n{prediction.get('notes', '')}")
                else:
                    # Display cache status
                    if cache_hit:
                        st.success(f"⚡ Cache HIT! Similarity: {similarity:.1%} (threshold: {SIMILARITY_THRESHOLD:.0%})")
                        st.caption(f"📦 Cached at: {cached_at}")
                    else:
                        st.info("🔍 Cache MISS - Fresh research completed and cached")
                    
                    # Main metrics - show both USD and converted price
                    col1, col2, col3, col4 = st.columns(4)
                    unit_price_usd = prediction.get("unit_price_usd")
                    unit_price = prediction.get("unit_price")
                    confidence = prediction.get("confidence", 0)
                    
                    col1.metric(
                        "Price (USD)", 
                        f"${unit_price_usd:,.4f}" if unit_price_usd else "N/A"
                    )
                    col2.metric(
                        f"Price ({target_currency})", 
                        f"{unit_price:,.2f}" if unit_price else "N/A"
                    )
                    col3.metric("Per", prediction.get("unit_of_measure", unit))
                    col4.metric("Confidence", f"{confidence:.0%}")
                    
                    # Status indicators
                    col5, col6, col7 = st.columns(3)
                    col5.metric("COO Research", "✅ Yes" if prediction.get("coo_research") else "⚠️ No")
                    col6.metric("Source Type", prediction.get("source_type", "unknown").upper())
                    col7.metric("Converted", "✅ Yes" if prediction.get("currency_converted") else "❌ No")
                    
                    # FX Rate
                    fx_rate = prediction.get("fx_rate")
                    if fx_rate and fx_rate.get("rate"):
                        st.caption(f"💱 FX: 1 {fx_rate.get('from', 'USD')} = {fx_rate.get('rate', 'N/A')} {fx_rate.get('to', target_currency)}")
                    
                    # Notes
                    notes = prediction.get("notes")
                    if notes:
                        with st.expander("📝 Research Notes"):
                            st.write(notes)
                    
                    # Sources
                    sources = prediction.get("sources", [])
                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)})"):
                            for src in sources:
                                st.markdown(f"**{src.get('title', 'Unknown')}** ({src.get('country', 'N/A')} - {src.get('type', 'N/A')})")
                                st.caption(f"Raw: {src.get('price_raw', 'N/A')} → {src.get('extracted_price', 'N/A')} {src.get('extracted_currency', '')} / {src.get('extracted_unit', '')}")
                                url = src.get('url', '#')
                                st.markdown(f"[{url[:50]}...]({url})")
                                st.divider()
                    
                    # Full JSON (including cache metadata)
                    with st.expander("📋 Full JSON Response"):
                        st.json(response)
                        
            except Exception as e:
                st.error(f"Research failed: {str(e)}")

# Sidebar: Cache Stats
with st.sidebar:
    st.subheader("📊 Cache Statistics")
    try:
        service = get_cached_price_service()
        stats = service.get_stats()
        st.metric("Cached Entries", stats["total_entries"])
        st.metric("Similarity Threshold", f"{stats['similarity_threshold']:.0%}")
        st.metric("Expiry (days)", stats["expiry_days"])
        
        if st.button("🗑️ Clear Cache"):
            cleared = service.clear_cache()
            st.success(f"Cleared {cleared} entries")
            st.rerun()
    except Exception as e:
        st.warning(f"Cache unavailable: {str(e)}")
