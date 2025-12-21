"""Price Research Agent - Streamlit UI with Redis Caching"""

import streamlit as st
from price_agent import (
    PriceResearchAgent, 
    ISO3166_COUNTRIES, 
    ISO4217_CURRENCIES,
    determine_source_type
)
from price_cache import get_cached_price_service, SIMILARITY_THRESHOLD

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
        with st.spinner("Checking cache and researching prices..."):
            try:
                # Use the cached price service
                service = get_cached_price_service()
                response = service.get_price(
                    description=description,
                    country_of_origin=countries[origin],
                    unit_of_measure=unit,
                    quantity=quantity,
                    preferred_currency=currency_sel if currency_sel != "Auto (Destination)" else None,
                )
                
                # Extract prediction from response
                prediction = response["prediction"]
                cache_hit = response["cache_hit"]
                similarity = response.get("similarity")
                cached_at = response.get("cached_at")
                
                if prediction.get("error"):
                    st.error(f"**Error:** {prediction['error']}\n\n{prediction.get('notes', '')}")
                else:
                    # Display cache status
                    if cache_hit:
                        st.success(f"⚡ Cache HIT! Similarity: {similarity:.1%} (threshold: {SIMILARITY_THRESHOLD:.0%})")
                        st.caption(f"📦 Cached at: {cached_at}")
                    else:
                        st.info("🔍 Cache MISS - Fresh research completed and cached")
                    
                    # Display results
                    st.success("Price research completed!")
                    
                    # Main metrics
                    col1, col2, col3 = st.columns(3)
                    unit_price = prediction.get("unit_price")
                    currency = prediction.get("currency", "USD")
                    col1.metric("Unit Price", f"{currency} {unit_price:,.4f}" if unit_price else "N/A")
                    col2.metric("Per", prediction.get("unit_of_measure", unit))
                    col3.metric("Confidence", f"{prediction.get('confidence', 0):.0%}")
                    
                    # Status indicators
                    col4, col5, col6 = st.columns(3)
                    col4.metric("COO Research", "✅ Yes" if prediction.get("coo_research") else "⚠️ No")
                    col5.metric("Source Type", prediction.get("source_type", "unknown").upper())
                    col6.metric("Currency Fallback", prediction.get("currency_fallback") or "None")
                    
                    # FX Rate
                    fx_rate = prediction.get("fx_rate")
                    if fx_rate and fx_rate.get("rate"):
                        st.caption(f"💱 FX: 1 {fx_rate.get('from', 'N/A')} = {fx_rate.get('rate', 'N/A')} {fx_rate.get('to', 'N/A')}")
                    
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
