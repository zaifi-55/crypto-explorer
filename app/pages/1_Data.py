import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.api import cache_data, build_ohlcv
from src.config import COIN_MAP, DEFAULT_DAYS, MAX_DAYS


st.set_page_config(
    page_title="Crypto Data Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Crypto Data Explorer")
st.markdown("Explore historical price and OHLC data for cryptocurrencies")


st.sidebar.header("🔧 Configuration")


coin = st.sidebar.selectbox(
    "Select Cryptocurrency", 
    list(COIN_MAP.keys()),
    index=0,
    help="Choose a cryptocurrency to analyze"
)


days = st.sidebar.slider(
    "Days of Historical Data",
    min_value=7,
    max_value=MAX_DAYS,  
    step=7,
    value=DEFAULT_DAYS,
    help=f"Select the number of days to fetch (Max: {MAX_DAYS} days for OHLC data due to API limitations)"
)


refresh_col, info_col = st.sidebar.columns(2)

with refresh_col:
    if st.button("🔄 Refresh", help="Force refresh data from API"):
        st.rerun()

with info_col:
    st.button("ℹ️ API Info", help="Click to see API rate limit information", disabled=True)






st.info("🚀 **Quick Start:** Select a coin and wait for data to load. Cached data loads instantly!")  

col1, col2 = st.columns([3, 1])

with col2:
    st.metric(
        label="Data Period",
        value=f"{days} days",
        delta=f"Until {datetime.now().strftime('%Y-%m-%d')}"
    )


st.subheader("📊 Data Loading")
progress_bar = st.progress(0)
status_text = st.empty()

try:
    
    status_text.text("Fetching price data...")
    progress_bar.progress(25)
    
    price_df = cache_data(coin, days) 
    if price_df.empty:
        st.error("❌ Failed to fetch price data. Please try again or select different parameters.")
        st.info("💡 **Troubleshooting:**")
        st.info("1. Check your internet connection")
        st.info("2. Try the API test in the sidebar")
        st.info("3. Wait a few minutes and retry")
        st.stop()
    
    progress_bar.progress(50)
    

    status_text.text("Fetching OHLC data...")
    progress_bar.progress(75)
    
    ohlcv_df = build_ohlcv(coin, days)
    
    progress_bar.progress(100)
    status_text.text("✅ Data loaded successfully!")
    
   
    import time
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("🔧 **Debug Information:**")
    st.code(f"Selected coin: {coin}")
    st.code(f"Days: {days}")
    st.code(f"Error type: {type(e).__name__}")
    st.info("Try using the API test tools in the sidebar.")
    st.stop()


col1, col2, col3, col4 = st.columns(4)

if not price_df.empty:
    current_price = price_df['price'].iloc[-1]
    price_change = price_df['price'].iloc[-1] - price_df['price'].iloc[0]
    price_change_pct = (price_change / price_df['price'].iloc[0]) * 100
    
    with col1:
        st.metric(
            label=f"Current {coin.capitalize()} Price",
            value=f"${current_price:,.2f}",
            delta=f"{price_change_pct:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Price Range",
            value=f"${price_df['price'].min():.2f} - ${price_df['price'].max():.2f}"
        )
    
    with col3:
        st.metric(
            label="Data Points",
            value=len(price_df)
        )
    
    with col4:
        if not ohlcv_df.empty:
            avg_volume = ohlcv_df['volume'].mean()
            st.metric(
                label="Avg Daily Volume",
                value=f"${avg_volume:,.0f}"
            )

st.markdown("---")


st.subheader(f"💵 {coin.capitalize()} Price History")

if not price_df.empty:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=price_df["date"],
        y=price_df["price"],
        mode="lines",
        name="Price (USD)",
        line=dict(color="#00cc96", width=2),
        hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> $%{y:,.2f}<extra></extra>"
    ))

    fig_price.update_layout(
        title=f"{coin.capitalize()} Price Trend ({days} days)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=400,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.error("No price data available to display chart")


st.subheader(f"📊 {coin.capitalize()} OHLC Candlestick Chart")

if not ohlcv_df.empty and len(ohlcv_df) > 0:
    fig_ohlc = go.Figure(data=[go.Candlestick(
        x=ohlcv_df["date"],
        open=ohlcv_df["open"],
        high=ohlcv_df["high"],
        low=ohlcv_df["low"],
        close=ohlcv_df["close"],
        name="OHLC",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        hovertext=[f"Volume: ${vol:,.0f}" for vol in ohlcv_df["volume"]]
    )])
    
    fig_ohlc.update_layout(
        title=f"{coin.capitalize()} OHLC Chart ({len(ohlcv_df)} data points)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig_ohlc, use_container_width=True)
    
    
    if 'volume' in ohlcv_df.columns and ohlcv_df['volume'].sum() > 0:
        st.subheader(f"📈 {coin.capitalize()} Trading Volume")
        
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=ohlcv_df["date"],
            y=ohlcv_df["volume"],
            name="Volume",
            marker_color='rgba(0, 204, 150, 0.6)',
            hovertemplate="<b>Date:</b> %{x}<br><b>Volume:</b> $%{y:,.0f}<extra></extra>"
        ))
        
        fig_volume.update_layout(
            title=f"{coin.capitalize()} Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume (USD)",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
        
else:
    st.warning("⚠️ OHLC data is not available for the selected period. This might be due to:")
    st.markdown(f"""
    - API limitations (max {MAX_DAYS} days for OHLC data)
    - Insufficient data for the chosen cryptocurrency
    - Network connectivity issues
    
    **Suggestions:**
    - Try selecting a shorter time period
    - Choose a different cryptocurrency
    - Click the refresh button to retry
    """)


st.markdown("---")
st.subheader("📋 Raw Data")

tab1, tab2 = st.tabs(["💵 Price Data", "📊 OHLCV Data"])

with tab1:
    if not price_df.empty:
        st.write(f"**{len(price_df)} price data points**")
        
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Price Statistics:**")
            st.write(f"- Mean: ${price_df['price'].mean():.2f}")
            st.write(f"- Median: ${price_df['price'].median():.2f}")
            st.write(f"- Std Dev: ${price_df['price'].std():.2f}")
        
        with col2:
            st.write("**Date Range:**")
            st.write(f"- From: {price_df['date'].min()}")
            st.write(f"- To: {price_df['date'].max()}")
        
       
        st.dataframe(
            price_df.sort_values('date', ascending=False),
            use_container_width=True,
            height=300
        )
    else:
        st.error("No price data available")

with tab2:
    if not ohlcv_df.empty:
        st.write(f"**{len(ohlcv_df)} OHLCV data points**")
        
        
        if len(ohlcv_df) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**OHLC Statistics:**")
                st.write(f"- Avg Open: ${ohlcv_df['open'].mean():.2f}")
                st.write(f"- Avg Close: ${ohlcv_df['close'].mean():.2f}")
                st.write(f"- Highest: ${ohlcv_df['high'].max():.2f}")
                st.write(f"- Lowest: ${ohlcv_df['low'].min():.2f}")
            
            with col2:
                st.write("**Volume Statistics:**")
                if ohlcv_df['volume'].sum() > 0:
                    st.write(f"- Avg Volume: ${ohlcv_df['volume'].mean():,.0f}")
                    st.write(f"- Total Volume: ${ohlcv_df['volume'].sum():,.0f}")
                else:
                    st.write("- Volume data not available")
        
        
        st.dataframe(
            ohlcv_df.sort_values('date', ascending=False),
            use_container_width=True,
            height=300
        )
    else:
        st.warning("No OHLCV data available for display")


st.markdown("---")
st.subheader("⬇️ Download Data")

def convert_df(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode("utf-8")

if not price_df.empty or not ohlcv_df.empty:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not price_df.empty:
            csv_price = convert_df(price_df)
            st.download_button(
                "📄 Download Price Data",
                data=csv_price,
                file_name=f"{coin}_price_{days}days_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download price data as CSV file"
            )
    
    with col2:
        if not ohlcv_df.empty:
            csv_ohlcv = convert_df(ohlcv_df)
            st.download_button(
                "📊 Download OHLCV Data",
                data=csv_ohlcv,
                file_name=f"{coin}_ohlcv_{days}days_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download OHLCV data as CSV file"
            )
    
    with col3:
        if not price_df.empty and not ohlcv_df.empty:
           
            combined_df = pd.merge(
                price_df[['date', 'price']], 
                ohlcv_df, 
                on='date', 
                how='outer'
            )
            csv_combined = convert_df(combined_df)
            st.download_button(
                "📦 Download Combined Data",
                data=csv_combined,
                file_name=f"{coin}_combined_{days}days_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download both price and OHLCV data combined"
            )

else:
    st.info("No data available for download. Please refresh or try different parameters.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    📊 Crypto Data Explorer | Data provided by CoinGecko API
</div>
""", unsafe_allow_html=True)