"""
Streamlit page for managing tracked stocks
Allows adding/removing custom stocks beyond the default 5
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config_manager import ConfigManager


def show():
    st.header("⚙️ Manage Stocks")
    
    st.write("Add or remove stocks to track. You start with 5 default stocks, but can add any Indian stock symbol.")
    
    # Initialize config manager
    cm = ConfigManager()
    
    # Show current status
    stats = cm.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Default Stocks", stats['default_stocks_count'])
    with col2:
        st.metric("Custom Stocks", stats['custom_stocks_count'])
    with col3:
        st.metric("Total Tracked", stats['total_tracked'])
    with col4:
        st.metric("Max Manageable", "100+")
    
    # Default stocks (read-only)
    st.subheader("📌 Default Stocks (Fixed)")
    st.info("These are default stocks that cannot be removed:")
    default_cols = st.columns(len(stats['default_stocks']))
    for idx, stock in enumerate(stats['default_stocks']):
        with default_cols[idx]:
            st.write(f"🔹 {stock.replace('.NS', '')}")
    
    # Custom stocks section
    st.subheader("➕ Custom Stocks (Your Additions)")
    
    if stats['custom_stocks_count'] > 0:
        st.write(f"You've added {stats['custom_stocks_count']} custom stocks:")
        
        # Display custom stocks with delete buttons
        for stock in stats['custom_stocks']:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"✓ {stock}")
            with col2:
                if st.button("🗑️", key=f"delete_{stock}", help=f"Remove {stock}"):
                    if cm.remove_stock(stock):
                        st.success(f"✓ Removed {stock}")
                        st.rerun()
                    else:
                        st.error(f"Failed to remove {stock}")
    else:
        st.info("No custom stocks added yet. Add one below!")
    
    # Add new stock
    st.subheader("🔍 Add New Stock")
    
    st.write("Enter Indian stock symbols (NSE). Examples: WIPRO, SBIN, MARUTI, BAJAJFINSV")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_stock = st.text_input(
            "Stock Symbol",
            placeholder="e.g., WIPRO (will add .NS automatically)",
            label_visibility="collapsed"
        )
    
    with col2:
        add_button = st.button("➕ Add", use_container_width=True, type="primary")
    
    if add_button:
        if new_stock.strip():
            is_valid, normalized = cm.validate_stock(new_stock)
            
            if not is_valid:
                st.error(f"❌ Invalid stock symbol: {new_stock}")
            else:
                # Check if already exists
                all_stocks = cm.get_all_stocks()
                if normalized in all_stocks:
                    st.error(f"❌ {normalized} is already being tracked!")
                else:
                    if cm.add_stock(new_stock):
                        st.success(f"✅ Added {normalized}!")
                        st.rerun()
                    else:
                        st.error(f"Failed to add {normalized}")
        else:
            st.warning("Please enter a stock symbol")
    
    # Popular stocks quick add
    st.subheader("⭐ Popular Indian Stocks")
    st.write("Quick add popular stocks:")
    
    popular_stocks = [
        ("WIPRO", "Technology"),
        ("SBIN", "Banking"),
        ("MARUTI", "Automotive"),
        ("BAJAJFINSV", "Financial"),
        ("HINDUNILVR", "FMCG"),
        ("KOTAKBANK", "Banking"),
        ("LT", "Infrastructure"),
        ("POWERGRID", "Energy"),
        ("ADANIPORTS", "Ports"),
        ("BRITANNIA", "FMCG"),
    ]
    
    cols = st.columns(5)
    for idx, (stock, sector) in enumerate(popular_stocks):
        with cols[idx % 5]:
            if st.button(f"{stock}\n({sector})", use_container_width=True):
                all_stocks = cm.get_all_stocks()
                normalized = stock + ".NS"
                
                if normalized in all_stocks:
                    st.warning(f"{normalized} already tracked")
                else:
                    if cm.add_stock(stock):
                        st.success(f"Added {stock}!")
                        st.rerun()
                    else:
                        st.warning(f"Could not add {stock}")
    
    # Reset option
    st.divider()
    st.subheader("🔄 Reset")
    
    if st.checkbox("Remove all custom stocks?", value=False):
        if st.button("🔄 Reset to Default (5 stocks only)", type="secondary"):
            cm.reset_custom_stocks()
            st.success("✓ Reset to default stocks")
            st.rerun()
    
    # Info section
    st.divider()
    st.subheader("ℹ️ Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **How it works:**
        - Add any Indian stock listed on NSE
        - Custom stocks are saved to `config/user_stocks.json`
        - They're automatically included in data collection
        - Model training can be done per stock
        - Predictions show all tracked stocks
        """)
    
    with col2:
        st.write("""
        **Stock Symbol Format:**
        - Use NSE symbols (e.g., WIPRO, not WIT)
        - .NS suffix added automatically
        - Examples: RELIANCE, TCS, INFY, HDFCBANK
        
        **Data Collection:**
        - Runs for all tracked stocks
        - May take longer with more stocks
        - News collected for active stocks only
        """)


if __name__ == "__main__":
    show()
