"""
Streamlit UI for AI Investment Advisor Agent.
Provides personalized investment recommendations using LangGraph + Gemini LLM.
"""

import os
import sys
import streamlit as st
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_ROOT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from agents.investment_advisor_agent import get_investment_advice


def show():
    st.header("🤖 AI Investment Advisor")
    st.markdown("""
    Get personalized investment recommendations powered by AI. 
    Analyzes latest market news, economic indicators, and geopolitical factors.
    """)
    
    # ── Setup session state ────────────────────────────────────────────────────
    if "advice_history" not in st.session_state:
        st.session_state.advice_history = []
    
    st.divider()
    
    # ── Investment Profile Section ────────────────────────────────────────────
    st.subheader("📋 Your Investment Profile")
    
    prof_col1, prof_col2, prof_col3 = st.columns(3)
    
    with prof_col1:
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["Conservative", "Moderate", "Aggressive"],
            index=1,
            help="Your comfort with market volatility"
        )
    
    with prof_col2:
        portfolio_type = st.selectbox(
            "Investment Style",
            ["Income", "Growth", "Balanced", "Speculative"],
            index=2,
            help="What are you optimizing for?"
        )
    
    with prof_col3:
        time_horizon = st.selectbox(
            "Time Horizon",
            ["Short-term (< 1 yr)", "Medium-term (1-3 yrs)", "Long-term (3+ yrs)"],
            index=1,
            help="When do you need the money?"
        )
    
    amount_col1, amount_col2 = st.columns(2)
    
    with amount_col1:
        investment_amount = st.number_input(
            "Investment Amount (₹)",
            min_value=10000,
            value=100000,
            step=10000,
            help="How much are you planning to invest?"
        )
    
    with amount_col2:
        existing_portfolio = st.selectbox(
            "Existing Portfolio",
            ["None", "Stocks", "Mutual Funds", "Mixed"],
            index=2,
            help="Do you have existing investments?"
        )
    
    st.divider()
    
    # ── Query Input ────────────────────────────────────────────────────────────
    st.subheader("❓ Your Investment Question")
    
    user_query = st.text_area(
        "What's your investment question or goal?",
        placeholder="""Examples:
- I'm a tech enthusiast, where should I invest?
- How to build a Rs 1 Lakh portfolio?
- Best sectors for next 6 months?
- Should I invest in AutoMobiles or IT?
- How to hedge against inflation?""",
        height=120,
        key="query_input"
    )
    
    # ── Additional Preferences ────────────────────────────────────────────────
    with st.expander("🎯 Sector Preferences (Optional)"):
        col1, col2, col3 = st.columns(3)
        selected_sectors = []
        
        sectors = ["Banking", "IT", "Auto", "Pharma", "Finance", "Energy", "Telecom", "FMCG"]
        
        for idx, sector in enumerate(sectors):
            col = [col1, col2, col3][idx % 3]
            with col:
                if st.checkbox(f"{sector}", key=f"sector_{sector}"):
                    selected_sectors.append(sector)
    
    st.divider()
    
    # ── Get Advice Button ──────────────────────────────────────────────────────
    col_submit, col_history = st.columns([3, 1])
    
    with col_submit:
        get_advice_btn = st.button("🚀 Get AI Recommendation", type="primary", width='stretch')
    
    with col_history:
        show_history = st.checkbox("📜 Show History")
    
    # ── Processing ────────────────────────────────────────────────────────────
    if get_advice_btn:
        if not user_query.strip():
            st.error("❌ Please enter your investment question.")
        else:
            with st.spinner("🤔 AI Agent analyzing market... This may take 10-30 seconds"):
                try:
                    # Build investment profile
                    investment_profile = {
                        "risk_tolerance": risk_tolerance.lower(),
                        "portfolio_type": portfolio_type.lower(),
                        "time_horizon": time_horizon,
                        "amount": investment_amount,
                        "existing_portfolio": existing_portfolio,
                    }
                    
                    # Get advice
                    advice = get_investment_advice(
                        user_query=user_query,
                        investment_profile=investment_profile,
                        include_sectors=selected_sectors if selected_sectors else None,
                    )
                    
                    # Store in history
                    st.session_state.advice_history.insert(0, {
                        "query": user_query,
                        "profile": investment_profile,
                        "advice": advice,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Display results
                    _display_recommendation(advice)
                    
                except Exception as e:
                    st.error(f"❌ Error getting recommendation: {str(e)}")
    
    # ── Show History ───────────────────────────────────────────────────────────
    if show_history and st.session_state.advice_history:
        st.divider()
        st.subheader("📜 Recent Recommendations")
        
        for idx, item in enumerate(st.session_state.advice_history[:5]):
            with st.expander(f"📌 {item['query'][:60]}... ({item['timestamp'][:10]})"):
                st.write(f"**Your Question:** {item['query']}")
                st.json(item['advice'])


def _display_recommendation(advice: dict):
    """Display formatted investment recommendation."""
    
    st.divider()
    st.subheader("💡 Investment Recommendation")
    
    # Display the AI-generated advice
    advice_text = advice.get("advice", "Unable to generate recommendation.")
    
    # Check if there was an error
    if advice.get("status") == "error":
        st.error(advice_text)
    else:
        # Display as a nice formatted box
        st.markdown(advice_text)
    
    # Show confidence level if available
    confidence = advice.get("confidence", 0)
    if confidence > 0:
        st.caption(f"✓ Confidence: {confidence:.0%}")
    
    st.divider()
    
    # Optional: Show timestamp
    if "timestamp" in advice:
        st.caption(f"Generated: {advice['timestamp'][:19]}")
    
    st.divider()
    
    # Debug: Show raw JSON for power users
    with st.expander("🔧 Raw Data (Advanced)"):
        st.json(advice)


if __name__ == "__main__":
    show()
