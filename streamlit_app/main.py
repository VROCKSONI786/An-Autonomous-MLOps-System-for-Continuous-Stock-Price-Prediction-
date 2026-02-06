import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="Stock Prediction MLOps",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "📥 Data Collection", "🤖 Model Training", "📈 Predictions", "📊 Performance Monitor"]
)

# Main title
st.markdown('<div class="main-header">📈 Stock Prediction MLOps System</div>', unsafe_allow_html=True)

# Route to different pages
if page == "🏠 Dashboard":
    from pages import dashboard
    dashboard.show()
elif page == "📥 Data Collection":
    from pages import data_collection
    data_collection.show()
elif page == "🤖 Model Training":
    from pages import model_training
    model_training.show()
elif page == "📈 Predictions":
    from pages import predictions
    predictions.show()
elif page == "📊 Performance Monitor":
    from pages import performance
    performance.show()