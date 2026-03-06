"""
secrets_bridge.py
─────────────────
Call this ONCE at the top of your Streamlit app to make Streamlit Cloud
secrets available as environment variables (so python-dotenv and os.getenv()
work identically in both local and cloud environments).

Usage — add to streamlit_app/main.py at the very top:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from .streamlit.secrets_bridge import inject_secrets
    inject_secrets()
"""
import os


def inject_secrets():
    """Push Streamlit secrets into os.environ if running on Streamlit Cloud."""
    try:
        import streamlit as st
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass   # local dev — .env file is used instead
