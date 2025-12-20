"""
Main entry point for Streamlit Cloud deployment
This file must be named 'streamlit_app.py' and located in the project root
"""
import sys
from pathlib import Path
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up environment for Streamlit Cloud
if 'GOOGLE_API_KEY' not in os.environ:
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets.get('general', {}):
            os.environ['GOOGLE_API_KEY'] = st.secrets['general']['GOOGLE_API_KEY']
    except:
        pass

# Import and run the Streamlit app
from ui.streamlit_app import main

if __name__ == "__main__":
    main()
