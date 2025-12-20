"""
Main entry point for Streamlit Cloud deployment
This file must be named 'streamlit_app.py' and located in the project root
"""
import os
import sys
import traceback

# Set environment variables BEFORE any other imports
os.environ['RAPIDOCR_HOME'] = '/tmp/rapidocr'
os. environ['HF_HOME'] = '/tmp/huggingface'
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['XDG_CACHE_HOME'] = '/tmp/cache'

# Create directories
for dir_path in ['/tmp/rapidocr', '/tmp/huggingface', '/tmp/torch', '/tmp/cache']:
    os.makedirs(dir_path, exist_ok=True)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Now import streamlit and handle errors
try:
    import streamlit as st
    
    # Show that we got this far
    st.write("🔄 Loading application...")
    
    # Import the main app
    from ui. streamlit_app import main
    
    if __name__ == "__main__": 
        try:
            main()
        except Exception as e:
            st. error(f"❌ **Error in main():** {str(e)}")
            st.code(traceback.format_exc())
            
except Exception as e:
    # If streamlit isn't even available, print to console
    print(f"FATAL ERROR: {e}")
    print(traceback.format_exc())
    
    # Try to show in streamlit if possible
    try:
        import streamlit as st
        st. error(f"❌ **Fatal Error:** {str(e)}")
        st.code(traceback.format_exc())
    except:
        pass
