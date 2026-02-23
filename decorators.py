import sys
from functools import wraps

def conditional_cache_resource(**cache_kwargs):
    """
    Apply @st.cache_resource only if Streamlit is actually running/running the script.
    When imported from a regular Python script (like api.py), it returns the original function.
    """
    def decorator(func):
        # Determine if we are in a Streamlit context
        # 1. Check if 'streamlit' is in sys.modules
        # 2. Check if the script is being run via 'streamlit run'
        is_streamlit = "streamlit" in sys.modules
        
        if is_streamlit:
            try:
                import streamlit as st
                # Even if imported, check if runtime is active to avoid 'missing ScriptRunContext' warnings
                # during import in FastAPI
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                if get_script_run_ctx() is not None:
                    return st.cache_resource(**cache_kwargs)(func)
                
                # If we are in streamlit but NOT in a script run context (e.g. initial import in FastAPI thread)
                # We return the original function to avoid the warning
                return func
            except Exception:
                return func
        return func
    return decorator
