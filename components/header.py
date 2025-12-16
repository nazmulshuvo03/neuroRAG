"""Title and description component"""
import streamlit as st
from config.settings import APP_TITLE, APP_ICON, APP_DESCRIPTION


def render_header():
    """Render the application title and description"""
    st.title(f"{APP_ICON} {APP_TITLE} Chatbot")
    st.markdown(APP_DESCRIPTION)
    st.markdown("---")
