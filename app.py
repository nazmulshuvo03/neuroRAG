"""
NeuroRAG - Neurodevelopmental Disorders Chatbot
Main application file that orchestrates all components

Structure:
1. Title of the application
2. Details of what this application does
3. Suggestions section (600px height, scrollable, clickable)
4. Empty space before chatbox
5. Chatbox (input + conversation history + message processing)
"""

import streamlit as st

# Configuration
from config.settings import APP_TITLE, APP_ICON, setup_api_key

# Utils
from utils.rag_utils import get_resources, get_chain

# Components
from components.header import render_header
from components.suggestions import render_suggestions
from components.chatbox import render_chatbox


def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON)
    
    # Setup API key
    setup_api_key()
    
    # Load resources (embeddings and vector store)
    vector_store = get_resources()
    rag_chain = get_chain(vector_store)
    
    # === UI Flow ===
    
    # 1. Title of the application
    # 2. Details of what this application does
    render_header()
    
    # 3. Suggestions section (600px height, scrollable, clickable)
    render_suggestions()
    
    # 4. Empty space before chatbox (handled in chatbox component)
    # 5. Chatbox (input + conversation + processing)
    render_chatbox(rag_chain)


if __name__ == "__main__":
    main()
