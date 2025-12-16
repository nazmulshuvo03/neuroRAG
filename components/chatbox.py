"""Chatbox component for user input and conversation display"""
import streamlit as st
from utils.rag_utils import process_message


def render_chatbox(rag_chain):
    """Render the chat interface with input box and message history"""
    
    # Add spacing before chatbox
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    st.markdown("### 💬 Conversation")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Check if a suggestion was selected
    default_input = ""
    if st.session_state.get("selected_suggestion"):
        default_input = st.session_state.selected_suggestion
        st.session_state.selected_suggestion = None  # Clear after using
    
    # Chat input box
    if default_input:
        # If suggestion was clicked, immediately process it
        user_input = default_input
    else:
        user_input = st.chat_input("Ask about Neurodevelopmental Disorders...")
    
    # Process user input
    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = process_message(user_input, rag_chain)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_message = f"❌ {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
        
        # Rerun to show the new messages
        st.rerun()
