"""Configuration settings for NeuroRAG application"""
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_api_key():
    """Load and validate API key from .env or Streamlit secrets"""
    if "GOOGLE_API_KEY" not in os.environ:
        if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        else:
            st.error("⚠️ GOOGLE_API_KEY not found! Please set it in .env file (local) or Streamlit secrets (deployment).")
            st.stop()

# App configuration
APP_TITLE = "NeuroRAG"
APP_ICON = "🧠"
APP_DESCRIPTION = """
**NeuroRAG** is a Retrieval-Augmented Generation (RAG) chatbot specialized in *Neurodevelopmental Disorders*. 
It uses **ChromaDB** as a vector database to store indexed medical documents, **Gemini** (Google's LLM) as the reasoning engine, 
and **HuggingFace embeddings** for semantic search. The app automatically detects if a GPU is available for faster processing.

**How to use:**
- Browse the **Suggestions** below and click any question to get an instant answer.
- Or type your own question in the chat input box at the bottom.
- For **medical questions about Neurodevelopmental Disorders**, answers are generated from the indexed documents.
- The bot can also respond to **greetings and general questions** in a helpful manner.
"""

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash-lite"
LLM_TEMPERATURE = 0.3
RETRIEVAL_K = 5

# Paths
CHROMA_DB_PATH = "chroma_db"

# Prompt template
SYSTEM_PROMPT = """
You are a helpful medical assistant specializing in Neurodevelopmental Disorders.

**Instructions:**
1. **First priority**: If the answer is found in the context below, use ONLY that information to answer.
2. **For greetings and basic questions** (like "hi", "hello", "what can you do?", "who are you?"):
   - Respond politely and briefly explain your capabilities as a specialized chatbot.
3. **For general questions unrelated to Neurodevelopmental Disorders**:
   - Provide a brief, helpful answer.
   - Add: "Note: This question is not directly related to our specialized database on Neurodevelopmental Disorders."
4. **For medical questions about Neurodevelopmental Disorders NOT in the context**:
   - Reply: "I cannot find this specific information in the provided documents."

<context>
{context}
</context>

Question: {input}

Remember: Prioritize information from the context when available. Be friendly and helpful for basic interactions.
"""
