"""RAG utilities for loading embeddings, vector store, and chain"""
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import torch
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from config.settings import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVAL_K,
    CHROMA_DB_PATH,
    SYSTEM_PROMPT
)


@st.cache_resource
def get_resources():
    """Load embeddings and vector store with automatic device detection"""
    try:
        # AUTOMATIC DEVICE DETECTION
        # If a GPU is available (your PC), use it. 
        # If not (Streamlit Cloud), switch to CPU automatically.
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Show a temporary info message about the device, then remove it
        _device_msg = st.empty()
        _device_msg.info(f"⚙️ System: Using {device.upper()} for processing.")
        # keep the message briefly so users see it, then remove it
        time.sleep(1.5)
        _device_msg.empty()

        # Load the embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
        )

        # Connect to the Database
        chroma_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), CHROMA_DB_PATH)
        
        if not os.path.exists(chroma_db_path):
            st.error(f"❌ ChromaDB folder not found at: {chroma_db_path}")
            st.info("Please ensure the 'chroma_db' folder is present in the repository.")
            st.stop()

        vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
        
        # Verify the vector store has documents
        try:
            collection = vector_store._collection
            count = collection.count()
            if count == 0:
                st.error(f"⚠️ Vector store is empty! Please run ingest.py first.")
                st.stop()
        except Exception as debug_e:
            st.warning(f"⚠️ Could not verify document count: {debug_e}")
        
        # Show a short success toast-like message then clear it
        _loaded_msg = st.empty()
        _loaded_msg.success("✅ Resources loaded successfully!")
        time.sleep(1.5)
        _loaded_msg.empty()
        
        return vector_store
        
    except Exception as e:
        st.error(f"❌ Error loading resources: {str(e)}")
        st.stop()


def get_chain(vector_store):
    """Create the RAG chain with Gemini and the vector store"""
    try:
        # Connect to Gemini (The Brain)
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

        # Turn the DB into a search engine (retrieve top K relevant chunks)
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

        # The Prompt Template (Instructions for the AI)
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

        # Create the thinking chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain
        
    except Exception as e:
        st.error(f"❌ Error creating RAG chain: {str(e)}")
        st.stop()


def process_message(user_input, rag_chain):
    """Process a user message and return the response"""
    try:
        response = rag_chain.invoke({"input": user_input})
        
        # Debug: Check what context was retrieved
        if isinstance(response, dict) and "context" in response:
            retrieved_docs = response.get("context", [])
            if not retrieved_docs:
                st.warning("⚠️ No relevant documents were retrieved from the database.")
        
        # Some LLM wrappers return 'answer' or 'output' — try both
        answer = response.get("answer") if isinstance(response, dict) else None
        if not answer:
            # fallback to common key names
            if isinstance(response, dict):
                answer = response.get("output") or response.get("text") or str(response)
            else:
                answer = str(response)
        return answer
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")
