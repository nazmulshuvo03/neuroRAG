import pysqlite3
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import torch
import streamlit as st
import os
from dotenv import load_dotenv

# 1. The Brain (Gemini)
# This library connects to Google's API
from langchain_google_genai import ChatGoogleGenerativeAI

# 2. The Memory (ChromaDB)
from langchain_chroma import Chroma

# 3. The Local Embeddings (HuggingFace)
# This runs on your RTX 3060 to understand the text
from langchain_huggingface import HuggingFaceEmbeddings

# 4. The "Chain" (The Logic)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load API Key
load_dotenv()

# Page Title
st.set_page_config(page_title="NeuroRAG", page_icon="🧠")
st.title("🧠 NeuroRAG Chatbot")


# --- Logic Section ---
@st.cache_resource
def get_resources():
    # AUTOMATIC DEVICE DETECTION
    # If a GPU is available (your PC), use it. 
    # If not (Streamlit Cloud), switch to CPU automatically.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    st.write(f"⚙️ System: Using {device.upper()} for processing.") # Optional: Show user what is running
    
    # Load the GPU-accelerated embedding model
    # We use the exact same model name as we did in ingest.py
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    # Connect to the Database we built earlier
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    return vector_store


def get_chain(vector_store):
    # Connect to Gemini (The Brain)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # Turn the DB into a search engine (retrieve top 5 relevant chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # The Prompt Template (Instructions for the AI)
    prompt = ChatPromptTemplate.from_template(
        """
    You are a helpful medical assistant. 
    Answer the user's question based ONLY on the context provided below.
    If the answer is not in the context, reply: "I cannot find this information in the provided documents."

    <context>
    {context}
    </context>

    Question: {input}
    """
    )

    # Create the thinking chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


# --- UI Section ---

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Resources
vector_store = get_resources()
rag_chain = get_chain(vector_store)

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Box
if user_input := st.chat_input("Ask about Neurodevelopmental Disorders..."):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            st.markdown(answer)

    # 3. Save AI Message
    st.session_state.messages.append({"role": "assistant", "content": answer})
