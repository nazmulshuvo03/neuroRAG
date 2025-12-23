import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st

load_dotenv()


@st.cache_resource
def get_resource():
    try:
        # 1. Load the embedding mode
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},  # we can use cuda here if we use GPU
        )

        # 2. Connect to the vector database we created
        chroma_db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
        vector_store = Chroma(
            persist_directory=chroma_db_path, embedding_function=embeddings
        )

        return vector_store

    except Exception as e:
        st.error(f"{str(e)}")
        st.stop()


def get_chain(vector_store):
    try:
        # 1. Connectonnect to LLM (Gemini in my case)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

        # 2. Turn the DB into a search engine
        retriver = vector_store.as_retriever(search_kwargs={"k": 5})

        # 3. The Prompt Template
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

        # 4. Create the thinking chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriver, document_chain)

        return retrieval_chain

    except Exception as e:
        st.error(f"{str(e)}")
        st.stop()


def process_user_message(user_input, rag_chain):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response.get("answer") if isinstance(response, dict) else None
                if not answer:
                    if isinstance(response, dict):
                        answer = (
                            response.get("output")
                            or response.get("text")
                            or str(response)
                        )
                    else:
                        answer = str(response)

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                error_message = f"Error generating response: {e}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )


# --- UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

vector_store = get_resource()
rag_chain = get_chain(vector_store)

st.markdown("### 💬 Conversation")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("<br>", unsafe_allow_html=True)

# Input Box
if user_input := st.chat_input("Ask about Neurodevelopmental Disorders..."):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                # 3. Save AI Message
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                error_message = f"❌ Error generating response: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
