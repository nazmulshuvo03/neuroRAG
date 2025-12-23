"""
We should have a directory callled 'data' in this same repo,
containing all the files needed to be used for RAG
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1. Load PDFs from data directory
data_path = os.path.join(os.path.dirname(__file__), "data")
loader = PyPDFDirectoryLoader(data_path)
documents = loader.load()

if not documents:
    print("No documents found!!")
    exit()

# 2. Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} text chunks.")

# 3. Create Embeddings on CPU
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # we can use cuda here if we use GPU
)

# 4. Save to Chroma Vector DB
vector_store = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory="chroma_db"
)
print("Success! Database created in 'chroma_db' folder.")
