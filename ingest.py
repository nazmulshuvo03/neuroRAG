import os
import torch  # Added this to check GPU availability
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1. Load PDFs
print("Loading PDFs from 'data' folder...")
loader = PyPDFDirectoryLoader("data")
documents = loader.load()

if not documents:
    print("No documents found! Check if your PDF files are in the 'data' folder.")
    exit()

print(f"Loaded {len(documents)} pages.")

# 2. Split Text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} text chunks.")

# 3. Create Embeddings on GPU
# Check if CUDA is actually available to PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

print("Loading Embedding Model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},  # <--- This tells it to use the RTX 3060
)

# 4. Save to Vector DB
print("Creating Vector Database...")
vector_store = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory="chroma_db"
)

print("Success! Database created in 'chroma_db' folder.")
