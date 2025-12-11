# NeuroRAG Chatbot 🧠

A RAG (Retrieval-Augmented Generation) chatbot specialized in Neurodevelopmental Disorders using Gemini AI, ChromaDB, and HuggingFace embeddings.

## Features

- 🤖 Powered by Google Gemini 1.5 Flash
- 📚 ChromaDB vector database for document retrieval
- 🔍 HuggingFace embeddings (sentence-transformers)
- ⚡ Automatic GPU/CPU detection
- 🌐 Works in both local and cloud deployment

## Local Development

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, will fall back to CPU)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nazmulshuvo03/neuroRAG.git
   cd neuroRAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will automatically:
- Detect if GPU is available (CUDA) or fall back to CPU
- Load the pre-built ChromaDB vector store from `chroma_db/` folder
- Start the Streamlit interface

## Streamlit Cloud Deployment

### Prerequisites

- Streamlit Cloud account
- Google API Key for Gemini

### Deployment Steps

1. **Push your code to GitHub** (make sure `chroma_db/` folder is included)

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the `app.py` file as the main file

3. **Configure Secrets**
   - In Streamlit Cloud dashboard, go to App Settings → Secrets
   - Add your secrets in TOML format:
     ```toml
     GOOGLE_API_KEY = "your-google-api-key-here"
     ```

4. **Deploy!**
   - Streamlit will automatically install dependencies from `requirements.txt`
   - The app will run on CPU in the cloud (GPU not needed)

## Project Structure

```
neuroRAG/
├── app.py                      # Main Streamlit application
├── ingest.py                   # Script to build ChromaDB (already done)
├── requirements.txt            # Python dependencies
├── chroma_db/                  # Pre-built vector database (included in git)
│   ├── chroma.sqlite3
│   └── [vector embeddings]
├── data/                       # Source documents
├── .env.example               # Example environment variables
└── .streamlit/
    └── secrets.toml.example   # Example Streamlit secrets
```

## Important Notes

- **ChromaDB is pre-built**: You don't need to run `ingest.py` - the `chroma_db/` folder is already included in the repository
- **API Key Required**: You must have a valid Google API key for Gemini
- **Device Detection**: The app automatically detects GPU (local) or CPU (cloud) and adjusts accordingly
- **Dependencies**: All required packages are in `requirements.txt`, including `pysqlite3-binary` for ChromaDB compatibility

## Getting a Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file (local) or Streamlit secrets (cloud)

## Troubleshooting

### "GOOGLE_API_KEY not found"
- **Local**: Make sure you have a `.env` file with `GOOGLE_API_KEY=your-key`
- **Cloud**: Add the API key in Streamlit Cloud app settings → Secrets

### "ChromaDB folder not found"
- Ensure the `chroma_db/` folder is committed to your git repository
- Check that the folder contains `chroma.sqlite3` and vector data

### Package import errors
- Make sure all packages from `requirements.txt` are installed
- On Streamlit Cloud, check the deployment logs for installation errors

## License

MIT License
