# NeuroRAG Project Structure

## Directory Layout

```
neuroRAG/
├── app.py                 # Main application entry point
├── app_old.py            # Backup of original monolithic app
├── ingest.py             # Data ingestion script
├── requirements.txt      # Python dependencies
├── chroma_db/           # Vector database storage
├── data/                # Source documents
│
├── config/              # Configuration package
│   ├── __init__.py
│   └── settings.py      # App settings, API keys, constants
│
├── utils/               # Utility functions package
│   ├── __init__.py
│   └── rag_utils.py     # RAG chain, embeddings, vector store
│
└── components/          # UI components package
    ├── __init__.py
    ├── header.py        # Title and description
    ├── suggestions.py   # Clickable question suggestions
    └── chatbox.py       # Chat input and conversation display
```

## Component Descriptions

### `config/settings.py`
- Application configuration constants
- API key management
- Model parameters (LLM, embeddings, retrieval settings)
- System prompts

### `utils/rag_utils.py`
- `get_resources()`: Load embeddings and vector store (cached)
- `get_chain()`: Create RAG chain with Gemini and ChromaDB
- `process_message()`: Process user queries and return responses

### `components/header.py`
- `render_header()`: Display app title and description

### `components/suggestions.py`
- `render_suggestions()`: Display 600px scrollable suggestion box
- Clickable buttons that populate the chat input

### `components/chatbox.py`
- `render_chatbox()`: Display conversation history and input box
- Handles message processing and response generation

### `app.py`
Main orchestrator that:
1. Sets up page config and API keys
2. Loads resources (vector store and RAG chain)
3. Renders UI components in order:
   - Header (title + description)
   - Suggestions (600px scrollable box)
   - Chatbox (with spacing)

## UI Flow

1. **Title**: Application name with icon
2. **Description**: What the app does and how to use it
3. **Suggestions**: 600px height box with scrollable clickable questions
4. **Empty Space**: Visual separation
5. **Chatbox**: Input field + conversation history + message processing

## Running the Application

```bash
streamlit run app.py
```

## Benefits of This Structure

✅ **Separation of Concerns**: Each file has a single responsibility
✅ **Reusability**: Components can be imported and reused
✅ **Maintainability**: Easy to find and update specific functionality
✅ **Testability**: Individual components can be tested separately
✅ **Scalability**: Easy to add new components or features
✅ **Configuration Management**: All settings in one place
