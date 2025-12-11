#!/bin/bash
# Quick setup verification script

echo "🔍 NeuroRAG Setup Verification"
echo "================================"
echo ""

# Check Python version
echo "✓ Checking Python version..."
python --version

# Check if .env file exists
if [ -f ".env" ]; then
    echo "✓ .env file found"
else
    echo "⚠️  .env file not found. Copy .env.example to .env and add your API key"
fi

# Check if chroma_db exists
if [ -d "chroma_db" ]; then
    echo "✓ chroma_db folder found"
    if [ -f "chroma_db/chroma.sqlite3" ]; then
        echo "✓ chroma.sqlite3 database found"
    else
        echo "⚠️  chroma.sqlite3 not found in chroma_db"
    fi
else
    echo "❌ chroma_db folder not found!"
    echo "   Make sure the chroma_db folder is present in the repository"
fi

# Check if requirements are installed
echo ""
echo "📦 Checking key packages..."
python -c "import streamlit" 2>/dev/null && echo "✓ streamlit installed" || echo "❌ streamlit not installed"
python -c "import langchain" 2>/dev/null && echo "✓ langchain installed" || echo "❌ langchain not installed"
python -c "import chromadb" 2>/dev/null && echo "✓ chromadb installed" || echo "❌ chromadb not installed"
python -c "import torch" 2>/dev/null && echo "✓ torch installed" || echo "❌ torch not installed"

# Check GPU availability
echo ""
echo "🖥️  GPU Check..."
python -c "import torch; print('✓ CUDA available - will use GPU' if torch.cuda.is_available() else '⚠️  CUDA not available - will use CPU')"

echo ""
echo "================================"
echo "Run 'streamlit run app.py' to start the app"
