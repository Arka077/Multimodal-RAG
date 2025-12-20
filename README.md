# Enhanced Multimodal RAG System

🚀 **[Live Demo: Try it now!](https://multimodal-docs-analyser.streamlit.app)** 🚀

A production-ready Retrieval-Augmented Generation (RAG) system powered by **Gemini API**, **LangChain**, and **LangGraph** for multimodal document understanding and question answering.

> 💡 **Quick Access:** The application is deployed and accessible at: **https://multimodal-docs-analyser.streamlit.app**

## 🌟 Features

### Core Capabilities
- 📄 **PDF Processing**: Extract text, tables, images with Docling
- 🖼️ **Vision Understanding**: Gemini 2.0 Flash for image analysis
- 🎵 **Audio Transcription**: OpenAI Whisper integration
- 🧠 **Knowledge Graph**: Automatic entity extraction and relationship mapping
- 🔍 **Hybrid Retrieval**: Vector (FAISS) + BM25 + Knowledge Graph fusion
- 🤖 **LangGraph Orchestration**: Stateful workflow management
- 💬 **Chat History**: Persistent conversation management
- 📊 **Visualizations**: Interactive KG and embedding space visualization
- 🤖 **AI-Enhanced Chunks**: Every chunk enriched with detailed Gemini summaries for better retrieval
- 🔄 **Multi-Key Support**: Automatic API key rotation when rate limits are hit
- 🧹 **Auto-Reset**: Knowledge base resets on app reload for fresh starts

### Technical Highlights
- ✅ **API-based Models**: No local model downloads required
- ✅ **Modular Architecture**: Clean separation of concerns
- ✅ **Type Safety**: Full type hints throughout
- ✅ **Production-Ready**: Error handling, logging, caching
- ✅ **Smart Fallback**: Multiple API keys prevent service interruption

## 📁 Project Structure

```
Rag/
├── config/                 # Configuration and settings
│   ├── __init__.py
│   └── settings.py        # API keys, paths, parameters
├── models/                # API client wrappers
│   ├── __init__.py
│   ├── gemini_client.py   # Gemini API for text & vision
│   └── whisper_client.py  # Whisper for audio
├── core/                  # Core processing logic
│   ├── __init__.py
│   ├── document_processor.py  # PDF, audio, image processing
│   ├── chunking.py           # Semantic text chunking
│   ├── knowledge_graph.py    # KG extraction and building
│   └── embeddings.py         # Embedding generation
├── retrieval/             # Retrieval components
│   ├── __init__.py
│   ├── vector_store.py       # FAISS vector search
│   ├── bm25_search.py        # BM25 keyword search
│   └── hybrid_retrieval.py   # Hybrid fusion + reranking
├── agents/                # LangGraph agents
│   ├── __init__.py
│   └── rag_agent.py          # RAG orchestration agent
├── ui/                    # User interface
│   ├── __init__.py
│   ├── knowledge_base.py     # KB management
│   └── streamlit_app.py      # Streamlit web interface
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── chat_manager.py       # Chat session management
│   └── visualizations.py     # KG and embedding viz
├── requirements.txt       # Python dependencies
├── main.py               # Application entry point
└── README.md             # This file
```

## 🚀 Quick Start

### Option 1: Use the Deployed App (Recommended)

Simply visit **[https://multimodal-docs-analyser.streamlit.app](https://multimodal-docs-analyser.streamlit.app)** to start using the application immediately!

### Option 2: Local Installation

#### 1. Installation

```bash
# Clone or navigate to the project directory
cd Rag

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Install system dependencies (for PDF processing)
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# macOS:
brew install tesseract poppler

# Windows: Download from official websites
```

#### 2. Configuration

Edit `config/settings.py` to set your API key:

```python
GEMINI_API_KEY: str = "YOUR_GEMINI_API_KEY_HERE"
```

#### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The Streamlit interface will launch at `http://localhost:8501`

## 📖 Usage Guide

### Adding Documents

1. Navigate to the **📚 Knowledge Base** tab
2. Click "Upload Files" and select documents:
   - PDFs (`.pdf`)
   - Images (`.png`, `.jpg`, `.jpeg`)
   - Audio (`.wav`, `.mp3`, `.m4a`, `.flac`)
3. Click "Add Files 📥"
4. Wait for processing (includes extraction, summarization, KG building, indexing)

### Asking Questions

1. Go to the **💬 Chat** tab
2. Type your question in the text box
3. Optional: Upload an image for visual queries
4. Press Enter or click "Send 📤"

**Query Examples:**
- "What are the main findings in the report?"
- "Show me the table with performance metrics"
- "Compare the results from Figure 1 and Figure 2"
- "Explain the methodology described in the audio"

### Exploring Visualizations

#### Knowledge Graph
- Navigate to **🕸 Knowledge Graph** tab
- Click "Generate Graph 🔄"
- Interact with the graph:
  - Hover over nodes for details
  - Gold borders indicate nodes with visual representations
  - Colors represent entity types

#### Embedding Space
- Go to **📊 Embeddings** tab
- Click "Generate Visualization 🔄"
- View 3D UMAP projection of document embeddings
- Colors represent document types

## 🔧 Configuration Options

In `config/settings.py`:

```python
# Model Configuration
GEMINI_MODEL = "gemini-2.0-flash-exp"    # Gemini model version
WHISPER_MODEL_SIZE = "base"              # Whisper model size

# Processing Parameters
MAX_CHUNK_SIZE = 600                     # Words per chunk
CHUNK_OVERLAP = 100                      # Overlap between chunks
TEMPERATURE = 0.3                        # Generation temperature

# Retrieval Parameters
DEFAULT_TOP_K = 50                       # Initial retrieval size
FINAL_TOP_K = 15                         # Final reranked results
RERANK_CONFIDENCE_THRESHOLD = 0.1        # Minimum confidence score
```

## 🏗️ Architecture Overview

### Document Processing Pipeline
1. **Extraction**: Docling extracts text, tables, images from PDFs
2. **Chunking**: Semantic chunking with overlap
3. **Enrichment**: Gemini generates summaries for all chunks
4. **Screenshot Generation**: Page snapshots for visual context

### Knowledge Graph Building
1. **Triple Extraction**: Gemini identifies (subject, predicate, object) triples
2. **Node Classification**: Entities categorized (person, organization, concept, etc.)
3. **Alias Detection**: Identifies synonyms and canonical forms
4. **Graph Construction**: NetworkX graph with rich metadata

### Retrieval Pipeline
1. **Query Analysis**: Intent detection, entity extraction
2. **Multi-source Retrieval**:
   - Vector search (Gemini embeddings + FAISS)
   - BM25 keyword search
   - Knowledge graph traversal
3. **Adaptive Fusion**: RRF with dynamic weighting
4. **Context Expansion**: Add parent/sibling chunks
5. **Reranking**: Cross-encoder with modality boosting

### Answer Generation
1. **Context Assembly**: Combine top-k chunks with citations
2. **Prompt Engineering**: Intent-specific templates
3. **LangGraph Workflow**: Stateful generation with Gemini
4. **Post-processing**: Visual enhancement, table formatting

## 🎯 Advanced Features

### Modality-Aware Retrieval
The system automatically detects query intent and boosts relevant content types:
- Audio queries → boost audio chunks
- Image queries → boost image chunks
- Table queries → boost table chunks

### Visual Enhancement
When visual intent is detected, the system automatically includes:
- Original images
- PDF page screenshots
- Audio playback widgets

### Chat Context
Maintains conversation history for:
- Follow-up questions
- Context-aware responses
- Session management

## 🐛 Troubleshooting

### Common Issues

**Issue**: "API key not valid"
- Solution: Check `config/settings.py` and ensure correct Gemini API key

**Issue**: "Module not found"
- Solution: Ensure all dependencies installed: `pip install -r requirements.txt`

**Issue**: "PDF processing failed"
- Solution: Install system dependencies (tesseract, poppler)

**Issue**: "Out of memory"
- Solution: Reduce batch sizes in `config/settings.py`

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Tips

1. **Batch Processing**: Process multiple files together
2. **Cache Management**: Screenshots and audio cached automatically
3. **Index Persistence**: Indices saved to disk for fast restart
4. **API Rate Limits**: Built-in rate limiting for Gemini API

## ☁️ Cloud Deployment

### Live Deployment

✨ **This application is already deployed and accessible at:**
### **[https://multimodal-docs-analyser.streamlit.app](https://multimodal-docs-analyser.streamlit.app)**

### Deploy Your Own Instance on Streamlit Cloud

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app" and select your repository
   - Set main file: `streamlit_app.py`
   - Add your `GOOGLE_API_KEY` in secrets

3. **Configure Secrets:**
   In app settings → Secrets, add:
   ```toml
   [general]
   GOOGLE_API_KEY = "your-api-key-here"
   ```

📖 **Full deployment guide:** See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md)

## 🔐 Security Notes

- API keys stored in `config/settings.py` (do not commit to version control)
- For Streamlit: Use `.streamlit/secrets.toml` (auto-ignored)
- For production: Use environment variables or secrets management
- Add sensitive files to `.gitignore`

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional document formats (Word, Excel, etc.)
- More embedding models
- Advanced KG reasoning
- Multi-language support

## 📧 Support

For issues and questions, please open an issue in the repository.

---

🌐 **Live Demo:** [multimodal-docs-analyser.streamlit.app](https://multimodal-docs-analyser.streamlit.app)

Built with ❤️ using Gemini, LangChain, and LangGraph
