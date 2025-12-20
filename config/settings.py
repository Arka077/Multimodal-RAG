"""
Configuration settings for the RAG system
"""
import os
from pathlib import Path
from typing import Optional

class Settings:
    """Global configuration settings"""
    
    # API Configuration - supports multiple Gemini keys with rotation
    @property
    def GEMINI_API_KEYS(self) -> list:
        """Get list of Gemini API keys from environment, Streamlit secrets, or fallback"""
        keys = []
        
        # Try environment variable for multiple keys (comma-separated)
        if os.environ.get('GOOGLE_API_KEYS'):
            keys = [k.strip() for k in os.environ.get('GOOGLE_API_KEYS').split(',')]
        elif os.environ.get('GOOGLE_API_KEY'):
            keys = [os.environ.get('GOOGLE_API_KEY')]
        
        # Try Streamlit secrets
        if not keys:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'general' in st.secrets:
                    if 'GOOGLE_API_KEYS' in st.secrets['general']:
                        # If it's a list
                        api_keys = st.secrets['general']['GOOGLE_API_KEYS']
                        if isinstance(api_keys, list):
                            keys = api_keys
                        else:
                            keys = [k.strip() for k in api_keys.split(',')]
                    elif 'GOOGLE_API_KEY' in st.secrets['general']:
                        keys = [st.secrets['general']['GOOGLE_API_KEY']]
            except:
                pass
        
        # Fallback - no keys available
        if not keys:
            raise ValueError("No GOOGLE_API_KEYS configured. Set in environment or secrets.toml")
        
        return keys
    
    @property
    def GEMINI_API_KEY(self) -> str:
        """Get current Gemini API key (for backward compatibility)"""
        return self.GEMINI_API_KEYS[0]
    
    GEMINI_MODEL: str = "gemini-2.5-flash"
    
    # Directory Structure
    BASE_DIR: Path = Path("./knowledge_base")
    INPUT_DIR: Path = BASE_DIR / "input_files"
    IMAGES_OUTPUT_DIR: Path = BASE_DIR / "extracted_images"
    SCREENSHOT_CACHE_DIR: Path = BASE_DIR / "screenshot_cache"
    AUDIO_CACHE_DIR: Path = BASE_DIR / "audio_cache"
    CHAT_HISTORY_DIR: Path = BASE_DIR / "chat_sessions"
    
    # Data Files
    PROCESSED_CHUNKS_JSON: Path = BASE_DIR / "all_processed_chunks.json"
    KG_JSONL_PATH: Path = BASE_DIR / "knowledge_graph.jsonl"
    NODE_METADATA_PATH: Path = BASE_DIR / "node_metadata.json"
    FAISS_INDEX_PATH: Path = BASE_DIR / "unified_index.index"
    CHUNK_IDS_PATH: Path = BASE_DIR / "chunk_ids.npy"
    PARENT_CHILD_MAP_PATH: Path = BASE_DIR / "parent_child_map.json"
    
    # Model Configuration
    WHISPER_MODEL_SIZE: str = "base"
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Local HuggingFace model
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    SPACY_MODEL: str = "en_core_web_sm"
    
    # Processing Parameters
    MAX_CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    MAX_NEW_TOKENS: int = 1024
    TEMPERATURE: float = 0.3
    
    # Retrieval Parameters
    DEFAULT_TOP_K: int = 50
    FINAL_TOP_K: int = 15
    RERANK_CONFIDENCE_THRESHOLD: float = 0.1
    
    # Device Configuration
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    def __init__(self):
        self._create_directories()
    
    def _create_directories(self):
        for dir_path in [
            self.BASE_DIR,
            self.INPUT_DIR,
            self.IMAGES_OUTPUT_DIR,
            self.SCREENSHOT_CACHE_DIR,
            self.AUDIO_CACHE_DIR,
            self.CHAT_HISTORY_DIR
        ]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def get_api_key(self, service: str = "gemini") -> Optional[str]:
        if service == "gemini":
            return self.GEMINI_API_KEY
        return None


# Global settings instance
settings = Settings()
