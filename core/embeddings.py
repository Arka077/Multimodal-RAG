"""
Embedding generation and management
"""
import numpy as np
import torch
from typing import List
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

from config.settings import settings


class EmbeddingManager:
    """Manage text embeddings using local HuggingFace models"""
    
    def __init__(self):
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model directly to the target device to avoid meta tensor issues
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
        
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded on {device}. Embedding dimension: {self.dimension}")
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return np.array(embeddings, dtype=np.float32)
    
    def embed_single(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text)
        return np.array(embedding, dtype=np.float32)
