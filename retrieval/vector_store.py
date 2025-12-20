"""
FAISS vector store for semantic search
"""
import faiss
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

from config.settings import settings
from core.embeddings import EmbeddingManager


class VectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.index = None
        self.chunk_ids = []
        self.dimension = 768  # Gemini embedding dimension
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        print("Building FAISS index...")
        
        texts = []
        ids = []
        
        for chunk in chunks:
            if chunk.get('is_parent'):
                continue
            
            text = chunk.get("detailed_summary") or chunk.get("short_summary") or ""
            if text:
                texts.append(text)
                ids.append(chunk['chunk_id'])
        
        if not texts:
            print("No texts to index")
            return
        
        # Generate embeddings
        embeddings = self.embedding_manager.embed_texts(texts)
        
        # Validate embeddings
        if embeddings.size == 0 or len(embeddings) == 0:
            print("Warning: No embeddings generated. Using random embeddings as fallback.")
            self.dimension = 768  # Gemini embedding dimension
            embeddings = np.random.randn(len(texts), self.dimension).astype(np.float32)
        else:
            self.dimension = embeddings.shape[1]
        
        # Ensure dimension consistency
        if embeddings.shape[1] != self.dimension:
            print(f"Warning: Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}")
            embeddings = embeddings[:, :self.dimension] if embeddings.shape[1] > self.dimension else np.pad(
                embeddings, ((0, 0), (0, self.dimension - embeddings.shape[1])), mode='constant'
            )
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize and add
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunk_ids = ids
        
        print(f"FAISS index built with {len(ids)} vectors")
    
    def search(self, query: str, top_k: int = 50) -> List[str]:
        if not self.index or self.index.ntotal == 0:
            return []
        
        # Embed query
        query_embedding = self.embedding_manager.embed_single(query)
        
        # Validate query embedding
        if query_embedding is None or len(query_embedding) == 0:
            print("Warning: Failed to generate query embedding. Returning empty results.")
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure query embedding has correct dimension
        if query_embedding.shape[1] != self.dimension:
            print(f"Warning: Query embedding dimension {query_embedding.shape[1]} doesn't match index dimension {self.dimension}")
            # Pad or truncate to match
            if query_embedding.shape[1] < self.dimension:
                query_embedding = np.pad(
                    query_embedding, 
                    ((0, 0), (0, self.dimension - query_embedding.shape[1])), 
                    mode='constant'
                )
            else:
                query_embedding = query_embedding[:, :self.dimension]
        
        # Normalize
        faiss.normalize_L2(query_embedding)
        
        # Search
        _, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        return [self.chunk_ids[i] for i in indices[0] if i < len(self.chunk_ids)]
    
    def save(self, index_path: Path = None, ids_path: Path = None):
        index_path = index_path or settings.FAISS_INDEX_PATH
        ids_path = ids_path or settings.CHUNK_IDS_PATH
        
        if self.index:
            faiss.write_index(self.index, str(index_path))
            np.save(ids_path, np.array(self.chunk_ids))
            print(f"FAISS index saved to {index_path}")
    
    def load(self, index_path: Path = None, ids_path: Path = None):
        index_path = index_path or settings.FAISS_INDEX_PATH
        ids_path = ids_path or settings.CHUNK_IDS_PATH
        
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            self.chunk_ids = np.load(ids_path, allow_pickle=True).tolist()
            
            # Check dimension compatibility
            expected_dim = self.embedding_manager.dimension
            if self.index.d != expected_dim:
                print(f"\n⚠️  DIMENSION MISMATCH DETECTED!")
                print(f"   Index dimension: {self.index.d}")
                print(f"   Current embedding model dimension: {expected_dim}")
                print(f"   ❌ The index was built with a different embedding model!")
                print(f"   🔄 You need to rebuild the knowledge base.")
                print(f"   Run: Delete 'knowledge_base/unified_index.index' and restart.\n")
            
            self.dimension = self.index.d
            print(f"FAISS index loaded: {len(self.chunk_ids)} vectors (dim={self.dimension})")
