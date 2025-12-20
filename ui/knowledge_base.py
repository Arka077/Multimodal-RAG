"""
Knowledge base management and orchestration
"""
import json
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

from config.settings import settings
from models import GeminiClient, WhisperClient
from core import DocumentProcessor, KnowledgeGraphBuilder, EmbeddingManager
from retrieval import VectorStore, BM25Search


class KnowledgeBaseManager:
    """Manage the entire knowledge base"""
    
    def __init__(self, auto_clear: bool = False):
        try:
            # Auto-clear if requested
            if auto_clear:
                self._clear_all_data()
            
            # Clients
            self.gemini = GeminiClient()
            self.whisper = WhisperClient()
            
            # Core components
            self.doc_processor = DocumentProcessor(self.gemini, self.whisper)
            self.embedding_manager = EmbeddingManager()
            self.kg_builder = KnowledgeGraphBuilder()  # No longer needs Gemini client
            
            # Storage
            self.chunks = []
            self.chunk_lookup = {}
            self.parent_child_map = {}
            self.node_metadata = {}
            self.knowledge_graph = nx.DiGraph()
            
            # Retrieval components
            self.vector_store = VectorStore(self.embedding_manager)
            self.bm25_search = BM25Search()
            
            # Load existing data
            self.load_from_disk()
        except Exception as e:
            import traceback
            print(f"ERROR in KnowledgeBaseManager.__init__:  {e}")
            print(traceback.format_exc())
            raise
    
    def load_from_disk(self):
        print("Loading knowledge base from disk...")
        
        # Load chunks
        if settings.PROCESSED_CHUNKS_JSON.exists():
            with open(settings.PROCESSED_CHUNKS_JSON, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
                self.chunk_lookup = {c['chunk_id']: c for c in self.chunks}
            print(f"  - Loaded {len(self.chunks)} chunks")
            self._build_parent_child_map()
        
        # Load vector store
        if settings.FAISS_INDEX_PATH.exists():
            self.vector_store.load()
        
        # Load node metadata
        if settings.NODE_METADATA_PATH.exists():
            with open(settings.NODE_METADATA_PATH, "r", encoding="utf-8") as f:
                self.node_metadata = json.load(f)
            print(f"  - Loaded node metadata")
        
        # Load knowledge graph
        if settings.KG_JSONL_PATH.exists():
            with open(settings.KG_JSONL_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    triple = json.loads(line)
                    self.knowledge_graph.add_edge(
                        triple["subject_canonical"],
                        triple["object_canonical"],
                        label=triple["predicate"]
                    )
            print(f"  - Loaded KG: {self.knowledge_graph.number_of_nodes()} nodes")
        
        # Build BM25
        if self.chunks:
            self.bm25_search.build_index(self.chunks)
        
        print("Knowledge base loaded successfully")
    
    def _clear_all_data(self):
        import shutil
        
        print("🧹 Clearing knowledge base...")
        
        # Files to remove
        files_to_clear = [
            settings.PROCESSED_CHUNKS_JSON,
            settings.FAISS_INDEX_PATH,
            settings.CHUNK_IDS_PATH,
            settings.KG_JSONL_PATH,
            settings.NODE_METADATA_PATH,
        ]
        
        cleared_count = 0
        for file_path in files_to_clear:
            if file_path.exists():
                try:
                    file_path.unlink()
                    cleared_count += 1
                except Exception as e:
                    print(f"   ⚠️  Could not remove {file_path.name}: {e}")
        
        # Clear cache directories
        cache_dirs = [
            settings.IMAGES_OUTPUT_DIR,
            settings.AUDIO_CACHE_DIR,
            settings.SCREENSHOT_CACHE_DIR,
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"   ⚠️  Could not clear {cache_dir.name}: {e}")
        
        if cleared_count > 0:
            print(f"   ✅ Cleared {cleared_count} knowledge base files")
        else:
            print(f"   📦 Knowledge base already empty")
    
    def _build_parent_child_map(self):
        self.parent_child_map = defaultdict(list)
        
        for chunk in self.chunks:
            if not chunk.get('is_parent') and 'parent_id' in chunk:
                self.parent_child_map[chunk['parent_id']].append(chunk['chunk_id'])
    
    def add_new_files_and_rebuild(self, file_paths: List[str]) -> str:
        print("\n=== Processing new files ===")
        
        # Process files
        new_chunks = self.doc_processor.process_files(file_paths)
        
        if not new_chunks:
            return "No new data was processed."
        
        # Add to chunks
        self.chunks.extend(new_chunks)
        self.chunk_lookup.update({c['chunk_id']: c for c in new_chunks})
        
        print(f"Added {len(new_chunks)} new chunks. Total: {len(self.chunks)}")
        
        # Rebuild knowledge base
        print("\n=== Rebuilding knowledge base ===")
        self._build_parent_child_map()
        self._rebuild_knowledge_graph()
        self._rebuild_vector_store()
        self._rebuild_bm25()
        self._save_all()
        
        return f"Successfully added {len(new_chunks)} chunks and rebuilt knowledge base."
    
    def _rebuild_knowledge_graph(self):
        print("\n🕸️  Rebuilding knowledge graph...")
        self.knowledge_graph, self.node_metadata = self.kg_builder.build_knowledge_graph(
            self.chunks
        )
        print(f"   ✅ KG built: {self.knowledge_graph.number_of_nodes()} nodes, {self.knowledge_graph.number_of_edges()} edges")
    
    def _rebuild_vector_store(self):
        print("\n📊 Rebuilding vector store...")
        self.vector_store.build_index(self.chunks)
        print(f"   ✅ Vector index built: {self.vector_store.index.ntotal if self.vector_store.index else 0} vectors")
    
    def _rebuild_bm25(self):
        print("\n🔍 Rebuilding BM25 index...")
        self.bm25_search.build_index(self.chunks)
        print(f"   ✅ BM25 index built")
    
    def _save_all(self):
        print("\n💾 Saving all artifacts...")
        
        # Save chunks
        with open(settings.PROCESSED_CHUNKS_JSON, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)
        
        # Save node metadata
        with open(settings.NODE_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.node_metadata, f, indent=2)
        
        # Save vector store
        self.vector_store.save()
        
        print(f"   ✅ Saved to:")
        print(f"      - Chunks: {settings.PROCESSED_CHUNKS_JSON}")
        print(f"      - Index: {settings.FAISS_INDEX_PATH}")
        print(f"      - KG: {settings.KG_JSONL_PATH}")
    
    def get_chunk_preview_data(self, chunk_id: str) -> Dict[str, Any]:
        chunk = self.chunk_lookup.get(chunk_id)
        if not chunk:
            return None
        
        preview = chunk.copy()
        
        # Add parent context
        if 'parent_id' in chunk and not chunk.get('is_parent'):
            parent = self.chunk_lookup.get(chunk['parent_id'], {})
            preview['parent_summary'] = parent.get('short_summary', '')
            
            # Add siblings
            siblings = self.parent_child_map.get(chunk['parent_id'], [])
            preview['siblings'] = [
                {
                    'chunk_id': sid,
                    'summary': self.chunk_lookup.get(sid, {}).get('short_summary', '')[:100]
                }
                for sid in siblings if sid != chunk_id
            ]
        
        return preview
